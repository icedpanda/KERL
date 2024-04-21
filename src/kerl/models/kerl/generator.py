import torch
import torch.nn.functional as F
import torch.nn as nn


from .modeling_bart import BartEncoderWrapper, BartDecoderModel
from .recommender import KERLRecommender

FINE_TUNE_SET = {
    "layers.5",
    "layernorm_embedding",  # last layer norm
}


class KERLGenerator(nn.Module):
    def __init__(
            self,
            rec_model: KERLRecommender,
            kg_vocab_mask,
            max_response_len=20,
    ):
        super(KERLGenerator, self).__init__()

        self.rec_model = rec_model

        self.encoder = BartEncoderWrapper.from_pretrained("facebook/bart-base")
        self.encoder.resize_token_embeddings(self.rec_model.vocab_size)
        self.decoder = BartDecoderModel(self.rec_model.vocab_size)
        self.dim_size = self.decoder.model.config.d_model
        self.max_response_len = max_response_len + 3
        self.top_k = 3

        self.decoder_start_id = self.decoder.model.config.decoder_start_token_id
        self.start_token_id = self.decoder.model.config.forced_bos_token_id
        self.eos_id = self.decoder.model.config.eos_token_id
        self.pad_id = self.decoder.model.config.pad_token_id
        self.register_buffer("kg_vocab_mask", kg_vocab_mask)

        self._build_conv()
        self._freeze_encoder()
        self._init_weights()

    def _build_conv(self):
        # for BART decoder MHA
        # project kg embedding to decoder token hidden size
        self.kg_embed_proj = nn.Linear(self.rec_model.user_dim, self.dim_size)
        self.kg_embed_att_proj = nn.Linear(self.rec_model.user_dim, self.dim_size)
        # copy projection (context + kg)
        self.copy_proj = nn.Linear(self.dim_size * 2, self.dim_size)
        # kg vocab
        self.copy_lm_head = nn.Linear(self.dim_size, self.rec_model.vocab_size, bias=False)
        # vocab
        self.lm_head = nn.Linear(self.dim_size, self.rec_model.vocab_size, bias=False)

    def _freeze_encoder(self):
        for name, param in self.encoder.encoder.named_parameters():
            param.requires_grad = any((name.startswith(layer) for layer in FINE_TUNE_SET))

    def _init_weights(self):
        modules = [self.kg_embed_proj, self.kg_embed_att_proj, self.copy_proj, self.copy_lm_head, self.lm_head]
        for module in modules:
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, context_tokens, context_entities, labels=None, mode="train"):

        entity_embed, entity_mask, user_entity_preference = self.get_user_kg_embeddings(context_entities)
        # project entity embedding to decoder hidden size for MHA
        entity_embed = self.kg_embed_proj(entity_embed)
        user_entity_preference = self.kg_embed_att_proj(user_entity_preference)

        if mode != "test":
            return self._decode_forced(context_tokens, entity_embed, entity_mask, user_entity_preference, labels)
        else:
            return self._decode_greedy(context_tokens, entity_embed, entity_mask, user_entity_preference)

    # train/validation
    def _decode_forced(self, context_tokens, entity_embed, kg_mask, user_kg_att_embed, labels):
        encoder_outs = self.encoder(
            input_ids=context_tokens["input_ids"],
            attention_mask=context_tokens["attention_mask"]).last_hidden_state

        # decode label as decoder input
        dialog_latent = self.decoder(
            encoder_outputs=encoder_outs,
            encoder_attention_mask=context_tokens["attention_mask"],
            kg_hidden_states=entity_embed,
            kg_attention_mask=kg_mask,
            labels=labels).last_hidden_state

        # copy mechanism
        user_kg_att_embed = user_kg_att_embed.unsqueeze(1).expand(-1, dialog_latent.size(1), -1)
        copy_embed = torch.cat([user_kg_att_embed, dialog_latent], dim=-1)
        # bs, seq, dim -> bs, seq, vocab
        copy_logits = self.copy_lm_head(self.copy_proj(copy_embed))
        # copy_logits = F.linear(self.copy_proj(copy_embed), self.decoder.model.decoder.embed_tokens.weight)
        # only copy from kg vocab
        copy_logits = copy_logits * self.kg_vocab_mask.unsqueeze(0).unsqueeze(0)
        # vocab
        vocab_logits = self.lm_head(dialog_latent)
        # vocab_logits = F.linear(dialog_latent, self.decoder.model.decoder.embed_tokens.weight)
        sum_logits = vocab_logits + copy_logits
        preds = torch.argmax(sum_logits, dim=-1).long()
        return sum_logits, preds

    def _decode_greedy(self, context_tokens, entity_embed, kg_mask, user_kg_att_embed):

        batch_size = context_tokens["input_ids"].size(0)
        logits = []
        user_kg_att_embed = user_kg_att_embed.unsqueeze(1)  # bs, 1, dim
        encoder_outs = self.encoder(
            input_ids=context_tokens["input_ids"],
            attention_mask=context_tokens["attention_mask"],
        ).last_hidden_state
        # no label for decoder input, manually set as input
        input_ids = torch.ones((batch_size, 1), dtype=torch.long, device=encoder_outs.device) * self.decoder_start_id
        start_token = torch.ones((batch_size, 1), dtype=torch.long, device=encoder_outs.device) * self.start_token_id

        input_ids = torch.cat((input_ids, start_token), dim=-1)
        unfinished_sents = input_ids.new(batch_size).fill_(1)
        sent_lengths = input_ids.new(batch_size).fill_(self.max_response_len)
        for i in range(self.max_response_len):
            dialog_latent = self.decoder(
                decoder_input_ids=input_ids,
                encoder_outputs=encoder_outs,
                encoder_attention_mask=context_tokens["attention_mask"],
                kg_hidden_states=entity_embed,
                kg_attention_mask=kg_mask,
            ).last_hidden_state

            # next token prediction
            # last token of dialog_latent
            dialog_latent = dialog_latent[:, -1:, :]  # bs, dim
            # copy mechanism
            copy_embed = torch.cat((user_kg_att_embed, dialog_latent), dim=-1)
            copy_logits = self.copy_lm_head(self.copy_proj(copy_embed))
            # copy_logits = F.linear(self.copy_proj(copy_embed), self.decoder.model.decoder.embed_tokens.weight)
            copy_logits = copy_logits * self.kg_vocab_mask.unsqueeze(0).unsqueeze(0)
            # vocab
            vocab_logits = self.lm_head(dialog_latent)
            # vocab_logits = F.linear(dialog_latent, self.decoder.model.decoder.embed_tokens.weight)
            sum_logits = vocab_logits + copy_logits
            logits.append(sum_logits)

            # preds = torch.argmax(sum_logits, dim=-1).long()
            # input_ids = torch.cat([input_ids, preds], dim=-1)
            # if ((input_ids == self.eos_id).sum(dim=-1) > 0).sum().item() == batch_size * 2:
            #     break
            # apply top-k sampling
            top_k_probs, top_k_indices = torch.topk(F.softmax(sum_logits, dim=-1), self.top_k)
            top_k_probs, top_k_indices = top_k_probs.squeeze(1), top_k_indices.squeeze(1)
            sampled_index = torch.multinomial(top_k_probs, 1)
            preds = top_k_indices.gather(index=sampled_index, dim=-1).squeeze(1)

            # preds = torch.argmax(sum_logits, dim=-1).long().squeeze(1)
            tokens_to_add = preds * unfinished_sents + self.pad_id * (1 - unfinished_sents)
            input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
            eos_in_sents = tokens_to_add == self.eos_id
            is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
            sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, i)
            unfinished_sents.mul_((~eos_in_sents).long())

            if unfinished_sents.max() == 0:
                break

        return torch.cat(logits, dim=1), input_ids

    def get_user_kg_embeddings(self, entity_lists):
        user_rep_list = []
        entity_masks = []
        bsz, max_len = entity_lists.shape

        # the whole kg embedding
        # kg model is frozen
        kg_embed = self.rec_model.get_kg_embeddings(self.rec_model.edge_index, self.rec_model.edge_type, "eval")
        for entity_list in entity_lists:
            if entity_list is None:
                user_rep_list.append(
                    torch.zeros(max_len, self.rec_model.user_embedding_dim, device=entity_lists.device))
                # 0 for padding, 1 for valid
                entity_masks.append(torch.zeros(max_len, device=entity_lists.device))
            else:
                user_rep = kg_embed[entity_list]
                if self.rec_model.learn_pos_embed:
                    user_rep = self.rec_model.pos_encoder(user_rep, entity_list)
                    user_rep = self.rec_model.layer_norm(user_rep)
                user_rep_list.append(user_rep)
                entity_mask = self.rec_model.create_mask(entity_list)
                entity_masks.append(entity_mask)

        entity_rep = torch.stack(user_rep_list, dim=0)
        entity_mask = torch.stack(entity_masks, dim=0)
        user_rep = self.rec_model.kg_attention(entity_rep, entity_mask.unsqueeze(-1))
        return entity_rep, entity_mask, user_rep
