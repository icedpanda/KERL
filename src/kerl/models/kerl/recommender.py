import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from .modeling_bart import BartEncoderModel
from ..utils.graph import RGCN
from ..utils.modules import EntityEncoder
from ..utils.modules import GateLayer
from ..utils.modules import LearnedPositionalEncoder
from ..utils.modules import SelfAttentionModule
from ..utils.pretrain import KGLoss
from ..utils.tools import edge_to_pyg_format

FINE_TUNE_SET = {
    "layers.4",
    "layers.5",
    "layernorm_embedding",  # last layer norm
}


class KERLRecommender(nn.Module):
    def __init__(
            self,
            edge,
            entity_description,
            n_entity: int,
            n_relation: int,
            n_base: int,
            vocab_size: int,
            use_context: bool = False,
            text_pooling_type: str = "cls",
            text_pooling_head: int = 8,
            kg_embedding_dim: int = 128,
            learn_pos_embeds: bool = True,
            use_kg_loss: bool = False,
            kg_margin: float = 1.0,
            use_description: bool = False,
            description_length: int = 40,
            max_n_positions: int = 30,
    ):
        super(KERLRecommender, self).__init__()

        edge_index, edge_type = edge_to_pyg_format(edge)
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_type", edge_type)

        self.n_entity = n_entity
        self.n_relation = n_relation
        self.num_bases = n_base
        self.kg_embedding_dim = kg_embedding_dim
        self.user_dim = self.kg_embedding_dim * 2
        self.relation_dim = self.user_dim
        self.learn_pos_embed = learn_pos_embeds
        # text encoder
        self.use_context = use_context
        self.vocab_size = vocab_size
        self.text_pooling = text_pooling_type
        self.text_pooling_head = text_pooling_head
        self.use_description = use_description
        self.description_length = description_length

        # learnable position embedding for entity
        self.max_position = max_n_positions
        self.use_kg_loss = use_kg_loss
        # TODO: make this a hyperparameter not hard coded
        if self.use_kg_loss:
            self.kg_margin = kg_margin
            self.norm_flag = False
            self.kg_norm = 2

        self.rec_b = nn.Linear(self.user_dim, self.n_entity)

        self._init_text_module()
        self._init_kg()
        self._init_entity_embedding()

        # buffer for saving kg embeddings
        # need tensor shape to reload the model, otherwise will raise error
        self.register_buffer("saved_entity_embeds", torch.randn(self.n_entity, self.kg_embedding_dim))
        self.register_buffer("text_embeds",
                             torch.randn(self.n_entity, self.description_length, self.item_encoder.hidden_size))
        self.register_buffer("description_input_ids", entity_description.input_ids)
        self.register_buffer("description_mask", entity_description.attention_mask)

        logger.debug("Recommendation Module initialized")
        logger.debug(f"number entity: {self.n_entity}", )

    def _init_text_module(self):
        self.bart_encoder = BartEncoderModel(
            vocab_size=self.vocab_size,
            pooler_type=self.text_pooling,
            num_heads=self.text_pooling_head,
            out_dim=self.user_dim,
        )
        self.text_dim = self.bart_encoder.h_dim
        self.gate = GateLayer(self.user_dim)
        self._init_cl_heads()
        self._freeze_text_encoder()
        logger.debug("Conversation History Encoder initialized")

    def _freeze_text_encoder(self):
        # freeze bart encoder parameters but keep last 2 layer trainable
        for name, param in self.bart_encoder.model.encoder.named_parameters():
            # use string startswith to match layer name
            # match a list of layer names
            param.requires_grad = any((name.startswith(layer) for layer in FINE_TUNE_SET))

    def _init_kg(self):
        self.kg_encoder = RGCN(
            in_channels=self.kg_embedding_dim,
            out_channels=self.user_dim,
            num_relations=self.n_relation,
            num_bases=self.num_bases)
        logger.debug("KG Encoder initialized")
        self.pos_encoder = LearnedPositionalEncoder(dim=self.user_dim, max_len=self.max_position)
        self.layer_norm = nn.LayerNorm(self.user_dim)
        self.kg_attention = SelfAttentionModule(dim=self.user_dim, da=self.user_dim)
        logger.debug("KG Module initialized")

    def _init_cl_heads(self):
        self.kg_proj = nn.Sequential(
            nn.Linear(self.user_dim, self.user_dim),
            nn.ReLU(),
            nn.Linear(self.user_dim, self.user_dim), )
        self.context_proj = nn.Sequential(
            nn.Linear(self.user_dim, self.user_dim),
            nn.ReLU(),
            nn.Linear(self.user_dim, self.user_dim), )
        module_list = [self.kg_proj, self.context_proj]
        for module in module_list:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.01)

    def _init_entity_embedding(self):
        # entity embedding
        if not self.use_description:
            logger.debug("use random entity embedding")
            self.entity_embeds = nn.Embedding(self.n_entity, self.kg_embedding_dim, padding_idx=0)
            nn.init.xavier_uniform_(self.entity_embeds.weight)
        else:
            logger.debug("use entity description embedding")
            self.item_encoder = EntityEncoder(pooler_type=self.text_pooling, output_dim=self.kg_embedding_dim)
            for name, param in self.item_encoder.model.named_parameters():
                param.requires_grad = False
        if self.use_kg_loss:
            # relation embedding for kg loss
            self.r_embeds = nn.Embedding(self.n_relation, self.relation_dim)
            self.kg_loss = KGLoss(p_norm=self.kg_norm,
                                  margin=self.kg_margin,
                                  norm_flag=self.norm_flag,
                                  pretrain_type="transE")

            emb_range = nn.Parameter(torch.Tensor([self.kg_margin / self.kg_embedding_dim]), requires_grad=False)
            nn.init.uniform_(tensor=self.r_embeds.weight, a=-emb_range.item(), b=emb_range.item())
        logger.debug("Entity Embedding initialized")

    def forward(self, context_entities, context_tokens, mode="train"):
        kg_embeddings = self.get_kg_embeddings(self.edge_index, self.edge_type, mode)
        user_kg_rep, user_context_embed = self.encoder_user(
            context_entities,
            kg_embeddings,
            context_tokens,
        )
        if self.use_context:
            user_embed = self.gate(user_kg_rep, user_context_embed)
            return F.linear(user_embed, kg_embeddings, self.rec_b.bias)

        return F.linear(user_kg_rep, kg_embeddings, self.rec_b.bias)

    def get_proj_kg_context(self, context_entities, context_tokens, mode="train"):
        """
        get the projection of kg and context for contrastive learning
        """
        kg_embeddings = self.get_kg_embeddings(self.edge_index, self.edge_type, mode=mode)

        user_kg_rep, user_context_embed = self.encoder_user(
            context_entities,
            kg_embeddings,
            context_tokens,
        )
        proj_kg = self.kg_proj(user_kg_rep)
        proj_context = self.context_proj(user_context_embed)
        return proj_kg, proj_context

    def calc_kg_loss(self, pos_heads, rels, pos_tails, neg_head, neg_tail):
        # we don't need this for evaluation/test, hard coded for now
        kg_embed = self.get_kg_embeddings(self.edge_index, self.edge_type, "train")
        r_embeds = self.r_embeds(rels)
        p_score = self.kg_loss(kg_embed[pos_heads], r_embeds, kg_embed[pos_tails])
        # negative_tail
        nt_score = self.kg_loss(kg_embed[pos_heads.unsqueeze(1)], r_embeds.unsqueeze(1), kg_embed[neg_tail])
        # negative_head
        nh_score = self.kg_loss(kg_embed[neg_head], r_embeds.unsqueeze(1), kg_embed[pos_tails.unsqueeze(1)])
        return p_score, torch.cat([nt_score, nh_score], dim=1)

    def get_kg_embeddings(self, edge_index, edge_type, mode="train"):
        entity_embeds = self.compute_entity_embedding() if mode == "train" else self.saved_entity_embeds
        return self.kg_encoder(entity_embeds, edge_index, edge_type)

    def encoder_context(self, context_tokens):
        return self.bart_encoder(
            context_tokens["input_ids"],
            context_tokens["attention_mask"],
            mode="rec",
        )

    def encoder_user(self, entity_lists, kg_embeddings, context_tokens):
        entity_rep = self.get_entity_rep(entity_lists, kg_embeddings)
        entity_mask = self.create_mask(entity_lists)
        user_context_rep = None
        if self.use_context:
            user_context_token_rep = self.encoder_context(context_tokens)
            user_context_rep = self.bart_encoder.get_pooled_output(
                user_context_token_rep,
                context_tokens["attention_mask"])

        entity_rep = self.kg_attention(entity_rep, entity_mask.unsqueeze(-1))
        return entity_rep, user_context_rep

    def get_entity_rep(self, entity_lists, kg_embeddings):
        entity_rep = kg_embeddings[entity_lists]
        if self.learn_pos_embed:
            # entity embeds + position embeds
            entity_rep = self.pos_encoder(entity_rep, entity_lists)
        entity_rep = self.layer_norm(entity_rep)
        return entity_rep

    @staticmethod
    def create_mask(entity_lists):
        """
        create mask for self attention
        1 for valid, 0 for padding
        """
        mask = torch.zeros_like(entity_lists, device=entity_lists.device)
        # entity pad id is 0
        mask[entity_lists != 0] = 1.0
        # batch, n_entity, 1
        return mask

    def compute_entity_embedding(self):
        if not self.use_description:
            return self.entity_embeds.weight

        batch_size = 4096 * 2
        num_batches = math.ceil(self.description_input_ids.shape[0] / batch_size)
        entity_embeddings = []
        for i in range(num_batches):
            embeds = self.text_embeds[i * batch_size: (i + 1) * batch_size]
            attention_mask = self.description_mask[i * batch_size: (i + 1) * batch_size]
            entity_embeds = self.item_encoder.encoder_pooler(embeds, attention_mask)
            entity_embeddings.append(entity_embeds)
        return torch.cat(entity_embeddings, dim=0)

    def save_entity_embedding(self):
        self.saved_entity_embeds = self.compute_entity_embedding()

    def compute_text_embeds(self):
        """
        precompute text embeds for all entities since we freeze the entity text encoder
        :return:
        """
        batch_size = 4096 * 2
        num_batches = math.ceil(self.description_input_ids.shape[0] / batch_size)
        text_embeds = []
        for i in range(num_batches):
            input_ids = self.description_input_ids[i * batch_size: (i + 1) * batch_size]
            attention_mask = self.description_mask[i * batch_size: (i + 1) * batch_size]
            text_embeds.append(self.item_encoder(input_ids, attention_mask, pool=False))

        return torch.cat(text_embeds, dim=0)
