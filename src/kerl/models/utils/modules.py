import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class SelfAttentionModule(nn.Module):
    """
    Self attention block. equation: 4 in KBRD paper.
    A structured self-attentive sentence embedding.:
    https://arxiv.org/abs/1703.03130
    """

    def __init__(self, dim: int, da: int):
        super(SelfAttentionModule, self).__init__()
        self.dim = dim
        self.da = da
        # INFO: same as the nn.Parameter in KBRD
        # nn.Linear equivalent to the matrix multiplication(y=Wx).
        # this is faster than iterating the list and pass per batch.
        self.weights_a = nn.Linear(self.dim, self.da, bias=False)
        self.weights_b = nn.Linear(self.da, 1, bias=False)
        self.mask_value = -1e9
        self.tanh = nn.Tanh()
        gain = nn.init.calculate_gain('tanh')
        nn.init.xavier_uniform_(self.weights_a.weight, gain=gain)
        nn.init.xavier_uniform_(self.weights_b.weight, gain=gain)

    def forward(self, inputs, mask=None):
        # use softmax as attention
        alpha = self.tanh(self.weights_a(inputs))
        alpha = self.weights_b(alpha)
        if mask is not None:
            alpha = alpha * mask
        alpha = F.softmax(alpha, dim=1)
        alpha = alpha.transpose(1, 2)
        return torch.matmul(alpha, inputs).squeeze(dim=1)


class AdditiveAttention(nn.Module):
    """AttentionPooling used to weight aggregate token embeddings into text embeddings
    Hierarchical Attention Networks for Document Classification
    https://aclanthology.org/N16-1174
    Arg:
        d_h: the last dimension of input
    """

    def __init__(self, d_h, hidden_size=768):
        super(AdditiveAttention, self).__init__()
        self.att_fc1 = nn.Linear(d_h, hidden_size)
        self.att_fc2 = nn.Linear(hidden_size, 1)
        self.att = nn.Tanh()
        gain = nn.init.calculate_gain('tanh')
        nn.init.xavier_uniform_(self.att_fc1.weight, gain=gain)
        nn.init.xavier_uniform_(self.att_fc2.weight, gain=gain)
        nn.init.constant_(self.att_fc1.bias, 0.01)
        nn.init.constant_(self.att_fc2.bias, 0.01)

    def forward(self, x, attn_mask=None):
        """
        Args:
            x: batch_size, token_seq, token_vector_dim
            attn_mask: batch_size, token_seq
        Returns:
            (shape) batch_size, text_vector_dim
        """
        bz = x.shape[0]
        e = self.att_fc1(x)
        e = self.att(e)
        alpha = self.att_fc2(e)

        alpha = torch.exp(alpha)
        if attn_mask is not None:
            alpha = alpha * attn_mask.unsqueeze(2)
        alpha = alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-8)

        x = torch.bmm(x.permute(0, 2, 1), alpha)
        x = torch.reshape(x, (bz, -1))  # batch_size
        return x


class LearnedPositionalEncoder(nn.Module):
    def __init__(self, dim, max_len=30, pad_idx=0):
        super(LearnedPositionalEncoder, self).__init__()
        self.pad_idx = pad_idx
        self.pos_embedding = nn.Embedding(num_embeddings=max_len, embedding_dim=dim, padding_idx=pad_idx)
        nn.init.xavier_uniform_(self.pos_embedding.weight)

    def forward(self, entity_embeds, entity_list=None):
        bsz, seq_len = entity_embeds.shape[:2]
        mask = entity_list.ne(self.pad_idx).int()
        positions = torch.cumsum(mask, dim=-1).long().type_as(mask) * mask.long() + self.pad_idx
        # batch_size, seq_len,
        # positions = torch.arange(seq_len, dtype=torch.long, device=entity_embeds.device).expand(bsz, -1)
        # pad idx not included in position embedding
        pos_embed = self.pos_embedding(positions)
        # pos_embed[pad_idx] = 0
        return entity_embeds + pos_embed


class GateLayer(nn.Module):
    def __init__(self, input_dim):
        super(GateLayer, self).__init__()
        self.fuse_layer = nn.Linear(input_dim * 2, input_dim)
        self.gate = nn.Linear(input_dim, 1)
        self.gate_act = nn.Sigmoid()
        nn.init.xavier_uniform_(self.fuse_layer.weight)
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.constant_(self.fuse_layer.bias, 0.01)
        nn.init.constant_(self.gate.bias, 0.01)

    def forward(self, input1, input2):
        features = torch.cat([input1, input2], dim=-1)
        gate = self.fuse_layer(features)
        gate = self.gate(gate)
        gate = self.gate_act(gate)
        return gate * input1 + (1 - gate) * input2


class EntityEncoder(nn.Module):
    def __init__(self, pooler_type="avg", output_dim=256):
        super().__init__()

        assert pooler_type in ["avg", "attention", "cls"], "pooler_type must be one of avg, attention, cls"
        # TODO: NOT HARD CODED model name
        self.model = AutoModel.from_pretrained("prajjwal1/bert-mini")
        self.pooler_type = pooler_type
        self.hidden_size = self.model.config.hidden_size
        self.dropout = nn.Dropout(p=0.2)
        # ffn
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, output_dim),
        )
        if self.pooler_type == "attention":
            # use same number of heads
            self.multi_head_attention = nn.MultiheadAttention(
                self.hidden_size,
                self.model.config.num_attention_heads,
                batch_first=True)
            self.attention_pooling_layer = AdditiveAttention(self.hidden_size, self.hidden_size)

        # init weights in ffn
        nn.init.xavier_uniform_(self.ffn[0].weight)
        nn.init.xavier_uniform_(self.ffn[2].weight)
        nn.init.constant_(self.ffn[0].bias, 0.01)
        nn.init.constant_(self.ffn[2].bias, 0.01)

    def forward(self, inputs, mask=None, pool=False):
        outs = self.model(inputs, attention_mask=mask).last_hidden_state
        return self.encoder_pooler(outs, mask) if pool else outs

    def encoder_pooler(self, hidden_states, attention_mask=None):
        """
        Pooling layer for encoder
        :param hidden_states: last hidden states from encoder
        :param attention_mask: attention mask for encoder
        :return:
        """
        hidden_states = self.dropout(hidden_states)
        if self.pooler_type == "avg":
            hidden_states = self.avg_pool(hidden_states, attention_mask)
        elif self.pooler_type == "attention":
            hidden_states = self.attention_pooling(hidden_states, attention_mask)
        else:
            hidden_states = self.cls_token_repr(hidden_states)
        # ffn
        hidden_states = self.ffn(hidden_states)
        return hidden_states

    @staticmethod
    def cls_token_repr(hidden_states):
        # original BART classification use eos token from decoder as cls token
        # here we use cls token from encoder
        return hidden_states[:, 0, :]

    @staticmethod
    def avg_pool(hidden_states, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        return torch.sum(hidden_states * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @staticmethod
    def max_pool(hidden_states, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        hidden_states[input_mask_expanded == 0] = -1e9
        return torch.max(hidden_states, 1)[0]

    def attention_pooling(self, hidden_states, attention_mask):
        # batch_size, hidden_size
        # torch multi-head attention use True for mask which means ignore. However, the attention mask from
        # transformers use True for mask which means keep. So we need to reverse the mask
        torch_attention_mask = torch.logical_not(attention_mask)
        mh_attention, _ = self.multi_head_attention(hidden_states, hidden_states, hidden_states, torch_attention_mask)
        return self.attention_pooling_layer(mh_attention, attention_mask)
