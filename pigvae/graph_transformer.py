import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Dropout, LayerNorm
from torch.nn.functional import softmax, relu
# from torch.nn.attention import SDPBackend
from collections import OrderedDict

"""
adapted from https://github.com/jadore801120/attention-is-all-you-need-pytorch
"""

class Transformer(nn.Module):
    def __init__(self, hidden_dim, num_heads, ppf_hidden_dim, num_layers, dropout = 0.1):
        super().__init__()
        self.num_layers = num_layers
        self.ppf_hidden_dim = ppf_hidden_dim #TBDeleted
        # self.embedding_layer = EmbeddingLayer(
        #     config.vocab_size, config.d_model, config.max_len
        # )
        self.blocks = nn.ModuleList(
            [Block(hidden_dim, num_heads, dropout) for _ in range(num_layers)]
        )

        # self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, x, mask):
        # output = self.embedding_layer(input_ids)

        for block in self.blocks:
            x = block(x, mask)

        output = x
        # output = self.head(output)
        return output



# class Transformer(torch.nn.Module):
#     def __init__(self, hidden_dim, k_dim, v_dim, num_heads, ppf_hidden_dim, num_layers):
#         super().__init__()
#         self.num_layers = num_layers
#         self.self_attn_layers = torch.nn.ModuleList([
#             SelfAttention(num_heads, hidden_dim, k_dim, v_dim)
#             for _ in range(num_layers)])
#         self.pff_layers = torch.nn.ModuleList([
#             PositionwiseFeedForward(hidden_dim, ppf_hidden_dim)
#             for _ in range(num_layers)])
#     #     self.init_weights()  # Apply custom weight initialization
        
#     # def init_weights(self):
#     #     for name, param in self.named_parameters():
#     #         if 'weight' in name and param.data.dim() == 2:
#     #             torch.nn.init.kaiming_uniform_(param)

#     def forward(self, x, mask):
#         for i in range(self.num_layers):
#             x = self.self_attn_layers[i](x, mask)
#             x = self.pff_layers[i](x)
#         return x
    
    
class Block(nn.Module):

    def __init__(
        self,
        hidden_dim,
        n_head,
        dropout
    ):
        super().__init__()
        self.attention_layer = SelfAttention(n_head, hidden_dim, dropout)
        self.feed_forward_layer = FeedForward(hidden_dim, dropout)

    def forward(self, x, attention_mask):
        out_attention = self.attention_layer(x, attention_mask)
        x = x + out_attention

        out_feed_forward = self.feed_forward_layer(x)
        x = x + out_feed_forward
        return x

    
def FeedForward(
    hidden_size,
    dropout=0.1
):
    return nn.Sequential(
        OrderedDict(
            [
                (
                    "ff_layernorm",
                    nn.LayerNorm(hidden_size)
                ),
                (
                    "pre_relu",
                    nn.Linear(
                        hidden_size,
                        4 * hidden_size,
                        bias=True,
                    ),
                ),
                ("relu", nn.ReLU()),
                (
                    "post_relu",
                    nn.Linear(
                        4 * hidden_size,
                        hidden_size,
                        bias=True,
                    ),
                ),
                ("dropout", nn.Dropout(dropout))
            ]
        )
    )
    
class ScaledDotProductWithEdgeAttention(torch.nn.Module):
    def __init__(self, k_dim, temperature, dropout=0.1):
        super().__init__()
        self.k_dim = k_dim
        self.temperature = temperature
        self.dropout = torch.nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):
        # q:  b x nh x nn x nn x dv
        # k:  b x nh x nn x nn x dv

        # k.T:  b x nh x nn x dv x nn
        # q x k.T --> b x nh x nn x nn x nn

        attn = torch.matmul(q, k.transpose(3, 4))
        attn = attn / self.temperature

        # attn: b x nh x nn x nn
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -2**15)

        attn = softmax(attn, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)  # output: b x nh x nn x nn x dv

        return output

class SelfAttention(torch.nn.Module):
    def __init__(self, n_head, hidden_dim, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.hidden_dim = hidden_dim
        scale = hidden_dim // n_head
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.input_projection = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.attention = ScaledDotProductWithEdgeAttention(
            k_dim=scale,
            temperature=scale ** 0.5
        )

    def forward(self, x, mask):
        # x: b x nn x nn x dv
        batch_size, num_nodes = x.size(0), x.size(1)
        x = self.layer_norm(x)
        projected = self.input_projection(x)
        
        device = x.device
        
        q_chunk, k_chunk, v_chunk = torch.chunk(projected, chunks=3, dim=-1)
        query = q_chunk.view(batch_size, num_nodes, num_nodes, self.n_head, -1).permute(0, 3, 1, 2, 4)
        key = k_chunk.view(batch_size, num_nodes, num_nodes, self.n_head, -1).permute(0, 3, 2, 1, 4)
        value = v_chunk.view(batch_size, num_nodes, num_nodes, self.n_head, -1).permute(0, 3, 2, 1, 4)
        
        attn_mask = mask.masked_fill(torch.eye(num_nodes, num_nodes, device=device).bool(), 0)
        attn_mask = attn_mask.unsqueeze(1).expand(-1, num_nodes, -1, -1)
        attn_mask = attn_mask * (torch.eye(
            num_nodes, num_nodes, device=device) == 0).bool().unsqueeze(0).unsqueeze(-2).expand(-1, -1, num_nodes, -1
                                                                                                )
        
        # with torch.nn.attention.sdpa_kernel(
        #     [
        #         SDPBackend.FLASH_ATTENTION,
        #         SDPBackend.EFFICIENT_ATTENTION,
        #         SDPBackend.MATH,
        #     ]
        # ):
        
        attention_output = scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                attn_mask=attn_mask,
                is_causal=False,
            )
        attention_output = attention_output.permute(0, 2, 3, 1, 4).contiguous()
        output = self.output_projection(attention_output.transpose(1, 2).flatten(-2))
        output = self.dropout(output)
        return output


class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, batch_size, num_nodes):
        x = self.pos_table[:, :num_nodes].clone().detach()
        x = x.expand(batch_size, -1, -1)
        return x

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_len):
        super(EmbeddingLayer, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_len, embed_dim)

    def forward(self, x):
        # x: (batch_size, seq_len)
        seq_len = x.size(1)
        positions = (
            torch.arange(seq_len, dtype=torch.long, device=x.device)
            .unsqueeze(0)
            .expand_as(x)
        )
        token_embeddings = self.token_embedding(x)
        position_embeddings = self.position_embedding(positions)
        embeddings = token_embeddings + position_embeddings
        return embeddings
    

# Efficient implementation equivalent to the following:
# https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros((*query.shape[:-2],  L, S), dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask[0].logical_not(), float("-inf"))
        else:
           attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)
    
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value

# import numpy as np
# import torch
# from torch.nn import Linear, Dropout, LayerNorm
# from torch.nn.functional import softmax, relu

# """
# adapted from https://github.com/jadore801120/attention-is-all-you-need-pytorch
# """


# class Transformer(torch.nn.Module):
#     def __init__(self, hidden_dim, k_dim, v_dim, num_heads, ppf_hidden_dim, num_layers):
#         super().__init__()
#         self.num_layers = num_layers
#         self.self_attn_layers = torch.nn.ModuleList([
#             SelfAttention(num_heads, hidden_dim, k_dim, v_dim)
#             for _ in range(num_layers)])
#         self.pff_layers = torch.nn.ModuleList([
#             PositionwiseFeedForward(hidden_dim, ppf_hidden_dim)
#             for _ in range(num_layers)])

#     def forward(self, x, mask):
#         for i in range(self.num_layers):
#             x = self.self_attn_layers[i](x, mask)
#             x = self.pff_layers[i](x)
#         return x


# class PositionwiseFeedForward(torch.nn.Module):

#     def __init__(self, d_in, d_hid, dropout=0.1):
#         super().__init__()
#         self.w_1 = Linear(d_in, d_hid)  # position-wise
#         self.w_2 = Linear(d_hid, d_in)  # position-wise
#         self.layer_norm = LayerNorm(d_in)
#         self.dropout = Dropout(dropout)

#     def forward(self, x):

#         residual = x

#         x = self.w_2(relu(self.w_1(x)))
#         x = self.dropout(x)
#         x += residual

#         x = self.layer_norm(x)

#         return x


# class ScaledDotProductWithEdgeAttention(torch.nn.Module):
#     def __init__(self, k_dim, temperature, dropout=0.1):
#         super().__init__()
#         self.k_dim = k_dim
#         self.temperature = temperature
#         self.dropout = torch.nn.Dropout(dropout)


#     def forward(self, q, k, v, mask=None):
#         # q:  b x nh x nn x nn x dv
#         # k:  b x nh x nn x nn x dv

#         # k.T:  b x nh x nn x dv x nn
#         # q x k.T --> b x nh x nn x nn x nn

#         attn = torch.matmul(q, k.transpose(3, 4))
#         attn = attn / self.temperature

#         # attn: b x nh x nn x nn
#         if mask is not None:
#             attn = attn.masked_fill(mask == 0, -2**15)

#         attn = softmax(attn, dim=-1)
#         attn = self.dropout(attn)
#         output = torch.matmul(attn, v)  # output: b x nh x nn x nn x dv

#         return output


# # TODO: add layer norm before attenion?
# class SelfAttention(torch.nn.Module):
#     def __init__(self, n_head, hidden_dim, k_dim, v_dim, dropout=0.1):
#         super().__init__()

#         self.n_head = n_head
#         self.q_dim = k_dim
#         self.k_dim = k_dim
#         self.v_dim = v_dim
#         self.hidden_dim = hidden_dim

#         self.w_qs = Linear(hidden_dim, n_head * self.q_dim, bias=False)
#         self.w_ks = Linear(hidden_dim, n_head * self.k_dim, bias=False)
#         self.w_vs = Linear(hidden_dim, n_head * v_dim, bias=False)
#         self.fc = Linear(n_head * v_dim, hidden_dim, bias=False)
#         self.attention = ScaledDotProductWithEdgeAttention(
#             k_dim=k_dim,
#             temperature=k_dim ** 0.5
#         )
#         self.dropout = Dropout(dropout)
#         self.layer_norm = LayerNorm(hidden_dim)

#     def forward(self, x, mask):
#         # x: b x nn x nn x dv

#         batch_size, num_nodes = x.size(0), x.size(1)
#         device = x.device

#         residual = x

#         # Pass through the pre-attention projection: b x lx x (n*dv)
#         # Separate different heads: b x nn x nn x nh x dv
#         q = self.w_qs(x).view(batch_size, num_nodes, num_nodes, self.n_head, self.q_dim)
#         k = self.w_ks(x).view(batch_size, num_nodes, num_nodes, self.n_head, self.k_dim)
#         v = self.w_vs(x).view(batch_size, num_nodes, num_nodes, self.n_head, self.v_dim)

#         # Transpose for attention dot product: b x nh x nn x nn x dv ; k and v edge features flip for block attention
#         q, k, v = q.permute(0, 3, 1, 2, 4), k.permute(0, 3, 2, 1, 4), v.permute(0, 3, 2, 1, 4)
#         # [bz, nh, nn1, nn2, dq]
        
#         print(f'Q: {q.shape}')
#         print(f'K: {k.shape}')
#         print(f'V: {v.shape}')

#         attn_mask = mask.masked_fill(torch.eye(num_nodes, num_nodes, device=device).bool(), 0)
#         attn_mask = attn_mask.unsqueeze(1).expand(-1, num_nodes, -1, -1)
#         attn_mask = attn_mask * (torch.eye(
#             num_nodes, num_nodes, device=device) == 0).bool().unsqueeze(0).unsqueeze(-2).expand(-1, -1, num_nodes, -1)
#         x = self.attention(q, k, v, mask=attn_mask.unsqueeze(1))  # unsqueeze For head axs broadcasting
        
#         print(f'M: {attn_mask.unsqueeze(1).shape}')
#         x = x.permute(0, 2, 3, 1, 4).contiguous()  # [bz, nn1, nn2, nh, dq]
#         x = x.view(batch_size, num_nodes, num_nodes, self.n_head * self.q_dim)
#         x = self.dropout(self.fc(x))
#         x += residual
#         x = self.layer_norm(x)

#         return x


# class PositionalEncoding(torch.nn.Module):

#     def __init__(self, d_hid, n_position=200):
#         super(PositionalEncoding, self).__init__()

#         # Not a parameter
#         self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

#     def _get_sinusoid_encoding_table(self, n_position, d_hid):
#         ''' Sinusoid position encoding table '''
#         # TODO: make it with torch instead of numpy

#         def get_position_angle_vec(position):
#             return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

#         sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
#         sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
#         sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

#         return torch.FloatTensor(sinusoid_table).unsqueeze(0)

#     def forward(self, batch_size, num_nodes):
#         x = self.pos_table[:, :num_nodes].clone().detach()
#         x = x.expand(batch_size, -1, -1)
#         return x
