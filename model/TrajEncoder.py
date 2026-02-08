import torch
import torch.nn as nn
from model.utils_model import MLP, clones, mask_padding
import math
from torch.autograd import Variable
import torch.nn.functional as F


class STEncoder(nn.Module):
    def __init__(self, embed_S, embed_T, embed_P, layer, embedding_size, hidden_size, output_size, num_layers, dropout):
        super(STEncoder, self).__init__()
        self.embed_S = embed_S
        self.embed_T = embed_T
        self.embed_P = embed_P
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.param_S = nn.Sequential(nn.Linear(embed_S.embed_size, hidden_size, bias=False),
                                     MLP(hidden_size, self.dropout))
        self.param_T = nn.Sequential(nn.Linear(embed_T.date2vec_size, hidden_size, bias=False),
                                     MLP(hidden_size, self.dropout))

        self.layers = clones(layer, num_layers)
        self.norm = LayerNorm(layer.size)

    def forward(self, s, t, rs, lengths, segmentwise_weight=None, mask=None):
        if mask is None:
            mask = mask_padding(lengths)
            mask = ~mask
        if segmentwise_weight is not None:
            target_size = (s.shape[0], lengths.max(), lengths.max())
            padded_tensor = torch.zeros(target_size)
            for i in range(s.shape[0]):
                original_tensor = torch.tensor(segmentwise_weight[i])
                original_size = original_tensor.size()
                padded_tensor[i, :original_size[0], :original_size[1]] = original_tensor
            segmentwise_weight = padded_tensor
            segmentwise_weight = segmentwise_weight.cuda()
        # Version 1
        x = self.param_S(self.embed_S(rs, s, lengths)) + self.param_T(self.embed_T(t, lengths))
        x = self.embed_P(x)
        for layer in self.layers:
            x = layer(x, mask, segmentwise_weight)
        x = self.norm(x)

        # Average pooling
        # (num_batch, seq_len, hidden_size) -> (num_batch, hidden_size)

        input_mask_expanded = mask.unsqueeze(-1).expand(x.size()).float()
        sum_embeddings = torch.sum(x * input_mask_expanded, 1)

        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        return sum_embeddings / sum_mask


class EncoderLayer(nn.Module):
    # Encoder is made up of self-attn and position-wise fully connected feed forward network
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask, segmentwise_weight=None):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask, segmentwise_weight))
        return self.sublayer[1](x, self.feed_forward)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # Apply residual connection to any sublayer with the same size
        return self.norm(x + self.dropout(sublayer(x)))


def attention(query, key, value, mask=None, dropout=None, extra_weights=None):
    """
    Compute 'Scaled Dot Product Attention'
    :param query:
    :param key:
    :param value:
    :param mask:
    :param dropout:
    :return:
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if extra_weights is not None:
        scores += extra_weights
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, extra_weights=None, weights_dim=None):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.extra_weights = extra_weights
        self.weights_dim = weights_dim
        if extra_weights:
            self.weights_param_l = nn.Linear(1, self.weights_dim)
            self.weights_param_r = nn.Linear(self.weights_dim, 1)

    def forward(self, query, key, value, mask=None, segment_weights=None):
        if mask is not None:
            #(num_batch, seq_len) -> (num_batch, num_head, seq_len, seq_len)
            mask = mask.unsqueeze(1).unsqueeze(-1)
        if self.extra_weights:
            assert segment_weights is not None
            segment_weights = 1.0 / torch.log(torch.exp(torch.tensor(1.0).cuda()) + segment_weights)
            segment_weights = self.weights_param_r(F.leaky_relu(self.weights_param_l(segment_weights.unsqueeze(-1)), negative_slope=0.2)).squeeze(-1)
            segment_weights = segment_weights.unsqueeze(1)
        else:
            segment_weights = None

        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout, extra_weights=segment_weights)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        del query
        del key
        del value

        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=2000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = d_model
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)








