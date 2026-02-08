import torch.nn as nn
from typing import Union, Tuple, Optional

from torch_geometric.nn.inits import glorot
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import GATConv
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax


class SegEmbedding(nn.Module):
    def __init__(self, configs):
        super(SegEmbedding, self).__init__()

        self.emb_cls = nn.Embedding(configs.num_cls_token, configs.seg_cls_dim)
        self.emb_length = nn.Embedding(configs.num_length_token, configs.seg_length_dim)
        self.emb_radian = nn.Embedding(configs.num_radian_token, configs.seg_radian_dim)
        self.emb_loc = nn.Embedding(configs.num_loc_token, configs.seg_loc_dim)
        self.emb_size = configs.seg_cls_dim + configs.seg_length_dim + configs.seg_radian_dim + configs.seg_loc_dim * 2  # 112

    def forward(self, inputs):
        """
        inputs: [batch, max_seq_length, [token_cls, token_length, token_radian, token_start, token_ends]
        """
        return torch.cat((
            self.emb_cls(inputs[:, 0]),
            self.emb_length(inputs[:, 3]),
            self.emb_radian(inputs[:, 4]),
            self.emb_loc(inputs[:, 1]),
            self.emb_loc(inputs[:, 2])), dim=1)


class GAT(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_heads, dropput, num_layers=1, edge_dim=None):
        """
        When edge_dim is None, we don't consider the edge weights.
        """
        super().__init__()
        assert num_layers >= 1
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropput)

        if num_layers == 1:
            self.layers.append(GATEdgeConv(input_size, output_size, num_heads, concat=False, dropout=dropput, negative_slope=0.2, edge_dim=edge_dim))
        else:
            self.layers.append(GATEdgeConv(input_size, hidden_size, num_heads, dropout=dropput, negative_slope=0.2, edge_dim=edge_dim))
            for _ in range(num_layers - 2):
                self.layers.append(GATEdgeConv(hidden_size * num_heads, hidden_size, num_heads, dropout=dropput, negative_slope=0.2, edge_dim=edge_dim))
            self.layers.append(GATEdgeConv(num_heads * hidden_size, output_size, num_heads, concat=False, dropout=dropput, negative_slope=0.2, edge_dim=edge_dim))

    def forward(self, x, edge_index, edge_weights):
        for l in range(self.num_layers):
            x = self.layers[l](x, edge_index, edge_weights)

        return x


def add_self_loops_v2(edge_index, edge_weight: Optional[torch.Tensor] = None,
                      edge_attr: Optional[torch.Tensor] = None, edge_attr_reduce: str = "mean",
                      fill_value: float = 1., num_nodes: Optional[int] = None):
    r"""Extended method of torch_geometric.utils.add_self_loops that
    supports :attr:`edge_attr`."""
    N = maybe_num_nodes(edge_index, num_nodes)

    loop_index = torch.arange(0, N, dtype=torch.long, device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)

    if edge_weight is not None:
        assert edge_weight.numel() == edge_index.size(1)
        loop_weight = edge_weight.new_full((N,), fill_value)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

    if edge_attr is not None:
        assert edge_attr.size(0) == edge_index.size(1)
        if edge_attr_reduce != "fill":
            loop_attr = scatter(edge_attr, edge_index[0], dim=0, dim_size=N,
                                reduce=edge_attr_reduce)
        else:
            loop_attr = edge_attr.new_full((N, edge_attr.size(1)), fill_value)
        edge_attr = torch.cat([edge_attr, loop_attr], dim=0)

    edge_index = torch.cat([edge_index, loop_index], dim=1)

    return edge_index, edge_weight, edge_attr


class GATEdgeConv(GATConv):
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.0,
                 add_self_loops: bool = True, bias: bool = True,
                 edge_dim: int = None, edge_attr_reduce_for_self_loops: str = "mean",
                 edge_attr_fill_value: float = 1.,
                 **kwargs):

        assert edge_attr_reduce_for_self_loops in ["mean", "sum", "add", "mul", "min", "max", "fill"]
        self.edge_dim = edge_dim
        self.edge_attr_reduce_for_self_loops = edge_attr_reduce_for_self_loops
        self.edge_attr_fill_value = edge_attr_fill_value

        super().__init__(in_channels, out_channels, heads, concat,
                         negative_slope, dropout, add_self_loops, bias, **kwargs)
        self.edge_dim = edge_dim  # TODO: a weird error, after calling super().__init__(), self.edge_dim will be "True" instead of "None"

        if edge_dim is not None:
            self.lin_e = nn.Linear(edge_dim, heads * out_channels)
            self.att_e = Parameter(torch.Tensor(1, heads, out_channels))
        else:
            self.lin_e = None
            self.register_parameter('att_e', None)

        self.reset_parameters_e()

    def reset_parameters_e(self):
        if self.edge_dim is not None:
            glorot(self.lin_e.weight)
            glorot(self.att_e)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_attr: OptTensor = None,
                size: Size = None, return_attention_weights=None):

        """
        # type: (Union[Tensor, OptPairTensor], Tensor, OptTensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, OptTensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], Tensor, OptTensor, Size, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, OptTensor, Size, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        alpha_l: OptTensor = None
        alpha_r: OptTensor = None
        alpha_e: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = x_r = self.lin_src(x).view(-1, H, C)
            alpha_l = (x_l * self.att_src).sum(dim=-1)
            alpha_r = (x_r * self.att_dst).sum(dim=-1)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = self.lin_src(x_l).view(-1, H, C)
            alpha_l = (x_l * self.att_src).sum(dim=-1)
            if x_r is not None:
                x_r = self.lin_dst(x_r).view(-1, H, C)
                alpha_r = (x_r * self.att_dst).sum(dim=-1)

        assert x_l is not None
        assert alpha_l is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                if size is not None:
                    num_nodes = min(size[0], size[1])
                edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
                edge_index, _, edge_attr = add_self_loops_v2(
                    edge_index, edge_attr=edge_attr,
                    edge_attr_reduce=self.edge_attr_reduce_for_self_loops,
                    fill_value=self.edge_attr_fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                assert edge_attr is None, \
                    "Using `edge_attr` not supported for SparseTensor `edge_index`."
                edge_index = set_diag(edge_index)

        if edge_attr is not None:
            if len(edge_attr.shape) == 1:
                edge_attr = edge_attr.unsqueeze(-1)
            edge_attr = self.lin_e(edge_attr).view(-1, H, C)
            alpha_e = (edge_attr * self.att_e).sum(dim=-1)

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor, alpha_e: OptTensor)
        out = self.propagate(edge_index, x=(x_l, x_r),
                             alpha=(alpha_l, alpha_r), alpha_e=alpha_e,
                             size=size)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                alpha_e: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = alpha if alpha_e is None else alpha + alpha_e
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)



