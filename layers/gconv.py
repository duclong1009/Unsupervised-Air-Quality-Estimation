import math
from typing import Optional, Tuple
import torch
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
from torch_geometric.nn.inits import zeros
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes


class GCN(nn.Module):
    def __init__(self, infea, outfea, act, bias=True):
        super(GCN, self).__init__()
        # define cac lop fc -> act 
        self.fc = nn.Linear(infea, outfea, bias=False)
        self.act = nn.ReLU() if act=='relu' else act 

        # init bias
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(outfea) )
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)
        
        # init weight 
        for m in self.modules():
            self.weights_init(m)
        
    def weights_init(self, m):
        # neu la lop fully connectedd
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight.data)

        if m.bias is not None:
            m.bias.data.fill_(0.0)
    
    def forward(self,seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(  torch.bmm(adj, torch.squeeze(adj, torch.squeeze(seq_fts, 0) )))
        else:
            out = torch.bmm(adj, seq_fts) 

        if self.bias is not None:
            out += self.bias 
            return self.act(out)


class GCNConv(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
    The adjacency matrix can include other values than :obj:`1` representing
    edge weights via the optional :obj:`edge_weight` tensor.

    Its node-wise formulation is given by:

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta} \sum_{j \in \mathcal{N}(v) \cup
        \{ i \}} \frac{e_{j,i}}{\sqrt{\hat{d}_j \hat{d}_i}} \mathbf{x}_j

    with :math:`\hat{d}_i = 1 + \sum_{j \in \mathcal{N}(i)} e_{j,i}`, where
    :math:`e_{j,i}` denotes the edge weight from source node :obj:`j` to target
    node :obj:`i` (default: :obj:`1.0`)

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        normalize (bool, optional): Whether to add self-loops and compute
            symmetric normalization coefficients on the fly.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(GCNConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.lin = Linear(in_channels, out_channels, bias=False,
                          weight_initializer='glorot')

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None


    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        x = self.lin(x)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        if self.bias is not None:
            out += self.bias

        return out


    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class DCN(MessagePassing):
    """An implementation of the Diffusion Convolution Layer.
    For details see: `"Diffusion Convolutional Recurrent Neural Network:
    Data-Driven Traffic Forecasting" <https://arxiv.org/abs/1707.01926>`_

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        K (int): Filter size :math:`K`.
        bias (bool, optional): If set to :obj:`False`, the layer
            will not learn an additive bias (default :obj:`True`).

    """

    def __init__(self, in_channels, out_channels, K, bias=True):
        super(DCN, self).__init__(aggr="add", flow="source_to_target")
        assert K > 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(torch.Tensor(2, K, in_channels, out_channels))

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.__reset_parameters()

    def __reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def forward(
          self,
          X: torch.FloatTensor,
          edge_index: torch.LongTensor,
          edge_weight: torch.FloatTensor,
        ) -> torch.FloatTensor:
        r"""Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph.

        Arg types:
            * **X** (PyTorch Float Tensor) - Node features.
            * **edge_index** (PyTorch Long Tensor) - Graph edge indices.
            * **edge_weight** (PyTorch Long Tensor, optional) - Edge weight vector.

        Return types:
            * **H** (PyTorch Float Tensor) - Hidden state matrix for all nodes.
        """
        adj_mat = to_dense_adj(edge_index, edge_attr=edge_weight)
        adj_mat = adj_mat.reshape(adj_mat.size(1), adj_mat.size(2))
        deg_out = torch.matmul(
            adj_mat, torch.ones(size=(adj_mat.size(0), 1)).to(X.device)
        )
        deg_out = deg_out.flatten()
        deg_in = torch.matmul(
            torch.ones(size=(1, adj_mat.size(0))).to(X.device), adj_mat
        )
        deg_in = deg_in.flatten()

        deg_out_inv = torch.reciprocal(deg_out)
        deg_in_inv = torch.reciprocal(deg_in)
        row, col = edge_index
        norm_out = deg_out_inv[row]
        norm_in = deg_in_inv[row]

        reverse_edge_index = adj_mat.transpose(0, 1)
        reverse_edge_index, vv = dense_to_sparse(reverse_edge_index)

        Tx_0 = X
        Tx_1 = X
        H = torch.matmul(Tx_0, (self.weight[0])[0]) + torch.matmul(
            Tx_0, (self.weight[1])[0]
        )

        if self.weight.size(1) > 1:
            Tx_1_o = self.propagate(edge_index, x=X, norm=norm_out, size=None)
            Tx_1_i = self.propagate(reverse_edge_index, x=X, norm=norm_in, size=None)
            H = (
                H
                + torch.matmul(Tx_1_o, (self.weight[0])[1])
                + torch.matmul(Tx_1_i, (self.weight[1])[1])
            )

        for k in range(2, self.weight.size(1)):
            Tx_2_o = self.propagate(edge_index, x=Tx_1_o, norm=norm_out, size=None)
            Tx_2_o = 2.0 * Tx_2_o - Tx_0
            Tx_2_i = self.propagate(
                reverse_edge_index, x=Tx_1_i, norm=norm_in, size=None
            )
            Tx_2_i = 2.0 * Tx_2_i - Tx_0
            H = (
                H
                + torch.matmul(Tx_2_o, (self.weight[0])[k])
                + torch.matmul(Tx_2_i, (self.weight[1])[k])
            )
            Tx_0, Tx_1_o, Tx_1_i = Tx_1, Tx_2_o, Tx_2_i

        if self.bias is not None:
            H += self.bias

        return H


@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    # type: (Tensor, OptTensor, Optional[int], bool, bool, Optional[int]) -> PairTensor  # noqa
    pass


@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    # type: (SparseTensor, OptTensor, Optional[int], bool, bool, Optional[int]) -> SparseTensor  # noqa
    pass


def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sparsesum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
