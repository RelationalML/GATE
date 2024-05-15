from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

#from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import (
    Adj,
    OptTensor,
    PairTensor,
    SparseTensor,
    #torch_sparse,
)
from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
    is_sparse,
)

import warnings
warnings.filterwarnings("ignore")
# import shutup
# shutup.please()

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures,LargestConnectedComponents
import networkx as nx
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.utils import segregate_self_loops,dense_to_sparse,\
index_to_mask,get_laplacian,erdos_renyi_graph,to_networkx
from torchmetrics.functional import pairwise_cosine_similarity
from torchmetrics import AUROC
import torch
from torch.nn import Linear
import torch.nn.functional as F 
#from torch_geometric.nn import GCNConv,GATv2Conv
import numpy as np
import matplotlib.pyplot as plt
import copy
import json
import pprint
import pickle
from scipy import stats
import pandas as pd
import os.path
from math import floor,ceil
from itertools import product
import scipy.sparse as sp
from torch.distributions.multivariate_normal import MultivariateNormal
from tsne_torch import TorchTSNE as TSNE
from sklearn.cluster import KMeans
import heapq
from scipy import stats
from msgpass import MessagePassing
from torchmetrics.functional.pairwise.helpers import _check_input

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.loader import NeighborSampler,GraphSAINTRandomWalkSampler, NeighborLoader

device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def getDataOGB(datasetName,numLayers=0):
    if datasetName in ['ogbn-arxiv','ogbn-products']:
        dataset = PygNodePropPredDataset(name=datasetName, transform=T.ToUndirected())
        data = dataset[0]
        print(data)
        split_idx = dataset.get_idx_split()       
        train_idx = split_idx['train']
        valid_idx = split_idx['valid']
        test_idx = split_idx['test']       
        data_loader = GraphSAINTRandomWalkSampler(data, batch_size=20000,
                                            walk_length=3,
                                            num_steps=30,
                                            sample_coverage=0)
        
        test_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
                                        batch_size=4096, shuffle=False, num_workers=2)
        
        # subgraph_loader = NeighborLoader(data, input_nodes=None, num_neighbors=[-1],
        #                            batch_size=4096, shuffle=False,
        #                            num_workers=os.cpu_count() - 2)
    elif datasetName in ['ogbn-mag']:
        dataset = PygNodePropPredDataset(name='ogbn-mag')
        rel_data = dataset[0]
        # We are only interested in paper <-> paper relations.
        data = Data(
            x=rel_data.x_dict['paper'],
            edge_index=rel_data.edge_index_dict[('paper', 'cites', 'paper')],
            y=rel_data.y_dict['paper'])
        data = T.ToUndirected()(data)
        data = data
        split_idx = {k: v['paper'] for k,v in dataset.get_idx_split().items()}
        train_idx = split_idx['train']
        valid_idx = split_idx['valid']
        test_idx = split_idx['test']       
        
        data_loader = NeighborSampler(data.edge_index, node_idx=train_idx, sizes= numLayers * [10],
                                       batch_size=20000, shuffle=False, num_workers=6)


        test_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
                                        batch_size=4096, shuffle=False, num_workers=6)
        

    return data,data.x.shape[1],len(torch.unique(data.y)),data_loader,train_idx,valid_idx,test_idx,test_loader,split_idx

class GATv2Conv(MessagePassing):



    r"""The GATv2 operator from the `"How Attentive are Graph Attention
    Networks?" <https://arxiv.org/abs/2105.14491>`_ paper, which fixes the
    static attention problem of the standard
    :class:`~torch_geometric.conv.GATConv` layer.
    Since the linear layers in the standard GAT are applied right after each
    other, the ranking of attended nodes is unconditioned on the query node.
    In contrast, in :class:`GATv2`, every node can attend to any other node.

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(\mathbf{\Theta}
        [\mathbf{x}_i \, \Vert \, \mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(\mathbf{\Theta}
        [\mathbf{x}_i \, \Vert \, \mathbf{x}_k]
        \right)\right)}.

    If the graph has multi-dimensional edge features :math:`\mathbf{e}_{i,j}`,
    the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(\mathbf{\Theta}
        [\mathbf{x}_i \, \Vert \, \mathbf{x}_j \, \Vert \, \mathbf{e}_{i,j}]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(\mathbf{\Theta}
        [\mathbf{x}_i \, \Vert \, \mathbf{x}_k \, \Vert \, \mathbf{e}_{i,k}]
        \right)\right)}.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). (default: :obj:`None`)
        fill_value (float or torch.Tensor or str, optional): The way to
            generate edge features of self-loops
            (in case :obj:`edge_dim != None`).
            If given as :obj:`float` or :class:`torch.Tensor`, edge features of
            self-loops will be directly given by :obj:`fill_value`.
            If given as :obj:`str`, edge features of self-loops are computed by
            aggregating all features of edges that point to the specific node,
            according to a reduce operation. (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"min"`, :obj:`"max"`, :obj:`"mul"`). (default: :obj:`"mean"`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        share_weights (bool, optional): If set to :obj:`True`, the same matrix
            will be applied to the source and the target node of every edge.
            (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, H * F_{out})` or
          :math:`((|\mathcal{V}_t|, H * F_{out})` if bipartite.
          If :obj:`return_attention_weights=True`, then
          :math:`((|\mathcal{V}|, H * F_{out}),
          ((2, |\mathcal{E}|), (|\mathcal{E}|, H)))`
          or :math:`((|\mathcal{V_t}|, H * F_{out}), ((2, |\mathcal{E}|),
          (|\mathcal{E}|, H)))` if bipartite
    """
    _alpha: OptTensor

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_l.reset_parameters()
        if self.lin_r is not None:
            self.lin_r.reset_parameters()
        self.lin_lAP.reset_parameters()
        self.lin_rAP.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        glorot(self.att)
        glorot(self.att2)
        zeros(self.bias)

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        share_weights: bool = False,
        attParamSharing = True,
        linWghtSharing = True,
        hasOmega = False,
        omegaInitVal : float = 1.0,
        defaultInit : bool = False,
        linAggrSharing : bool = False, #change default later to True
        alphasActivation : str = 'relu',
        **kwargs,
    ):
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.share_weights = share_weights
        self.attParamSharing = attParamSharing
        self.linWghtSharing = linWghtSharing
        self.hasOmega = hasOmega
        self.linAggrSharing = linAggrSharing
        self.alphasActivation = alphasActivation

        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, heads * out_channels, bias=bias)
                                #weight_initializer='glorot')
            
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(in_channels, heads * out_channels,
                                        bias=bias)#, weight_initializer='glorot')
            if linAggrSharing:
                self.lin_s =  self.lin_l
            else:
                self.lin_s = Linear(in_channels, heads * out_channels, bias=bias)

        else: #not usually the case.. not sure what case this is.. different input sizes for node self and nbts to compute alphas

            self.lin_l = Linear(in_channels[0], heads * out_channels,
                                bias=bias)#, weight_initializer='glorot')
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(in_channels[1], heads * out_channels,
                                    bias=bias)#, weight_initializer='glorot')

        self.att = Parameter(torch.empty(1, heads, out_channels))
        #self.omega = Parameter(torch.ones(heads * out_channels,))
        if self.attParamSharing:
            self.att2 = self.att
        else:
            self.att2 = Parameter(torch.empty(1, heads, out_channels))

        if linWghtSharing:
           self.lin_lAP = self.lin_l
           self.lin_rAP = self.lin_r
        else:
            self.lin_lAP = Linear(in_channels, heads * out_channels, bias=bias)
            if share_weights:
                self.lin_rAP = self.lin_lAP
            else:
                self.lin_rAP = Linear(in_channels, heads * out_channels, bias=bias)
            
        if self.hasOmega:
            self.omega = Parameter(torch.zeros(heads * out_channels,))
            self.omega.data.fill_(omegaInitVal)

        #self.signParam = Parameter(torch.zeros(heads * out_channels,))

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False),
                                   #weight_initializer='glorot')
        else:
            self.lin_edge = None

        if bias and concat:
            self.bias = Parameter(torch.empty(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        if defaultInit:
            self.reset_parameters()

    

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None,
                return_attention_weights: bool = None):
        # type: (Union[Tensor, PairTensor], Tensor, OptTensor, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], Tensor, OptTensor, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""Runs the forward pass of the module.

        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        x_s: OptTensor = None
        x_lAP: OptTensor = None
        x_rAP: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2
            x_l = self.lin_l(x).view(-1, H, C)
            x_s = self.lin_s(x).view(-1, H, C)
            if self.linWghtSharing:
                if self.share_weights :
                    x_r = x_l
                else:
                    x_r = self.lin_r(x).view(-1, H, C)
            if self.linWghtSharing:
                x_lAP = x_l
                x_rAP = x_r
            else:
                x_lAP = self.lin_lAP(x).view(-1, H, C)
                if self.share_weights:
                    x_rAP = x_lAP
                else:
                    x_rAP = self.lin_rAP(x).view(-1, H, C)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2 
            x_s = self.lin_s(x_l).view(-1, H, C)   
            x_l = self.lin_l(x_l).view(-1, H, C)   
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)

            if self.linWghtSharing:
                x_lAP = x_l
                x_rAP = x_r
            else:
                x_lAP = self.lin_lAP(x[0]).view(-1, H, C)
                if self.share_weights:
                    x_rAP = x_lAP
                else:
                    x_rAP = self.lin_rAP(x[1]).view(-1, H, C)

        #added for attn perceptron weight sharing
        # if self.linWghtSharing:
        #     x_lAP = x_l
        #     x_rAP = x_r
        # else:
        #     x_lAP = self.lin_lAP(x).view(-1, H, C)
        #     if self.share_weights:
        #         x_rAP = x_lAP
        #     else:
        #         x_rAP = self.lin_rAP(x).view(-1, H, C)


        # print(x_l.shape)
        # print(x_lAP.shape)
        # print(x_r.shape)
        # print(x_rAP.shape)
        assert x_l is not None
        assert x_s is not None
        #assert x_r is not None #can be none if linWghtSharing==False
        assert x_lAP is not None
        assert x_rAP is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    print('Not implemented :( ')
                    #edge_index = torch_sparse.set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        # print('X_L',x_l)
        # print('X_R',x_r)
        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=(x_l, x_r), edge_attr=edge_attr,
                             size=None,edgeIndex = edge_index, 
                             xAP=(x_lAP,x_rAP), x_s=x_s) #x_l corresponds to x_j in 

        alpha = self._alpha
        assert alpha is not None
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                if is_torch_sparse_tensor(edge_index):
                    # TODO TorchScript requires to return a tuple
                    adj = 0 #set_sparse_value(edge_index, alpha) ######CHANGEDDDD
                    return out, (adj, alpha)
                else:
                    return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    
    def message(self, x_j: Tensor, x_i: Tensor, edge_attr: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int],edgeIndex: OptTensor, 
                xAP_lAP: Tensor, xAP_rAP: Tensor, x_s: Tensor) -> Tensor: #, x_s: Tensor
        
        # print(x_j.shape)
        # print(x_s.shape)
        # print('x_j==xAP_s',torch.equal(x_j,x_s))
        # print('x_j==xAP_lAP',torch.equal(x_j,xAP_lAP))
        # print('x_i==xAP_rAP',torch.equal(x_i,xAP_rAP))
        # print('x_i==x_j',torch.equal(x_i,x_j))

        # input("Press Enter to continue...") 
        #x = x_i + x_j
        x = xAP_lAP + xAP_rAP
        
        #sign = torch.sign((F.tanh(x)*self.signParam).sum(dim=-1))

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            x = x + edge_attr

        #Replaced laeky_relu by relu so -ve and +ve values of attn parameters a1 and a2, respectively, correspond 
        #-ve and +ve values of e_{uv} and e_{vv}
        if self.attParamSharing:
            x = F.leaky_relu(x, self.negative_slope)
        else:
            if self.alphasActivation == 'relu':
                x = F.relu(x)#, self.negative_slope)
            elif self.alphasActivation == 'leaky_relu':
                x = F.leaky_relu(x,self.negative_slope)


        alpha1 = (x * self.att).sum(dim=-1)
        ijNotEq = torch.tensor(edgeIndex[0]!=edgeIndex[1],dtype=torch.float).unsqueeze(-1)
        ijEq = torch.tensor(edgeIndex[0]==edgeIndex[1],dtype=torch.float).unsqueeze(-1)#.unsqueeze(-1)
        alpha1 = alpha1*ijNotEq
        alpha2 = (x * self.att2).sum(dim=-1)
        alpha2 = alpha2*ijEq
        alpha = alpha1+alpha2
        
        # print('alpha1:',alpha1.unsqueeze(-1).shape)
        # print('alpha2:',alpha2.shape)
        # print('ijNotEq:',ijNotEq.shape)
        # print('ijEq:',ijEq.shape)
        


        #print('e: ',alpha.shape)
        alpha = softmax(alpha, index, ptr, size_i)
        
        # print('ALPHA SHAPE: ',alpha.shape)
        # print('ALPHA SHAPE unsqueeze: ',alpha.unsqueeze(-1).shape)
        # print('NODE J SHAPE: ' ,x_j.shape)
        # print('MESSAGE J SHAPE: ' ,(x_j * alpha.unsqueeze(-1)  * self.omega).shape)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        #x_jP = x_j
        #x_j = (x_j*ijNotEq.unsqueeze(-1)) + (x_s*ijEq.unsqueeze(-1))

        # print('x_j==x_jP',torch.equal(x_j,x_jP))

        # input('Enterrrr...')


        if self.hasOmega:
            alpha = alpha.unsqueeze(-1)
            ijEq = ijEq.unsqueeze(-1)
            return  x_j*(ijEq-(self.omega*(ijEq-alpha)))
        else:
            # print(edgeIndex.shape)
            # print(x_j.shape)
            # print(alpha.shape)
            #input('Enter......')
            return  x_j * alpha.unsqueeze(-1) 
        #(f-(o*(f-a)))*b
        #

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
    
    def printOmega(self):
        print(self.omega)

def sq_euclidean_dist(x,y,zero_diagonal=True):
        x, y, zero_diagonal = _check_input(x, y, zero_diagonal)
        # upcast to float64 to prevent precision issues
        _orig_dtype = x.dtype
        x = x.to(torch.float64)
        y = y.to(torch.float64)
        x_norm = (x * x).sum(dim=1, keepdim=True)
        y_norm = (y * y).sum(dim=1)
        distance = (x_norm + y_norm - 2 * x.mm(y.T)).to(_orig_dtype)
        if zero_diagonal:
            distance.fill_diagonal_(0)
        return distance#.sqrt()

class GATv2(torch.nn.Module):
    def __init__(self, numLayers, dims, heads, concat, weightSharing, selfLoops = True, 
                 attnDropout=0,bias=False, activation='relu',useIdMap=False, useResLin=False,
                 attParamSharing=True,linWghtSharing=True,linLastLayer=False,hasOmega=False,
                 omegaInitVal=1,defaultInit=False,linAggrSharing=True,alphasActivation='relu'):
        super().__init__()
        self.numLayers = numLayers
        self.heads = heads
        self.weightSharing = weightSharing
        self.selfLoops = selfLoops
        self.dropout = attnDropout
        self.bias=bias
        if activation=='relu':
            self.activation = F.relu
        elif activation=='elu':
            self.activation = F.elu # as used previously
        self.useIdMap = useIdMap
        self.useResLin = useResLin
        self.linLastLayer = linLastLayer
        if self.linLastLayer:
            self.layers = torch.nn.ModuleList(
            [GATv2Conv(dims[j]*heads[j],dims[j+1],bias=bias,
                       heads=heads[j+1],concat=concat[j],add_self_loops=selfLoops,
                       share_weights=weightSharing,dropout=attnDropout,
                       attParamSharing=attParamSharing,linWghtSharing=linWghtSharing,
                       hasOmega=hasOmega,omegaInitVal=omegaInitVal,defaultInit=defaultInit,
                       linAggrSharing=linAggrSharing,alphasActivation=alphasActivation) 
                       for j in range(self.numLayers-1)]
            +[Linear(dims[numLayers-1],dims[numLayers],bias=self.bias)])
        else:
            self.layers = torch.nn.ModuleList(
            [GATv2Conv(dims[j]*heads[j],dims[j+1],bias=bias,
                       heads=heads[j+1],concat=concat[j],add_self_loops=selfLoops,
                       share_weights=weightSharing,dropout=attnDropout,
                       attParamSharing=attParamSharing,linWghtSharing=linWghtSharing,
                       hasOmega=hasOmega,omegaInitVal=omegaInitVal,defaultInit=defaultInit,
                       linAggrSharing=linAggrSharing,alphasActivation=alphasActivation) 
                       for j in range(self.numLayers)])
        
        if self.useIdMap:
            self.residual = torch.nn.ModuleList(
               [torch.nn.Linear(dims[0]*heads[0],dims[1]*heads[1],bias=False),
                torch.nn.Linear(dims[self.numLayers-1]*heads[self.numLayers-1],dims[self.numLayers],bias=False)])
            # self.residual = torch.nn.ModuleList(
            #    [torch.nn.Linear(dims[j]*heads[j],dims[j+1]*heads[j+1],bias=False) for j in [0,self.numLayers-1]])
        if self.useResLin:
            self.residual = torch.nn.ModuleList(
               [torch.nn.Linear(dims[j],dims[j+1],bias=False) for j in range(numLayers)])
     
    def forward(self, x, edge_index, saint, getSelfAttnCoef=False,getMetric=[False,False],adj=None,masks={},classMasks=[],classWise=False,getEntropy=False):
         #leakyrelu for computing alphas have negative_slope=0.2 (as set in GAT and used in GATv2)
        attnCoef = [0] * len(self.layers)
        entropy = [0] * len(self.layers)
        dirEnGlb = None#{k: [0] * (self.numLayers+1) for k in ['All']+list(masks.keys())}
        dirEnNbr = None#{k: [0] * (self.numLayers+1) for k in ['All']+list(masks.keys())}
        madGlb = None #{k: [0] * (self.numLayers+1)  for k in ['All']+list(masks.keys())}
        madNbr = None #{k: [0] * (self.numLayers+1)  for k in ['All']+list(masks.keys())}
        numClasses = len(classMasks)
        dirEnGlbClassWise=None
        dirEnNbrClassWise=None
        dirEnDisClassWise=None
        madGlbClassWise=None
        madNbrClassWise=None
        madDisClassWise=None
        dirEnGlb1ClsVsAll=None
        dirEnNbr1ClsVsAll=None
        dirEnDis1ClsVsAll=None
        madGlb1ClsVsAll=None
        madNbr1ClsVsAll=None
        madDis1ClsVsAll=None
        if adj!=None:
            adjPrime = (1-adj).fill_diagonal_(0)
        with torch.no_grad():
            if getMetric[0]:
                dirEnGlb = {k: [0] * (self.numLayers+1) for k in ['All']+list(masks.keys())}
                dirEnNbr = {k: [0] * (self.numLayers+1) for k in ['All']+list(masks.keys())}
                #dirEnLcl = {k: [0] * (self.numLayers+1) for k in ['All']+list(masks.keys())}
                
                d = sq_euclidean_dist(x.detach().clone(),x.detach().clone())#torch.square(pairwise_euclidean_distance(x.detach().clone()))
                md=torch.mul(adj,d)
                dirEnGlb['All'][0] = torch.sum(d)/2
                dirEnNbr['All'][0] = torch.nansum(md)/2#torch.nanmean(torch.sum(md,axis=1)/torch.count_nonzero(adj[1],axis=1))
                
                for k,v in masks.items():
                    dirEnGlb[k][0]=torch.sum(d[v,v])/2
                    #dirEnLcl[k][0]=torch.sum(md[v,:])
                    dirEnNbr[k][0]=torch.nansum(md[v,:][:,v])/2 #torch.nanmean(torch.sum(md[v,:],axis=1)/torch.count_nonzero(adj[1][v,:],axis=1))
                if classWise: 
                    dirEnGlbClassWise = torch.FloatTensor(self.numLayers+1,numClasses,numClasses)
                    dirEnNbrClassWise = torch.FloatTensor(self.numLayers+1,numClasses,numClasses)
                    dirEnDisClassWise = torch.FloatTensor(self.numLayers+1,numClasses,numClasses)
                    
                    dirEnGlb1ClsVsAll = torch.FloatTensor(self.numLayers+1,numClasses)
                    dirEnNbr1ClsVsAll = torch.FloatTensor(self.numLayers+1,numClasses)
                    dirEnDis1ClsVsAll = torch.FloatTensor(self.numLayers+1,numClasses)
                    mdPrime = torch.mul(1-adj,d)
                    for c1 in range(numClasses):
                        m1 = classMasks[c1]
                        m2 = torch.logical_not(m1)
                        dirEnGlb1ClsVsAll[0][c1]=torch.sum(d[m1,:][:,m2])
                        dirEnNbr1ClsVsAll[0][c1]=torch.nansum(md[m1,:][:,m2])
                        dirEnDis1ClsVsAll[0][c1]=torch.nansum(mdPrime[m1,:][:,m2])
                        for c2 in range(numClasses):
                            m2 = classMasks[c2]
                            dirEnGlbClassWise[0][c1][c2]=torch.sum(d[m1,:][:,m2])
                            dirEnNbrClassWise[0][c1][c2]=torch.nansum(md[m1,:][:,m2])
                            dirEnDisClassWise[0][c1][c2]=torch.nansum(mdPrime[m1,:][:,m2])        
            if getMetric[1]:
                madGlb = {k: [0] * (self.numLayers+1)  for k in ['All']+list(masks.keys())}
                madNbr = {k: [0] * (self.numLayers+1)  for k in ['All']+list(masks.keys())}
        
                d = 1 - pairwise_cosine_similarity(x.detach().clone(),zero_diagonal=False)            
                md=torch.mul(adj,d)
                madGlb['All'][0] = torch.mean(d)    
                madNbr['All'][0] = torch.nanmean(torch.sum(md,axis=1)/torch.count_nonzero(adj,axis=1))
                for k,v in masks.items():
                    madGlb[k][0]=torch.mean(d[v,v])
                    madNbr[k][0]=torch.nanmean(torch.sum(md[v,:][:,v],axis=1)/torch.count_nonzero(adj[v,:][:,v],axis=1))
                if classWise:
                    madGlbClassWise = torch.FloatTensor(self.numLayers+1,numClasses,numClasses)
                    madNbrClassWise = torch.FloatTensor(self.numLayers+1,numClasses,numClasses)
                    madDisClassWise = torch.FloatTensor(self.numLayers+1,numClasses,numClasses)
                    
                    madGlb1ClsVsAll = torch.FloatTensor(self.numLayers+1,numClasses)
                    madNbr1ClsVsAll = torch.FloatTensor(self.numLayers+1,numClasses)
                    madDis1ClsVsAll = torch.FloatTensor(self.numLayers+1,numClasses)
                    mdPrime = torch.mul(1-adj,d)
                    for c1 in range(numClasses):
                        m1 = classMasks[c1]
                        m2 = torch.logical_not(m1)
                        madGlb1ClsVsAll[0][c1]=torch.mean(d[m1,:][:,m2])
                        madNbr1ClsVsAll[0][c1]=torch.nanmean(torch.sum(md[m1,:][:,m2],axis=1)/torch.count_nonzero(adj[m1,:][:,m2],axis=1))
                        madDis1ClsVsAll[0][c1]=torch.nanmean(torch.sum(mdPrime[m1,:][:,m2],axis=1)/torch.count_nonzero(adjPrime[m1,:][:,m2],axis=1))
                        for c2 in range(numClasses):
                            m2 = classMasks[c2]
                            madGlbClassWise[0][c1][c2]=torch.mean(d[m1,:][:,m2])
                            madNbrClassWise[0][c1][c2]=torch.nanmean(torch.sum(md[m1,:][:,m2],axis=1)/torch.count_nonzero(adj[m1,:][:,m2],axis=1))
                            madDisClassWise[0][c1][c2]=torch.nanmean(torch.sum(mdPrime[m1,:][:,m2],axis=1)/torch.count_nonzero(adjPrime[m1,:][:,m2],axis=1))
                
                # for k,v in masks.items():
                #     md=torch.mul(adj[1][v,:],d[v,:])
                #     metrics[1][k][0] = torch.nanmean(torch.sum(md,axis=1)/torch.count_nonzero(adj[1][v,:],axis=1))
        if saint:

            for i in range(self.numLayers):#len(self.GATv2Convs)-1):
                #print(i,': ',self.GATv2Convs[i].printOmega())
                if self.linLastLayer and i==self.numLayers-1:
                    x_new=self.layers[i](x)
                else:
                    x_new,a = self.layers[i](x,edge_index,return_attention_weights=getSelfAttnCoef)
                    a=(a[0].detach(),a[1].detach())
                    attnCoef[i] = (a[1][torch.where(torch.eq(a[0][0],a[0][1])==True)[0]]).squeeze(-1) #record only self-attn-coefs (a_ii)
                    
                    if getEntropy:
                        #print(a[0][0],a[0][1])
                        eqIdx = (a[0][0]==a[0][1]).nonzero().squeeze()
                        nodes = a[0][0][eqIdx]
                        entropy[i] = torch.zeros(len(nodes))
                        #print(eqIdx)
                        for n,node in enumerate(nodes):
                            selfIdx = eqIdx[n]
                            #print(selfIdx)
                            #print(a[0][0][selfIdx],node)
                            nbrsIdx = (a[0][1]==node).nonzero().squeeze() 
                            #print(nbrsIdx)
                            #print(a[0][0][nbrsIdx])
                            # print(a[0][1][nbrsIdx])
                            nbrsIdx = nbrsIdx[nbrsIdx!=selfIdx]
                            #print(nbrsIdx)
                            #print(a[0][0][nbrsIdx])
                            #print(a[0][1][nbrsIdx])
                            deg = len(nbrsIdx)
                            eL=(1.0/deg)*((a[1][selfIdx]*torch.log2(a[1][selfIdx]))[0])
                            eR = (((deg-1)*1.0)/deg) * ((a[1][nbrsIdx]*torch.log2(a[1][nbrsIdx])).sum())
                            entropy[i][n] = -1*(eL+eR)

                if self.useIdMap:
                    # print(i)
                    # print(x.shape)
                    # print(x_new.shape)
                    if i==0:#in [0,numLayers-1]:
                        x_new = x_new + self.residual[0](x)
                    elif i==self.numLayers-1:
                        x_new = x_new + self.residual[1](x)
                    else:
                        x_new = x  + x_new
                if self.useResLin:
                    x_new = x_new + self.residual[i](x)
                x=x_new
                if i <(self.numLayers-1):
                    x = self.activation(x)#x.relu() #F.relu(x,inplace=True)
                    if self.dropout>0:
                        x = F.dropout(x, p=self.dropout, training=self.training)
        else:
            
            for i, (ei,_,size) in enumerate(edge_index): 
                x_target = x[:size[1]]
                new_x = self.layers[i]((x,x_target),ei)
                x_new,a = self.layers[i]((x,x_target),ei,return_attention_weights=getSelfAttnCoef)
                a=(a[0].detach(),a[1].detach())
                attnCoef[i] = (a[1][torch.where(torch.eq(a[0][0],a[0][1])==True)[0]]).squeeze(-1) #record only self-attn-coefs (a_ii)
                
                if getEntropy:
                    #print(a[0][0],a[0][1])
                    eqIdx = (a[0][0]==a[0][1]).nonzero().squeeze()
                    nodes = a[0][0][eqIdx]
                    entropy[i] = torch.zeros(len(nodes))
                    #print(eqIdx)
                    for n,node in enumerate(nodes):
                        selfIdx = eqIdx[n]
                        #print(selfIdx)
                        #print(a[0][0][selfIdx],node)
                        nbrsIdx = (a[0][1]==node).nonzero().squeeze() 
                        #print(nbrsIdx)
                        #print(a[0][0][nbrsIdx])
                        # print(a[0][1][nbrsIdx])
                        nbrsIdx = nbrsIdx[nbrsIdx!=selfIdx]
                        #print(nbrsIdx)
                        #print(a[0][0][nbrsIdx])
                        #print(a[0][1][nbrsIdx])
                        deg = len(nbrsIdx)
                        eL=(1.0/deg)*((a[1][selfIdx]*torch.log2(a[1][selfIdx]))[0])
                        eR = (((deg-1)*1.0)/deg) * ((a[1][nbrsIdx]*torch.log2(a[1][nbrsIdx])).sum())
                        entropy[i][n] = -1*(eL+eR)

                if self.useIdMap:
                    # print(i)
                    # print(x.shape)
                    # print(x_new.shape)
                    if i==0:#in [0,numLayers-1]:
                        x_new = x_new + self.residual[0](x)
                    elif i==self.numLayers-1:
                        x_new = x_new + self.residual[1](x)
                    else:
                        x_new = x  + x_new
                if self.useResLin:
                    x_new = x_new + self.residual[i](x)
                x=x_new
                if i <(self.numLayers-1):
                    x = self.activation(x)#x.relu() #F.relu(x,inplace=True)
                    if self.dropout>0:
                        x = F.dropout(x, p=self.dropout, training=self.training)
                


        #x,a = self.GATv2Convs[len(self.GATv2Convs)-1](x,edge_index,return_attention_weights=getAttnCoef)
            
            with torch.no_grad():
                if getMetric[0]:
                    d = sq_euclidean_dist(x.detach().clone(),x.detach().clone())#torch.square(pairwise_euclidean_distance(x.detach().clone()))
                    md=torch.mul(adj,d)
                    dirEnGlb['All'][i+1] = torch.sum(d)/2
                    dirEnNbr['All'][i+1] = torch.nansum(md)/2#torch.nanmean(torch.sum(md,axis=1)/torch.count_nonzero(adj[1],axis=1))
                    
                    for k,v in masks.items():
                        dirEnGlb[k][i+1]=torch.sum(d[v,v])/2
                        #dirEnLcl[k][0]=torch.sum(md[v,:])
                        dirEnNbr[k][i+1]=torch.nansum(md[v,:][:,v])/2 #torch.nanmean(torch.sum(md[v,:],axis=1)/torch.count_nonzero(adj[1][v,:],axis=1))
                    if classWise: 
                        mdPrime = torch.mul(1-adj,d)
                        for c1 in range(numClasses):
                            m1 = classMasks[c1]
                            m2 = torch.logical_not(m1)
                            dirEnGlb1ClsVsAll[i+1][c1]=torch.sum(d[m1,:][:,m2])
                            dirEnNbr1ClsVsAll[i+1][c1]=torch.nansum(md[m1,:][:,m2])
                            dirEnDis1ClsVsAll[i+1][c1]=torch.nansum(mdPrime[m1,:][:,m2])
                            for c2 in range(numClasses):
                                m2 = classMasks[c2]
                                dirEnGlbClassWise[i+1][c1][c2]=torch.sum(d[m1,:][:,m2])
                                dirEnNbrClassWise[i+1][c1][c2]=torch.nansum(md[m1,:][:,m2])
                                dirEnDisClassWise[i+1][c1][c2]=torch.nansum(mdPrime[m1,:][:,m2])        


                if getMetric[1]:
                    d = 1 - pairwise_cosine_similarity(x.detach().clone(),zero_diagonal=False)            
                    md=torch.mul(adj,d)
                    madGlb['All'][i+1] = torch.mean(d)    
                    madNbr['All'][i+1] = torch.nanmean(torch.sum(md,axis=1)/torch.count_nonzero(adj,axis=1))
                    for k,v in masks.items():
                        madGlb[k][i+1]=torch.mean(d[v,v])
                        madNbr[k][i+1]=torch.nanmean(torch.sum(md[v,:][:,v],axis=1)/torch.count_nonzero(adj[v,:][:,v],axis=1))
                    if classWise:
                        mdPrime = torch.mul(1-adj,d)
                        for c1 in range(numClasses):
                            m1 = classMasks[c1]
                            m2 = torch.logical_not(m1)
                            madGlb1ClsVsAll[i+1][c1]=torch.mean(d[m1,:][:,m2])
                            madNbr1ClsVsAll[i+1][c1]=torch.nanmean(torch.sum(md[m1,:][:,m2],axis=1)/torch.count_nonzero(adj[m1,:][:,m2],axis=1))
                            madDis1ClsVsAll[i+1][c1]=torch.nanmean(torch.sum(mdPrime[m1,:][:,m2],axis=1)/torch.count_nonzero(adjPrime[m1,:][:,m2],axis=1))
                            for c2 in range(numClasses):
                                m2 = classMasks[c2]
                                madGlbClassWise[i+1][c1][c2]=torch.mean(d[m1,:][:,m2])
                                madNbrClassWise[i+1][c1][c2]=torch.nanmean(torch.sum(md[m1,:][:,m2],axis=1)/torch.count_nonzero(adj[m1,:][:,m2],axis=1))
                                madDisClassWise[i+1][c1][c2]=torch.nanmean(torch.sum(mdPrime[m1,:][:,m2],axis=1)/torch.count_nonzero(adjPrime[m1,:][:,m2],axis=1))
                    
        #attnCoef[len(self.GATv2Convs)-1] =  (a[0].detach(),a[1].adj())
        smoothnessMetrics={
            'dirEnGl':dirEnGlb,
            'dirEnNb':dirEnNbr,
            'madGl':madGlb,
            'madNb':madNbr,
            'dirEnGlClassWise':dirEnGlbClassWise,
            'dirEnNbClassWise':dirEnNbrClassWise,
            'dirEnDiClassWise':dirEnDisClassWise,
            'madGlClassWise':madGlbClassWise,
            'madNbClassWise':madNbrClassWise,
            'madDiClassWise':madDisClassWise,
            'dirEnGl1ClsVsAll':dirEnGlb1ClsVsAll,
            'dirEnNb1ClsVsAll':dirEnNbr1ClsVsAll,
            'dirEnDi1ClsVsAll':dirEnDis1ClsVsAll,
            'madGl1ClsVsAll':madGlb1ClsVsAll,
            'madNb1ClsVsAll':madNbr1ClsVsAll,
            'madDi1ClsVsAll':madDis1ClsVsAll
        }
        return x,attnCoef,smoothnessMetrics,entropy
    
    def inference(self, x, subgraph_loader):        
        for i, layer in enumerate(self.layers[:-1]):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x_source = x[n_id].to(device)
                x_target = x_source[:size[1]]  # Target nodes are always placed first.
                new_x = layer((x_source, x_target), edge_index)
                new_x = self.activation(new_x)
                x_target = new_x
                xs.append(x_target.cpu())
            x = torch.cat(xs, dim=0)
        xs = []
        for batch_size, n_id, adj in subgraph_loader:
            edge_index, _, size = adj.to(device)
            x_source = x[n_id].to(device)
            x_target = x_source[:size[1]]  # Target nodes are always placed first.
            new_x = self.layers[-1]((x_source, x_target), edge_index)
            xs.append(new_x.cpu())
        x = torch.cat(xs, dim=0)
        return x


def computeStatSumry(arr,quantiles):
  r = {'mean': arr.mean(),
        'std': arr.std()}
  quantiles=torch.cat((torch.tensor([0,1],device=device),quantiles),dim=0)
  p = torch.quantile(arr,quantiles)
  r['min'] = p[0]
  r['max'] = p[1]
  for i in range(2,len(quantiles)):
    r[str(int(quantiles[i]*100))+'%ile'] = p[i]
  return r

def computeAlphaStatSumry(alphas,quantiles):
    return [computeStatSumry(alphas[1][np.where(np.equal(alphas[0][0],alphas[0][1])==True)[0]],quantiles),
       computeStatSumry(alphas[1][np.where(np.equal(alphas[0][0],alphas[0][1])==False)[0]],quantiles)]

def printExpSettings(expID,expSetting):
    print('Exp: '+str(expID))
    for k,v in expSetting.items():
        for k2,v2 in expSetting[k].items():
            if(k2==expID):
                print(k,': ',v2)

def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def initializeParams(params,initScheme,initA1, initA2, activation, paramTypes, attnParamTypes):
    numLayers = len(params)
    with torch.no_grad():
        d = {'attn':initA1, 'attn2':initA2}
        for k in attnParamTypes:
            v = d[k]
            for l in range(numLayers):
                if k in params[l].keys():
                    if v=='XavN':
                        torch.nn.init.xavier_normal_(params[l][k].data)
                    elif v=='posXavN':
                        torch.nn.init.xavier_normal_(params[l][k].data)
                        params[l][k].data = torch.abs(params[l][k].data)
                    elif v=='negXavN':
                        torch.nn.init.xavier_normal_(params[l][k].data)
                        params[l][k].data = -1*torch.abs(params[l][k].data)
                    elif v=='negXavN*2':
                        torch.nn.init.xavier_normal_(params[l][k].data)
                        params[l][k].data = -2*torch.abs(params[l][k].data)
                    elif v=='XavU' or  v=='posXavU': #U in (0,1)
                        torch.nn.init.xavier_uniform_(params[l][k].data)
                    elif v=='negXavU':
                        torch.nn.init.xavier_uniform_(params[l][k].data)
                        params[l][k].data = -1*(params[l][k].data)
                    else:
                        v=float(v)
                        params[l][k].data.fill_(v)
        if(initScheme[:2]!='LL'):
            for l in range(numLayers):
                for f in paramTypes:#)-set(['attn','attn2']):
                    if(initScheme=='xavierN'):
                        torch.nn.init.xavier_normal_(params[l][f].data)
                    if(initScheme=='xavierU'):
                        torch.nn.init.xavier_uniform_(params[l][f].data)
                    if(initScheme=='kaimingN'):
                        torch.nn.init.kaiming_normal_(params[l][f].data,mode='fan_in',nonlinearity=activation)
                    if(initScheme=='kaimingU'):
                        torch.nn.init.kaiming_uniform_(params[l][f].data,mode='fan_in',nonlinearity=activation)
        elif(initScheme[:2]=='LL'):
            #attn params defined above globally
            #for l in range(numLayers):
                #torch.nn.init.xavier_normal_(params[l]['attn'].data)#.fill_(0)
                #torch.nn.init.xavier_normal_(params[l]['attn2'].data)#.fill_(0)# = torch.zeros(params[l]['attn2'].data.shape,device=device) ##LL attnWeights are 0
                #params[l]['attn'].data.fill_(-1)
                #params[l]['attn2'].data.fill_(0)
            for f in paramTypes:#)-set(['attn','attn2']):
                #print(l,f,params[0][f].data,params[0][f])
                firstLayerDeltaDim = (ceil(params[0][f].data.shape[0]/2),params[0][f].data.shape[1])
                if initScheme=='LLxavierU':
                    firstLayerDelta = torch.nn.init.xavier_uniform_(torch.empty(firstLayerDeltaDim[0],firstLayerDeltaDim[1],device=device))
                if initScheme=='LLxavierN':
                    firstLayerDelta = torch.nn.init.xavier_normal_(torch.empty(firstLayerDeltaDim[0],firstLayerDeltaDim[1],device=device))
                if initScheme=='LLkaimingU':
                    firstLayerDelta = torch.nn.init.kaiming_uniform_(torch.empty(firstLayerDeltaDim[0],firstLayerDeltaDim[1],device=device),nonlinearity=activation)
                if initScheme=='LLkaimingN':
                    firstLayerDelta = torch.nn.init.kaiming_normal_(torch.empty(firstLayerDeltaDim[0],firstLayerDeltaDim[1],device=device),nonlinearity=activation)
                if initScheme=='LLortho':
                    firstLayerDelta = torch.nn.init.orthogonal_(torch.empty(firstLayerDeltaDim[0],firstLayerDeltaDim[1],device=device))
                if initScheme=='LLidentity':
                   firstLayerDelta = torch.nn.init.eye_(torch.empty(firstLayerDeltaDim[0],firstLayerDeltaDim[1],device=device))
                if initScheme!='LLidentityHid':
                    params[0][f].data = torch.cat((firstLayerDelta,-firstLayerDelta),dim=0) #BUG CHECK
                
                if f in params[numLayers-1].keys():
                    finalLayerDeltaDim= (params[numLayers-1][f].data.shape[0],ceil(params[numLayers-1][f].data.shape[1]/2))
                    if initScheme=='LLxavierU':
                        finalLayerDelta = torch.nn.init.xavier_uniform_(torch.empty(finalLayerDeltaDim[0],finalLayerDeltaDim[1],device=device))
                    if initScheme=='LLxavierN':
                        finalLayerDelta = torch.nn.init.xavier_normal_(torch.empty(finalLayerDeltaDim[0],finalLayerDeltaDim[1],device=device))
                    if initScheme=='LLkaimingU':
                        finalLayerDelta = torch.nn.init.kaiming_uniform_(torch.empty(finalLayerDeltaDim[0],finalLayerDeltaDim[1],device=device),nonlinearity=activation)
                    if initScheme=='LLkaimingN':
                        finalLayerDelta = torch.nn.init.kaiming_normal_(torch.empty(finalLayerDeltaDim[0],finalLayerDeltaDim[1],device=device),nonlinearity=activation)
                    if initScheme=='LLortho':
                        finalLayerDelta = torch.nn.init.orthogonal_(torch.empty(finalLayerDeltaDim[0],finalLayerDeltaDim[1],device=device))
                    if initScheme=='LLidentity':
                        finalLayerDelta = torch.nn.init.eye_(torch.empty(finalLayerDeltaDim[0],finalLayerDeltaDim[1],device=device))
                    if initScheme!='LLidentityHid':
                        params[numLayers-1][f].data = torch.cat((finalLayerDelta,-finalLayerDelta),dim=1) #BUG CHECK

            for l in range(1,numLayers-1):
                    for f in paramTypes:#)-set(['attn','attn2']):
                        dim = params[l][f].data.shape
                        if initScheme=='LLxavierU':
                            delta = torch.nn.init.xavier_uniform_(torch.empty(ceil(dim[0]/2),ceil(dim[1]/2),device=device))
                        if initScheme=='LLxavierN':
                            delta = torch.nn.init.xavier_normal_(torch.empty(ceil(dim[0]/2),ceil(dim[1]/2),device=device))
                        if initScheme=='LLkaimingU':
                            delta = torch.nn.init.kaiming_uniform_(torch.empty(ceil(dim[0]/2),ceil(dim[1]/2),device=device),nonlinearity=activation)
                        if initScheme=='LLkaimingN':
                            delta = torch.nn.init.kaiming_normal_(torch.empty(ceil(dim[0]/2),ceil(dim[1]/2),device=device),nonlinearity=activation)
                        if initScheme=='LLortho':
                            delta = torch.nn.init.orthogonal_(torch.empty(ceil(dim[0]/2),ceil(dim[1]/2),device=device))
                        if initScheme=='LLidentity' or initScheme=='LLidentityHid':
                            delta = torch.nn.init.eye_(torch.empty(ceil(dim[0]/2),ceil(dim[1]/2),device=device))
                        delta = torch.cat((delta, -delta), dim=0)
                        delta = torch.cat((delta, -delta), dim=1)
                        params[l][f].data = delta
        # if(initScheme=='xavrWzeroA'):
        #     for l in range(numLayers):
        #         torch.nn.init.zeros_(params[l]['attn'].data)
        #         for f in set(paramTypes)-set(['attn','attn2']):
        #             torch.nn.init.xavier_normal_(params[l][f].data)
        if(initScheme=='identityAll'):
            #for l in range(numLayers):
                #torch.nn.init.zeros_(params[l]['attn'].data)
                for f in paramTypes:#)-set(['attn','attn2']):
                    torch.nn.init.eye_(params[l][f].data)
        if(initScheme=='identityHid'):
            #for l in range(numLayers):
                #torch.nn.init.zeros_(params[l]['attn'].data)
            for f in paramTypes:#)-set(['attn','attn2']):
                torch.nn.init.xavier_normal_(params[0][f].data)
                torch.nn.init.xavier_normal_(params[numLayers-1][f].data)
            for l in range(1,numLayers-1):
                for f in set(paramTypes)-set(['attn','attn2']):
                    torch.nn.init.eye_(params[l][f].data)
        if(initScheme=='LLidentityHid'):
            #for l in range(numLayers):
                #torch.nn.init.zeros_(params[l]['attn'].data)
            for f in paramTypes:#)-set(['attn','attn2']):
                torch.nn.init.xavier_normal_(params[0][f].data)
                torch.nn.init.xavier_normal_(params[numLayers-1][f].data)
    for l in range(numLayers):
        for f in paramTypes+attnParamTypes:
            if f in params[l].keys():
                params[l][f].data.requires_grad=True #because of initialization update
    return params

def deepCopyParamsToNumpy(params):
    paramsCopy = [{} for i in range(len(params))]    
    for l in range(len(params)):
        for p in params[l].keys():
            paramsCopy[l][p] = params[l][p].data.detach().cpu().numpy()
    return paramsCopy


path = 'ExpResults_OGB/'
expSetting = pd.read_csv('ExpSettings_OGB.csv',index_col='expId').fillna('').to_dict()
#Add ExpIDs to run here corresponding to ExpSettings.csv 
loadfromCheckpoint = False
expIDtoLoadFrom = None
checkpointEveryXEpoch = 10
expIDs = [1]#range(1,3+1) #Add ExpIDs to run here
trainLossToConverge =  0.00001
printLossEveryXEpoch = 1
saveParamGradStatSumry = False
saveNeuronLevelL2Norms = False
saveLayerWiseForbNorms = False
saveWeightsAtMaxValAcc = False
saveNodeSmoothnessVals = False
saveNodeSmoothnessValsClassWise = False
saveAttentionParams = False
saveAlphas = False
saveAlphasSummary = False
saveBiases = False
saveOmega = False
getAlphaEntropy = False
randomLabels=False
oneHotFeatures = False

quantiles = torch.tensor((np.array(range(1,10,1))/10),dtype=torch.float32,device=device)
qLabels = [str(int(q*100))+'%ile' for q in quantiles]
labels = ['min','max','mean','std']+qLabels 

for expID in expIDs:
    runID = int(expSetting['runID'][expID])
    runIDs = [runID]
    datasetName = str(expSetting['dataset'][expID])
    optim = str(expSetting['optimizer'][expID])
    numLayers = int(expSetting['numLayers'][expID])
    numEpochs = int(expSetting['maxEpochs'][expID])
    lr = float(expSetting['initialLR'][expID])
    hiddenDims = [int(expSetting['hiddenDim'][expID])] * (numLayers-1)
    mulLastAttHead = bool(expSetting['mulLastAttHead'][expID])
    #data input always has 1 attention head, decide for last layer
    if mulLastAttHead:
        heads = [1] + ([int(expSetting['attnHeads'][expID])] * (numLayers)) 
    else:
        heads = [1] + ([int(expSetting['attnHeads'][expID])] * (numLayers-1)) + [1] 
    concat = ([True] * (numLayers-1)) + [False] #concat attn heads for all layers except the last, avergae for last (doesn't matter when num of heads for last layer=1)
    attnDropout = float(expSetting['attnDropout'][expID])
    wghtDecay =  float(expSetting['wghtDecay'][expID])
    activation = str(expSetting['activation'][expID])
    weightSharing = bool(expSetting['weightSharing'][expID])
    dataTransform = str(expSetting['dataTransform'][expID]) #removeIsolatedNodes,useLCC 
    initScheme=str(expSetting['initScheme'][expID])
    scalScheme=str(expSetting['scalScheme'][expID])
    lrDecayFactor = float(expSetting['lrDecayFactor'][expID])
    useIdMap = bool(expSetting['useIdMap'][expID])
    useResLin = bool(expSetting['useResLin'][expID])
    if lrDecayFactor<1:
        lrDecayPatience = float(expSetting['lrDecayPatience'][expID])
    scalHPstr = [0,0,0]
    if len(str(expSetting['scalHP'][expID]))>0:
         scalHPstr=[float(x) for x in str(expSetting['scalHP'][expID]).split('|')] #e.g. (low,high) for uniform, (mean,std) for normal, (const) for const. Third parameter is beta
    initA1=None
    initA2= None
    if len(str(expSetting['initA1'][expID]))>0:
        if str(expSetting['initA1'][expID])[0]!='-' and str(expSetting['initA1'][expID]).isnumeric()==False:
            initA1 = str(expSetting['initA1'][expID])
        else:
            initA1 = float(expSetting['initA1'][expID])
    attParamSharing=False
    if len(str(expSetting['initA2'][expID]))>0:
        if str(expSetting['initA2'][expID]).isnumeric()==False:
            initA2 = str(expSetting['initA2'][expID])
        else:
            initA2 = float(expSetting['initA2'][expID])
    else:
        attParamSharing=True
    linWghtSharing = bool(expSetting['linWghtSharing'][expID])
    linAggrSharing = bool(expSetting['linAggrSharing'][expID])
    bias = bool(expSetting['bias'][expID])
    linLastLayer = bool(expSetting['linLastLayer'][expID])  
    hasOmega = bool(expSetting['hasOmega'][expID])  
    omegaInitVal = float(expSetting['omegaInitVal'][expID]) 

    if str(expSetting['Note'][expID])=='randLabels':
        randomLabels=True
    if str(expSetting['Note'][expID])=='OneHotFeats':
        oneHotFeatures = True
    if str(expSetting['Note'][expID])=='randLabels+OneHotFeats':
        randomLabels=True
        oneHotFeatures = True

    selfLoops = True

    
    data,input_dim,output_dim, data_loader, train_idx,valid_idx,test_idx,test_loader,split_idx = getDataOGB(datasetName,numLayers) 
    
    print('X: ',data.x.shape)
    print('E: ',data.edge_index.shape)
    print('NumFeats: ',input_dim)
    print('NumClasses: ',output_dim)

    data = data.to(device)
    symLapSp = None#get_laplacian(data.edge_index,normalization='sym')
    symLap = None#torch.sparse.FloatTensor(symLapSp[0],symLapSp[1] , torch.Size([data.x.shape[0],data.x.shape[0]])).to_dense()
    adj = None#torch.sparse.FloatTensor(data.edge_index,torch.ones(data.edge_index.shape[1],device=device), torch.Size([data.x.shape[0],data.x.shape[0]])).to_dense()
    masks = {}
    if saveNodeSmoothnessVals:
        adj = torch.sparse.FloatTensor(data.edge_index,torch.ones(data.edge_index.shape[1],device=device), torch.Size([data.x.shape[0],data.x.shape[0]])).to_dense()
        masks['Train']=data.train_mask
        #print(data.train_mask)
        #print(data.train_mask.nonzero().flatten())
        masks['Val']=data.val_mask
        masks['Test']=data.test_mask
    classes = torch.unique(data.y)
    classMasks = [None] * len(classes)
    #masks['Class']=[None] * len(classes)
    if saveNodeSmoothnessValsClassWise:
        for o in range(len(classes)):
            classMasks[o]=data.y==classes[o]
    dims = [input_dim]+hiddenDims+[output_dim]
    paramTypes = ['feat']
    if not linAggrSharing:
        paramTypes = paramTypes + ['feat3']
    if linWghtSharing:
        if weightSharing:
            paramTypes = paramTypes#['feat']
        else:
            paramTypes = paramTypes + ['feat2']#,'attn','attn2']    
    else:
        if weightSharing:
            paramTypes = paramTypes + ['featAtt']
        else:
            paramTypes = paramTypes + ['featAtt','featAtt2']
    if attParamSharing:
        attnParamTypes = ['attn']
    else:
        attnParamTypes = ['attn','attn2']
    print('*******')
    printExpSettings(expID,expSetting)
    print('*******')
    
    for run in runIDs:#range(numRuns):

        print('-- RUN ID: '+str(run))
        set_seeds(run)

        model = GATv2(numLayers,dims,heads,concat, weightSharing, selfLoops, attnDropout,bias=bias,
                      activation=activation,useIdMap=useIdMap,useResLin=useResLin,
                      attParamSharing=attParamSharing,
                      linWghtSharing=linWghtSharing,linLastLayer=linLastLayer,
                      hasOmega=hasOmega,omegaInitVal=omegaInitVal,
                      linAggrSharing=linAggrSharing).to(device)
        # print(model)
        # for name,param in model.named_paramete,rs():
        #     print(name,param.shape)
        
        if optim=='SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wghtDecay)
        if optim=='Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wghtDecay)
        criterion = torch.nn.CrossEntropyLoss()
        if lrDecayFactor<1:
            lrScheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=lrDecayFactor, patience=lrDecayPatience) #based on valAcc
        
        # define variables to track required elements
        trainLoss = torch.zeros(numEpochs, dtype=torch.float32, device = device)
        valLoss = torch.zeros(numEpochs, dtype=torch.float32, device = device)
        trainAcc = torch.zeros(numEpochs, dtype=torch.float32, device = device)
        valAcc = torch.zeros(numEpochs, dtype=torch.float32, device = device)
        testAcc = torch.zeros(numEpochs, dtype=torch.float32, device = device)
            
        selfAttnCoef=None
        selfAttnCoefSumry=None
        # #extra records of parameters for studying training dynamics
        if saveParamGradStatSumry:
            paramStatSumry = [{} for i in range(numLayers)]
            for i in range(numLayers):
                for f in paramTypes+attnParamTypes:
                    paramStatSumry[i][f] = {x2:{x:torch.zeros(numEpochs,device=device) for x in labels} for x2 in ['wght','grad']}
        if saveNeuronLevelL2Norms:
            featL2Norms = [{} for i in range(numLayers)]
            attnWghtsSq = [{x:torch.zeros((numEpochs,dims[i+1]),device=device)
                               for x in attnParamTypes} for i in range(numLayers)]
            for i in range(numLayers):
                for f in set(paramTypes):#-set(['attn','attn2']): #incoming: row-wise of W matrix, and outgoing is col-wise of W matrix
                    featL2Norms[i][f] =  {'row':torch.zeros((numEpochs,dims[i+1]),device=device),'col':torch.zeros((numEpochs,dims[i]),device=device)}
        if saveLayerWiseForbNorms:
            forbNorms = [{f:{x:torch.zeros(numEpochs, device=device) for x in ['wght','grad']}
                             for f in paramTypes+attnParamTypes} for i in range(numLayers)]
        if saveAttentionParams:
            attnParams = [{f:{x:torch.zeros((numEpochs,dims[i+1]), device=device) for x in ['wght','grad']}
                             for f in attnParamTypes} for i in range(numLayers)]
        if saveAlphas:
            selfAttnCoef = [torch.zeros((numEpochs,data.x.shape[0]),device=device) for i in range(numLayers)]
        if saveAlphasSummary:
            selfAttnCoefSumry = [{x:torch.zeros(numEpochs,device=device) for x in labels} for i in range(numLayers)]
                

        if saveOmega:
            omegaVals = [{x:torch.zeros((numEpochs,dims[i+1]), device=device) for x in ['wght','grad']}
                         for i in range(numLayers)]   
        
        entropyVals=None
        dirEnGl = None#{k: torch.zeros(numEpochs,numLayers+1) for k in list(masks.keys())+['All']}
        dirEnNb = None#{k: torch.zeros(numEpochs,numLayers+1) for k in list(masks.keys())+['All']}
        madGl = None#{k: torch.zeros(numEpochs,numLayers+1)  for k in list(masks.keys())+['All']}
        madNb = None#{k: torch.zeros(numEpochs,numLayers+1)  for k in list(masks.keys())+['All']}
        if saveNodeSmoothnessVals:
            dirEnGl = {k: torch.zeros(numEpochs,numLayers+1) for k in list(masks.keys())+['All']}
            dirEnNb = {k: torch.zeros(numEpochs,numLayers+1) for k in list(masks.keys())+['All']}
            madGl = {k: torch.zeros(numEpochs,numLayers+1)  for k in list(masks.keys())+['All']}
            madNb = {k: torch.zeros(numEpochs,numLayers+1)  for k in list(masks.keys())+['All']}
        
        smoothnessValsClassWise = None
        smoothnessVals1ClsVsAll = None
        smoothnessValsClassWiseNameList = ['dirEnGlClassWise','dirEnNbClassWise','dirEnDiClassWise',
                                           'madGlClassWise','madNbClassWise','madDiClassWise']
        smoothnessVals1ClsVsAllNameList = ['dirEnGl1ClsVsAll','dirEnNb1ClsVsAll','dirEnDi1ClsVsAll',
                                           'madGl1ClsVsAll','madNb1ClsVsAll','madDi1ClsVsAll']
        if saveNodeSmoothnessValsClassWise:
            smoothnessValsClassWise = {k: torch.zeros(numEpochs,numLayers+1,len(classes),len(classes)) 
                                       for k in smoothnessValsClassWiseNameList}
            smoothnessVals1ClsVsAll = {k: torch.zeros(numEpochs,numLayers+1,len(classes))
                                       for k in smoothnessVals1ClsVsAllNameList}
        if getAlphaEntropy:
            entropyVals = [torch.zeros((numEpochs,data.x.shape[0]),device=device) for i in range(numLayers)]

            
        biasParamTypes = set()
        biases = [{} for i in range(numLayers)]
        biasParamNameMapping = {'bias':'gnnB','lin_l':'featB','lin_r':'feat2B','lin_lAP':'featAttB','lin_rAP':'featAtt2B'}
        #map default param names to custom names to match visualization scripts later
        modelParamNameMapping = {'omega':'omega','att':'attn',
                                 'lin_l':'feat','lin_r':'feat2','lin_s':'feat3',
                                 'att2':'attn2','lin_lAP':'featAtt',
                                 'lin_rAP':'featAtt2','weight':'feat'}
        params = [{} for i in range(numLayers)]
        for name,param in model.named_parameters():
            paramNameTokens = name.split('.')
            #print(name,paramNameTokens)
            if paramNameTokens[2] in ['att','att2','weight','omega'] or \
            (paramNameTokens[2] in ['lin_l','lin_r','lin_s','lin_lAP','lin_rAP'] and paramNameTokens[3]=='weight'):
                params[int(paramNameTokens[1])][modelParamNameMapping[paramNameTokens[2]]] = param
            if paramNameTokens[2] in ['bias'] or \
            (paramNameTokens[2] in ['lin_l','lin_r','lin_lAP','lin_rAP'] and paramNameTokens[3]=='bias'):
                biases[int(paramNameTokens[1])][biasParamNameMapping[paramNameTokens[2]]] = param
                biasParamTypes.add(biasParamNameMapping[paramNameTokens[2]])
        biasParamTypes = list(biasParamTypes)
        
        if saveBiases:
            biasParams = [{f:{x:torch.zeros((numEpochs,dims[i+1]), device=device) for x in ['wght','grad']}
                             for f in biasParamTypes} for i in range(numLayers)]
         
        params = initializeParams(params,initScheme,initA1,initA2,activation,paramTypes,attnParamTypes)
        paramsAtMaxValAcc = None
        #print('After init')
        for l in range(numLayers):
            for f in biasParamTypes:
                torch.nn.init.zeros_(biases[l][f].data)
                biases[l][f].data.requires_grad=True
       

        initialParamsCopy  = deepCopyParamsToNumpy(params)

        maxValAcc = 0
        continueTraining = True      
        epoch=0

        if loadfromCheckpoint:
            if expIDtoLoadFrom==None:
                expIDtoLoadFrom=expID
            checkpoint = torch.load(path+'CheckpointsOGB/Exp'+str(expIDtoLoadFrom)+'_run'+str(run)+'.pt')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['trainedEpochs']+1
            trainLoss[:epoch] = checkpoint['trainLoss'][:epoch]
            valLoss[:epoch] = checkpoint['valLoss'][:epoch]
            trainAcc[:epoch] = checkpoint['trainAcc'][:epoch]
            valAcc[:epoch] = checkpoint['valAcc'][:epoch]
            testAcc[:epoch] = checkpoint['testAcc'][:epoch]
            if saveAlphasSummary:
                selfAttnCoefSumry = checkpoint['selfAttnCoefSumry']

        while(epoch<numEpochs and continueTraining):
            
            #record required quantities of weights used in a layer
            if saveParamGradStatSumry:
                for l in range(numLayers):
                    for p in paramTypes+attnParamTypes:
                        if p in params[l].keys():
                            for k,v in computeStatSumry(params[l][p].data.detach(),quantiles).items():
                                paramStatSumry[l][p]['wght'][k][epoch] = v
            if saveNeuronLevelL2Norms:
                for l in range(numLayers):
                    for p in paramTypes+attnParamTypes:
                        if p in params[l].keys():
                            wghts=params[l][p].data.detach()
                            if p in attnParamTypes:
                                attnWghtsSq[l][p][epoch] = torch.pow(wghts,2)
                            else:
                                featL2Norms[l][p]['row'][epoch] = torch.pow(wghts,2).sum(axis=1)
                                featL2Norms[l][p]['col'][epoch] = torch.pow(wghts,2).sum(axis=0)
            if saveLayerWiseForbNorms:
                for l in range(numLayers):
                    for p in paramTypes+attnParamTypes:
                            if p in params[l].keys():
                                forbNorms[l][p]['wght'][epoch] = torch.sqrt(torch.pow(params[l][p].data.detach(),2).sum())
            
            if saveAttentionParams:
                for l in range(numLayers):
                    for p in attnParamTypes:
                        if p in params[l].keys():#print('W: E',epoch,'L',l,'P',p,params[l][p].data.detach())
                            attnParams[l][p]['wght'][epoch] = params[l][p].data.detach()
            if saveBiases:
                for l in range(numLayers):
                    for p in biasParamTypes:
                        if p in biases[l].keys():
                            biasParams[l][p]['wght'][epoch] = biases[l][p].data.detach()
            if saveOmega:
                for l in range(numLayers):
                    if 'omega' in params[l].keys():
                        omegaVals[l]['wght'][epoch] = params[l]['omega'].data.detach()        

            
            model.train()

            totalTrainLoss = 0
            totalValLoss = 0

            

            
            if type(data_loader) is GraphSAINTRandomWalkSampler:
                for batch in data_loader:
                  batch_size = 20000
                  #batch.to(device)
                  optimizer.zero_grad()
                  out,attnCoef,smoothnessMetrics,alphaEntropy  = model(batch.x.to(device), batch.edge_index.to(device), 
                                       saint=True, getSelfAttnCoef= saveAlphas or saveAlphasSummary,
                                                        getMetric=[saveNodeSmoothnessVals,saveNodeSmoothnessVals],
                                                        adj=adj,masks=masks,classMasks=classMasks,
                                                        classWise=saveNodeSmoothnessValsClassWise,getEntropy=getAlphaEntropy)
                  #out = out[split_idx['train']]
                  out = out[:batch_size] 
                  batch_y = batch.y[:batch_size].to(device)
                  batch_y = torch.reshape(batch_y,(-1,))
                  #batch_y = torch.reshape(batch_y[split_idx['train']], (-1,))
                  batchTrainLoss = criterion(out, batch_y)
                  batchTrainLoss.backward()
                  optimizer.step()
                  totalTrainLoss += float(batchTrainLoss.detach())

            else:
                for batch_size, n_id, adjs in data_loader:# `adjs` holds a list of `(edge_index, e_id, size)` tuples.
                    adjs = [adj.to(device) for adj in adjs]
                    x = data.x[n_id].to(device)
                    y = data.y[n_id[:batch_size]].squeeze().to(device)

                    optimizer.zero_grad()
                    out,attnCoef,smoothnessMetrics,alphaEntropy  = model(x, adjs, saint=False, getSelfAttnCoef= saveAlphas or saveAlphasSummary,
                                                        getMetric=[saveNodeSmoothnessVals,saveNodeSmoothnessVals],
                                                        adj=adj,masks=masks,classMasks=classMasks,
                                                        classWise=saveNodeSmoothnessValsClassWise,getEntropy=getAlphaEntropy)
                    batchTrainLoss = F.cross_entropy(out, y)
                    batchTrainLoss.backward()
                    optimizer.step()
                    totalTrainLoss += float(batchTrainLoss.detach())
            
            
            
            trainLoss[epoch] = totalTrainLoss / len(data_loader)
            valLoss[epoch] = totalValLoss / len(data_loader)
            
 
            if saveNodeSmoothnessVals:
                for k in ['All']+list(masks.keys()):
                    dirEnGl[k][epoch]=torch.FloatTensor(smoothnessMetrics['dirEnGl'][k])
                    dirEnNb[k][epoch]=torch.FloatTensor(smoothnessMetrics['dirEnNb'][k])
                    madGl[k][epoch]=torch.FloatTensor(smoothnessMetrics['madGl'][k])
                    madNb[k][epoch]=torch.FloatTensor(smoothnessMetrics['madNb'][k])
            if saveNodeSmoothnessValsClassWise:
                for k in smoothnessValsClassWiseNameList:
                    smoothnessValsClassWise[k][epoch] = torch.FloatTensor(smoothnessMetrics[k])
                for k in smoothnessVals1ClsVsAllNameList:
                    smoothnessVals1ClsVsAll[k][epoch] = torch.FloatTensor(smoothnessMetrics[k])
            #record quantities again for the gradients in the epoch 
            if saveParamGradStatSumry:
                for l in range(numLayers):
                    for p in paramTypes+attnParamTypes:
                        if p in params[l].keys():
                            for k,v in computeStatSumry(params[l][p].grad.detach(),quantiles).items():
                                paramStatSumry[l][p]['grad'][k][epoch] = v
            if saveLayerWiseForbNorms:
                for l in range(numLayers):
                    for p in paramTypes+attnParamTypes:
                        if p in params[l].keys():
                            forbNorms[l][p]['grad'][epoch] = torch.sqrt(torch.pow(params[l][p].grad.detach(),2).sum())
            if saveAttentionParams:
                for l in range(numLayers):
                    for p in attnParamTypes:
                        if p in params[l].keys():                        
                        #print('G: E',epoch,'L',l,'P',p,params[l][p].grad.detach())
                            attnParams[l][p]['grad'][epoch] = params[l][p].grad.detach()
            if saveAlphas:
                for l in range(numLayers):    
                    selfAttnCoef[l][epoch] = attnCoef[l]        
            if saveAlphasSummary:
                 for l in range(numLayers):    
                    for k,v in computeStatSumry(attnCoef[l].detach(),quantiles).items():
                        selfAttnCoefSumry[l][k][epoch] = v
            
            if saveBiases:
                for l in range(numLayers):
                    for p in biasParamTypes:
                        if p in biases[l].keys():
                            biasParams[l][p]['grad'][epoch] = biases[l][p].grad.detach()
            if saveOmega:
                for l in range(numLayers):
                    if 'omega' in params[l].keys():
                        omegaVals[l]['grad'][epoch] = params[l]['omega'].grad.detach()
            if getAlphaEntropy:
                for l in range(numLayers):
                    entropyVals[l][epoch] = alphaEntropy[l]


            model.eval()
            with torch.no_grad():
                evaluator = Evaluator(name=datasetName)
                out = model.inference(data.x, subgraph_loader=test_loader)
                y_pred = out.argmax(dim=-1, keepdim=True)
                trainAcc[epoch] = evaluator.eval({
                    'y_true': data.y[split_idx['train']],
                    'y_pred': y_pred[split_idx['train']],
                })['acc']
                valAcc[epoch] = evaluator.eval({
                    'y_true': data.y[split_idx['valid']],
                    'y_pred': y_pred[split_idx['valid']],
                })['acc']
                testAcc[epoch] = evaluator.eval({
                    'y_true': data.y[split_idx['test']],
                    'y_pred': y_pred[split_idx['test']],
                })['acc']

            
            if saveWeightsAtMaxValAcc and valAcc[epoch]>maxValAcc:
                paramsAtMaxValAcc  = deepCopyParamsToNumpy(params)
                maxValAcc = valAcc[epoch]

            if(trainLoss[epoch]<=trainLossToConverge):
                continueTraining=False

            if lrDecayFactor<1:
                lrScheduler.step(valAcc[epoch])

           

            if(epoch%printLossEveryXEpoch==0 or epoch==numEpochs-1):
                print(f'--Epoch: {epoch:03d}, Train Loss: {trainLoss[epoch]:.4f}, Train Acc: {trainAcc[epoch]:.4f}, Val Acc: {valAcc[epoch]:.4f}, Test Acc: {testAcc[epoch]:.4f}')
            if(epoch>0 and epoch%checkpointEveryXEpoch==0):
                stateDict = {'trainedEpochs': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'trainLoss':trainLoss,
                'valLoss' : valLoss,
                'trainAcc' : trainAcc,
                'valAcc': valAcc,
                'testAcc': testAcc}
                if saveAlphasSummary:
                    stateDict['selfAttnCoefSumry']=selfAttnCoefSumry

                torch.save(stateDict, path+'CheckpointsOGB/Exp'+str(expID)+'_run'+str(run)+'.pt')
                               
                expDict = {'expID':expID,  
                'trainedEpochs':epoch,
                'trainLoss':trainLoss[:epoch].detach().cpu().numpy(),
                'valLoss':valLoss[:epoch].detach().cpu().numpy(),
                'trainAcc':trainAcc[:epoch].detach().cpu().numpy(),
                'valAcc':valAcc[:epoch].detach().cpu().numpy(),
                'testAcc':testAcc[:epoch].detach().cpu().numpy()   
                }                
                with open(path+'dictExp'+str(expID)+'_run'+str(run)+'.pkl', 'wb') as f:
                    pickle.dump(expDict,f)

                if saveAlphasSummary:     
                    selfAttnCoefSumryCheckpoint = [{x:torch.zeros(numEpochs,device=device) for x in labels} for i in range(numLayers)]
                    for l in range(numLayers):   
                        for x in labels:
                            selfAttnCoefSumryCheckpoint[l][x] = selfAttnCoefSumry[l][x][:epoch].cpu().numpy()
                    saveAlphasSumryCheckpointDict = {'expID':expID,
                                'numLayers':numLayers,
                                'trainedEpochs':epoch,   
                                'selfAttnCoefSumry':selfAttnCoefSumryCheckpoint
                        }
                    with open(path+'selfAttnCoefSumryExp'+str(expID)+'_run'+str(run)+'.pkl', 'wb') as f:
                        pickle.dump(saveAlphasSumryCheckpointDict,f)
            epoch+=1

        finalParamsCopy  = deepCopyParamsToNumpy(params)

        trainLoss = trainLoss[:epoch].detach().cpu().numpy()
        valLoss = valLoss[:epoch].detach().cpu().numpy()
        trainAcc = trainAcc[:epoch].detach().cpu().numpy()
        valAcc = valAcc[:epoch].detach().cpu().numpy()
        testAcc = testAcc[:epoch].detach().cpu().numpy()
        
        if saveNodeSmoothnessVals:
            for k in ['All']+list(masks.keys()):
                dirEnGl[k] = dirEnGl[k][:epoch].detach().cpu().numpy()
                dirEnNb[k] = dirEnNb[k][:epoch].detach().cpu().numpy()
                madGl[k] = madGl[k][:epoch].detach().cpu().numpy()
                madNb[k] = madNb[k][:epoch].detach().cpu().numpy()
        if saveNodeSmoothnessValsClassWise:
            for k in smoothnessValsClassWiseNameList:
                smoothnessValsClassWise[k] = smoothnessValsClassWise[k][:epoch].detach().cpu().numpy()
            for k in smoothnessVals1ClsVsAllNameList:
                smoothnessVals1ClsVsAll[k] = smoothnessVals1ClsVsAll[k][:epoch].detach().cpu().numpy()
        
        #print('Max or Convergence Epoch: ', epoch)
        print('Max Validation Acc At Epoch: ', np.argmax(valAcc)+1)
        print('Test Acc at Max Val Acc:', testAcc[np.argmax(valAcc)]*100)
        if saveNodeSmoothnessVals:
            for k in ['All']+list(masks.keys()):
                print('Dir Energy Global at Max Val Acc ('+k+'):', dirEnGl[k][np.argmax(valAcc)])
                print('Dir Energy Neighbors at Max Val Acc ('+k+'):', dirEnNb[k][np.argmax(valAcc)])
                print('MAD Global at Max Val Acc ('+k+'):', madGl[k][np.argmax(valAcc)])
                print('MAD Neighbors at Max Val Acc ('+k+'):', madNb[k][np.argmax(valAcc)])

        if saveParamGradStatSumry:
            for l in range(numLayers):
                for p in paramTypes+attnParamTypes:
                    if p in paramStatSumry[l].keys():
                        for x in labels:
                            paramStatSumry[l][p]['wght'][x] = paramStatSumry[l][p]['wght'][x][:epoch].cpu().numpy()
                            paramStatSumry[l][p]['grad'][x] = paramStatSumry[l][p]['grad'][x][:epoch].cpu().numpy()

        if saveNeuronLevelL2Norms:
            for l in range(numLayers):
                attnWghtsSq[l] = attnWghtsSq[l][0:epoch,:].T.cpu().numpy()
                for p in paramTypes:#-set(['attn','attn2']):
                    if p in featL2Norms[l].keys():
                        featL2Norms[l][p]['row'] = featL2Norms[l][p]['row'][0:epoch,:].T.cpu().numpy()
                        featL2Norms[l][p]['col'] = featL2Norms[l][p]['col'][0:epoch,:].T.cpu().numpy()

        if saveLayerWiseForbNorms:
            for l in range(numLayers):
                for p in paramTypes+attnParamTypes:
                    if p in forbNorms[l].keys():
                        forbNorms[l][p]['wght'] = forbNorms[l][p]['wght'][:epoch].cpu().numpy()
                        forbNorms[l][p]['grad'] = forbNorms[l][p]['grad'][:epoch].cpu().numpy()

        if saveAttentionParams:
            for l in range(numLayers):
                for p in attnParamTypes:
                    if p in attnParams[l].keys():
                        attnParams[l][p]['wght'] = attnParams[l][p]['wght'][:epoch].cpu().numpy()
                        attnParams[l][p]['grad'] = attnParams[l][p]['grad'][:epoch].cpu().numpy()
        if saveAlphas:
            for l in range(numLayers):
                selfAttnCoef[l] = selfAttnCoef[l][:epoch].cpu().numpy()
            
        if saveAlphasSummary:     
            for l in range(numLayers):   
                for x in labels:
                    selfAttnCoefSumry[l][x] = selfAttnCoefSumry[l][x][:epoch].cpu().numpy()

        if saveBiases:
            for l in range(numLayers):
                for p in biasParamTypes:
                    if p in biasParams[l].keys():
                        biasParams[l][p]['wght'] = biasParams[l][p]['wght'][:epoch].cpu().numpy()
                        biasParams[l][p]['grad'] = biasParams[l][p]['grad'][:epoch].cpu().numpy()
        
        if saveOmega:
            for l in range(numLayers):
                omegaVals[l]['wght'] = omegaVals[l]['wght'][:epoch].cpu().numpy()
                omegaVals[l]['grad'] = omegaVals[l]['grad'][:epoch].cpu().numpy()
        if getAlphaEntropy:
            for l in range(numLayers):
                entropyVals[l] = entropyVals[l][:epoch].cpu().numpy()

        
        expDict = {'expID':expID,  
                'trainedEpochs':epoch,
                'trainLoss':trainLoss,
                'valLoss':valLoss,
                'trainAcc':trainAcc,
                'valAcc':valAcc,
                'testAcc':testAcc,                
        }

        with open(path+'dictExp'+str(expID)+'_run'+str(run)+'.pkl', 'wb') as f:
            pickle.dump(expDict,f)

        if saveParamGradStatSumry:
            saveParamStatSumry = {'expID':expID,
                        'numLayers':numLayers,
                        'trainedEpochs':epoch,
                        'quantiles':quantiles.cpu().numpy(),
                        'statSumry':paramStatSumry
                    }
            with open(path+'paramStatSumryExp'+str(expID)+'_run'+str(run)+'.pkl', 'wb') as f:
                pickle.dump(saveParamStatSumry,f)

        if saveNeuronLevelL2Norms:
            saveNeuronLevelAttnAndFeatL2Norms = {
                        'expID':expID,
                        'numLayers':numLayers,
                        'trainedEpochs':epoch,
                        'featL2Norms':featL2Norms,
                        'attnWghtsSq':attnWghtsSq
                    }
            with open(path+'neuronLevelAttnAndFeatL2Norms'+str(expID)+'_run'+str(run)+'.pkl', 'wb') as f:
                pickle.dump(saveNeuronLevelAttnAndFeatL2Norms,f)

        if saveLayerWiseForbNorms:
            saveForbNorms = {'expID':expID,
                        'numLayers':numLayers,
                        'trainedEpochs':epoch,
                        'forbNorms':forbNorms
                }
            with open(path+'forbNormsExp'+str(expID)+'_run'+str(run)+'.pkl', 'wb') as f:
                pickle.dump(saveForbNorms,f)

        if saveNodeSmoothnessVals:
            saveNodeSmoothnessMetrics = {'expID':expID,
                        'numLayers':numLayers,
                        'trainedEpochs':epoch,
                        'dirEnGlobal':dirEnGl,
                        'dirEnNeighbors':dirEnNb,
                        'madGlobal':madGl,
                        'madNeighbors':madNb,
                        'smoothnessValsClassWise':smoothnessValsClassWise,
                        'smoothnessVals1ClsVsAll':smoothnessVals1ClsVsAll
                }
            with open(path+'nodeSmoothnessMetricsExp'+str(expID)+'_run'+str(run)+'.pkl', 'wb') as f:
                pickle.dump(saveNodeSmoothnessMetrics,f)

        if saveAttentionParams:
            saveAttnParams = {'expID':expID,
                        'numLayers':numLayers,
                        'trainedEpochs':epoch,
                        'attnParams':attnParams,                        
                        'selfAttnCoef':selfAttnCoef,
                        'selfAttnCoefSumry':selfAttnCoefSumry
                }
            with open(path+'attnParamsExp'+str(expID)+'_run'+str(run)+'.pkl', 'wb') as f:
                pickle.dump(saveAttnParams,f)
        
        if saveAlphas:
            saveAlphas = {'expID':expID,
                        'numLayers':numLayers,
                        'trainedEpochs':epoch,                     
                        'selfAttnCoef':selfAttnCoef,  
                        'selfAttnCoefSumry':selfAttnCoefSumry
                }
            with open(path+'selfAttnCoefExp'+str(expID)+'_run'+str(run)+'.pkl', 'wb') as f:
                pickle.dump(saveAlphas,f)

        if saveAlphasSummary:
            saveAlphasSumry = {'expID':expID,
                        'numLayers':numLayers,
                        'trainedEpochs':epoch,   
                        'selfAttnCoefSumry':selfAttnCoefSumry
                }
            with open(path+'selfAttnCoefSumryExp'+str(expID)+'_run'+str(run)+'.pkl', 'wb') as f:
                pickle.dump(saveAlphasSumry,f)


        if saveBiases:
            saveBiasParams = {'expID':expID,
                        'numLayers':numLayers,
                        'trainedEpochs':epoch,
                        'biasParams':biasParams,     
                }
            with open(path+'biasParamsExp'+str(expID)+'_run'+str(run)+'.pkl', 'wb') as f:
                pickle.dump(saveBiasParams,f)
        
        if saveOmega:
            saveOmegaVals = {'expID':expID,
                        'numLayers':numLayers,
                        'trainedEpochs':epoch,
                        'omegaVals':omegaVals,
                        'selfAttnCoef':selfAttnCoef                
                }
            with open(path+'omegaValsExp'+str(expID)+'_run'+str(run)+'.pkl', 'wb') as f:
                pickle.dump(saveOmegaVals,f)

        if getAlphaEntropy:
            saveAlphaEntropy = {'expID':expID,
                        'numLayers':numLayers,
                        'trainedEpochs':epoch,
                        'alphaEntropy':entropyVals                
                }
            with open(path+'alphaEntropyValsExp'+str(expID)+'_run'+str(run)+'.pkl', 'wb') as f:
                pickle.dump(saveAlphaEntropy,f)