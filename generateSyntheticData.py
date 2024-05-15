
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
from typing import Optional, Tuple, Union
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures,LargestConnectedComponents
import networkx as nx
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.utils import segregate_self_loops,dense_to_sparse,\
index_to_mask,get_laplacian,erdos_renyi_graph,to_networkx
from torchmetrics.functional import pairwise_cosine_similarity
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

device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
            if linWghtSharing:
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
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2
            x_l = self.lin_l(x_l).view(-1, H, C)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)

        #added for attn perceptron weight sharing
        if self.linWghtSharing:
            x_lAP = x_l
            x_rAP = x_r
        else:
            x_lAP = self.lin_lAP(x).view(-1, H, C)
            if self.share_weights:
                x_rAP = x_lAP
            else:
                x_rAP = self.lin_rAP(x).view(-1, H, C)


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
        x_j = (x_j*ijNotEq.unsqueeze(-1)) + (x_s*ijEq.unsqueeze(-1))

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
     
    def forward(self, x, edge_index, getSelfAttnCoef=False,getMetric=[False,False],adj=None,masks={},classMasks=[],classWise=False,getEntropy=False):
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

def generateSyntheticDataNeighborDependent(synID, structure, numNodes, numFeats, 
                          featsMean, featsCovar, numClass, aggrIters, 
                          lastLayerDim, trainFrac, valFrac, testFrac):
                         
    
    if structure[:2]=='ER':        
        edge_index = erdos_renyi_graph(numNodes,float(structure.split("_")[1]),False)
    if structure=='Cora':        
        d=Planetoid(root='data/Planetoid', name='Cora')[0]
        edge_index = d.edge_index
        numNodes = d.x.shape[0]
    #print(edge_index.shape)
    mean = torch.zeros((numFeats,)).fill_(featsMean)
    if featsCovar == 'randStdN':
        r = torch.normal(0,1,(numFeats,numFeats))#torch.rand((numFeats,numFeats))
        cov = torch.matmul(r,r.T)
    elif featsCovar == 'identity':
        cov = torch.eye(numFeats)
    dist = MultivariateNormal(loc=mean, covariance_matrix=cov)
    x = dist.sample((numNodes,))

    xEmb = TSNE(n_components=2, perplexity=30, n_iter=1000, verbose=False).fit_transform(x)
    # plt.scatter(xEmb[:,0],xEmb[:,1])    
    # plt.savefig('SyntheticData/D'+str(synID)+'_Feats.png', bbox_inches='tight')
    # # plt.show()
    # plt.clf() 

    depth = aggrIters
    width = numFeats
    inputDim = numFeats
    outputDim = numClass
    if lastLayerDim=='numFeats':
        dims = [inputDim]+([width] * depth)#(depth-1))+[outputDim]
    elif lastLayerDim=='numClasses':
        dims = [inputDim]+([width] * (depth-1))+[outputDim]
    heads = [1] * len(dims)
    concat = [True] * len(dims)
    device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = Data(x=x, edge_index=edge_index, num_nodes=numNodes).to(device)
    
    randGATmodel = GATv2(depth,dims,heads,concat, weightSharing=True, selfLoops=False, 
                         attnDropout=0,bias=False,
                      activation='relu',useIdMap=False,useResLin=False,
                      attParamSharing=False, linWghtSharing=True,linLastLayer=False,
                      hasOmega=False,omegaInitVal=1,defaultInit=True).to(device)
    
    aggrEmb,a,s,e = randGATmodel(data.x, data.edge_index, getSelfAttnCoef=False,getEntropy=False)
    aggrEmb = aggrEmb.detach().cpu().numpy()
    #print(aggrEmb.shape)
    kmeans = KMeans(init="k-means++", n_clusters=2, n_init=4).fit(aggrEmb) #random_state=0)
    y = kmeans.labels_
    # print(y.shape)
    if aggrEmb.shape[1]!=2:
       aggrEmb = TSNE(n_components=2, perplexity=30, n_iter=1000, verbose=False).fit_transform(aggrEmb)
    

    data.y=torch.tensor(y,dtype=torch.int64,device=device)

    data.train_mask = torch.zeros(numNodes, dtype=torch.bool, device=device)
    data.val_mask = torch.zeros(numNodes, dtype=torch.bool, device=device)
    data.test_mask = torch.zeros(numNodes, dtype=torch.bool, device=device)
    masks = np.split(np.random.permutation(
                    np.arange(numNodes)),
                    np.cumsum([int(trainFrac*numNodes),
                            int(valFrac*numNodes),
                            int(testFrac*numNodes)][:-1]))
    data.train_mask[masks[0]]=1
    data.val_mask[masks[1]]=1
    data.test_mask[masks[2]]=1
    # print(data.train_mask.sum()+data.val_mask.sum()+data.test_mask.sum())
    # print(data.y[data.train_mask].sum())
    # print(data.y[data.val_mask].sum())
    # print(data.y[data.test_mask].sum())
    # print(data.y[torch.logical_or(torch.logical_or(data.train_mask,data.val_mask),data.test_mask)].sum())

    plt.scatter(aggrEmb[:,0],aggrEmb[:,1],c=y,alpha=0.5)
    plt.savefig('SyntheticData/D'+str(synID)+'_AggrFeats.png', bbox_inches='tight')
    #plt.show()
    plt.clf() 
    xEmb = TSNE(n_components=2, perplexity=30, n_iter=1000, verbose=False).fit_transform(x)
    plt.scatter(xEmb[:,0],xEmb[:,1],c=y,alpha=0.5)    
    plt.savefig('SyntheticData/D'+str(synID)+'_OrgFeats.png', bbox_inches='tight')
    #plt.show()
    plt.clf() 


    with open('SyntheticData/D'+str(synID)+'.pkl', 'wb') as f:
                pickle.dump(data,f)

def generateSyntheticDataSelfSufficient(synID, structure, numNodes, randLabels,
                                        numClass, trainFrac, valFrac, testFrac):
    if structure[:2]=='ER':        
        edge_index = erdos_renyi_graph(numNodes,float(structure.split("_")[1]),False)      
        data = Data(edge_index=edge_index)
        data.y = torch.randint(low=0,high=numClass,size=(numNodes,))
        data.train_mask = torch.zeros(numNodes, dtype=torch.bool, device=device)
        data.val_mask = torch.zeros(numNodes, dtype=torch.bool, device=device)
        data.test_mask = torch.zeros(numNodes, dtype=torch.bool, device=device)
        masks = np.split(np.random.permutation(
                        np.arange(numNodes)),
                        np.cumsum([int(trainFrac*numNodes),
                                int(valFrac*numNodes),
                                int(testFrac*numNodes)][:-1]))
        data.train_mask[masks[0]]=1
        data.val_mask[masks[1]]=1
        data.test_mask[masks[2]]=1    

    if structure=='Cora':        
        d=Planetoid(root='data/Planetoid', name='Cora')[0]
        edge_index = d.edge_index   
        data = Data(edge_index=edge_index)
        numNodes = d.x.shape[0]
        data.train_mask = d.train_mask
        data.val_mask = d.val_mask
        data.test_mask = d.test_mask
        if numClass==None:
            numClass = len(torch.unique(d.y))
        if randLabels:
            data.y = torch.randint(low=0,high=numClass,size=d.y.shape)
        else:
            data.y = d.y
    
    data.x = torch.tensor(F.one_hot(data.y).clone().detach(),dtype = torch.float32)
    with open('SyntheticData/D'+str(synID)+'.pkl', 'wb') as f:
                pickle.dump(data,f)

         

dataGenSettings = pd.read_csv('SyntheticDataSettings.csv',index_col='graphID').fillna('').to_dict()

graphIDs = [21,22,23] #set IDs from SyntheticDataSettings.csv
for synID in graphIDs:
    structure = str(dataGenSettings['structure'][synID])
    numNodes = int(dataGenSettings['numNodes'][synID])
    numFeats = int(dataGenSettings['numFeats'][synID])
    featsMean = float(dataGenSettings['featsMean'][synID])
    featsCovar = str(dataGenSettings['featsCovar'][synID])
    numClass = int(dataGenSettings['numClass'][synID])
    aggrIters = int(dataGenSettings['aggrIters'][synID])
    lastLayerDim = str(dataGenSettings['lastLayerDim'][synID])
    trainFrac = float(dataGenSettings['trainFrac'][synID])
    valFrac = float(dataGenSettings['valFrac'][synID])
    testFrac = float(dataGenSettings['testFrac'][synID])

    generateSyntheticDataNeighborDependent(synID, structure, numNodes, numFeats, 
                          featsMean, featsCovar, numClass, aggrIters, 
                          lastLayerDim, trainFrac, valFrac, testFrac)


# generateSyntheticDataSelfSufficient(24, 'Cora', None, False,None, 0.5, 0.25, 0.25)
# generateSyntheticDataSelfSufficient(25, 'Cora', None, True,None, 0.5, 0.25, 0.25)    
# generateSyntheticDataSelfSufficient(26, 'ER_0.01', 1000, True,2, 0.5, 0.25, 0.25)
# generateSyntheticDataSelfSufficient(27, 'ER_0.01', 1000, True,8, 0.5, 0.25, 0.25)
    
