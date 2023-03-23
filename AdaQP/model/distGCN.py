import torch
from typing import Any
from torch import Tensor
from dgl import DGLHeteroGraph
from torch.nn.parameter import Parameter
from torch.nn import init
import torch.nn as nn
import torch.nn.functional as F

from .ops import distAggConv

class distGCNConv(nn.Module):
    '''distGCNConv layer transmits 1-hop features(embeddings) and gradients during forward and backward pass'''

    def __init__(self, in_feats: int, out_feats: int, weight=True, bias: bool = True, activation: Any = None):
        super(distGCNConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        if weight:
            self.weight = Parameter(torch.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('weight', None)
        if bias:
            self.bias = Parameter(torch.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self._activation = activation

    def reset_parameters(self):
        if self.weight is not None:
            init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)
    
    def set_allow_zero_in_degree(self, set_value: bool):
        self._allow_zero_in_degree = set_value

    def forward(self, feats: Tensor, graph: DGLHeteroGraph, layer: int) -> Tensor:
        weight = self.weight
        # call distAggConv.forward() to aggregate first then mult W
        rst = distAggConv.apply(feats, graph, layer, self.training)
        if weight is not None:
            rst = torch.matmul(rst, weight)  # calculate on GPU
        if self.bias is not None:
            rst = rst + self.bias
        if self._activation is not None:
            rst = self._activation(rst)
        return rst

class distGCN(nn.Module):
    def __init__(self, in_feats: int, h_feats: int, num_classes: int, num_layers: int, drop_rate: float, use_norm: bool = True):
        super(distGCN, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(distGCNConv(in_feats, h_feats))
        if use_norm:
            self.norms = nn.ModuleList()
            self.norms.append(nn.LayerNorm(h_feats))
        # append hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(distGCNConv(h_feats, h_feats))
            if use_norm:
                self.norms.append(nn.LayerNorm(h_feats))
        # append last layer
        self.convs.append(distGCNConv(h_feats, num_classes))
        # set drop rate
        self.drop_rate = drop_rate

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if hasattr(self, 'norms'):
            for bn in self.norms:
                bn.reset_parameters()

    def forward(self, g: DGLHeteroGraph, feats: Tensor) -> Tensor:
        for i, conv in enumerate(self.convs[:-1]):
            feats = conv(feats, g, i)
            feats = F.dropout(feats, p=self.drop_rate, training=self.training)
            if hasattr(self, 'norms'):
                feats = self.norms[i](feats)
            feats = F.relu(feats, inplace=True)
        feats = self.convs[-1](feats, g, i + 1)
        return feats