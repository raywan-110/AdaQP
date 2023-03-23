import dgl
import torch
from typing import Any
from torch import Tensor
from dgl import DGLHeteroGraph
from torch.nn.parameter import Parameter
from torch.nn import init
import torch.nn as nn
import torch.nn.functional as F

from .ops import distAggSAGE

class distSAGEConv(nn.Module):
    '''distGCNConv layer transmits 1-hop features(embeddings) and gradients during forward and backward pass'''

    def __init__(self, in_feats: int, out_feats: int, aggregator_type: str ='mean', bias: int = True, norm: str = None, activation: Any = None):
        super(distSAGEConv, self).__init__()
        valid_agg_types = ('mean', 'gcn')
        if aggregator_type not in valid_agg_types:
            raise dgl.DGLError(
                'Invalid aggregator_type. Must be one of {}. '
                'But got {!r} instead.'.format(valid_agg_types, aggregator_type)
            )
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self._activation = activation
        self._aggregator_type = aggregator_type
        if bias:
            self.bias = Parameter(torch.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)
        if aggregator_type != 'gcn':
            self.fc_self = nn.Linear(self._in_feats, self._out_feats, bias=False)
        self.fc_neigh = nn.Linear(self._in_feats, self._out_feats, bias=False)

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if self._aggregator_type != 'gcn':
            init.xavier_uniform_(self.fc_self.weight, gain=gain)
        init.xavier_uniform_(self.fc_neigh.weight, gain=gain)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, local_feats: Tensor, graph: DGLHeteroGraph, layer: int) -> Tensor:
        '''training and transductive inference'''
        # call distAggSAGE.forward() to aggregate feats from neighbors
        h_neigh = distAggSAGE.apply(local_feats, graph, layer, self.training)
        if self._aggregator_type != 'gcn':
            rst = self.fc_self(local_feats) + self.fc_neigh(h_neigh)  # mean
        else:
            rst = self.fc_neigh(h_neigh)  # gcn
        # bias term
        if self.bias is not None:
            rst = rst + self.bias
        # activation
        if self._activation is not None:
            rst = self._activation(rst)
        # normalization
        if self._norm is not None:
            rst = self._norm(rst)
        return rst


class distSAGE(nn.Module):
    def __init__(self, in_feats: int, h_feats: int, num_classes: int, num_layers: int, drop_rate: float, aggregator_type: str = 'mean', use_norm: bool = True):
        super(distSAGE, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(distSAGEConv(in_feats, h_feats, aggregator_type=aggregator_type))
        if use_norm:
            self.norms = nn.ModuleList()
            self.norms.append(nn.LayerNorm(h_feats))
        # append hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(distSAGEConv(h_feats, h_feats, aggregator_type=aggregator_type))
            if use_norm:
                self.norms.append(nn.LayerNorm(h_feats))
        # append last layer
        self.convs.append(distSAGEConv(h_feats, num_classes, aggregator_type=aggregator_type))
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
            feats = F.relu(feats)
        feats = self.convs[-1](feats, g, i + 1)

        return feats