import dgl
import torch
from enum import Enum
from typing import Any, Tuple
from dgl import DGLHeteroGraph
from torch import Tensor
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd
from dgl import function as fn

from .op_util import msg_all2all_GLOO
from ..manager import GraphEngine as engine

class ProprogationMode(Enum):
    Forward = 0
    Backward = 1
    
def GCN_aggregation(graph: dgl.DGLGraph, feats: Tensor, mode: ProprogationMode = ProprogationMode.Forward):
    with graph.local_scope():
        aggregate_fn = fn.copy_src('h', 'm')
        if mode == ProprogationMode.Forward:
            norm1 = graph.ndata['out_degrees'].float().clamp(min=1).pow(-0.5) # out degrees for forward
            norm2 = graph.ndata['in_degrees'].float().clamp(min=1).pow(-0.5) # in degrees for forward
        elif mode == ProprogationMode.Backward:
            norm1 = graph.ndata['in_degrees'].float().clamp(min=1).pow(-0.5) # in degrees for backward
            norm2 = graph.ndata['out_degrees'].float().clamp(min=1).pow(-0.5) # out degrees for backward
        else:
            raise ValueError(f'Invalid mode {mode}')
        feats = feats * norm1.view(-1, 1)
        graph.srcdata['h'] = feats
        graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
        rst = graph.dstdata['h'] * norm2.view(-1, 1)
        return rst

def SAGE_aggregation(graph: dgl.DGLGraph, feats: Tensor, mode: ProprogationMode = ProprogationMode.Forward, aggregator_type='mean'):
    with graph.local_scope():
        aggregate_fn = fn.copy_src('h', 'm')
        if mode == ProprogationMode.Forward:
            graph.srcdata['h'] = feats
            if aggregator_type == 'mean':
                graph.update_all(aggregate_fn, fn.mean(msg='m', out='neigh'))
                h_neigh = graph.dstdata['neigh']
            elif aggregator_type == 'gcn':
                graph.update_all(aggregate_fn, fn.sum(msg='m', out='neigh'))
                degs = graph.ndata['in_degrees'].float().clamp(min=1).view(-1, 1)
                h_neigh = (graph.dstdata['neigh'] + graph.srcdata['h']) / (degs + 1)
            else:
                raise ValueError(f'Invalid aggregator_type {aggregator_type}')
        elif mode == ProprogationMode.Backward:
            if aggregator_type == 'mean':
                norm = graph.ndata['out_degrees'].float().clamp(min=1).pow(-1).view(-1, 1)
                feats = feats * norm
                graph.srcdata['h'] = feats
                graph.update_all(aggregate_fn, fn.sum(msg='m', out='neigh'))
                h_neigh = graph.dstdata['neigh']
            elif aggregator_type == 'gcn':
                norm = graph.ndata['out_degrees'].float().clamp(min=1) + 1
                norm = torch.pow(norm, -1).view(-1, 1)
                feats = feats * norm
                graph.srcdata['h'] = feats
                graph.update_all(aggregate_fn, fn.sum(msg='m', out='neigh'))
                h_neigh = graph.dstdata['neigh'] + graph.srcdata['h']
            else:
                raise ValueError(f'Invalid aggregator_type {aggregator_type}')
        else:
            raise ValueError(f'Invalid mode {mode}')
        return h_neigh

class distAggConv(Function):
    '''
    customized distributed aggregation Function class which aggregates features from both local and remote neighbors for GCN.
    '''
    @staticmethod
    @custom_fwd
    def forward(ctx, local_messages: Tensor, graph: DGLHeteroGraph, layer: int, is_train: bool) -> Tensor:
        # exchange messages (features/embeddings)
        send_messages = local_messages[engine.ctx.total_send_idx]
        remote_messages = msg_all2all_GLOO(send_messages, f'forward{layer}', is_train)
        full_messages = torch.cat([local_messages, remote_messages], dim=0)
        # aggregate messages
        with engine.ctx.timer.record(f'forward{layer}_full_aggregation'):
            rst = GCN_aggregation(graph, full_messages, mode=ProprogationMode.Forward)
        len_local = local_messages.shape[0]
        return_embeddings = rst[:len_local]
        ctx.saved = layer
        return return_embeddings
    
    @staticmethod
    @custom_bwd
    def backward(ctx: Any, *grad_outputs: Tuple[Tensor, ...]) -> Tensor:
        layer = ctx.saved
        local_messages = grad_outputs[0]
        send_messages = local_messages[engine.ctx.total_send_idx]
        # exchange messages (embedding gradients)
        remote_messages = msg_all2all_GLOO(send_messages, f'backward{layer}', is_train=True)
        full_messages = torch.cat([local_messages, remote_messages], dim=0)
        # aggregate messages
        with engine.ctx.timer.record(f'backward{layer}_full_aggregation'):
            rst = GCN_aggregation(engine.ctx.bwd_graph, full_messages, mode=ProprogationMode.Backward)
        len_local = local_messages.shape[0]
        return_gradients = rst[:len_local]
        return return_gradients, None, None, None

class distAggSAGE(Function):
    '''
    customized distributed aggregation Function class which aggregates features from both local and remote neighbors for GraphSAGE.
    '''
    @staticmethod
    @custom_fwd
    def forward(ctx, local_messages: Tensor, graph: DGLHeteroGraph, layer: int, is_train: bool) -> Tensor:
        # exchange messages (features/embeddings)
        send_messages = local_messages[engine.ctx.total_send_idx]
        remote_messages = msg_all2all_GLOO(send_messages, f'forward{layer}', is_train)
        full_messages = torch.cat([local_messages, remote_messages], dim=0)
        with engine.ctx.timer.record(f'forward{layer}_full_aggregation'):
            rst = SAGE_aggregation(graph, full_messages, mode=ProprogationMode.Forward, aggregator_type=engine.ctx.agg_type)
        len_local = local_messages.shape[0]
        return_embeddings = rst[:len_local]
        ctx.saved = layer
        return return_embeddings
    
    @staticmethod
    @custom_bwd
    def backward(ctx: Any, *grad_outputs: Tuple[Tensor, ...]) -> Tensor:
        layer = ctx.saved
        local_messages = grad_outputs[0]
        send_messages = local_messages[engine.ctx.total_send_idx]
        # exhange messages (embedding gradients)
        remote_messages = msg_all2all_GLOO(send_messages, f'backward{layer}', is_train=True)
        full_messages = torch.cat([local_messages, remote_messages], dim=0)
        # aggregate messages
        with engine.ctx.timer.record(f'backward{layer}_full_aggregation'):
            rst = SAGE_aggregation(engine.ctx.bwd_graph, full_messages, mode=ProprogationMode.Backward, aggregator_type=engine.ctx.agg_type)
        len_local = local_messages.shape[0]
        return_gradients = rst[:len_local]
        return return_gradients, None, None, None