import dgl
import torch
from typing import Any, Tuple, Union
from functools import wraps
from dgl import DGLHeteroGraph
from torch import Tensor
from contextlib import contextmanager
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd
from dgl import function as fn

from .op_util import msg_all2all_GLOO
from ..helper import ProprogationMode, BitType
from ..manager import DecompGraph
from ..manager import GraphEngine as engine
    
def GCN_aggregation(graph: DGLHeteroGraph, feats: Tensor, mode: ProprogationMode = ProprogationMode.Forward):
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

def SAGE_aggregation(graph: DGLHeteroGraph, feats: Tensor, mode: ProprogationMode = ProprogationMode.Forward, aggregator_type='mean'):
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

class DistAggConv(Function):
    '''
    customized distributed aggregation Function class which aggregates features from both local and remote neighbors for GCN.
    '''
    @staticmethod
    @custom_fwd
    def forward(ctx, local_messages: Tensor, graph: Union[DGLHeteroGraph, DecompGraph], layer: int, is_train: bool) -> Tensor:
        if not engine.ctx.use_parallel:
            return full_graph_propagation(ctx, local_messages, graph, layer, is_train, ProprogationMode.Forward, DistAggConv.__name__)
        else:
            return decomposed_graph_propagation(ctx, local_messages, graph, layer, is_train, ProprogationMode.Forward, DistAggConv.__name__)
        # # exchange messages (features/embeddings)
        # send_messages = local_messages[engine.ctx.total_send_idx]
        # remote_messages = msg_all2all_GLOO(send_messages, f'forward{layer}', is_train)
        # full_messages = torch.cat([local_messages, remote_messages], dim=0)
        # # aggregate messages
        # with engine.ctx.timer.record(f'forward{layer}_full_aggregation'):
        #     rst = GCN_aggregation(graph, full_messages, mode=ProprogationMode.Forward)
        # len_local = local_messages.shape[0]
        # return_messages = rst[:len_local]
        # ctx.saved = layer
        # return return_messages
    
    @staticmethod
    @custom_bwd
    def backward(ctx: Any, *grad_outputs: Tuple[Tensor, ...]) -> Tensor:
        layer = ctx.saved
        local_messages = grad_outputs[0]
        if not engine.ctx.use_parallel:
            return full_graph_propagation(ctx, local_messages, engine.ctx.bwd_graph, layer, True, ProprogationMode.Backward, DistAggConv.__name__)
        else:
            return decomposed_graph_propagation(ctx, local_messages, engine.ctx.bwd_graph, layer, True, ProprogationMode.Backward, DistAggConv.__name__)
        # send_messages = local_messages[engine.ctx.total_send_idx]
        # # exchange messages (embedding gradients)
        # remote_messages = msg_all2all_GLOO(send_messages, f'backward{layer}', is_train=True)
        # full_messages = torch.cat([local_messages, remote_messages], dim=0)
        # # aggregate messages
        # with engine.ctx.timer.record(f'backward{layer}_full_aggregation'):
        #     rst = GCN_aggregation(engine.ctx.bwd_graph, full_messages, mode=ProprogationMode.Backward)
        # len_local = local_messages.shape[0]
        # return_gradients = rst[:len_local]
        # return return_gradients, None, None, None

class DistAggSAGE(Function):
    '''
    customized distributed aggregation Function class which aggregates features from both local and remote neighbors for GraphSAGE.
    '''
    @staticmethod
    @custom_fwd
    def forward(ctx, local_messages: Tensor, graph: Union[DGLHeteroGraph, DecompGraph], layer: int, is_train: bool) -> Tensor:
        if not engine.ctx.use_parallel:
            return full_graph_propagation(ctx, local_messages, graph, layer, is_train, ProprogationMode.Forward, DistAggSAGE.__name__)
        else:
            return decomposed_graph_propagation(ctx, local_messages, graph, layer, is_train, ProprogationMode.Forward, DistAggSAGE.__name__)
        # exchange messages (features/embeddings)
        # send_messages = local_messages[engine.ctx.total_send_idx]
        # remote_messages = msg_all2all_GLOO(send_messages, f'forward{layer}', is_train)
        # full_messages = torch.cat([local_messages, remote_messages], dim=0)
        # with engine.ctx.timer.record(f'forward{layer}_full_aggregation'):
        #     rst = SAGE_aggregation(graph, full_messages, mode=ProprogationMode.Forward, aggregator_type=engine.ctx.agg_type)
        # len_local = local_messages.shape[0]
        # return_messages = rst[:len_local]
        # ctx.saved = layer
        # return return_messages
    
    @staticmethod
    @custom_bwd
    def backward(ctx: Any, *grad_outputs: Tuple[Tensor, ...]) -> Tensor:
        layer = ctx.saved
        local_messages = grad_outputs[0]
        if not engine.ctx.use_parallel:
            return full_graph_propagation(ctx, local_messages, engine.ctx.bwd_graph, layer, True, ProprogationMode.Backward, DistAggSAGE.__name__)
        else:
            return decomposed_graph_propagation(ctx, local_messages, engine.ctx.bwd_graph, layer, True, ProprogationMode.Backward, DistAggSAGE.__name__)
        # send_messages = local_messages[engine.ctx.total_send_idx]
        # # exhange messages (embedding gradients)
        # remote_messages = msg_all2all_GLOO(send_messages, f'backward{layer}', is_train=True)
        # full_messages = torch.cat([local_messages, remote_messages], dim=0)
        # # aggregate messages
        # with engine.ctx.timer.record(f'backward{layer}_full_aggregation'):
        #     rst = SAGE_aggregation(engine.ctx.bwd_graph, full_messages, mode=ProprogationMode.Backward, aggregator_type=engine.ctx.agg_type)
        # len_local = local_messages.shape[0]
        # return_gradients = rst[:len_local]
        # return return_gradients, None, None, None

'''
*************************************************
*********** forward/backward functions **********
*************************************************
'''

@contextmanager
def central_compute_ctx(is_train: bool):
    # enter central compute context
    if engine.ctx.bit_type == BitType.QUANT and is_train:
        engine.ctx.quant_cpu_event.wait() # wait for quant kernel launch in marginal thread
        torch.cuda.current_stream().wait_event(engine.ctx.quant_cuda_event) # wait for completion of quant kernel
        engine.ctx.quant_cpu_event.clear() # clear event for next iteration
    yield
    # leave central compute context
    if engine.ctx.bit_type == BitType.QUANT and is_train:
        engine.ctx.comp_cuda_event.record(torch.cuda.current_stream()) # record event for next iteration
        engine.ctx.comp_cpu_event.set() # set event for next iteration

def full_graph_propagation(ctx, local_messages: Tensor, graph: Union[DGLHeteroGraph, DecompGraph], layer: int, is_train: bool, mode: ProprogationMode, class_name: str) -> Union[Tensor, Tuple[Tensor, None, None, None]]:
    # exchange messages (features/embeddings/embedding gradients)
    send_messages = local_messages[engine.ctx.total_send_idx]
    name = f'forward{layer}' if mode == ProprogationMode.Forward else f'backward{layer}'
    remote_messages = msg_all2all_GLOO(send_messages, name, is_train)
    full_messages = torch.cat([local_messages, remote_messages], dim=0)
    # aggregate messages
    with engine.ctx.timer.record(f'{name}_full_aggregation'):
        if class_name == 'DistAggConv':
            rst = GCN_aggregation(graph, full_messages, mode=mode)
        elif class_name == 'DistAggSAGE':
            rst = SAGE_aggregation(graph, full_messages, mode=mode, aggregator_type=engine.ctx.agg_type)
        else:
            raise ValueError(f'Invalid class_name {class_name}')
    len_local = local_messages.shape[0]
    return_messages = rst[:len_local]
    # store layer info in forward propagation 
    if mode == ProprogationMode.Forward:
        ctx.saved = layer
        return return_messages
    # return gradients in backward propagation
    else:
        return return_messages, None, None, None

def decomposed_graph_propagation(ctx, local_messages: Tensor, graph: Union[DGLHeteroGraph, DecompGraph], layer: int, is_train: bool, mode: ProprogationMode, class_name: str) -> Union[Tensor, Tuple[Tensor, None, None, None]]:
    # fetch needed messages for central and marginal graphs
    assert isinstance(graph, DecompGraph), f'graph must be a DecompGraph, but got {type(graph)}'
    messages_from_marginal, messages_from_central = graph.get_copy_buffers(layer)
    messages_from_marginal.copy_(local_messages[graph.src_marginal_idx])
    messages_from_central.copy_(local_messages[graph.src_central_idx])
    torch.cuda.current_stream().synchronize() # wait for completion of copy
    # async exchange messages (features/embeddings/embedding gradients)
    send_messages = local_messages[engine.ctx.total_send_idx]
    name = f'forward{layer}' if mode == ProprogationMode.Forward else f'backward{layer}'
    response = engine.ctx.marginal_pool.apply_async(msg_all2all_GLOO, args=(send_messages, name, is_train))
    # computation on marginal graph should wait for completion of quantization
    with central_compute_ctx(is_train):
        central_messages = torch.concat([local_messages[:engine.ctx.num_central], messages_from_marginal], dim=0)
        with engine.ctx.timer.record(f'{name}_central_aggregation'):
            if class_name == 'DistAggConv':
                central_rst = GCN_aggregation(graph.central_graph, central_messages, mode=mode)[:engine.ctx.num_central]
            elif class_name == 'DistAggSAGE':
                central_rst = SAGE_aggregation(graph.central_graph, central_messages, mode=mode, aggregator_type=engine.ctx.agg_type)[:engine.ctx.num_central]
            else:
                raise ValueError(f'Invalid class_name {class_name}')
    # TODO check if need to wait dequantization
    remote_messages = response.get() # block until completion of async messages exchange
    marginal_messages = torch.concat([messages_from_central, local_messages[engine.ctx.num_central:engine.ctx.num_inner], remote_messages], dim=0)
    with engine.ctx.timer.record(f'{name}_marginal_aggregation'):
        if class_name == 'DistAggConv':
            marginal_rst = GCN_aggregation(graph.marginal_graph, marginal_messages, mode=mode)[len(graph.src_central_idx):len(graph.src_central_idx) + engine.ctx.num_marginal]
        elif class_name == 'DistAggSAGE':
            marginal_rst = SAGE_aggregation(graph.marginal_graph, marginal_messages, mode=mode, aggregator_type=engine.ctx.agg_type)[len(graph.src_central_idx):len(graph.src_central_idx) + engine.ctx.num_marginal]
        else:
            raise ValueError(f'Invalid class_name {class_name}')
    return_messages = torch.concat([central_rst, marginal_rst], dim=0)
    # store layer info in forward propagation 
    if mode == ProprogationMode.Forward:
        ctx.saved = layer
        return return_messages
    # return gradients in backward propagation
    else:
        return return_messages, None, None, None