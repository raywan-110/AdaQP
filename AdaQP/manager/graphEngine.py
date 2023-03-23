import logging
import dgl
from typing import Tuple, Union
from dgl import DGLHeteroGraph

from ..util import Timer
from .conversion import *
from .processing import *
from ..communicator import BitType
from ..communicator import Communicator as comm

class GraphEngine(object):
    '''
    manage the graph and all kind of nodes feats (e.g., features, labels, mask, etc.).
    '''
    def __init__(self, part_dir='part_data', dataset='dataset', msg_precision_type: str = 'full', use_parallel=False):
        # load original graph and feats
        original_graph, original_feats, gpb, is_bidirected = convert_partition(part_dir, dataset)
        # get send_idx, recv_idx, and scores
        original_send_idx, recv_idx, scores = get_send_recv_idx_scores(original_graph, original_feats, gpb)
        # reorder graph nodes ID order (from 0 to N-1: central->marginal->remote)
        reordered_graph, reordered_feats, reordered_send_idx, num_marginal, num_central = reorder_graph(original_graph, original_feats, original_send_idx)
        send_idx, total_idx = convert_send_idx(reordered_send_idx)
        reordered_graph.ndata['in_degrees'] = reordered_feats['in_degrees']
        reordered_graph.ndata['out_degrees'] = reordered_feats['out_degrees']
        num_inner = torch.count_nonzero(reordered_feats['inner_node']).item()
        num_remote = reordered_graph.num_nodes() - num_inner
        assert num_inner == num_central + num_marginal
        logging.info(f'<worker{comm.get_rank()} local graph total nodes: {reordered_graph.num_nodes()} edges: {reordered_graph.num_edges()} remote nodes: {num_remote} central nodes: {num_central} marginal nodes: {num_marginal}>')        
        # TODO decompose local graph according to the `use_parallel` flag
        if use_parallel:
            pass
        # set ctx
        GraphEngine.ctx = self
        # set tags
        self._is_bidirected = is_bidirected
        self._use_parallel = use_parallel
        if msg_precision_type == 'full':
            self._bit_type = BitType.FULL
        elif msg_precision_type == 'quantized':
            self._bit_type = BitType.QUANTIZED
        else:
            raise NotImplementedError(f'only full and quantized are supported now, {msg_precision_type} is undifined.')
        # set node info and idx
        self._num_remove, self._num_inner, self._num_marginal, self._num_central = num_remote, num_inner, num_marginal, num_central
        self._send_idx, self._recv_idx, self._scores = send_idx, recv_idx, scores
        self._total_send_idx = total_idx
        # set masks
        self._train_mask = torch.nonzero(reordered_feats['train_mask']).squeeze()
        self._val_mask = torch.nonzero(reordered_feats['val_mask']).squeeze()
        self._test_mask = torch.nonzero(reordered_feats['test_mask']).squeeze()
        # set device
        self._device = comm.ctx.device
        # set forward graph and nodes feats
        self._graph = reordered_graph
        self._nodes_feats = reordered_feats
        # set backward graph
        self._bwd_graph = self._get_bp_graph(reordered_graph, use_parallel, is_bidirected)
        # pop unnecessary feats
        self._nodes_feats.pop('train_mask')
        self._nodes_feats.pop('val_mask')
        self._nodes_feats.pop('test_mask')
        self._nodes_feats.pop('part_id')
        self._nodes_feats.pop(dgl.NID)
        self._nodes_feats.pop('feat')
        self._nodes_feats.pop('label')        
        # move to device
        self._move()
        # init the timer for recording the time cost
        self.timer = Timer(device=self._device)
        # graphSAGE aggregator type if needed
        self._agg_type: str = None

    def _get_bp_graph(self, fp_graph: Union[DGLHeteroGraph, Tuple[DGLHeteroGraph, DGLHeteroGraph]], use_parallel: bool, is_bidirected: bool) -> DGLHeteroGraph:
        if use_parallel:
            pass
        else:
            if not is_bidirected:
                bp_graph = dgl.reverse(fp_graph, copy_ndata=False, copy_edata=False)
            else:
                bp_graph = fp_graph
        return bp_graph
    
    def _move(self):
        if self._use_parallel:
            pass
        else:
            self._graph = self._graph.to(self._device)
            if not self._is_bidirected:
                self.bp_graph = self.bp_graph.to(self._device)
            self._nodes_feats['feat'] = self._nodes_feats['feat'].to(self._device)
            self._nodes_feats['label'] = self._nodes_feats['label'].to(self._device)
    

    # getter methods
    @property
    def device(self):
        return self._device

    @property
    def is_bidirected(self):
        return self._is_bidirected
    
    @property
    def use_parallel(self):
        return self._use_parallel
    
    @property
    def bit_type(self):
        return self._bit_type
    
    @property
    def agg_type(self):
        assert self._agg_type is not None, 'please set the aggregator type first.'
        return self._agg_type
    
    @property
    def num_remove(self):
        return self._num_remove
    
    @property
    def num_inner(self):
        return self._num_inner
    
    @property
    def num_marginal(self):
        return self._num_marginal
    
    @property
    def num_central(self):
        return self._num_central
    
    @property
    def send_idx(self):
        return self._send_idx
    
    @property
    def recv_idx(self):
        return self._recv_idx
    
    @property
    def scores(self):
        return self._scores
    
    @property
    def total_send_idx(self):
        return self._total_send_idx
    
    @property
    def bwd_graph(self):
        return self._bwd_graph

    
        
        
        