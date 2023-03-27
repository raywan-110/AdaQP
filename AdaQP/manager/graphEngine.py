import dgl
from typing import Tuple, Union
from dgl import DGLHeteroGraph

from ..util import Timer, Recorder
from .conversion import *
from .processing import *
from ..communicator import Communicator as comm
from ..helper import BitType, DistGNNType

class GraphEngine(object):
    '''
    manage the graph and all kind of nodes feats (e.g., features, labels, mask, etc.).
    '''
    def __init__(self, epoches: int, part_dir, dataset, msg_precision_type: str, model_type: DistGNNType, use_parallel=False):
        # load original graph and feats
        original_graph, original_feats, gpb, is_bidirected = convert_partition(part_dir, dataset)
        # get send_idx, recv_idx, and scores
        original_send_idx, recv_idx, scores = get_send_recv_idx_scores(original_graph, original_feats, gpb, part_dir, dataset, model_type)
        # reorder graph nodes ID order (from 0 to N-1: central->marginal->remote)
        reordered_graph, reordered_feats, reordered_send_idx, num_marginal, num_central = reorder_graph(original_graph, original_feats, original_send_idx)
        send_idx, total_idx = convert_send_idx(reordered_send_idx)
        reordered_graph.ndata['in_degrees'] = reordered_feats['in_degrees']
        reordered_graph.ndata['out_degrees'] = reordered_feats['out_degrees']
        num_inner = torch.count_nonzero(reordered_feats['inner_node']).item()
        num_remote = reordered_graph.num_nodes() - num_inner
        assert num_inner == num_central + num_marginal
        # TODO decompose local graph according to the `use_parallel` flag
        if use_parallel:
            pass
        # set tags
        self._is_bidirected = is_bidirected
        self._use_parallel = use_parallel
        if msg_precision_type == 'full':
            self._bit_type = BitType.FULL
        elif msg_precision_type == 'quant':
            self._bit_type = BitType.QUANT
        else:
            raise NotImplementedError(f'only full and quant are supported now, {msg_precision_type} is undifined.')
        # set node info and idx
        self._num_remove, self._num_inner, self._num_marginal, self._num_central = num_remote, num_inner, num_marginal, num_central
        self._send_idx, self._recv_idx, self._scores = send_idx, recv_idx, scores
        self._total_send_idx = total_idx
        # set feats and labels
        self.feats = reordered_feats['feat']
        self.labels = reordered_feats['label']
        # set masks
        self.train_mask = torch.nonzero(reordered_feats['train_mask']).squeeze()
        self.val_mask = torch.nonzero(reordered_feats['val_mask']).squeeze()
        self.test_mask = torch.nonzero(reordered_feats['test_mask']).squeeze()
        # set device
        self._device = comm.ctx.device
        # set forward graph and nodes feats
        self.graph = reordered_graph
        # pop unnecessary feats
        reordered_feats.pop('in_degrees')
        reordered_feats.pop('out_degrees')
        reordered_feats.pop('feat')
        reordered_feats.pop('label')
        reordered_feats.pop('train_mask')
        reordered_feats.pop('val_mask')
        reordered_feats.pop('test_mask')
        reordered_feats.pop('part_id')
        reordered_feats.pop(dgl.NID)   
        # move to device
        self._move()
        # set backward graph after moving 
        self._set_bwd_graph(use_parallel, is_bidirected)
        # init the timer for recording the time cost
        self.timer = Timer(device=self._device)
        # init the recorder for recording metrics
        self.recorder = Recorder(epoches)
        # graphSAGE aggregator type if needed
        self._agg_type: str = None
        # set ctx
        GraphEngine.ctx = self
    
    def __repr__(self):
        return  f'<GraphEngine(rank: {comm.get_rank()}, total nodes: {self.graph.num_nodes()}, edges: {self.graph.num_edges()} remote nodes: {self.num_remove} central nodes: {self.num_central}, marginal nodes: {self.num_marginal})>'

    def _set_bwd_graph(self, use_parallel: bool, is_bidirected: bool) -> DGLHeteroGraph:
        if use_parallel:
            pass
        else:
            if not is_bidirected:
                self.bwd_graph = dgl.reverse(self.graph, copy_ndata=False, copy_edata=False)
            else:
                self.bwd_graph = self.graph

    def _move(self):
        if self._use_parallel:
            pass
        else:
            # model the (foward) graph
            self.graph = self.graph.to(self._device)
            # move feats and labels
            self.feats = self.feats.to(self._device)
            self.labels = self.labels.to(self._device)
            # move masks
            self.train_mask = self.train_mask.to(self._device)
            self.val_mask = self.val_mask.to(self._device)
            self.test_mask = self.test_mask.to(self._device)
    
    '''
    *************************************************
    ***************** getter methods ****************
    *************************************************
    '''

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
    
    @agg_type.setter
    def agg_type(self, agg_type: str):
        self._agg_type = agg_type
    
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


    
        
        
        