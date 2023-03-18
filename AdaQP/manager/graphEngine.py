import logging
import dgl
from typing import Tuple, Union
from dgl import DGLHeteroGraph

from .conversion import *
from .processing import *
from ..communicator import Communicator as comm

class GraphEngine(object):
    '''
    manage the graph and all kind of nodes feats (e.g., features, labels, mask, etc.).
    '''
    def __init__(self, part_dir='part_data', dataset='dataset', use_parallel=False):
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
        assert num_inner == num_central + num_marginal
        logging.info(f'<worker{comm.get_rank()} local graph total nodes: {reordered_graph.num_nodes()} edges: {reordered_graph.num_edges()} remote nodes: {reordered_graph.num_nodes() - num_inner} central nodes: {num_central} marginal nodes: {num_marginal}>')        
        # TODO decompose local graph according to the `use_parallel` flag
        if use_parallel:
            pass
        # set tags
        self.is_bidirected = is_bidirected
        self.use_parallel = use_parallel
        # set idxs
        self.num_inner, self.num_marginal, self.num_central = num_inner, num_marginal, num_central
        self.send_idx, self.recv_idx, self.scores = send_idx, recv_idx, scores
        self.total_send_idx = total_idx
        # set masks
        self.train_mask = torch.nonzero(reordered_feats['train_mask']).squeeze()
        self.val_mask = torch.nonzero(reordered_feats['val_mask']).squeeze()
        self.test_mask = torch.nonzero(reordered_feats['test_mask']).squeeze()
        # set device
        self.device = comm.ctx.get_device()
        # set forward graph and nodes feats
        self.graph = reordered_graph
        self.nodes_feats = reordered_feats
        # set backward graph
        self.bp_graph = self._get_bp_graph(reordered_graph, use_parallel, is_bidirected)
        # pop unnecessary feats
        self.nodes_feats.pop('train_mask')
        self.nodes_feats.pop('val_mask')
        self.nodes_feats.pop('test_mask')
        self.nodes_feats.pop('part_id')
        self.nodes_feats.pop(dgl.NID)
        self.nodes_feats.pop('feat')
        self.nodes_feats.pop('label')        
        # move to device
        self._move()
        GraphEngine.ctx = self

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
        if self.use_parallel:
            pass
        else:
            self.graph = self.graph.to(self.device)
            if not self.is_bidirected:
                self.bp_graph = self.bp_graph.to(self.device)
            self.nodes_feats['feat'] = self.nodes_feats['feat'].to(self.device)
            self.nodes_feats['label'] = self.nodes_feats['label'].to(self.device)
        
        
        