import dgl
import torch
from dgl import DGLHeteroGraph
from dgl.distributed import GraphPartitionBook
from torch import Tensor, BoolTensor
from typing import Dict, Tuple

from ..communicator import Communicator as comm
from ..communicator import Basic_Buffer_Type

def convert_partition(part_dir: str, dataset: str) -> Tuple[dgl.DGLHeteroGraph, Dict[str, Tensor], GraphPartitionBook, bool]:
    rank, world_size = comm.get_rank(), comm.get_world_size()
    part_config = f'{part_dir}/{dataset}/{world_size}part/{dataset}.json'
    g, nodes_feats, _, gpb, _, node_type, _ = dgl.distributed.load_partition(part_config, rank)
    # set graph degrees for GNNs aggregation
    save_dir = f'graph_degrees/{dataset}'
    # load global degrees information 
    in_degrees_global, out_degrees_global = torch.load(f'{save_dir}/in_degrees.pt'), torch.load(f'{save_dir}/out_degrees.pt')
    degree_ids = g.ndata['orig_id']
    nodes_feats['in_degrees'] = in_degrees_global[degree_ids]
    nodes_feats['out_degrees'] = out_degrees_global[degree_ids]
    # TODO handle these settings
    is_bidirected = False
    if all(nodes_feats['in_degrees'] == nodes_feats['out_degrees']):
        is_bidirected = True
    else:
        is_bidirected = False
    # move all the features to nodes_feats
    node_type = node_type[0]
    # save original degrees for fp and bp
    nodes_feats[dgl.NID] = g.ndata[dgl.NID]
    nodes_feats['part_id'] = g.ndata['part_id']
    nodes_feats['inner_node'] = g.ndata['inner_node'].bool()
    nodes_feats['label'] = nodes_feats[node_type + '/label']
    nodes_feats['feat'] = nodes_feats[node_type + '/feat']
    nodes_feats['train_mask'] = nodes_feats[node_type + '/train_mask'].bool()
    nodes_feats['val_mask'] = nodes_feats[node_type + '/val_mask'].bool()
    nodes_feats['test_mask'] = nodes_feats[node_type + '/test_mask'].bool()

    # remove redundant feats
    nodes_feats.pop(node_type + '/val_mask')
    nodes_feats.pop(node_type + '/test_mask')
    nodes_feats.pop(node_type + '/label')
    nodes_feats.pop(node_type + '/feat')
    nodes_feats.pop(node_type + '/train_mask')
    # only remain topology of graph
    g.ndata.clear()
    g.edata.clear()
    return g, nodes_feats, gpb, is_bidirected

def reorder_graph(local_graph: DGLHeteroGraph, nodes_feats: Dict[str, Tensor], send_idx: Basic_Buffer_Type) -> Tuple[DGLHeteroGraph, Dict[str, Tensor], Basic_Buffer_Type, int, int]:
    # get inner mask and remote nodes
    inner_mask = nodes_feats['inner_node']
    num_inner = torch.count_nonzero(inner_mask).item()
    remote_nodes = local_graph.nodes()[~inner_mask]
    # get marginal dst nodes
    _, v = local_graph.out_edges(remote_nodes)
    marginal_nodes = torch.unique(v)
    # set marginal mask and central mask
    marginal_mask = torch.zeros_like(inner_mask, dtype=torch.bool)
    marginal_mask[marginal_nodes] = True
    central_mask = torch.concat([~marginal_mask[:num_inner], marginal_mask[num_inner:]])
    num_marginal, num_central = torch.count_nonzero(marginal_mask).item(), torch.count_nonzero(central_mask).item()
    new_graph, nodes_feats, send_idx = _reorder(local_graph, nodes_feats, send_idx, num_inner, num_central, marginal_mask, central_mask)
    assert new_graph.num_nodes() == local_graph.num_nodes()
    assert new_graph.num_edges() == local_graph.num_edges()
    return new_graph, nodes_feats, send_idx, num_marginal, num_central

def _reorder(input_graph: DGLHeteroGraph, nodes_feats: Dict[str, Tensor], send_idx: Basic_Buffer_Type, num_inner: int, num_central: int, m_mask: BoolTensor, c_mask: BoolTensor) -> Tuple[DGLHeteroGraph, Dict[str, Tensor], Basic_Buffer_Type]:
    '''reorder local nodes and return new graph and feats dict'''
    # get the new ids
    new_id = torch.zeros(size=(num_inner,), dtype=torch.long)
    new_id[c_mask[:num_inner]] = torch.arange(num_central, dtype=torch.long)
    new_id[m_mask[:num_inner]] = torch.arange(num_central, num_inner, dtype=torch.long)
    u, v = input_graph.edges()
    u[u < num_inner] = new_id[u[u < num_inner].long()]
    v[v < num_inner] = new_id[v[v < num_inner].long()]
    reordered_graph = dgl.graph((u, v))
    # reorder all the feats
    for key in nodes_feats:
        nodes_feats[key][new_id] = nodes_feats[key].clone()[0:num_inner]
    # reorder send idx according to the new order
    for key in send_idx:
        send_idx[key] = new_id[send_idx[key]]
    return reordered_graph, nodes_feats, send_idx

def convert_send_idx(original_send_idx: Basic_Buffer_Type) -> Tuple[Dict[int, Tuple[int, int]], Tensor]:
    '''
    convert original layout of send_idx into offset fashion.
    
    `ATTENTION`: this function should be called after graph reordering.
    '''
    offset = 0
    converted_send_idx: Dict[int, Tuple[int, int]] = {}
    total_idx = []
    for k, v in original_send_idx.items():
        converted_send_idx[k] = (offset, offset + len(v))
        offset += len(v)
        total_idx.append(v)
    total_idx = torch.cat(total_idx)
    return converted_send_idx, total_idx