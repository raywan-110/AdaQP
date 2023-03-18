import os
import dgl
import torch
import logging
from torch import Tensor
from dgl import DGLHeteroGraph
from dgl.distributed import GraphPartitionBook
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import RedditDataset
from typing import Dict, Tuple

from ..communicator import Communicator as comm
from ..communicator import Basic_Buffer_Type
from ..model import DistGNNType
from .dataset import Amazon2m, load_yelp

def process_obg_dataset(dataset: str, raw_dir: str) -> DGLHeteroGraph:
    '''
    process the ogb dataset, return a dgl graph with node features, labels and train/val/test masks.
    '''
    data = DglNodePropPredDataset(name=dataset, root=raw_dir)
    graph, labels = data[0]
    labels = labels[:, 0]
    graph.ndata['label'] = labels
    # split the dataset into tain/val/test before partitioning
    splitted_idx = data.get_idx_split()
    train_nid, val_nid, test_nid = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
    train_mask = torch.zeros((graph.number_of_nodes(),), dtype=torch.bool)
    train_mask[train_nid] = True
    val_mask = torch.zeros((graph.number_of_nodes(),), dtype=torch.bool)
    val_mask[val_nid] = True
    test_mask = torch.zeros((graph.number_of_nodes(),), dtype=torch.bool)
    test_mask[test_nid] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    return graph

def graph_patition_store(dataset: str, partition_size: int, raw_dir: str = 'dataset', part_dir: str = 'part_data'):
    '''
    partition the dataset and store them as partition grpah.
    
    `ATTENTION`:
    
    we set HALO hop as 1 to save cross-device neighboring nodes' indices for constructing send/recv idx.
    '''
    # the dir to store graph partition
    partition_dir = '{}/{}/{}part'.format(part_dir, dataset, partition_size)
    if os.path.exists(partition_dir):
        return
    if 'ogbn' in dataset:
        # load and process ogbn-dataset
        graph = process_obg_dataset(dataset, raw_dir)
    elif dataset == 'reddit':
        data = RedditDataset(raw_dir=raw_dir)
        graph = data[0]
    elif dataset == 'amazon2m':
        data = Amazon2m(raw_dir=raw_dir)
        graph = data[0]
    elif dataset == 'yelp':
        graph = load_yelp(raw_dir=raw_dir)
    else:
        raise ValueError(f'no such dataset: {dataset}')
    # add self loop
    graph.edata.clear()
    graph = dgl.remove_self_loop(graph)
    graph = dgl.add_self_loop(graph)
    # save global degrees
    in_degrees = graph.in_degrees()
    out_degrees = graph.out_degrees()
    save_dir = f'graph_info/{dataset}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(in_degrees, f'{save_dir}/in_degrees.pt')
    torch.save(out_degrees, f'{save_dir}/out_degrees.pt')
    # partition the whole graph
    logging.info('begin partition...')
    dgl.distributed.partition_graph(graph, graph_name=dataset, num_parts=partition_size,
                                    out_path=partition_dir, num_hops=1, balance_edges=False)

def get_send_recv_idx_scores(local_graph: dgl.DGLHeteroGraph, nodes_feats: Dict[str, Tensor], gpb: GraphPartitionBook, dataset: str, model_type: DistGNNType, part_dir: str = 'part_data') -> Tuple[Basic_Buffer_Type, Basic_Buffer_Type, Basic_Buffer_Type]:
    '''
    get send/recv idx and agg scores for each node in the local graph. 
    '''
    rank, world_size = comm.get_rank(), comm.get_world_size()
    current_partition_dir = f'{part_dir}/{dataset}/{world_size}part/part{rank}'
    except_list = [None for _ in range(world_size)]
    except_info = None
    # try loading send/recv idx from disk
    try:
        send_idx = torch.load(f'{current_partition_dir}/send_idx.pt')
        recv_idx = torch.load(f'{current_partition_dir}/recv_idx.pt')
        agg_scores = torch.load(f'{current_partition_dir}/agg_scores.pt')
    except IOError as e:
        except_info = str(e)
    # check if all processes have loaded send/recv idx successfully
    comm.all_gather_any(except_list, except_info)
    # if not, build send/recv idx and store them to the disk
    if not all([except_info is None for except_info in except_list]):
        fail_idx = [i for i, except_info in enumerate(except_list) if except_info is not None]
        logging.info(f'<worder {fail_idx} failed to load send/recv idx from disk, begin building...>')
        send_idx, recv_idx, agg_scores = _build_store_send_recv_idx_scores(local_graph, nodes_feats, gpb, current_partition_dir, model_type)
    return send_idx, recv_idx, agg_scores
        

def _build_store_send_recv_idx_scores(local_graph: dgl.DGLHeteroGraph, nodes_feats: Dict[str, Tensor], gpb: GraphPartitionBook, part_dir: str, model_type: DistGNNType) -> Tuple[Basic_Buffer_Type, Basic_Buffer_Type, Basic_Buffer_Type]:
    '''
    build send/recv idx for each partition and store them to the disk.
    '''
    rank, world_size = comm.get_rank(), comm.get_world_size()
    temp_send_idx: Dict[int, Tensor] = {}
    recv_idx: Dict[int, Tensor] = {}
    scores: Dict[int, Tensor] = {}
    outer_start = len(torch.nonzero(nodes_feats['inner_node']))
    # build recv_idx and temp_idx
    temp_buffer: Dict[int , Tuple(Tensor, Tensor)] = {}
    for i in range(world_size):
        if i is not rank:
            belong2i = (nodes_feats['part_id'] == i)
            start = gpb.partid2nids(i)[0].item()
            remote_ids = nodes_feats[dgl.NID][belong2i] - start
            # get forward & backward aggreagtion scores for remote neighbors
            agg_score = _get_agg_scores(belong2i, local_graph, nodes_feats, model_type)
            local_ids = torch.nonzero(belong2i).view(-1) - outer_start
            temp_buffer[i] = (remote_ids, agg_score)
            recv_idx[i] = local_ids  # data recv from i
    # print recv_idx in debug mode
    for k, v in recv_idx.items():
        logging.debug(f'<worker{rank} recv {len(v)} nodes from worker{k}>')
    # build temp_send_idx and scores
    temp_buffer_list = [None for _ in range(world_size)]
    comm.all_gather_any(temp_buffer_list, temp_buffer)
    for i in range(world_size):
        if i is not rank:
            if rank in temp_buffer_list[i].keys():
                temp_send_idx[i] = temp_buffer_list[i][rank][0]   # data from i to rank
                scores[i] = temp_buffer_list[i][rank][1]  # score from i to rank
    # print temp_send_idx in debug mode
    for k, v in temp_send_idx.items():
        logging.debug(f'<worker{rank} send {len(v)} nodes to worker{k}>')
    # store send_idx, recv_idx and scores to disk
    torch.save(f'{part_dir}/send_idx.pt', temp_send_idx)
    torch.save(f'{part_dir}/recv_idx.pt', recv_idx)
    torch.save(f'{part_dir}/agg_scores.pt', scores)
    return temp_send_idx, recv_idx, scores

def _get_agg_scores(local_graph: dgl.DGLHeteroGraph, belong_mask: Tensor, nodes_feats: Dict[str, Tensor], model_type: DistGNNType) -> Tuple[Tensor, Tensor]:
    '''
    return aggregation scores for each node in the local graph.
    '''
    fp_neighbor_ids = local_graph.out_edges(local_graph.nodes()[belong_mask])[1]
    fp_local_degrees = local_graph.out_degrees(local_graph.nodes()[belong_mask])
    bp_neighbor_ids = local_graph.in_edges(local_graph.nodes()[belong_mask])[0]
    bp_local_degrees = local_graph.in_degrees(local_graph.nodes()[belong_mask])
    if model_type is DistGNNType.DistGCN:
        # construct forward score
        fp_global_degrees = nodes_feats['out_degrees'][belong_mask]
        score = torch.pow(nodes_feats['in_degrees'][fp_neighbor_ids].float().clamp(min=1), -0.5).split(fp_local_degrees.tolist())
        fp_agg_score = torch.tensor([sum(score[i] * torch.pow(fp_global_degrees[i].float().clamp(min=1), -0.5)) for i in range(len(score))])
        # construct backward score
        bp_global_degrees = nodes_feats['in_degrees'][belong_mask]
        score = torch.pow(nodes_feats['out_degrees'][bp_neighbor_ids].float().clamp(min=1), -0.5).split(bp_local_degrees.tolist())
        bp_agg_score = torch.tensor([sum(score[i] * torch.pow(bp_global_degrees[i].float().clamp(min=1), -0.5)) for i in range(len(score))])
    elif model_type is DistGNNType.DistGraphSAGE:
        # construct forward score
        score = torch.pow(nodes_feats['in_degrees'][fp_neighbor_ids].float().clamp(min=1), -1).split(fp_local_degrees.tolist())
        fp_agg_score = torch.tensor([sum(value) for value in score])
        # construct backward score
        score = torch.pow(nodes_feats['out_degrees'][bp_neighbor_ids].float().clamp(min=1), -1).split(bp_local_degrees.tolist())
        bp_agg_score = torch.tensor([sum(value) for value in score])
    else:
        raise NotImplementedError(f'{model_type} is not implemented yet.')
    return (fp_agg_score, bp_agg_score)
        
        

        
    
    

