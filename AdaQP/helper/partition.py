import os
import dgl
import torch
from dgl import DGLHeteroGraph
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import RedditDataset

from .dataset import AmazonProducts, load_yelp

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
    elif dataset == 'amazonProducts':
        data = AmazonProducts(raw_dir=raw_dir)
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
    save_dir = f'graph_degrees/{dataset}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(in_degrees, f'{save_dir}/in_degrees.pt')
    torch.save(out_degrees, f'{save_dir}/out_degrees.pt')
    # partition the whole graph
    print('<begin partition...>')
    dgl.distributed.partition_graph(graph, graph_name=dataset, num_parts=partition_size,
                                    out_path=partition_dir, num_hops=1, balance_edges=False)

