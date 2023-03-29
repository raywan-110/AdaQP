import os
import ssl
import sys
import urllib
import json
import dgl
import torch
from dgl.data.dgl_dataset import DGLDataset
from dgl import DGLHeteroGraph
from typing import Optional
from sklearn.preprocessing import StandardScaler
import numpy as np
import scipy.sparse as sp

# Amazon dataset
def download_url(url: str, folder: str, log: bool = True,
                 filename: Optional[str] = None):
    r"""Downloads the content of an URL to a specific folder.

    Args:
        url (string): The url.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """

    if filename is None:
        filename = url.rpartition('/')[2]
        filename = filename if filename[0] == '?' else filename.split('?')[0]
    path = os.path.join(folder, filename)
    if os.path.exists(path):  # pragma: no cover
        if log:
            print(f'Using existing file {filename}', file=sys.stderr)
        return path
    if log:
        print(f'Downloading {url}', file=sys.stderr)
    if not os.path.exists(folder):
        os.makedirs(folder)
    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)
    with open(path, 'wb') as f:
        # workaround for https://bugs.python.org/issue42853
        while True:
            chunk = data.read(10 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    return path

class AmazonProducts(DGLDataset):
    def __init__(self, raw_dir: str=None, force_reload: bool=False, verbose: bool=False):
        _url = 'https://docs.google.com/uc?export=download&id={}&confirm=t'
        super(AmazonProducts, self).__init__(name='amazonProducts', url=_url, raw_dir=raw_dir, force_reload=force_reload, verbose=verbose)
    
    def download(self):
        adj_full_id = '17qhNA8H1IpbkkR-T2BmPQm8QNW5do-aa'
        feats_id = '10SW8lCvAj-kb6ckkfTOC5y0l8XXdtMxj'
        class_map_id = '1LIl4kimLfftj4-7NmValuWyCQE8AaE7P'
        role_id = '1npK9xlmbnjNkV80hK2Q68wTEVOFjnt4K'
        if os.path.exists(self.raw_path):  # pragma: no cover
            return
        path = download_url(self.url.format(adj_full_id), self.raw_path)
        os.rename(path, os.path.join(self.raw_path, 'adj_full.npz'))

        path = download_url(self.url.format(feats_id), self.raw_path)
        os.rename(path, os.path.join(self.raw_path, 'feats.npy'))

        path = download_url(self.url.format(class_map_id), self.raw_path)
        os.rename(path, os.path.join(self.raw_path, 'class_map.json'))

        path = download_url(self.url.format(role_id), self.raw_path)
        os.rename(path, os.path.join(self.raw_path, 'role.json'))

    def process(self):
        f = np.load(os.path.join(self.raw_path, 'adj_full.npz'))
        # graph
        adj = sp.csr_matrix((f['data'], f['indices'], f['indptr']), f['shape'])
        adj = adj.tocoo()
        self._graph = dgl.from_scipy(adj)
        # features and labels
        amazon_data = np.load(os.path.join(self.raw_path, 'feats.npy'))
        amazon_data = torch.from_numpy(amazon_data).to(torch.float32)
        ys = [-1] * amazon_data.size(0)
        with open(os.path.join(self.raw_path, 'class_map.json')) as f:
            class_map = json.load(f)
            for key, item in class_map.items():
                ys[int(key)] = item
        labels = torch.tensor(ys, dtype=torch.float32)
        # train/val/test indices
        with open(os.path.join(self.raw_path, 'role.json')) as f:
            role = json.load(f)
        train_mask = torch.zeros(amazon_data.size(0), dtype=torch.bool)
        train_mask[torch.tensor(role['tr'])] = True
        val_mask = torch.zeros(amazon_data.size(0), dtype=torch.bool)
        val_mask[torch.tensor(role['va'])] = True
        test_mask = torch.zeros(amazon_data.size(0), dtype=torch.bool)
        test_mask[torch.tensor(role['te'])] = True
        # add all the data to graph 
        self._graph.ndata['train_mask'] = train_mask
        self._graph.ndata['val_mask'] = val_mask
        self._graph.ndata['test_mask'] = test_mask
        self._graph.ndata['feat'] = amazon_data
        self._graph.ndata['label'] = labels
        # reorder graph
        self._graph = dgl.reorder_graph(self._graph, node_permute_algo='rcmk', edge_permute_algo='dst', store_ids=False)  

    @property
    def num_classes(self):
        return 107
    
    @property
    def num_labels(self):
        return self.num_classes
    
    def __getitem__(self, idx):
        assert idx == 0, "AmazonProducts Dataset only has noe graph"
        return self._graph
    
    def __len__(self):
        return 1

# yelp dataset
def load_yelp(raw_dir='/dataset') -> DGLHeteroGraph:
    prefix = f'{raw_dir}/yelp/'

    with open(prefix + 'class_map.json') as f:
        class_map = json.load(f)
    with open(prefix + 'role.json') as f:
        role = json.load(f)

    adj_full = sp.load_npz(prefix + 'adj_full.npz')
    feats = np.load(prefix + 'feats.npy')
    n_node = feats.shape[0]

    g = dgl.from_scipy(adj_full)
    node_data = g.ndata

    label = list(class_map.values())
    node_data['label'] = torch.tensor(label)

    node_data['train_mask'] = torch.zeros(n_node, dtype=torch.bool)
    node_data['val_mask'] = torch.zeros(n_node, dtype=torch.bool)
    node_data['test_mask'] = torch.zeros(n_node, dtype=torch.bool)
    node_data['train_mask'][role['tr']] = True
    node_data['val_mask'][role['va']] = True
    node_data['test_mask'][role['te']] = True

    assert torch.all(torch.logical_not(torch.logical_and(node_data['train_mask'], node_data['val_mask'])))
    assert torch.all(torch.logical_not(torch.logical_and(node_data['train_mask'], node_data['test_mask'])))
    assert torch.all(torch.logical_not(torch.logical_and(node_data['val_mask'], node_data['test_mask'])))
    assert torch.all(
        torch.logical_or(torch.logical_or(node_data['train_mask'], node_data['val_mask']), node_data['test_mask']))

    train_feats = feats[node_data['train_mask']]
    scaler = StandardScaler()
    scaler.fit(train_feats)
    feats = scaler.transform(feats)

    node_data['feat'] = torch.tensor(feats, dtype=torch.float)

    return g
