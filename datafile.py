import os.path as osp
import numpy as np
import torch
from torch_geometric.datasets import Planetoid, CitationFull, WikiCS, Coauthor, Amazon, WikipediaNetwork, Reddit
import torch_geometric.transforms as T
from torch_geometric.datasets import KarateClub
from sklearn.preprocessing import label_binarize
import scipy.io
from torch_geometric.data import Data
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import torch_geometric.transforms as transforms


def get_dataset(path, name):
    assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP', 'Karate', 'WikiCS', 'Coauthor-CS', 'Coauthor-Phy', 'Reddit', 'ogbn-products',
                    'Amazon-Computers', 'Amazon-Photo', 'ogbn-arxiv', 'ogbg-code', 'Chameleon', 'Squirrel', 'arxiv-year']
    name = 'dblp' if name == 'DBLP' else name
    root_path = osp.expanduser('./datasets')
    if name == 'Chameleon':
        #raw = dict(np.load(path+'/chameleon.npz', allow_pickle=True))
        #return Data(x=torch.Tensor(raw['features']), y=torch.LongTensor(raw['label']),
        #            edge_index=torch.LongTensor(raw['edges']).t())
        return WikipediaNetwork(root=path, name='chameleon', geom_gcn_preprocess=True, transform=T.NormalizeFeatures())
    if name == 'Squirrel':
        #raw = dict(np.load(path + '/squirrel.npz', allow_pickle=True))
        #return Data(x=torch.Tensor(raw['features']), y=torch.LongTensor(raw['label']),
        #            edge_index=torch.LongTensor(raw['edges']).t())
        return WikipediaNetwork(root=path, name='squirrel', geom_gcn_preprocess=True, transform=T.NormalizeFeatures())
    if name == 'Coauthor-CS':
        return Coauthor(root=path, name='cs', transform=T.NormalizeFeatures())

    if name == 'Coauthor-Phy':
        return Coauthor(root=path, name='physics', transform=T.NormalizeFeatures())

    if name == 'WikiCS':
        return WikiCS(root=path, transform=T.NormalizeFeatures())

    if name == 'Amazon-Computers':
        return Amazon(root=path, name='computers', transform=T.NormalizeFeatures())

    if name == 'Amazon-Photo':
        return Amazon(root=path, name='photo', transform=T.NormalizeFeatures())

    if name == 'Karate':
        return KarateClub(root=path, transform=T.NormalizeFeatures())

    if name == "arxiv-year":
        # arxiv-year uses the same graph and features as ogbn-arxiv, but with different labels
        return PygNodePropPredDataset(name="ogbn-arxiv", transform=transforms.ToSparseTensor(), root=path)

    if name == 'ogbn-products':
       return PygNodePropPredDataset('ogbn-products', root=path,  transform=T.NormalizeFeatures())
    if name.startswith('ogbn'):
        return PygNodePropPredDataset(root=osp.join(root_path, 'OGB'), name=name, transform=T.NormalizeFeatures())

    return (CitationFull if name == 'dblp' else Planetoid)(osp.join(root_path, 'Citation'), name, transform=T.NormalizeFeatures())


def get_path(base_path, name):
    if name in ['Cora', 'CiteSeer', 'PubMed']:
        return base_path
    else:
        return osp.join(base_path, name)

def load_fb100(filename):
    # e.g. filename = Rutgers89 or Cornell5 or Wisconsin87 or Amherst41
    # columns are: student/faculty, gender, major,
    #              second major/minor, dorm/house, year/ high school
    # 0 denotes missing entry
    mat = scipy.io.loadmat('./datasets/facebook100/' + filename + '.mat')
    A = mat['A']
    metadata = mat['local_info']
    return A, metadata

def load_fb100_dataset(filename):
    A, metadata = load_fb100(filename)
    dataset = NCDataset(filename)
    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    metadata = metadata.astype(np.int)
    label = metadata[:, 1] - 1  # gender label, -1 means unlabeled

    # make features into one-hot encodings
    feature_vals = np.hstack(
        (np.expand_dims(metadata[:, 0], 1), metadata[:, 2:]))
    features = np.empty((A.shape[0], 0))
    for col in range(feature_vals.shape[1]):
        feat_col = feature_vals[:, col]
        feat_onehot = label_binarize(feat_col, classes=np.unique(feat_col))
        features = np.hstack((features, feat_onehot))

    node_feat = torch.tensor(features, dtype=torch.float)
    num_nodes = metadata.shape[0]
    num_node_features = node_feat.shape[1]

    dataset.graph = {'edge_index': edge_index,
                     'num_node_features': num_node_features,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}
    dataset.label = torch.tensor(label)
    return dataset, A
class NCDataset(object):
    def __init__(self, name, root='./datasets'):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None,
                    but when something is passed, it uses its information. Useful for debugging for external contributers.

        Usage after construction:

        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]

        Where the graph is a dictionary of the following form:
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/
        """

        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = {}
        self.label = None


    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))

def load_fb100_data(filename):
    A, metadata = load_fb100(filename)
    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    metadata = metadata.astype(np.int)
    label = metadata[:, 1] - 1  # gender label, -1 means unlabeled

    # make features into one-hot encodings
    feature_vals = np.hstack(
        (np.expand_dims(metadata[:, 0], 1), metadata[:, 2:]))
    features = np.empty((A.shape[0], 0))
    for col in range(feature_vals.shape[1]):
        feat_col = feature_vals[:, col]
        feat_onehot = label_binarize(feat_col, classes=np.unique(feat_col))
        features = np.hstack((features, feat_onehot))

    node_feat = torch.tensor(features, dtype=torch.float)

    data = Data(x=node_feat, y=torch.tensor(label), edge_index=edge_index)
    return data
