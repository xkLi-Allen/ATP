import networkx as nx
import numpy as np
import os.path as osp
import pickle as pkl
import scipy.sparse as sp
import torch


from dataset.base_data import Graph
from dataset.base_dataset import NodeDataset
from dataset.utils import pkl_read_file, download_to, remove_self_loops, coomatrix_to_torch_tensor, edge_homophily, node_homophily, linkx_homophily


class Planetoid(NodeDataset):
    '''
    Dataset description: (Paper with Code): https://paperswithcode.com/datasets
    Directed infomation:    Undirected network from original open source dataset (cora & citeseer & pubmed)
                            Notably, citeseer contains isolated nodes.

    -> cora:        unsigned & undirected & unweighted homogeneous simplex network
    -> citeseer:    unsigned & undirected & unweighted homogeneous simplex network
    -> pubmed:      unsigned & undirected & unweighted homogeneous simplex network

    We remove all self-loops from the original dataset, which is consistent with reports in original paper -> NeurIPS'21 Large Scale Learning on Non-Homophilous Graphs: New Benchmarks and Strong Simple Methods, LINKX: https://arxiv.org/pdf/2110.14446.pdf

    cora:       The Cora dataset consists of 2708 scientific publications classified into one of seven classes. 
                The citation network consists of 5429 links. 
                Each publication in the dataset is described by a 0/1-valued word vector indicating the absence/presence of the corresponding word from the dictionary. 
                The dictionary consists of 1433 unique words.
                2,708 nodes, 5,278 edges, 1,433 feature dimensions, 7 classes num.
                Edge homophily: 0.81, Node homophily:0.8252, Linkx homophily:0.7657.

    citeseer:   The CiteSeer dataset consists of 3312 scientific publications classified into one of six classes. 
                The citation network consists of 4732 links. 
                Each publication in the dataset is described by a 0/1-valued word vector indicating the absence/presence of the corresponding word from the dictionary. 
                The dictionary consists of 3703 unique words.
                3,327 nodes, 4,552 edges, 3,703 feature dimensions, 6 classes num.
                Edge homophily: 0.7355, Node homophily:0.7166, Linkx homophily:0.6267.

    pubmed:     The Pubmed dataset consists of 19717 scientific publications from PubMed database pertaining to diabetes classified into one of three classes. 
                The citation network consists of 44338 links. 
                Each publication in the dataset is described by a TF/IDF weighted word vector from a dictionary which consists of 500 unique words.
                19,717 nodes, 44,324 edges, 500 feature dimensions, 3 classes num.
                Edge homophily: 0.8024, Node homophily:0.7924, Linkx homophily:0.6641.

    split:
        official: 
            train_idx: classes num * 20
            val_idx: 500
            test_idx: 1000
    '''

    def __init__(self, name="cora", root="./dataset/homo_data/", split="official", k=None):
        name = name.lower()
        if name not in ["cora", "citeseer", "pubmed"]:
            raise ValueError("Dataset name not supported!")
        super(Planetoid, self).__init__(root + "Planetoid/", name, k)

        self.read_file()
        self.split = split
        self.train_idx, self.val_idx, self.test_idx = self.generate_split(
            split)

    @property
    def raw_file_paths(self):
        filenames = ["x", "tx", "allx", "y",
                     "ty", "ally", "graph", "test.index"]
        return [osp.join(self.raw_dir, "ind.{}.{}".format(self.name, filename)) for filename in filenames]

    @property
    def processed_file_paths(self):
        filename = "graph"
        return osp.join(self.processed_dir, "{}.{}".format(self.name, filename))

    def read_file(self):
        self.data = pkl_read_file(self.processed_file_paths)
        self.edge = self.data.edge
        self.node = self.data.node
        self.x = self.data.x
        self.y = self.data.y
        self.adj = self.data.adj
        self.edge_type = self.data.edge_type
        self.num_features = self.data.num_features
        self.num_classes = self.data.num_classes
        self.num_node = self.data.num_node
        self.num_edge = self.data.num_edge

    def download(self):
        url = "https://github.com/kimiyoung/planetoid/raw/master/data"
        for filepath in self.raw_file_paths:
            file_url = url + '/' + osp.basename(filepath)
            print(file_url)
            download_to(file_url, filepath)

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        np.seterr(divide='ignore', invalid='ignore')
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def process(self):
        objects = []
        for raw_file in self.raw_file_paths[:-1]:
            objects.append(pkl_read_file(raw_file))

        x, tx, allx, y, ty, ally, graph = tuple(objects)

        test_idx_reorder = []
        with open(self.raw_file_paths[-1], 'r') as rf:
            try:
                for line in rf:
                    test_idx_reorder.append(int(line.strip()))
            except IOError as e:
                print(e)
                exit(1)
        test_idx_range = np.sort(test_idx_reorder)

        if self.name == "citeseer":
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vectors into the right position
            test_idx_range_full = range(
                min(test_idx_reorder), max(test_idx_reorder) + 1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range - min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range - min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        features = self.normalize(features)
        features = np.array(features.todense())
        num_node = features.shape[0]

        edge_index = sp.coo_matrix(
            nx.adjacency_matrix(nx.from_dict_of_lists(graph)))
        edge_index = coomatrix_to_torch_tensor(edge_index)
        undi_edge_index = torch.unique(edge_index, dim=1)
        undi_edge_index = remove_self_loops(undi_edge_index)[0]
        row, col = undi_edge_index
        edge_weight = torch.ones(len(row))
        edge_type = "UUU"

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]
        labels = np.argmax(labels, 1)
        labels = torch.LongTensor(labels)

        g = Graph(row, col, edge_weight, num_node,
                  edge_type, x=features, y=labels)
        with open(self.processed_file_paths, 'wb') as rf:
            try:
                pkl.dump(g, rf)
            except IOError as e:
                print(e)
                exit(1)

    def generate_split(self, split):
        if split == "official":
            train_idx = range(self.num_classes * 20)
            val_idx = range(self.num_classes * 20, self.num_classes * 20 + 500)
            test_idx = range(self.num_node - 1000, self.num_node)
        elif split == "random":
            raise NotImplementedError
        else:
            raise ValueError("Please input valid split pattern!")

        return train_idx, val_idx, test_idx
