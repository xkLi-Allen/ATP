import torch
import numpy as np
import os.path as osp
import pickle as pkl
import pandas as pd

from dataset.base_data import Graph
# from configs.data_config import data_args
from dataset.base_dataset import NodeDataset
from ogb.nodeproppred import PygNodePropPredDataset
from dataset.utils import pkl_read_file, remove_self_loops, to_undirected, edge_homophily, node_homophily, linkx_homophily, set_spectral_adjacency_reg_features


class Ogbn(NodeDataset):
    '''
    Dataset description: (Open Graph Benchmark): https://ogb.stanford.edu/docs/nodeprop/
    Directed infomation:    Undirected network (ogbn-products)
                            Directed network (ogbn-arxiv) -> we implement it as an undirected graph.

    -> ogbn-arxiv:     unsigned & undirected & unweighted homogeneous simplex network    
    -> ogbn-products:  unsigned & undirected & unweighted homogeneous simplex network

    We remove all multiple edges and self-loops from the original dataset. The above phenomenon result in a different number of edges compared to the original report -> NeurIPS'21 Large Scale Learning on Non-Homophilous Graphs: New Benchmarks and Strong Simple Methods, LINKX https://arxiv.org/pdf/2110.14446.pdf
    -> Multiple edges do not affect the number of edges, but may lead to the aggregation of edge weights.
    -> ogbn-arxiv: (1,166,243 -> 1,157,799)
    -> ogbn-products: (61,859,140 -> 61,859,012)

    ogbn-arxiv:     The ogbn-arxiv dataset is a directed graph, representing the citation network between all Computer Science (CS) arXiv papers.
                    169,343 nodes, 1,157,799 edges, 128 feature dimensions, 40 classes num.
                    Edge homophily: 0.6542, Node homophily:0.6353, Linkx homophily:0.4211.

    ogbn-products:  The ogbn-products dataset is an undirected and unweighted graph, representing an Amazon product co-purchasing network. 
                    Nodes represent products sold in Amazon, and edges between two products indicate that the products are purchased together. 
                    2,449,029 nodes, 61,859,012 edges, 100 feature dimensions, 47 classes num.
                    Edge homophily: 0.8076, Node homophily:0.833, Linkx homophily:0.4591.

    split:
        ogbn-arxiv:
            official:   We propose to train on papers published until 2017, 
                        validate on those published in 2018, 
                        and test on those published since 2019.
                        train/val/test = 90,941/29,799/48,603

        ogbn-products:
            official:   We sort the products according to their sales ranking 
                        and use the top 8% for training, 
                        next top 2% for validation, 
                        and the rest for testing. 
    '''

    def __init__(self, name="arxiv", root="./dataset/homo_data/", split="official", k=None):
        name = name.lower()
        if name not in ["arxiv", "products"]:
            raise ValueError("Dataset name not found!")
        super(Ogbn, self).__init__(root + "ogbn/", name, k)

        self.read_file()
        self.train_idx, self.val_idx, self.test_idx = self.generate_split(
            split)

    @property
    def raw_file_paths(self):
        filepath = "ogbn_" + self.name + "/processed/geometric_data_processed.pt"
        return osp.join(self.raw_dir, filepath)

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
        # import time
        # t1 = time.time()
        # edge_weight = self.edge.edge_weight
        # indices = torch.vstack((self.edge.row, self.edge.col)).long()
        # edge_num_node = indices.max().item() + 1
        # features = set_spectral_adjacency_reg_features(edge_num_node, indices, edge_weight)
        # print(time.time()-t1)

    def download(self):
        PygNodePropPredDataset("ogbn-" + self.name, self.raw_dir)

    def process(self):
        dataset = PygNodePropPredDataset("ogbn-" + self.name, self.raw_dir)

        data = dataset[0]
        features, labels = data.x.numpy().astype(
            np.float32), data.y.to(torch.long).squeeze(1)
        num_node = data.num_nodes

        if self.name == "arxiv":
            undi_edge_index = torch.unique(data.edge_index, dim=1)
            undi_edge_index = to_undirected(undi_edge_index)
        elif self.name == "products":
            undi_edge_index = data.edge_index
        undi_edge_index = torch.unique(undi_edge_index, dim=1)
        undi_edge_index = remove_self_loops(undi_edge_index)[0]

        row, col = undi_edge_index
        edge_weight = torch.ones(len(row))
        edge_type = "UUU"

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
            if self.name == "arxiv":
                root = "dataset/homo_data/ogbn/arxiv/raw/ogbn_arxiv"
                split_type = "time"

            elif self.name == "products":
                root = "dataset/homo_data/ogbn/products/raw/ogbn_products"
                split_type = "sales_ranking"

            path = osp.join(root, 'split', split_type)
            train_idx = torch.from_numpy(pd.read_csv(osp.join(
                path, 'train.csv.gz'), compression='gzip', header=None).values.T[0]).to(torch.long)
            val_idx = torch.from_numpy(pd.read_csv(osp.join(
                path, 'valid.csv.gz'), compression='gzip', header=None).values.T[0]).to(torch.long)
            test_idx = torch.from_numpy(pd.read_csv(osp.join(
                path, 'test.csv.gz'), compression='gzip', header=None).values.T[0]).to(torch.long)

        elif split == "random":
            raise NotImplementedError

        else:
            raise ValueError("Please input valid split pattern!")

        return train_idx, val_idx, test_idx
