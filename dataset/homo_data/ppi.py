import json

import torch
from torch_geometric.datasets import PPI
from itertools import product
import pickle as pkl
from torch_geometric.utils import remove_self_loops
from torch_geometric.data import Data
from dataset.base_data import Graph
from dataset.base_dataset import NodeDataset
import os
import os.path as osp
import numpy as np
from dataset.utils import pkl_read_file


class PPI(NodeDataset):
    def __init__(self, name, root='./', split="official"):
        super(PPI, self).__init__(root + "PPI", name, k=None)
        self.name = 'ppi'
        self.split = split
        self.read_file()
        if split == "official":
            self.edge = self.data.edge
            self.node = self.data.node
            self.x = self.data.x
            self.y = self.data.y
            self.adj = self.data.adj
            self.edge_type = self.data.edge_type
            self.num_features = self.data.num_features
            self.num_classes = 121
            self.num_node = self.data.num_node
            self.num_edge = self.data.num_edge
            self.train_idx, self.val_idx, self.test_idx = self.generate_split(
                split)

    def read_file(self):
        # if self.split == "train":
        #     self.data = pkl_read_file(self.processed_file_paths[0])
        # elif self.split == 'val':
        #     self.data = pkl_read_file(self.processed_file_paths[1])
        # elif self.split == 'test':
        #     self.data = pkl_read_file(self.processed_file_paths[2])
        #
        try:
            if self.split == 'train':
                self.data, self.slices = torch.load(
                    self.processed_file_paths[0])
                # print(self.slices)
            elif self.split == 'val':
                self.data, self.slices = torch.load(
                    self.processed_file_paths[1])
            elif self.split == 'test':
                self.data, self.slices = torch.load(
                    self.processed_file_paths[2])
            elif self.split == "official":
                self.data = pkl_read_file(self.processed_file_paths[4])
        except:

            graph = torch.load(self.processed_file_paths[3])
            undi_edge_index = graph[0].edge_index
            undi_edge_index = torch.unique(undi_edge_index, dim=1)
            undi_edge_index = remove_self_loops(undi_edge_index)[0]
            row = undi_edge_index[0]
            col = undi_edge_index[1]

            # row = graph[0].edge_index[0]
            # col = graph[0].edge_index[1]
            edge_weight = torch.ones(len(row))
            num_node = graph[0].num_nodes
            edge_type = "image__to__image"
            features, labels = graph[0].x.numpy().astype(
                np.float32), graph[0].y.to(torch.long).squeeze(1)

            g = Graph(row, col, edge_weight, num_node,
                      edge_type, x=features, y=labels)
            with open(self.processed_file_paths[4], 'wb') as rf:
                try:
                    pkl.dump(g, rf)
                except IOError as e:
                    print(e)
                    exit(1)

        # if self.split == 'train':
        #     self.data, self.slices = torch.load(self.processed_file_paths[0])
        # elif self.split == 'val':
        #     self.data, self.slices = torch.load(self.processed_file_paths[1])
        # elif self.split == 'test':
        #     self.data, self.slices = torch.load(self.processed_file_paths[2])
        # self.dataset = PPI(root, split=split)

    @property
    def raw_file_paths(self):
        splits = ['train', 'valid', 'test']
        files = ['feats.npy', 'graph_id.npy', 'graph.json', 'labels.npy']
        filenames = [f'{split}_{name}' for split,
                     name in product(splits, files)]

        return [os.path.join(self.raw_dir, filename) for filename in filenames]

    @property
    def processed_file_paths(self):
        return [os.path.join(self.processed_dir, 'train.pt'), os.path.join(self.processed_dir, 'val.pt'),
                os.path.join(self.processed_dir, 'test.pt'),  os.path.join(
                    self.processed_dir, 'graph.pt'),
                os.path.join(self.processed_dir, 'modify_graph.pt')]

    def process(self):
        pass

    def download(self):
        self.dataset.download()

    def normalize(self, mx):
        pass

    def generate_split(self, split):
        if split == "official":
            train_idx = range(0, 44906)
            val_idx = range(44906, 51320)
            test_idx = range(51320, 56944)
        elif split == "random":
            raise NotImplementedError
        else:
            raise ValueError("Please input valid split pattern!")

        return train_idx, val_idx, test_idx
