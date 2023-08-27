import json
import os

import scipy.sparse as sp
import dgl
import numpy as np
from dataset.base_data import Graph
import torch
import pickle as pkl

from dataset.utils import pkl_read_file


class PPI_small:
    def __init__(self, name, root='./', split="official"):
        try:
            self.data = pkl_read_file(
                os.path.join(root, name, 'graph_small.pt'))

        except:
            adj_full = sp.load_npz(os.path.join(root, name, 'adj_full.npz'))
            G = dgl.from_scipy(adj_full)
            nodes_num = G.num_nodes()
            role = json.load(open(os.path.join(root, name, 'role.json'), 'r'))
            tr = list(role['tr'])
            te = list(role['te'])
            va = list(role['va'])
            mask = np.zeros((nodes_num,), dtype=bool)
            train_mask = mask.copy()
            train_mask[tr] = True
            val_mask = mask.copy()
            val_mask[va] = True
            test_mask = mask.copy()
            test_mask[te] = True

            G.ndata['train_mask'] = torch.tensor(train_mask, dtype=torch.bool)
            G.ndata['val_mask'] = torch.tensor(val_mask, dtype=torch.bool)
            G.ndata['test_mask'] = torch.tensor(test_mask, dtype=torch.bool)

            feats = np.load(os.path.join(root, name, 'feats.npy'))
            G.ndata['feat'] = torch.tensor(feats, dtype=torch.float)

            class_map = json.load(
                open(os.path.join(root, name, 'class_map.json'), 'r'))
            labels = np.array([class_map[str(i)] for i in range(nodes_num)])
            G.ndata['label'] = torch.tensor(labels, dtype=torch.float)
            print(type(adj_full))
            labels = torch.tensor(labels, dtype=torch.float)
            # row = G.edge_index[0]
            # col = G.edge_index[1]

            # num_node = G.num_nodes
            edge_type = "image__to__image"
            row, col = adj_full.nonzero()
            edge_weight = torch.ones(len(row))
            g = Graph(row=row, col=col, edge_weight=edge_weight, num_node=nodes_num,
                      edge_type=edge_type, x=feats, y=labels, sp_adj=adj_full)
            with open(os.path.join(root, name, "graph_small.pt"), 'wb') as rf:
                try:
                    pkl.dump(g, rf)
                except IOError as e:
                    print(e)
                    exit(1)
            self.data = pkl_read_file(
                os.path.join(root, name, 'graph_small.pt'))
        self.name = "ppi_small"
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
        # self.data.lables = self.data.lables.to(torch.long)
        self.train_idx, self.val_idx, self.test_idx = self.generate_split(
            split)

    def generate_split(self, split):
        if split == "official":
            train_idx = range(0, 9716)
            val_idx = range(9716, 11541)
            test_idx = range(11541, 14755)
        elif split == "random":
            raise NotImplementedError
        else:
            raise ValueError("Please input valid split pattern!")

        return train_idx, val_idx, test_idx
