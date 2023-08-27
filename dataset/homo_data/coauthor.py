import os.path as osp
import pickle as pkl
import torch
from dataset.base_data import Graph

from dataset.base_dataset import NodeDataset
from dataset.utils import download_to, pkl_read_file, random_split_dataset, read_npz
from dataset.node_split import node_class_split


class Coauthor(NodeDataset):
    def __init__(self, name="cs", root="./", split="random", k=None):
        self.name = name
        if name not in ['cs', 'phy']:
            raise ValueError("Dataset name not supported!")
        super(Coauthor, self).__init__(root + "coauthor/", name, k)

        self.read_file()
        self.split = split
        self.train_idx, self.val_idx, self.test_idx = self.generate_split(
            split)

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

    @property
    def raw_file_paths(self):
        return osp.join(self.raw_dir, f'ms_academic_{self.name.lower()}.npz')

    @property
    def processed_file_paths(self):
        filename = "graph"
        return osp.join(self.processed_dir, "{}.{}".format(self.name, filename))

    def download(self):
        url = "https://github.com/shchur/gnn-benchmark/raw/master/data/npz"
        file_url = url + '/' + osp.basename(self.raw_file_paths)
        print(file_url)
        download_to(file_url, self.raw_file_paths)

    def process(self):
        x, edge_index, y = read_npz(self.raw_file_paths)
        num_node = x.shape[0]
        row, col = edge_index
        edge_weight = torch.ones(size=(len(row),))
        node_type = "paper"
        edge_type = "paper__to__paper"
        g = Graph(row, col, edge_weight, num_node,
                  edge_type, x=x.numpy(), y=y)
        with open(self.processed_file_paths, 'wb') as rf:
            try:
                pkl.dump(g, rf)
            except IOError as e:
                print(e)
                exit(1)

    def generate_split(self, split):
        if split == "random":
            train_idx, val_idx, test_idx = random_split_dataset(self.num_node)

        elif split == "official":
            train_idx, val_idx, test_idx, seed_idx_list, _ = node_class_split(
                name=self.name, data=self.data, split=split, cache_node_split=osp.join(self.processed_dir, "official_split"), official_split=None, node_split_id=0, train_size_per_class=int(20), val_size_per_class=int(30))

        else:
            raise ValueError("Please input valid split pattern!")

        return train_idx, val_idx, test_idx
