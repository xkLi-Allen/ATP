import itertools
import math
from tqdm import tqdm
import torch
import time
import scipy.sparse as sp
from torch_geometric.datasets import Planetoid
from scipy.sparse import csr_matrix, save_npz, load_npz
import networkx as nx
import os
import numpy as np

from dataset.homo_data.planetoid import Planetoid
from dataset.homo_data.ogbn import Ogbn
from dataset.homo_data.ogbn_100m import Ogbn_papers100m
from dataset.homo_data.flickr import Flickr
from dataset.homo_data.reddit import Reddit
from dataset.homo_data.ppi_small import PPI_small
from dataset.homo_data.amazon import Amazon
from dataset.homo_data.coauthor import Coauthor
from utils import label_node_homogeneity
import matplotlib.pyplot as plt


class GraphData:
    def __init__(self, root, name, args):
        self.data_name = name
        self.args = args
        self.root = os.path.join(root)
        self.load_homo_simplex_dataset(name=args.data_name, root=args.root,
                                       split=args.split)
        # self.load_data()
        self.dataset.mining_matrix = None
        self.dataset.mining_list = None

    def node_homo(self):

        G = nx.Graph()
        G.add_nodes_from(range(self.dataset.data.num_node))
        G.add_edges_from(zip(self.dataset.data.edge.col.tolist(),
                             self.dataset.data.edge.row.tolist()))
        degrees = dict(G.degree())
        max_degree = max(degrees.values())
        boundaries = []
        for i in range(10, 101, 10):
            # print(i)
            boundaries.append(int(max_degree*i/100))
        result = []
        prev_boundary = 0
        for boundary in boundaries:
            subset = [i for i, value in enumerate(
                degrees) if prev_boundary <= value <= boundary]
            result.append(subset)
            prev_boundary = boundary

        node_homogeneity = []
        for l in result:
            homogeneity = label_node_homogeneity(
                G=self.dataset.data, node_index=l)
            node_homogeneity.append(homogeneity)

        # 绘制图表
        x_ticks = [f"{i}-{i + 10}%" for i in range(0, 101, 10)]
        x = np.arange(len(x_ticks))
        y1 = [len(l) for l in result]
        print(y1)
        y2 = node_homogeneity

        fig, ax1 = plt.subplots()

        ax2 = ax1.twinx()
        ax1.bar(x, y1, alpha=0.7, color='b', align='center')
        ax2.plot(x, y2, '-o', color='r')

        ax1.set_xlabel('Percentage of Nodes')
        ax1.set_ylabel('Percentage of Nodes in Subset', color='b')
        ax2.set_ylabel('Average Node Homophily', color='r')
        ax1.set_xticks(x)
        ax1.set_xticklabels(x_ticks)

        plt.title('Node Subset Analysis')
        plt.show()
        exit(0)
        # homo = label_node_homogeneity(G=self.dataset.data)
        # print("the average homo is {}".format(homo))

    def calculate_node_homo(self):
        num_nodes = self.dataset.data.num_node
        homophily_node = 0
        for edge_u in tqdm(range(num_nodes)):
            hit = 0
            # 遍历所有节点，通过边找出他们的邻居

            edge_v_list = self.dataset.data.edge.col[torch.where(
                torch.tensor(self.dataset.data.edge.row) == edge_u)].tolist()

            if isinstance(edge_v_list, list) and len(edge_v_list) != 0:
                for i in range(len(edge_v_list)):
                    edge_v = edge_v_list[i]
                    if self.dataset.data.y[edge_u] == self.dataset.data.y[edge_v]:
                        hit += 1
                homophily_node += hit / len(edge_v_list)

            else:
                if self.dataset.data.y[edge_u] == self.dataset.data.y[edge_v_list]:
                    hit += 1
                homophily_node += hit
        homophily_node /= num_nodes
        print(
            "the node-level homophily of {} is: {}".format(self.data_name, homophily_node))

        num_edges = len(self.dataset.data.edge.row)
        homophily_edge = 0
        for i in tqdm(range(num_edges)):
            if self.dataset.data.y[self.dataset.data.edge.row[i]] == self.dataset.data.y[self.dataset.data.edge.col[i]]:
                homophily_edge += 1
        homophily_edge /= num_edges
        print(
            "the edge-level homophily of {} is: {}".format(self.data_name, homophily_edge))
        with open('output.txt', 'a') as f:
            f.write(f"{self.dataset.name}\n ")
            f.write(
                "the node-level homophily of {} is: {}".format(self.data_name, homophily_node))
            f.write("\n")
            f.write(
                "the edge-level homophily of {} is: {}".format(self.data_name, homophily_edge))
            f.write("\n")
        exit(0)

    def drop_nodes(self, ratio=0.1):
        try:
            loaded_matrix = load_npz(os.path.join("/mnt/ssd2/home/xkli/mjy",
                                                  '{} by {} adj_matrix.npz'.format(self.data_name, ratio)))
            self.dataset.adj = loaded_matrix
            print("already has the drop node matrix!")
        except:
            print("drop node base on degree, ratio {}".format(ratio))
            G = nx.Graph()
            G.add_nodes_from(range(self.dataset.data.num_node))
            G.add_edges_from(zip(self.dataset.data.edge.col.tolist(),
                                 self.dataset.data.edge.row.tolist()))
            degrees = dict(G.degree())
            print(degrees)

            # drop训练集
            # sorted_nodes = sorted(degrees, key=degrees.get, reverse=True)
            # top_10_percent = int(len(self.dataset.train_idx) * ratio)
            # top_nodes = []
            # i = 0
            # for u in sorted_nodes:
            #     if u in self.dataset.train_idx:
            #         top_nodes.append(u)
            #         i += 1
            #     if i == top_10_percent:
            #         break

            # drop所有
            sorted_nodes = sorted(degrees, key=degrees.get, reverse=True)
            top_10_percent = int(self.dataset.data.num_node * ratio)
            top_nodes = []
            i = 0
            for u in sorted_nodes:
                top_nodes.append(u)
                i += 1
                if i == top_10_percent:
                    break
            # top_nodes = sorted_nodes[:top_10_percent]
            # print(top_nodes)

            # for u, v in tqdm(zip(self.dataset.data.edge.row.tolist(), self.dataset.data.edge.col.tolist())):
            #     # print(u, v)
            #     if (u in top_nodes) or (v in top_nodes):
            #         pass
            #     else:
            #         row_new.append(u)
            #         col_new.append(v)
            #         weight_new.append(1)
            # self.dataset.adj = csr_matrix((weight_new, (row_new, col_new)), shape=(
            #     self.dataset.data.num_node, self.dataset.data.num_node))
            # save_npz(os.path.join("/mnt/ssd2/home/xkli/mjy",
            #                       '{} by {} adj_matrix.npz'.format(self.data_name, ratio)), self.dataset.adj)

            for u in tqdm(top_nodes):
                for v in set(self.dataset.adj.getrow(u).indices):
                    row_i = self.dataset.adj.indptr[u]
                    row_i_next = self.dataset.adj.indptr[u + 1]
                    row_j = self.dataset.adj.indptr[v]
                    row_j_next = self.dataset.adj.indptr[v + 1]
                    pos_ij = np.searchsorted(
                        self.dataset.adj.indices[row_i:row_i_next], v) + row_i
                    pos_ji = np.searchsorted(
                        self.dataset.adj.indices[row_j:row_j_next], u) + row_j
                    # 本来有边直接改
                    if pos_ij < row_i_next and self.dataset.adj.indices[pos_ij] == v:
                        self.dataset.adj.data[pos_ij] = 0
                        self.dataset.adj.data[pos_ji] = 0

            t = time.time()
            indices_with_zero_weight = np.where(self.dataset.adj.data == 0)[0]
            self.dataset.adj.data = np.delete(
                self.dataset.adj.data, indices_with_zero_weight)
            self.dataset.adj.indices = np.delete(
                self.dataset.adj.indices, indices_with_zero_weight)

            self.dataset.adj.indptr[1:] -= np.searchsorted(
                indices_with_zero_weight, self.dataset.adj.indptr[1:])
            save_npz(os.path.join("/mnt/ssd2/home/xkli/mjy",
                                  '{} by {} adj_matrix.npz'.format(self.data_name, ratio)), self.dataset.adj)
            print(time.time()-t)
            exit(0)
            # print(self.dataset.adj)
        # self.dataset.data.adj = sp.coo_matrix((torch.ones([len(self.dataset.data.edge.row)]),
        #                                        (row_new, col_new)),
        #                                       shape=(self.dataset.data.num_node, self.dataset.data.num_node))

    def load_homo_simplex_dataset(self, name, root, split):
        if name.lower() in ('cora', 'citeseer', 'pubmed'):
            dataset = Planetoid(name, root, split)
            self.root = os.path.join(root, "Planetoid")
        if name.lower() in ('arxiv', 'products'):
            dataset = Ogbn(name, root, split)
            self.root = os.path.join(root, "ogbn")
        if name.lower() in ('flickr'):
            dataset = Flickr(name, root, split)
            self.root = os.path.join(root, "Flickr")
        if name.lower() in ('reddit'):
            dataset = Reddit(name, root, split)
            self.root = os.path.join(root, "Reddit")

        if name.lower() in ('papers100m'):
            dataset = Ogbn_papers100m()
            self.root = os.path.join(root, "papers100M")

        if name.lower() in ('ppi_small'):
            dataset = PPI_small(name, root, split)
            self.root = os.path.join(root, "ppi_small")

        if name.lower() in ('computers', 'photo'):
            dataset = Amazon(name, root, split)
            self.root = os.path.join(root, "amazon")

        if name.lower() in ('cs', 'phy'):
            dataset = Coauthor(name, root, split)
            self.root = os.path.join(root, "coauthor")


        self.dataset = dataset
        self.dataset.data.adj = sp.coo_matrix((torch.ones([len(self.dataset.data.edge.row)]),
                                               (self.dataset.data.edge.row, self.dataset.data.edge.col)),
                                              shape=(self.dataset.data.num_node, self.dataset.data.num_node))

        return dataset



    def mining(self, method,  a=0.5, b=0.5, c=0.5):
        print("Begin to mining the data!")
        start_time = time.time()
        if method is None:
            raise NotImplemented
        elif method in [ "Engienvector_centrality",
                         "clustering_coefficients",
                        "degree_centrality", "together", ]:
            try:
                if method == "together":
                    centrality1 = torch.load(os.path.join(
                        self.root, self.data_name, "degree_centrality"))
                    centrality2 = torch.load(os.path.join(
                        self.root, self.data_name, "clustering_coefficients"))

                    try:
                        centrality3 = torch.load(os.path.join(
                            self.root, self.data_name, "Engienvector_centrality"))
                    except:
                        print("there is no engienvector for the current dataset!")
                        centrality3 = torch.load(os.path.join(
                            self.root, self.data_name, "degree_centrality"))
                        c = 0

                    for i, centrality in enumerate([centrality1, centrality2, centrality3]):
                        num_outliers = int(0.05 * centrality.size()[0])
                        arr = centrality

                        _, indices = torch.topk(arr.abs(), num_outliers)
                        outliers = arr[indices]

                        arr[indices] = 1.0
                        min_value = arr.min()
                        max_value = arr[arr != 1.0].max()

                        centrality = (arr - min_value) / \
                            (max_value - min_value)
                        centrality[arr == 1.0] = 1.0
                        if i == 0:
                            centrality1 = centrality
                        if i == 1:
                            centrality2 = centrality
                        if i == 2:
                            centrality3 = centrality


                    centrality = a*centrality1 + b*centrality2 + c*centrality3
                    min_val = torch.min(centrality)
                    max_val = torch.max(centrality)

                    centrality = (centrality - min_val) / (max_val - min_val)
                    self.dataset.mining_list = centrality
                    print("already has the required matrix")
                    return centrality

                else:
                    centrality = torch.load(os.path.join(
                        self.root, self.data_name, method))


                    min_val = torch.min(centrality)
                    max_val = torch.max(centrality)
                    centrality = (centrality - min_val) / (max_val - min_val)

                    self.dataset.mining_list = centrality

                    centrality1 = torch.load(os.path.join(
                        self.root, self.data_name, "degree_centrality"))
                    self.dataset.ori_degree = centrality1
                    min_val = torch.min(centrality1)
                    max_val = torch.max(centrality1)
                    centrality1 = (centrality1 - min_val) / (max_val - min_val)

                    centrality2 = torch.load(os.path.join(
                        self.root, self.data_name, "clustering_coefficients"))
                    min_val = torch.min(centrality2)
                    max_val = torch.max(centrality2)
                    centrality2 = (centrality2 - min_val) / (max_val - min_val)

                    self.dataset.degree_list = centrality1
                    self.dataset.cluster_list = centrality2

                    print("already has the required matrix")
                    return centrality



            except:
                print("there is no {} for {}".format(method, self.data_name))
                G = nx.Graph()
                G.add_nodes_from(range(self.dataset.data.num_node))
                matrix = self.dataset.adj.tocsr()
                rows, cols = matrix.nonzero()
                G.add_edges_from(zip(rows, cols))
                # G.add_edges_from(
                #     zip(self.dataset.data.edge.col.tolist(), self.dataset.data.edge.row.tolist()))
                if method == "Betweenness_centrality":
                    centrality = nx.betweenness_centrality(G)

                elif method == "Engienvector_centrality":
                    centrality = nx.eigenvector_centrality(G)

                elif method == 'Closeness_centrality':
                    centrality = nx.closeness_centrality(G)

                elif method == 'clustering_coefficients':
                    centrality = nx.clustering(G)
                elif method == "pagerank":
                    centrality = nx.pagerank(G)
                elif method == "degree_centrality":
                    centrality = nx.degree_centrality(G)
                elif method == "together":
                    centrality = nx.degree_centrality(G)
                    centrality = torch.tensor(
                        [centrality[node] for node in range(self.dataset.data.num_node)])
                    torch.save(centrality, os.path.join(
                        self.root, self.data_name, "degree_centrality"))

                    centrality = nx.clustering(G)
                    centrality = torch.tensor(
                        [centrality[node] for node in range(self.dataset.data.num_node)])
                    torch.save(centrality, os.path.join(
                        self.root, self.data_name, "clustering_coefficients"))

                    centrality = nx.eigenvector_centrality(G)
                    centrality = torch.tensor(
                        [centrality[node] for node in range(self.dataset.data.num_node)])
                    torch.save(centrality, os.path.join(
                        self.root, self.data_name, "Engienvector_centrality"))

                if method != "together":
                    centrality = torch.tensor(
                        [centrality[node] for node in range(self.dataset.data.num_node)])
                    torch.save(centrality, os.path.join(
                        self.root, self.data_name, method))
                    print("Spending {:.4f}s in finishing mining task".format(
                        time.time() - start_time))

                    min_val = torch.min(centrality)
                    max_val = torch.max(centrality)

                    centrality = (centrality - min_val) / (max_val - min_val)
                    self.dataset.mining_list = centrality
                if method == "together":

                    centrality1 = torch.load(os.path.join(
                        self.root, self.data_name, "degree_centrality"))
                    centrality2 = torch.load(os.path.join(
                        self.root, self.data_name, "clustering_coefficients"))

                    try:
                        centrality3 = torch.load(os.path.join(
                            self.root, self.data_name, "Engienvector_centrality"))
                    except:
                        print("there is no engienvector for the current dataset!")
                        centrality3 = torch.load(os.path.join(
                            self.root, self.data_name, "degree_centrality"))
                        c = 0

                    for i, centrality in enumerate([centrality1, centrality2, centrality3]):
                        num_outliers = int(0.05 * centrality.size()[0])
                        print(num_outliers)
                        arr = centrality

                        _, indices = torch.topk(arr.abs(), num_outliers)
                        outliers = arr[indices]

                        arr[indices] = 1.0

                        min_value = arr.min()
                        max_value = arr[arr != 1.0].max()

                        centrality = (arr - min_value) / \
                            (max_value - min_value)
                        centrality[arr == 1.0] = 1.0
                        if i == 0:
                            centrality1 = centrality
                        if i == 1:
                            centrality2 = centrality
                        if i == 2:
                            centrality3 = centrality


                    centrality = a*centrality1 + b*centrality2 + c*centrality3
                    min_val = torch.min(centrality)
                    max_val = torch.max(centrality)

                    centrality = (centrality - min_val) / (max_val - min_val)
                    self.dataset.mining_list = centrality
                    print("already has the required matrix")

                return centrality


