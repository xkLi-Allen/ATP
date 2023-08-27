import torch
import platform
import numpy as np
import torch.nn as nn
import scipy.sparse as sp

from torch import Tensor
from operators.utils import csr_sparse_dense_matmul, cuda_csr_sparse_dense_matmul
from torch.nn import CosineSimilarity, PairwiseDistance


class GraphOp:
    def __init__(self, prop_steps):
        self.prop_steps = prop_steps
        self.adj = None

    def construct_adj(self, adj, attribute_list=None):
        raise NotImplementedError

    def propagate(self, adj, feature, adapt=True, norm=None, degree=None):
        self.adj = self.construct_adj(adj)

        if not isinstance(adj, sp.csr_matrix):
            raise TypeError(
                "The adjacency matrix must be a scipy csr sparse matrix!")
        elif not isinstance(feature, np.ndarray):
            if isinstance(feature, Tensor):
                feature = feature.numpy()
            else:
                raise TypeError("The feature matrix must be a numpy.ndarray!")
        elif self.adj.shape[1] != feature.shape[0]:
            raise ValueError(
                "Dimension mismatch detected for the adjacency and the feature matrix!")

        prop_feat_list = [feature]
        # print(feature)
        # print(self.adj)

        # norm_fea_inf = torch.mm(
        #     norm.float(), torch.tensor(feature, dtype=torch.float))
        # print(norm_fea_inf)
        if adapt == False:
            for _ in range(self.prop_steps):
                # if platform.system() == "Linux":
                try:
                    feat_temp = csr_sparse_dense_matmul(
                        self.adj, prop_feat_list[-1])
                except:
                    feat_temp = self.adj.dot(prop_feat_list[-1])
                prop_feat_list.append(feat_temp)

            # d = {}
            # for i in range(feature.shape[0]):
            #     d[i] = degree[i]*feature.shape[0]

            # sorted_nodes_last = sorted(d, key=d.get, reverse=False)
            # sorted_nodes_top = sorted(d, key=d.get, reverse=True)
            # # print(feature.shape[0])
            # top_10_percent = int(feature.shape[0] * 0.20)
            # top_20_percent = int(feature.shape[0] * 0.20)
            # top_nodes = []
            # last_nodes = []
            # i = 0
            # # print(d[11450])
            # for u in sorted_nodes_top:
            #     # print(d[u])
            #     # if d[u] < 50:
            #     #     break

            #     top_nodes.append(u)
            #     i += 1
            #     if i == top_10_percent:
            #         break
            # i = 0
            # for u in sorted_nodes_last:
            #     # if d[u] > 3:
            #     #     break
            #     last_nodes.append(u)
            #     i += 1
            #     if i == top_20_percent:
            #         break

            # # print(degree[last_nodes[0]])
            # # print(degree[top_nodes[0]])
            # l1 = []
            # l2 = []
            # l3 = []
            # # print(top_nodes)
            # # print(prop_feat_list[1])
            # for i in range(self.prop_steps+1):
            #     dist = (torch.tensor(
            #         prop_feat_list[i]) - norm_fea_inf).norm(2, 1)
            #     dist = dist.view(-1, 1)
            #     # print(dist)
            #     select_dist1 = dist[top_nodes]
            #     select_dist2 = dist[last_nodes]

            #     l1.append(torch.mean(dist[top_nodes]).item())
            #     l2.append(torch.mean(select_dist2).item())
            #     l3.append(torch.mean(dist).item())

            # print(l1)
            # print(l2)
            # print(l3)

            # exit(0)
            #     prop_feat_list[i] = prop_feat_list[i] + \
            #         np.multiply(dist.numpy(), feature)
            # print(prop_feat_list)

        else:
            cos_sim_module = torch.nn.CosineSimilarity(dim=1)
            euclidean_dist_module = torch.nn.PairwiseDistance(p=2)

            try:
                feat_temp = csr_sparse_dense_matmul(
                    self.adj, prop_feat_list[-1])
            except:
                feat_temp = self.adj.dot(prop_feat_list[-1])
            prop_feat_list.append(feat_temp)
            for _ in range(1, self.prop_steps):
                # cos_sim = cos_sim_module(torch.tensor(
                #     prop_feat_list[-1]), torch.tensor(feature))

                # min_val = torch.min(cos_sim)
                # max_val = torch.max(cos_sim)
                # cos_sim = (cos_sim - min_val) / \
                #     (max_val - min_val)
                # cos_sim = cos_sim.view(-1, 1)
                # cos_sim = cos_sim.numpy()

                # euclidean_dist = euclidean_dist_module(
                #     torch.tensor(prop_feat_list[-1]), torch.tensor(feature))
                # min_val = torch.min(euclidean_dist)
                # max_val = torch.max(euclidean_dist)
                # euclidean_dist = (euclidean_dist - min_val) / \
                #     (max_val - min_val)
                # euclidean_dist = euclidean_dist.view(-1, 1)
                # euclidean_dist = euclidean_dist.numpy()

                degree = degree.view(-1, 1)
                # degree = torch.tensor(np.zeros_like(degree))
                try:
                    feat_temp = csr_sparse_dense_matmul(
                        self.adj, prop_feat_list[-1]+np.multiply(prop_feat_list[0], 1-degree.numpy()))
                    # feat_temp = csr_sparse_dense_matmul(self.adj,
                    #                                     np.multiply(prop_feat_list[-1], degree.numpy())+np.multiply(prop_feat_list[0], 1-degree.numpy()))
                    # feat_temp = csr_sparse_dense_matmul(
                    #     self.adj, prop_feat_list[-1]+prop_feat_list[0])

                except:
                    feat_temp = self.adj.dot(
                        prop_feat_list[-1]+np.multiply(prop_feat_list[0], 1-degree.numpy()))
                    # feat_temp = self.adj.dot(
                    #     np.multiply(prop_feat_list[-1], degree.numpy())+np.multiply(prop_feat_list[0], 1-degree.numpy()))
                    # feat_temp = self.adj.dot(
                    #     prop_feat_list[-1]+prop_feat_list[0])
                prop_feat_list.append(feat_temp)

        return [torch.FloatTensor(feat) for feat in prop_feat_list]




# Might include training parameters
class MessageOp(nn.Module):
    def __init__(self, start=None, end=None):
        super(MessageOp, self).__init__()
        self.aggr_type = None
        self.start, self.end = start, end

    def aggr_type(self):
        return self.aggr_type

    def combine(self, feat_list):
        return NotImplementedError

    def aggregate(self, feat_list):
        if not isinstance(feat_list, list):
            return TypeError("The input must be a list consists of feature matrices!")
        for feat in feat_list:
            if not isinstance(feat, Tensor):
                raise TypeError("The feature matrices must be tensors!")

        return self.combine(feat_list)


