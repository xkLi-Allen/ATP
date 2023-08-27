import torch
import numpy as np

from torch import Tensor
from scipy.sparse import csr_matrix


# Base class for adjacency matrix
class Edge:
    def __init__(self, row, col, edge_weight, edge_type, num_node, edge_attrs=None, sp_adj=None):
        if not isinstance(edge_type, str):
            raise TypeError("Edge type must be a string!")
        self.edge_type = edge_type

        if (not isinstance(row, (list, np.ndarray, Tensor))) or (not isinstance(col, (list, np.ndarray, Tensor))) or (
                not isinstance(edge_weight, (list, np.ndarray, Tensor))):
            raise TypeError("Row, col and edge_weight must be a list, np.ndarray or Tensor!")
        self.row = row
        self.col = col
        self.edge_weight = edge_weight
        self.edge_attrs = edge_attrs
        self.num_edge = len(row)

        if sp_adj != None:
            self.sparse_matrix = sp_adj
        elif isinstance(row, Tensor) or isinstance(col, Tensor):
            self.sparse_matrix = csr_matrix((edge_weight.numpy(), (row.numpy(), col.numpy())),
                                              shape=(num_node, num_node))
        else:
            self.sparse_matrix = csr_matrix((edge_weight, (row, col)), shape=(num_node, num_node))

    def edge_index(self):
        return self.row, self.col


# Base class or storing node information
class Node:
    def __init__(self, num_node, x=None, y=None, node_ids=None):
        if not isinstance(num_node, int):
            raise TypeError("Num nodes must be a integer!")
        elif (node_ids is not None) and (not isinstance(node_ids, (list, np.ndarray, Tensor))):
            raise TypeError("Node IDs must be a list, np.ndarray or Tensor!")
        self.num_node = num_node
        if node_ids is not None:
            self.node_ids = node_ids
        else:
            self.node_ids = range(num_node)
        self.x = x
        self.y = y



# Base class for homogeneous graph
class Graph:
    def __init__(self, row, col, edge_weight, num_node, edge_type, x=None, y=None, node_ids=None,
                 edge_attr=None, sp_adj = None):
        self.edge = Edge(row, col, edge_weight, edge_type, num_node, edge_attr)
        if node_ids is None:
            self.node_ids = range(num_node)
        else:
            self.node_ids = node_ids
        self.node = Node(num_node, x, y, node_ids)

        self.num_node = self.node.num_node
        self.num_edge = self.edge.num_edge
        self.adj = self.edge.sparse_matrix
        self.edge_type = self.edge.edge_type
        self.x = self.node.x
        self.y = self.node.y
        self.num_features = self.x.shape[1]
        if self.y is not None:
            self.num_classes = self.y.max() + 1
        else:
            self.num_classes = None
        self.row_sum = self.adj.sum(axis=1)
        self.node_degrees = torch.LongTensor(self.row_sum).squeeze(1)

