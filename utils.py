import torch
import numpy as np
import scipy.sparse as sp
import numpy.ctypeslib as ctl
import os.path as osp
import random
from ctypes import c_int
from scipy.sparse import coo_matrix
from tqdm import tqdm


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def get_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def label_node_homogeneity(G, node_index):
    num_nodes = G.num_node
    homophily = 0
    for edge_u in tqdm(node_index):
        hit = 0
        # 遍历所有节点，通过边找出他们的邻居
        edge_v_list = G.edge.col[torch.where(G.edge.row == edge_u)]
        if len(edge_v_list) != 0:
            for i in range(len(edge_v_list)):
                edge_v = edge_v_list[i]
                if G.y[edge_u] == G.y[edge_v]:
                    hit += 1
            homophily += hit / len(edge_v_list)
    homophily /= num_nodes
    return homophily
