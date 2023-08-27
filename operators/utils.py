import torch
import numpy as np
import os.path as osp
import scipy.sparse as sp
import numpy.ctypeslib as ctl

from ctypes import c_int
from torch import Tensor
from torch_sparse import coalesce
from scipy.sparse import csr_matrix
from torch_scatter import scatter_add
from scipy.sparse.linalg import eigsh
from torch_geometric.utils import add_self_loops, to_scipy_sparse_matrix


def csr_sparse_dense_matmul(adj, feature):
    file_path = osp.abspath(__file__)
    dir_path = osp.split(file_path)[0]

    ctl_lib = ctl.load_library("./csrc/libmatmul.so", dir_path)

    arr_1d_int = ctl.ndpointer(
        dtype=np.int32,
        ndim=1,
        flags="CONTIGUOUS"
    )

    arr_1d_float = ctl.ndpointer(
        dtype=np.float32,
        ndim=1,
        flags="CONTIGUOUS"
    )
    ctl_lib.FloatCSRMulDenseOMP.argtypes = [arr_1d_float, arr_1d_float, arr_1d_int, arr_1d_int, arr_1d_float,
                                            c_int, c_int]
    ctl_lib.FloatCSRMulDenseOMP.restypes = None

    answer = np.zeros(feature.shape).astype(np.float32).flatten()
    data = adj.data.astype(np.float32)
    indices = adj.indices
    indptr = adj.indptr
    mat = feature.flatten()
    mat_row, mat_col = feature.shape

    ctl_lib.FloatCSRMulDenseOMP(
        answer, data, indices, indptr, mat, mat_row, mat_col)

    return answer.reshape(feature.shape)


def cuda_csr_sparse_dense_matmul(adj, feature):
    file_path = osp.abspath(__file__)
    dir_path = osp.split(file_path)[0]

    ctl_lib = ctl.load_library("./csrc/libcudamatmul.so", dir_path)

    arr_1d_int = ctl.ndpointer(
        dtype=np.int32,
        ndim=1,
        flags="CONTIGUOUS"
    )
    arr_1d_float = ctl.ndpointer(
        dtype=np.float32,
        ndim=1,
        flags="CONTIGUOUS"
    )
    ctl_lib.FloatCSRMulDense.argtypes = [arr_1d_float, c_int, arr_1d_float, arr_1d_int, arr_1d_int, arr_1d_float, c_int,
                                         c_int]
    ctl_lib.FloatCSRMulDense.restypes = c_int

    answer = np.zeros(feature.shape).astype(np.float32).flatten()
    data = adj.data.astype(np.float32)
    data_nnz = len(data)
    indices = adj.indices
    indptr = adj.indptr
    mat = feature.flatten()
    mat_row, mat_col = feature.shape

    ctl_lib.FloatCSRMulDense(answer, data_nnz, data,
                             indices, indptr, mat, mat_row, mat_col)

    return answer.reshape(feature.shape)


def adj_to_symmetric_norm(adj, r):
    adj = adj + sp.eye(adj.shape[0])
    degrees = np.array(adj.sum(1))
    r_inv_sqrt_left = np.power(degrees, r - 1).flatten()
    r_inv_sqrt_left[np.isinf(r_inv_sqrt_left)] = 0.
    r_mat_inv_sqrt_left = sp.diags(r_inv_sqrt_left)

    r_inv_sqrt_right = np.power(degrees, -r).flatten()
    r_inv_sqrt_right[np.isinf(r_inv_sqrt_right)] = 0.
    r_mat_inv_sqrt_right = sp.diags(r_inv_sqrt_right)

    adj_normalized = adj.dot(
        r_mat_inv_sqrt_left).transpose().dot(r_mat_inv_sqrt_right)
    return adj_normalized


def adj_to_symmetric_norm_att(adj, att):
    adj = adj + sp.eye(adj.shape[0])
    degrees = np.array(adj.sum(1))
    att = np.array(att)  # 将指数值转换为numpy数组
    att = att.reshape((-1, 1))
    r_inv_sqrt_left = np.power(degrees, att - 1).flatten()
    r_inv_sqrt_left[np.isinf(r_inv_sqrt_left)] = 0.
    r_mat_inv_sqrt_left = sp.diags(r_inv_sqrt_left)

    r_inv_sqrt_right = np.power(degrees, -att).flatten()
    r_inv_sqrt_right[np.isinf(r_inv_sqrt_right)] = 0.
    r_mat_inv_sqrt_right = sp.diags(r_inv_sqrt_right)

    adj_normalized = adj.dot(
        r_mat_inv_sqrt_left).transpose().dot(r_mat_inv_sqrt_right)
    return adj_normalized


def adj_to_directed_symmetric_mag_norm(adj, r, q):
    num_nodes = adj.shape[0]
    row, col = torch.tensor(adj.row, dtype=torch.long), torch.tensor(
        adj.col, dtype=torch.long)         # r,c
    # weight
    edge_weight = torch.tensor(adj.data)

    row, col = torch.cat([row, col], dim=0), torch.cat(
        [col, row], dim=0)       # r->r+c, c->c+r
    # r|c , c|r (2, edgex2) -> {A(u,v)}, {A(v,u)}
    edge_index = torch.stack([row, col], dim=0)
    # weight, weight -> weight x {+A(u,v)}, {+A(v,u)}
    sym_attr = torch.cat([edge_weight, edge_weight], dim=0)
    # weight, -weight -> weight x {+A(u,v)}, {-A(v,u)}
    theta_attr = torch.cat([edge_weight, -edge_weight], dim=0)
    # -> weight x {+A(u,v), +A(v,u)}, weight x {+A(v,u), -A(v,u)}
    edge_attr = torch.stack([sym_attr, theta_attr], dim=1)
    edge_index_sym, edge_attr = coalesce(edge_index, edge_attr,                 # edge_attr[:, 0]: weight x {+A(u,v) + +A(v,u)} = weight x (A(u,v) + A(v,u))
                                         num_nodes, num_nodes, "add")                     # edge_attr[:, 1]: weight x {+A(u,v) + -A(v,u)} = weight x (A(u,v) - A(v,u))
    # edge_index_sym: raw edge_index
    edge_weight_sym = edge_attr[:, 0]
    # {weight x (A(u,v) + A(v,u))} / 2 -> A_s(u,v)
    edge_weight_sym = edge_weight_sym/2
    loop_weight_sym = torch.ones((num_nodes))
    edge_weight_sym = torch.hstack((edge_weight_sym, loop_weight_sym))
    loop_edge_u_v = torch.linspace(0, num_nodes-1, steps=num_nodes, dtype=int)
    loop_edge_index_u = torch.hstack((edge_index_sym[0], loop_edge_u_v))
    loop_edge_index_v = torch.hstack((edge_index_sym[1], loop_edge_u_v))
    edge_index_sym = torch.vstack((loop_edge_index_u, loop_edge_index_v))

    theta_weight = edge_attr[:, 1]
    loop_weight = torch.zeros((num_nodes))
    theta_weight = torch.hstack((theta_weight, loop_weight))

    row, col = edge_index_sym[0], edge_index_sym[1]
    deg = scatter_add(edge_weight_sym, row, dim=0,
                      dim_size=num_nodes)          # D_s(u,u)

    # exp(i\theta^{q}) -> exp(i x 2\pi x q x {weight x (A(u,v) - A(v,u))})
    edge_weight_q = torch.exp(1j * 2 * np.pi * q * theta_weight)

    deg_inv_sqrt_left = torch.pow(deg, r-1)
    deg_inv_sqrt_left.masked_fill_(deg_inv_sqrt_left == float('inf'), 0)
    deg_inv_sqrt_right = torch.pow(deg, -r)
    deg_inv_sqrt_right.masked_fill_(deg_inv_sqrt_right == float('inf'), 0)
    edge_weight = deg_inv_sqrt_left[row] * edge_weight_sym * \
        deg_inv_sqrt_right[col] * edge_weight_q

    edge_weight_real = edge_weight.real
    edge_weight_imag = edge_weight.imag

    real_adj_normalized = csr_matrix((edge_weight_real.numpy(
    ), (row.numpy(), col.numpy())), shape=(num_nodes, num_nodes))
    imag_adj_normalized = csr_matrix((edge_weight_imag.numpy(
    ), (row.numpy(), col.numpy())), shape=(num_nodes, num_nodes))

    return real_adj_normalized, imag_adj_normalized




def one_dim_weighted_add(feat_list, weight_list):
    if not isinstance(feat_list, list) or not isinstance(weight_list, Tensor):
        raise TypeError(
            "This function is designed for list(feature) and tensor(weight)!")
    elif len(feat_list) != weight_list.shape[0]:
        raise ValueError(
            "The feature list and the weight list have different lengths!")
    elif len(weight_list.shape) != 1:
        raise ValueError("The weight list should be a 1d tensor!")

    feat_shape = feat_list[0].shape
    feat_reshape = torch.vstack(
        [feat.contiguous().view(1, -1).squeeze(0) for feat in feat_list])
    weighted_feat = (feat_reshape * weight_list.view(-1, 1)
                     ).sum(dim=0).view(feat_shape)
    return weighted_feat


def two_dim_weighted_add(feat_list, weight_list):
    if not isinstance(feat_list, list) or not isinstance(weight_list, Tensor):
        raise TypeError(
            "This function is designed for list(feature) and tensor(weight)!")
    elif len(feat_list) != weight_list.shape[1]:
        raise ValueError(
            "The feature list and the weight list have different lengths!")
    elif len(weight_list.shape) != 2:
        raise ValueError("The weight list should be a 2d tensor!")

    feat_reshape = torch.stack(feat_list, dim=2)
    weight_reshape = weight_list.unsqueeze(dim=2)
    weighted_feat = torch.bmm(feat_reshape, weight_reshape).squeeze(dim=2)
    return weighted_feat
