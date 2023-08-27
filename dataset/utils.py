import ssl
import sys
import torch
import urllib
import warnings
import numpy as np
import pickle as pkl
import os.path as osp
import scipy.sparse as sp
import torch.nn.functional as F

from torch import Tensor
from torch import FloatTensor
from scipy.sparse import coo_matrix
from torch_scatter import scatter_add
from sklearn.preprocessing import StandardScaler
from torch_geometric.utils import remove_self_loops
from torch_geometric.utils import to_scipy_sparse_matrix


def read_npz(path):
    with np.load(path) as f:
        return parse_npz(f)


def parse_npz(f):
    x = sp.csr_matrix((f['attr_data'], f['attr_indices'],
                      f['attr_indptr']), f['attr_shape']).todense()
    x = torch.from_numpy(x).to(torch.float)
    x[x > 0] = 1

    adj = sp.csr_matrix((f['adj_data'], f['adj_indices'],
                        f['adj_indptr']), f['adj_shape']).tocoo()
    row = torch.from_numpy(adj.row).to(torch.long)
    col = torch.from_numpy(adj.col).to(torch.long)
    edge_index = torch.stack([row, col], dim=0)
    edge_index = remove_self_loops(edge_index)
    edge_index = to_undirected(edge_index)

    y = torch.from_numpy(f['labels']).to(torch.long)

    return x, edge_index, y


def random_split_dataset(n_samples):
    val_idx = np.random.choice(
        list(range(n_samples)), size=int(n_samples * 0.2), replace=False)
    remain = set(range(n_samples)) - set(val_idx)
    test_idx = np.random.choice(
        list(remain), size=int(n_samples * 0.2), replace=False)
    train_idx = list(remain - set(test_idx))

    return train_idx, val_idx, test_idx


def file_exist(filepaths):
    if isinstance(filepaths, list):
        for filepath in filepaths:
            if not osp.exists(filepath):
                return False
        return True
    else:
        if osp.exists(filepaths):
            return True
        else:
            return False


def pkl_read_file(filepath):
    file = None
    with open(filepath, 'rb') as rf:
        try:
            if sys.version_info > (3, 0):
                file = pkl.load(rf, encoding="latin1")
            else:
                file = pkl.load(rf)
        except IOError as e:
            print(e)
            exit(1)
    return file


def download_to(url, path):
    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)

    with open(path, 'wb') as wf:
        try:
            wf.write(data.read())
        except IOError as e:
            print(e)
            exit(1)


def to_undirected(edge_index):
    if isinstance(edge_index, sp.csr_matrix) or isinstance(edge_index, sp.coo_matrix):
        row, col = edge_index.row, edge_index.col
        row, col = torch.from_numpy(row), torch.from_numpy(col)
    else:
        row, col = edge_index
        if not isinstance(row, Tensor) or not isinstance(col, Tensor):
            row, col = torch.from_numpy(row), torch.from_numpy(col)
    new_row = torch.hstack((row, col))
    new_col = torch.hstack((col, row))
    new_edge_index = torch.stack((new_row, new_col), dim=0)
    return new_edge_index


def coomatrix_to_torch_tensor(edge_index):
    if isinstance(edge_index, sp.csr_matrix) or isinstance(edge_index, sp.coo_matrix):
        row, col = edge_index.row, edge_index.col
        row, col = torch.from_numpy(row), torch.from_numpy(col)
    else:
        row, col = edge_index
    edge_index = torch.stack((row, col), dim=0)
    return edge_index


def edge_homophily(A, labels, ignore_negative=False):
    src_node, targ_node = A.nonzero()
    matching = labels[src_node] == labels[targ_node]
    labeled_mask = (labels[src_node] >= 0) * (labels[targ_node] >= 0)
    if ignore_negative:
        matching = matching[labeled_mask].numpy()
    else:
        matching = matching.numpy()
    edge_hom = np.sum(matching == 1) / len(matching)
    return edge_hom


def node_homophily(A, labels):
    src_node, targ_node = A.nonzero()
    edge_idx = torch.tensor(
        np.vstack((src_node, targ_node)), dtype=torch.long).contiguous()
    num_nodes = A.shape[0]
    return node_homophily_edge_idx(edge_idx, labels, num_nodes)


def node_homophily_edge_idx(edge_idx, labels, num_nodes):
    edge_index = remove_self_loops(edge_idx)[0]
    hs = torch.zeros(num_nodes)
    degs = torch.bincount(edge_index[0, :]).float()
    iso_nodes = num_nodes - len(degs)
    for _ in (range(iso_nodes)):
        tmp = torch.zeros((1))
        degs = torch.hstack((degs, tmp))
    matches = (labels[edge_index[0, :]] == labels[edge_index[1, :]]).float()
    hs = hs.scatter_add(0, edge_index[0, :], matches) / degs
    return hs[degs != 0].mean().item()


def compat_matrix_edge_idx(A, labels):
    src_node, targ_node = A.nonzero()
    edge_idx = torch.tensor(
        np.vstack((src_node, targ_node)), dtype=torch.long).contiguous()
    edge_index = remove_self_loops(edge_idx)[0]
    src_node, targ_node = edge_index[0, :], edge_index[1, :]
    labeled_nodes = (labels[src_node] >= 0) * (labels[targ_node] >= 0)
    label = labels.squeeze()
    c = label.max()+1
    H = torch.zeros((c, c)).to(edge_index.device)
    src_label = label[src_node[labeled_nodes]]
    targ_label = label[targ_node[labeled_nodes]]
    label_idx = torch.cat(
        (src_label.unsqueeze(0), targ_label.unsqueeze(0)), axis=0)
    for k in range(c):
        sum_idx = torch.where(src_label == k)[0]
        add_idx = targ_label[sum_idx]
        scatter_add(torch.ones_like(add_idx).to(
            H.dtype), add_idx, out=H[k, :], dim=-1)
    H = H / torch.sum(H, axis=1, keepdims=True)
    return H


def linkx_homophily(edge_index, label):
    label = label.squeeze()
    c = label.max()+1
    H = compat_matrix_edge_idx(edge_index, label)
    nonzero_label = label[label >= 0]
    counts = nonzero_label.unique(return_counts=True)[1]
    proportions = counts.float() / nonzero_label.shape[0]
    val = 0
    for k in range(c):
        class_add = torch.clamp(H[k, k] - proportions[k], min=0)
        if not torch.isnan(class_add):
            # only add if not nan
            val += class_add
    val /= c-1
    return val.item()


def even_quantile_labels(vals, nclasses, verbose=True):
    """ partitions vals into nclasses by a quantile based split,
    where the first class is less than the 1/nclasses quantile,
    second class is less than the 2/nclasses quantile, and so on

    vals is np array
    returns an np array of int class labels
    """
    label = -1 * np.ones(vals.shape[0], dtype=np.int64)
    interval_lst = []
    lower = -np.inf
    for k in range(nclasses - 1):
        upper = np.nanquantile(vals, (k + 1) / nclasses)
        interval_lst.append((lower, upper))
        inds = (vals >= lower) * (vals < upper)
        label[inds] = k
        lower = upper
    label[vals >= lower] = nclasses - 1
    interval_lst.append((lower, np.inf))
    if verbose:
        print('Class Label Intervals:')
        for class_idx, interval in enumerate(interval_lst):
            print(f'Class {class_idx}: [{interval[0]}, {interval[1]})]')
    return label


def remove_self_loops(edge_index, edge_attr=None):
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    if edge_attr is None:
        return edge_index, None
    else:
        return edge_index, edge_attr[mask]


def remove_self_loops_weights(edge_weight, edge_index):
    mask = edge_index[0] != edge_index[1]
    edge_weight = edge_weight[mask]
    return edge_weight


def set_hermitian_features(A, k: int = 2):
    """ create Hermitian feature  (rw normalized)
    Args:
        k (int):  Half of the dimension of features. Default is 2.
    """
    A = A
    H = (A-A.transpose()) * 1j
    # (np.real(H).power(2) + np.imag(H).power(2)).power(0.5)
    H_abs = np.abs(H)
    D_abs_inv = sp.diags(1/np.array(H_abs.sum(1))[:, 0])
    H_rw = D_abs_inv.dot(H)
    u, _, _ = sp.linalg.svds(H_rw, k=k)
    features_SVD = np.concatenate((np.real(u), np.imag(u)), axis=1)
    scaler = StandardScaler().fit(features_SVD)
    features_SVD = scaler.transform(features_SVD)
    features = FloatTensor(features_SVD)
    return features


def sqrtinvdiag(M: sp.spmatrix) -> sp.csc_matrix:
    """Inverts and square-roots a positive diagonal matrix.

    Args:
        M (scipy sparse matrix): matrix to invert
    Returns:
        scipy sparse matrix of inverted square-root of diagonal
    """

    d = M.diagonal()
    dd = [1 / max(np.sqrt(x), 1 / 999999999) for x in d]

    return sp.dia_matrix((dd, [0]), shape=(len(d), len(d))).tocsc()


def separate_positive_negative(num_nodes, edge_index, edge_weight):
    ind = edge_weight > 0
    edge_index_p = edge_index[:, ind]
    edge_weight_p = edge_weight[ind]
    ind = edge_weight < 0
    edge_index_n = edge_index[:, ind]
    edge_weight_n = - edge_weight[ind]
    A_p = to_scipy_sparse_matrix(
        edge_index_p, edge_weight_p, num_nodes=num_nodes)
    A_n = to_scipy_sparse_matrix(
        edge_index_n, edge_weight_n, num_nodes=num_nodes)
    return edge_index_p, edge_weight_p, edge_index_n, edge_weight_n, A_p, A_n


def set_signed_Laplacian_features(num_nodes, edge_index, edge_weight, k=2):
    """generate the graph features using eigenvectors of the signed Laplacian matrix.

    Args:
        k (int): The dimension of the features. Default is 2.
    """
    edge_index_p, edge_weight_p, edge_index_n, edge_weight_n, A_p, A_n = separate_positive_negative(
        num_nodes, edge_index, edge_weight)
    A = (A_p - A_n).tocsc()
    D_p = sp.diags(A_p.sum(axis=0).tolist(), [0]).tocsc()
    D_n = sp.diags(A_n.sum(axis=0).tolist(), [0]).tocsc()
    Dbar = (D_p + D_n)
    d = sqrtinvdiag(Dbar)
    normA = d * A * d
    # normalized symmetric signed Laplacian
    L = sp.eye(A_p.shape[0], format="csc") - normA
    (vals, vecs) = sp.linalg.eigs(
        L, int(k), maxiter=A_p.shape[0], which='LR')
    vecs = vecs / vals  # weight eigenvalues by eigenvectors, since smaller eigenvectors are more likely to be informative
    features = FloatTensor(vecs)
    return features


def set_spectral_adjacency_reg_features(num_nodes, edge_index, edge_weight, k=2, normalization=None, tau_p=None, tau_n=None,
                                        eigens=None, mi=None):
    """generate the graph features using eigenvectors of the regularised adjacency matrix.

    Args:
        k (int): The dimension of the features. Default is 2.
        normalization (string): How to normalise for cluster size:

            1. :obj:`none`: No normalization.

            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{A} <- \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2}`

            3. :obj:`"rw"`: Random-walk normalization
            :math:`\mathbf{A} <- \mathbf{D}^{-1} \mathbf{A}`

            4. :obj:`"sym_sep"`: Symmetric normalization for the positive and negative parts separately.

            5. :obj:`"rw_sep"`: Random-walk normalization for the positive and negative parts separately.

        tau_p (int): Regularisation coefficient for positive adjacency matrix.
        tau_n (int): Regularisation coefficient for negative adjacency matrix.
        eigens (int): The number of eigenvectors to take. Defaults to k.
        mi (int): The maximum number of iterations for which to run eigenvlue solvers. Defaults to number of nodes.
    """
    edge_index_p, edge_weight_p, edge_index_n, edge_weight_n, A_p, A_n = separate_positive_negative(
        num_nodes, edge_index, edge_weight)
    A = (A_p - A_n).tocsc()
    A_p = sp.csc_matrix(A_p)
    A_n = sp.csc_matrix(A_n)
    D_p = sp.diags(A_p.sum(axis=0).tolist(), [0]).tocsc()
    D_n = sp.diags(A_n.sum(axis=0).tolist(), [0]).tocsc()
    Dbar = (D_p + D_n)
    d = sqrtinvdiag(Dbar)
    size = A_p.shape[0]
    if eigens == None:
        eigens = k

    if mi == None:
        mi = size

    if tau_p == None or tau_n == None:
        tau_p = 0.25 * np.mean(Dbar.data) / size
        tau_n = 0.25 * np.mean(Dbar.data) / size

    p_tau = A_p.copy().astype(np.float32)
    n_tau = A_n.copy().astype(np.float32)
    p_tau.data += tau_p
    n_tau.data += tau_n

    Dbar_c = size - Dbar.diagonal()

    Dbar_tau_s = (p_tau + n_tau).sum(axis=0) + \
        (Dbar_c * abs(tau_p - tau_n))[None, :]

    Dbar_tau = sp.diags(Dbar_tau_s.tolist(), [0])

    if normalization is None:
        matrix = A
        delta_tau = tau_p - tau_n

        def mv(v):
            return matrix.dot(v) + delta_tau * v.sum()

    elif normalization == 'sym':
        d = sqrtinvdiag(Dbar_tau)
        matrix = d * A * d
        dd = d.diagonal()
        tau_dd = (tau_p - tau_n) * dd

        def mv(v):
            return matrix.dot(v) + tau_dd * dd.dot(v)

    elif normalization == 'sym_sep':

        diag_corr = sp.diags([size * tau_p] * size).tocsc()
        dp = sqrtinvdiag(D_p + diag_corr)

        matrix = dp * A_p * dp

        diag_corr = sp.diags([size * tau_n] * size).tocsc()
        dn = sqrtinvdiag(D_n + diag_corr)

        matrix = matrix - (dn * A_n * dn)

        dpd = dp.diagonal()
        dnd = dn.diagonal()
        tau_dp = tau_p * dpd
        tau_dn = tau_n * dnd

        def mv(v):
            return matrix.dot(v) + tau_dp * dpd.dot(v) - tau_dn * dnd.dot(v)

    else:
        raise NameError('Error in choosing normalization!')

    matrix_o = sp.linalg.LinearOperator(matrix.shape, matvec=mv)

    (w, v) = sp.linalg.eigs(matrix_o, int(eigens), maxiter=mi, which='LR')

    v = v * w  # weight eigenvalues by eigenvectors, since larger eigenvectors are more likely to be informative

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        features = FloatTensor(v)

    return feature_normalization(features)


def set_in_out_degree_features(num_node, edge_index, signed, edge_weight):
    cpu_edge_index = edge_index.cpu()
    if signed:
        if edge_weight is None:
            raise ValueError(
                'Edge weight input should not be None when generating features based on edge signs!')
        else:
            edge_weight = edge_weight.cpu().numpy()
        A = coo_matrix((edge_weight, (cpu_edge_index[0], cpu_edge_index[1])),
                       shape=(num_node, num_node), dtype=np.float32).tocsr()
        A_abs = A.copy()
        A_abs.data = np.abs(A_abs.data)
        A_p = (A_abs + A)/2
        A_n = (A_abs - A)/2
        out_pos_degree = np.sum(A_p, axis=0).T
        out_neg_degree = np.sum(A_n, axis=0).T
        in_pos_degree = np.sum(A_p, axis=1)
        in_neg_degree = np.sum(A_n, axis=1)
        degree = torch.from_numpy(
            np.c_[in_pos_degree, in_neg_degree, out_pos_degree, out_neg_degree]).float()
    else:
        if edge_weight is None:
            edge_weight = np.ones(len(cpu_edge_index.T))
        else:
            edge_weight = np.abs(edge_weight.cpu().numpy())
        A = coo_matrix((edge_weight, (cpu_edge_index[0], cpu_edge_index[1])),
                       shape=(num_node, num_node), dtype=np.float32).tocsr()
        out_degree = np.sum(A, axis=0).T
        in_degree = np.sum(A, axis=1)
        degree = torch.from_numpy(np.c_[in_degree, out_degree]).float()
    return degree


def feature_normalization(features):
    features = F.normalize(features, p=1, dim=1)
    features = features.numpy()
    m = features.mean(axis=0)
    s = features.std(axis=0, ddof=0, keepdims=True) + 1e-12
    features -= m
    features /= s
    return torch.FloatTensor(features)
