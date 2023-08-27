from tqdm import tqdm
import math
import torch
import random
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
import networkx as nx
from sklearn.cluster import KMeans
from tasks.clustering_metrics import clustering_metrics
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
import torch.nn as nn


def accuracy(output, labels):
    pred = output.max(1)[1].type_as(labels)
    correct = pred.eq(labels).double()
    correct = correct.sum()
    return (correct / len(labels)).item()


def adjust_learning_rate(optimizer, lr, epoch):
    if epoch <= 50:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr * epoch / 50


def add_labels(features, labels, idx, num_classes):
    onehot = np.zeros([features.shape[0], num_classes])
    onehot[idx, labels[idx]] = 1
    return np.concatenate([features, onehot], axis=-1)


def link_cls_train(model, train_query_edges, train_labels, device, optimizer, loss_fn):
    model.train()
    model.base_model.query_edges = train_query_edges
    optimizer.zero_grad()
    train_output = model.model_forward(None, device)
    loss_train = loss_fn(train_output, train_labels)
    acc_train = accuracy(train_output, train_labels)
    loss_train.backward()
    optimizer.step()
    return loss_train.item(), acc_train


def link_cls_evaluate(model, val_query_edges, test_query_edges, val_labels, test_labels, device):
    model.eval()
    model.base_model.query_edges = val_query_edges
    val_output = model.model_forward(None, device)
    model.base_model.query_edges = test_query_edges
    test_output = model.model_forward(None, device)
    acc_val = accuracy(val_output, val_labels)
    acc_test = accuracy(test_output, test_labels)
    return acc_val, acc_test


def node_cls_evaluate(model, val_idx, test_idx, labels, device):
    model.eval()
    val_output = model.model_forward(val_idx, device)
    test_output = model.model_forward(test_idx, device)
    if val_output.shape[0] != len(val_idx):
        val_output = val_output[val_idx]

    if test_output.shape[0] != len(test_idx):
        test_output = test_output[test_idx]

    acc_val = accuracy(val_output, labels[val_idx])
    acc_test = accuracy(test_output, labels[test_idx])
    return acc_val, acc_test


def node_test_wrong(model, val_idx, test_idx, labels, device):
    model.eval()
    misclassified_indices = []
    test_output = model.model_forward(test_idx, device)
    if test_output.shape[0] != len(test_idx):
        test_output = test_output[test_idx]
    pred = test_output.max(1)[1].type_as(labels[test_idx])
    correct = pred.eq(labels[test_idx]).double()
    correct = correct.sum()
    acc_test = (correct / len(labels[test_idx])).item()

    misclassified = [idx for idx, (p, l) in enumerate(
        zip(pred, labels[test_idx])) if p != l]
    misclassified_indices = [test_idx[i] for i in misclassified]
    return acc_test, misclassified_indices


def node_cls_evaluate_f1(model, val_idx, test_idx, labels, device):
    model.eval()
    preds_val = []
    preds_test = []
    val_output = model.model_forward(val_idx, device)
    test_output = model.model_forward(test_idx, device)
    if val_output.shape[0] != len(val_idx):
        val_output = val_output[val_idx]
    preds_val.append(torch.argmax(val_output, dim=-1))
    if test_output.shape[0] != len(test_idx):
        test_output = test_output[test_idx]

    preds_test.append(torch.argmax(test_output, dim=-1))

    preds_val = torch.cat(preds_val, dim=0)
    preds_test = torch.cat(preds_test, dim=0)
    acc_val = f1_score(labels[val_idx].cpu(), preds_val.cpu(), average="micro")
    acc_test = f1_score(labels[test_idx].cpu(),
                        preds_test.cpu(), average="micro")

    return acc_val, acc_test


def node_cls_mini_batch_evaluate(model, val_idx, val_loader, test_idx, test_loader, labels, device):
    model.eval()
    correct_num_val, correct_num_test = 0, 0
    for batch in val_loader:
        val_output = model.model_forward(batch, device)

        if val_output.shape[0] != len(batch):
            val_output = val_output[batch]

        pred = val_output.max(1)[1].type_as(labels)
        correct_num_val += pred.eq(labels[batch]).double().sum()
    acc_val = correct_num_val / len(val_idx)

    for batch in test_loader:
        test_output = model.model_forward(batch, device)

        if test_output.shape[0] != len(batch):
            test_output = test_output[batch]

        pred = test_output.max(1)[1].type_as(labels)
        correct_num_test += pred.eq(labels[batch]).double().sum()
    acc_test = correct_num_test / len(test_idx)

    return acc_val.item(), acc_test.item()


def node_cls_train(model, train_idx, labels, device, optimizer, loss_fn):
    model.train()
    optimizer.zero_grad()
    train_output = model.model_forward(train_idx, device)
    if train_output.shape[0] != len(train_idx):
        train_output = train_output[train_idx]
    loss_train = loss_fn(train_output, labels[train_idx])
    acc_train = accuracy(train_output, labels[train_idx])
    loss_train.backward()
    optimizer.step()

    return loss_train.item(), acc_train


def node_cls_train_f1(model, train_idx, labels, device, optimizer, loss_fn):
    preds = []
    model.train()
    optimizer.zero_grad()
    train_output = model.model_forward(train_idx, device)
    if train_output.shape[0] != len(train_idx):
        train_output = train_output[train_idx]
    if isinstance(loss_fn, nn.BCEWithLogitsLoss):
        preds.append((train_output > 0).float())
    else:
        preds.append(torch.argmax(train_output, dim=-1))
    loss_train = loss_fn(train_output, labels[train_idx])

    loss_train.backward()
    optimizer.step()
    preds = torch.cat(preds, dim=0)
    acc_train = f1_score(labels[train_idx].cpu(),
                         preds.cpu(), average="micro")
    return loss_train.item(), acc_train


def node_cls_mini_batch_train(model, train_idx, train_loader, labels, device, optimizer, loss_fn):
    model.train()
    correct_num = 0
    loss_train_sum = 0.
    for batch in train_loader:
        train_output = model.model_forward(batch, device)

        if train_output.shape[0] != len(batch):
            train_output = train_output[batch]

        loss_train = loss_fn(train_output, labels[batch])

        pred = train_output.max(1)[1].type_as(labels)
        correct_num += pred.eq(labels[batch]).double().sum()
        loss_train_sum += loss_train.item()

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

    loss_train = loss_train_sum / len(train_loader)
    acc_train = correct_num / len(train_idx)

    return loss_train, acc_train.item()


def node_cls_mini_batch_train_f1(model, train_idx, train_loader, labels, device, optimizer, loss_fn):
    model.train()
    correct_num = 0
    loss_train_sum = 0.
    preds = []
    for batch in train_loader:
        train_output = model.model_forward(batch, device)

        if train_output.shape[0] != len(batch):
            train_output = train_output[batch]

        loss_train = loss_fn(train_output, labels[batch])

        if isinstance(loss_fn, nn.BCEWithLogitsLoss):
            preds.append((train_output > 0).float())
        else:
            preds.append(torch.argmax(train_output, dim=-1))

        loss_train_sum += loss_train.item()

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
    preds = torch.cat(preds, dim=0)
    loss_train = loss_train_sum / len(train_loader)
    result = f1_score(labels[train_idx].cpu(), preds.cpu(), average="micro")
    return loss_train, result


def node_cls_mini_batch_evaluate_f1(model, val_idx, val_loader, test_idx, test_loader, labels, device, loss_fn):
    model.eval()
    correct_num_val, correct_num_test = 0, 0
    preds_val = []
    preds_test = []
    for batch in val_loader:
        val_output = model.model_forward(batch, device)

        if val_output.shape[0] != len(batch):
            val_output = val_output[batch]
        if isinstance(loss_fn, nn.BCEWithLogitsLoss):
            preds_val.append((val_output > 0).float())
        else:
            preds_val.append(torch.argmax(val_output, dim=-1))

    for batch in test_loader:
        test_output = model.model_forward(batch, device)

        if test_output.shape[0] != len(batch):
            test_output = test_output[batch]

        if isinstance(loss_fn, nn.BCEWithLogitsLoss):
            preds_test.append((test_output > 0).float())
        else:
            preds_test.append(torch.argmax(test_output, dim=-1))

    preds_val = torch.cat(preds_val, dim=0)
    preds_test = torch.cat(preds_test, dim=0)
    acc_val = f1_score(labels[val_idx].cpu(), preds_val.cpu(), average="micro")
    acc_test = f1_score(labels[test_idx].cpu(),
                        preds_test.cpu(), average="micro")
    return acc_val, acc_test


def modify_r(adj, softlabel):
    softlabel_list = np.argmax(softlabel, axis=1)
    nei_with_same_label = []
    for u in tqdm(range(len(softlabel_list))):
        u_neighbors = set(adj.getrow(u).indices)
        nei = 0
        for v in u_neighbors:
            if softlabel_list[u] == softlabel_list[v]:
                nei += 1
        if len(u_neighbors) == 0:
            nei_with_same_label.append(0)
        else:
            nei_with_same_label.append(nei/len(u_neighbors))
    nei_with_same_label = torch.tensor(nei_with_same_label)
    centrality = nei_with_same_label
    # print(torch.mean(nei_with_same_label))
    # min_val = torch.min(nei_with_same_label)
    # max_val = torch.max(nei_with_same_label)

    # centrality = (nei_with_same_label - min_val) / (max_val - min_val)
    return centrality


def modify_adj(adj, softlabel, threshold=0.8):
    G = nx.from_scipy_sparse_matrix(adj)
    confidence, _ = torch.max(softlabel, dim=1)
    print(confidence)
    mask = confidence >= threshold
    selected_nodes = torch.nonzero(mask).squeeze()
    for u in tqdm(selected_nodes):
        u_neighbors = set(adj.getrow(u).indices)
        u = u.item()
        # print("for node {}".format(u))
        for v in u_neighbors:
            common_neighbors = sorted(nx.common_neighbors(G, u, v))
            if len(common_neighbors) == 0:
                continue
            adamic_adar_index = sum(1 / math.log(nx.degree(G, n))
                                    for n in common_neighbors)
            # print(adamic_adar_index)
            w = adamic_adar_index
            row_i = adj.indptr[u]
            row_i_next = adj.indptr[u + 1]
            row_j = adj.indptr[v]
            row_j_next = adj.indptr[v + 1]
            pos_ij = np.searchsorted(adj.indices[row_i:row_i_next], v) + row_i
            pos_ji = np.searchsorted(adj.indices[row_j:row_j_next], u) + row_j
            # 本来有边直接改
            if pos_ij < row_i_next and adj.indices[pos_ij] == v:
                # print("there is a edge and change it!")
                adj.data[pos_ij] = w
                adj.data[pos_ji] = w

    return adj


# def modify_adj(num_class, adj, feature, train_idx, soft_label, true_label, window_size=10, ):
    soft_label = torch.tensor(soft_label)
    confidence, pre_label = torch.max(soft_label, dim=1)
    entropies = -torch.sum(soft_label * torch.log2(soft_label), dim=1)
    train_map = {}
    question_nodes = []
    row_new = []
    col_new = []
    weight_new = []
    for u in tqdm(range(len(soft_label))):
        if u in train_idx:
            train_map[(u, true_label[u])] = entropies[u]
        u_neighbors = set(adj.getrow(u).indices)
        # print(u_neighbors)
        neigh_entropy = []
        for v in u_neighbors:
            neigh_entropy.append(entropies[v])
            if pre_label[u] == pre_label[v] and v not in question_nodes:
                row_new.append(u)
                col_new.append(v)
                weight_new.append(1)

        mean_entro = sum(neigh_entropy) / len(neigh_entropy)

        # 直接删掉所有边过于残暴？
        # the current entropy smaller than the mean of its neighbors
        if mean_entro < entropies[u]:
            question_nodes.append(u)
            # print("the current entropy smaller than the mean of its neighbors")
            # for v in u_neighbors:
            #     if pre_label[u] == pre_label[v] and v not in question_nodes:
            #         row_new.remove(u)
            #         col_new.remove(v)
            #         weight_new.remove(1)

    super_nodes = []
    s = set([x[1].item() for x in train_map.keys()])

    for category in s:
        sorted_elements = sorted([(index, value) for (index, cat), value in train_map.items() if cat == category],
                                 key=lambda x: x[1])[:window_size]
        # print(f"Category {category}: {sorted_elements}")
        indices = [idx for idx, val in sorted_elements]
        super_nodes.extend(indices)

    # print(adj)
    # print(len(question_nodes))
    # print(len(super_nodes))

    # 防止重复
    marked_pairs = {}

    for i, u in tqdm(enumerate(question_nodes)):
        for j, v in enumerate(super_nodes):
            pair = (u, v)
            if pair in marked_pairs:
                continue
            marked_pairs[(v, u)] = 1
            marked_pairs[(u, v)] = 1
            if u == v:
                continue
            euclidean_distance = np.linalg.norm(feature[u, :] - feature[v, :])
            similarity = 1 / (1 + euclidean_distance)
            # print("The similarity between the {} and {} nodes is:{}".format(u, v, similarity))
            row_new.append(u)
            row_new.append(v)
            col_new.append(v)
            col_new.append(u)
            weight_new.append(similarity)
            weight_new.append(similarity)
    new_adj = sp.csr_matrix((weight_new, (row_new, col_new)), shape=(
        len(soft_label), len(soft_label)))
    # print(new_adj)
    return new_adj

# def cluster_loss(train_output, y_pred, cluster_centers):

#     for i in range(len(cluster_centers)):
#         if i == 0:
#             dist = torch.norm(train_output - cluster_centers[i], p=2, dim=1, keepdim=True)
#         else:
#             dist = torch.cat((dist, torch.norm(train_output - cluster_centers[i], p=2, dim=1, keepdim=True)), 1)

#     loss = 0.
#     loss_tmp = -dist.mean(1).sum()
#     loss_tmp += 2 * np.sum(dist[j, x] for j, x in zip(range(dist.shape[0]), y_pred))
#     loss = loss_tmp / dist.shape[0]
#     return loss


# def clustering_train(model, train_idx, labels, device, optimizer, loss_fn, n_clusters, n_init):
#     model.train()
#     optimizer.zero_grad()

#     train_output = model.model_forward(train_idx, device)

#     # calc loss
#     kmeans = KMeans(n_clusters=n_clusters, n_init=n_init)
#     y_pred = kmeans.fit_predict(train_output.data.cpu().numpy()) # cluster_label
#     cluster_centers = torch.FloatTensor(kmeans.cluster_centers_).to(device)

#     loss_train = loss_fn(train_output, y_pred, cluster_centers)
#     loss_train.backward()
#     optimizer.step()

#     # calc acc, nmi, adj
#     labels = labels.cpu().numpy()
#     cm = clustering_metrics(labels, y_pred)
#     acc, nmi, adjscore = cm.evaluationClusterModelFromLabel()

#     return loss_train.item(), acc, nmi, adjscore


# def sparse_to_tuple(sparse_mx):
#     if not sp.isspmatrix_coo(sparse_mx):
#         sparse_mx = sparse_mx.tocoo()
#     coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
#     values = sparse_mx.data
#     shape = sparse_mx.shape
#     return coords, values, shape


# # input full edge_features, pos_edges and neg_edges to calc roc_auc, avg_prec score
# def edge_predict_score(edge_feature, pos_edges, neg_edges, threshold):
#     labels = torch.cat((torch.ones(len(pos_edges)), torch.zeros(len(neg_edges))))
#     all_edges = torch.cat((pos_edges, neg_edges))
#     edge_pred = edge_feature[all_edges[:, 0], all_edges[:, 1]].reshape(-1)
#     edge_pred = torch.sigmoid(edge_pred)
#     # edge_pred = edge_pred > threshold
#     roc_auc = roc_auc_score(labels, edge_pred)
#     avg_prec = average_precision_score(labels, edge_pred)
#     return roc_auc, avg_prec


# def edge_predict_train(model, train_node_index, with_params, pos_edges, neg_edges,
#                        device, optimizer, loss_fn, threshold):
#     if with_params is True:
#         model.train()
#         optimizer.zero_grad()

#     train_output = model.model_forward(train_node_index, device)
#     edge_feature = torch.mm(train_output, train_output.t())
#     labels = torch.cat((torch.ones(len(pos_edges)), torch.zeros(len(neg_edges)))).to(device)
#     train_edge = torch.cat((pos_edges, neg_edges)).to(device)
#     edge_pred = edge_feature[train_edge[:, 0], train_edge[:, 1]].reshape(-1)
#     edge_pred = torch.sigmoid(edge_pred)

#     # print("-----------------------------")
#     # print("edge_features:  ", edge_feature[:200])
#     # print("edge_pred:\n", edge_pred[len(pos_edges)-50:len(pos_edges)+50])
#     # print("labels:\n",labels[len(pos_edges)-50:len(pos_edges)+50])
#     # print("-----------------------------")

#     loss = loss_fn(edge_pred, labels)
#     if with_params is True:
#         loss.backward()
#         optimizer.step()

#     labels = labels.cpu().data
#     edge_pred = edge_pred.cpu().data
#     edge_pred = edge_pred > threshold
#     roc_auc = roc_auc_score(labels, edge_pred)
#     avg_prec = average_precision_score(labels, edge_pred)
#     return loss.item(), roc_auc, avg_prec


# def edge_predict_eval(model, train_node_index, val_pos_edges, val_neg_edges,
#                       test_pos_edges, test_neg_edges, device, threshold):
#     model.eval()
#     train_output = model.model_forward(train_node_index, device)
#     edge_feature = torch.mm(train_output, train_output.t()).cpu().data

#     roc_auc_val, avg_prec_val = edge_predict_score(edge_feature, val_pos_edges, val_neg_edges, threshold)
#     roc_auc_test, avg_prec_test = edge_predict_score(edge_feature, test_pos_edges, test_neg_edges, threshold)

#     return roc_auc_val, avg_prec_val, roc_auc_test, avg_prec_test


# def mini_batch_edge_predict_train(model, train_node_index, with_params, train_loader,
#                                   device, optimizer, loss_fn, threshold):
#     if with_params is True:
#         model.train()
#         optimizer.zero_grad()

#     loss_train = 0.
#     roc_auc_sum = 0.
#     avg_prec_sum = 0.

#     output = model.model_forward(train_node_index, device)
#     output = output.cpu()
#     edge_feature = torch.mm(output, output.t())
#     edge_feature = torch.sigmoid(edge_feature)

#     for batch, label in train_loader:
#         edge_pred = edge_feature[batch[:, 0], batch[:, 1]].reshape(-1)
#         # print("-----------------------------")
#         # print("edge_pred:\n", edge_pred.data[:100])
#         # print("labels:\n",label.data[:100])
#         # print("roc_auc_partial: ",roc_auc_score(label.data, edge_pred.data[:100]))
#         # print("-----------------------------")
#         pred_label = edge_pred > threshold
#         roc_auc_sum += roc_auc_score(label.data, pred_label.data)
#         avg_prec_sum += average_precision_score(label.data, pred_label.data)

#         edge_pred = edge_pred.to(device)
#         label = label.to(device)
#         loss_train += loss_fn(edge_pred, label)

#     if with_params is True:
#         loss_train.backward()
#         optimizer.step()

#     loss_train = loss_train.item() / len(train_loader)
#     roc_auc = roc_auc_sum / len(train_loader)
#     avg_prec = avg_prec_sum / len(train_loader)

#     return loss_train, roc_auc, avg_prec


# def mini_batch_edge_predict_eval(model, train_node_index, val_loader, test_loader, device, threshold):
#     model.eval()
#     roc_auc_val_sum, avg_prec_val_sum = 0., 0.
#     roc_auc_test_sum, avg_prec_test_sum = 0., 0.

#     output = model.model_forward(train_node_index, device)
#     output = output.cpu().data
#     edge_feature = torch.mm(output, output.t())
#     edge_feature = torch.sigmoid(edge_feature)

#     for batch, label in val_loader:
#         edge_pred = edge_feature[batch[:, 0], batch[:, 1]].reshape(-1)
#         label_pred = edge_pred > threshold
#         roc_auc_val_sum += roc_auc_score(label, label_pred)
#         avg_prec_val_sum += average_precision_score(label, label_pred)

#     roc_auc_val = roc_auc_val_sum / len(val_loader)
#     avg_prec_val = avg_prec_val_sum / len(val_loader)

#     for batch, label in test_loader:
#         edge_pred = edge_feature[batch[:, 0], batch[:, 1]].reshape(-1)
#         label_pred = edge_pred > threshold
#         roc_auc_test_sum += roc_auc_score(label, edge_pred)
#         avg_prec_test_sum += average_precision_score(label, edge_pred)

#     roc_auc_test = roc_auc_test_sum / len(test_loader)
#     avg_prec_test = avg_prec_test_sum / len(test_loader)

#     return roc_auc_val, avg_prec_val, roc_auc_test, avg_prec_test


# def mix_pos_neg_edges(pos_edges, neg_edges, mix_size):
#     start, end = 0, mix_size
#     mix_edges = torch.cat((pos_edges[start:end], neg_edges[start:end]))
#     mix_labels = torch.cat((torch.ones(end - start), torch.zeros(end - start)))

#     start += mix_size
#     end += mix_size
#     while end < len(pos_edges):
#         tmp_edges = torch.cat((pos_edges[start:end], neg_edges[start:end]))
#         tmp_labels = torch.cat((torch.ones(end - start), torch.zeros(end - start)))
#         mix_edges = torch.cat((mix_edges, tmp_edges))
#         mix_labels = torch.cat((mix_labels, tmp_labels))
#         start += mix_size
#         end += mix_size

#     tmp_edges = torch.cat((pos_edges[start:], neg_edges[start:]))
#     tmp_labels = torch.cat((torch.ones(len(pos_edges) - start), torch.zeros(len(neg_edges) - start)))
#     mix_edges = torch.cat((mix_edges, tmp_edges))
#     mix_labels = torch.cat((mix_labels, tmp_labels))

#     return mix_edges, mix_labels
