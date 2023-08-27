import gc
import time
import torch
import numpy as np
import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F
from operators.message_operator.message_operator.learnable_weighted_messahe_op import LearnableWeightedMessageOp
from tasks.utils import modify_adj, modify_r
from models.utils import scipy_sparse_mat_to_torch_sparse_tensor
from operators.graph_operator.symmetrical_simgraph_laplacian_operator import SymLaplacianGraphOp
import os

class BaseSGModel(nn.Module):
    def __init__(self, prop_steps, feat_dim, output_dim):
        super(BaseSGModel, self).__init__()
        self.prop_steps = prop_steps
        self.feat_dim = feat_dim
        self.output_dim = output_dim

        self.naive_graph_op = None
        self.pre_graph_op, self.pre_msg_op = None, None
        self.post_graph_op, self.post_msg_op = None, None
        self.base_model = None
        self.processed_feat_list_all = None


        self.processed_feat_list = None
        self.processed_feature = None
        self.pre_msg_learnable = False
        
        self.attention_r_list = None
        # self.attention_r = SimLearnableWeightedMessageOp(0, 4, "gate", self.feat_dim*(self.prop_steps+1))




    def preprocess(self, adj, feature, emb_path = None, adapt=False, degree=None):
        if self.pre_graph_op is not None:
            node_sum = adj.shape[0]
            edge_sum = adj.sum()/2
            row_sum = (adj.sum(1) + 1)
            norm_a_inf = row_sum/ (2*edge_sum+node_sum)
            norm_a_inf = torch.Tensor(norm_a_inf).view(-1, node_sum)
            if emb_path is not None and os.path.exists(emb_path):
                self.processed_feat_list = torch.load(emb_path)
                print("load the exist feature")
            else:
                self.processed_feat_list = self.pre_graph_op.propagate(
                    adj, feature, adapt, norm_a_inf, degree=degree)
                if emb_path is not None:
                    torch.save(self.processed_feat_list, emb_path)

            

            # self.processed_feat_list_all = self.processed_feat_list
            # if len(self.processed_feat_list_all) > 1:
            #     attention_r_list = []
            #     for i in range(len(self.processed_feat_list_all)):
            #         t = torch.hstack(self.processed_feat_list_all[i])
            #         attention_r_list.append(t)
            #     self.attention_r_list = attention_r_list
            # self.pre_msg_learnable = True
            if self.pre_msg_op.aggr_type in [
                    "proj_concat", "learnable_weighted", "iterate_learnable_weighted"]:
                self.pre_msg_learnable = True
            else:
                self.pre_msg_learnable = False
                self.processed_feature = self.pre_msg_op.aggregate(
                    self.processed_feat_list)

        else:
            if self.naive_graph_op is not None:
                self.base_model.adj = self.naive_graph_op.construct_adj(adj)
                if not isinstance(self.base_model.adj, sp.csr_matrix):
                    raise TypeError(
                        "The adjacency matrix must be a scipy csr sparse matrix!")
                elif self.base_model.adj.shape[1] != feature.shape[0]:
                    raise ValueError(
                        "Dimension mismatch detected for the adjacency and the feature matrix!")
                self.base_model.adj = scipy_sparse_mat_to_torch_sparse_tensor(
                    self.base_model.adj)
            self.pre_msg_learnable = False
            self.processed_feature = torch.FloatTensor(feature)
        


    def postprocess(self, adj, output):
        if self.post_graph_op is not None:
            if self.post_msg_op.aggr_type in [
                    "proj_concat", "learnable_weighted", "iterate_learnable_weighted"]:
                raise ValueError(
                    "Learnable weighted message operator is not supported in the post-processing phase!")
            print("here")
            output = output.cpu().detach().numpy()
            # adj = modify_adj(adj, softlabel=output)
            centrality = modify_r(adj, output)

            self.post_graph_op = SymLaplacianGraphOp(
                self.prop_steps, r=1-centrality)

            output = self.post_graph_op.propagate(adj, output)
            output = self.post_msg_op.aggregate(output)
            

        return output

    # a wrapper of the forward function
    def model_forward(self, idx, device):
        return self.forward(idx, device)

    def forward(self, idx, device):
        processed_feature = None
        if self.base_model.adj != None:
            # t1 = time.time()
            self.base_model.adj = self.base_model.adj.to(device)
            processed_feature = self.processed_feature.to(device)
            # print(f"gcn to device: {time.time()-t1:.5f}")

        else:
            
            if self.pre_msg_learnable is False:
                # t1 = time.time()
                processed_feature = self.processed_feature[idx].to(device)
                # print(f"sgc to device: {time.time()-t1:.5f}")
            else:
                # transferred_feat_list = [feat.to(
                #     device) for feat in self.processed_feat_list]
                transferred_feat_list = [feat[idx].to(
                    device) for feat in self.processed_feat_list]
                processed_feature = self.pre_msg_op.aggregate(
                    transferred_feat_list)
        # gc.collect()
        output = self.base_model(processed_feature)
        # del processed_feature
        # torch.cuda.empty_cache()
        return output



   