import time
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import os
from operators.graph_operator.symmetrical_simgraph_ppr_operator import PprGraphOp
from tasks.clustering_metrics import clustering_metrics
import gc
# Node Classification with Label use and Label Reuse trick
# NOTE: When use this trick, the input_dim of model should be feature_dim + num_classes instead of feature_dim
from tasks.base_task import BaseTask
from tasks.utils import add_labels, node_cls_evaluate_f1, node_cls_mini_batch_evaluate_f1, node_cls_mini_batch_train, node_cls_mini_batch_train_f1, node_cls_train, node_cls_mini_batch_evaluate, \
    node_cls_evaluate, accuracy, node_cls_train_f1
from torch.optim.lr_scheduler import StepLR


class NodeClassificationWithLabelUse(BaseTask):
    def __init__(self, args, dataset, model, lr, weight_decay, epochs, device, normalize_times, loss_fn=nn.CrossEntropyLoss(), seed=42,
                 train_batch_size=None, eval_batch_size=None, label_reuse_batch_size=None,
                 mask_rate=0.5, use_labels=True, reuse_start_epoch=0, label_iters=0):
        super(NodeClassificationWithLabelUse, self).__init__()
        self.args = args
        self.dataset = dataset
        self.labels = self.dataset.y
        self.normalize_record = {"val_acc": [], "test_acc": []}
        self.model = model
        self.optimizer = Adam(model.parameters(), lr=lr,
                              weight_decay=weight_decay)
        self.normalize_times = normalize_times
        self.epochs = epochs

        self.loss_fn = loss_fn
        self.device = device
        self.seed = seed
        self.mask_rate = mask_rate
        self.use_labels = use_labels
        self.reuse_start_epoch = reuse_start_epoch
        self.label_iters = label_iters
        self.label_reuse_batch_size = label_reuse_batch_size
        self.scheduler = StepLR(self.optimizer, step_size=600, gamma=0.1)

        if args.m_r == True:
            self.emb_path = os.path.join("/mnt/ssd2/home/xkli/mjy", "embedding",
                                         f"{self.dataset.name}_smoothed_embs_K_{args.prop_steps}_use_label_{args.use_label}_modify_r_{args.m_r}_{args.a}_{args.b}_{args.c}.pt")
        else:
            self.emb_path = os.path.join("/mnt/ssd2/home/xkli/mjy", "embedding",
                                         f"{self.dataset.name}_smoothed_embs_K_{args.prop_steps}_use_label_{args.use_label}_modify_r_{args.m_r}.pt")
        print(self.emb_path)

        self.mini_batch = False
        if train_batch_size is not None:
            self.mini_batch = True
            self.train_loader = DataLoader(
                self.dataset.train_idx, batch_size=train_batch_size, shuffle=True, drop_last=False)
            self.val_loader = DataLoader(
                self.dataset.val_idx, batch_size=eval_batch_size, shuffle=False, drop_last=False)
            self.test_loader = DataLoader(
                self.dataset.test_idx, batch_size=eval_batch_size, shuffle=False, drop_last=False)
            self.all_eval_loader = DataLoader(
                range(self.dataset.data.num_node), batch_size=eval_batch_size, shuffle=False, drop_last=False)

        if self.label_iters > 0 and self.use_labels is False:
            raise ValueError(
                "When using label reuse, it's essential to enable label use!")

        for i in range(self.normalize_times):
            if i == 0:
                normalize_times_st = time.time()
            self.execute()

        if self.normalize_times > 1:
            print("Optimization Finished!")
            print("Total training time is: {:.4f}s".format(
                time.time() - normalize_times_st))
            print("Mean Val ± Std Val: {}±{}, Mean Test ± Std Test: {}±{}".format(round(np.mean(self.normalize_record["val_acc"]), 4), round(np.std(
                self.normalize_record["val_acc"], ddof=1), 4), round(np.mean(self.normalize_record["test_acc"]), 4), round(np.std(self.normalize_record["test_acc"], ddof=1), 4)))
            s = "Mean Val ± Std Val: {}±{}, Mean Test ± Std Test: {}±{}".format(round(np.mean(self.normalize_record["val_acc"]), 4), round(np.std(
                self.normalize_record["val_acc"], ddof=1), 4), round(np.mean(self.normalize_record["test_acc"]), 4), round(np.std(self.normalize_record["test_acc"], ddof=1), 4))
            with open('scalablegnn_6/output.txt', 'a') as f:

                f.write(f"{self.dataset.name}\n ")
                f.write(f"prop_steps : {self.args.prop_steps} model_name: {self.args.model_name}  num_layers: {self.args.num_layers} hidden_layer: {self.args.hidden_dim} simhomo_name: {self.args.data_name}   use_label: {self.args.use_label} m_r: {self.args.m_r}  r_way: {self.args.r_way}  lr: {self.args.lr}  dropout: {self.args.dropout}  weight_decay: {self.args.weight_decay} num_epochs: {self.args.num_epochs}\n")
                f.write(s)
                f.write(f"\n")

            print("finish task")
            print()

    @property
    def test_acc(self):
        return self.test_acc

    def execute(self):

        self.model = self.model.to(self.device)
        self.labels = self.labels.to(self.device)

        t_total = time.time()
        best_val = 0.
        best_test = 0.
        # self.model.load_state_dict(torch.load("best_model.pkl"))

        features = self.dataset.x
        if self.use_labels:
            # add label feature if use_label is True
            train_idx = np.array(self.dataset.train_idx)
            mask = np.random.rand(train_idx.shape[0]) < self.mask_rate
            train_labels_idx = train_idx[mask]
            train_pred_idx = train_idx[~mask]
            features = add_labels(
                self.dataset.x, self.dataset.y, train_labels_idx, self.dataset.num_classes)

            pre_time_st = time.time()

            # self.model.preprocess(
            #     self.dataset.adj, features, self.emb_path)
            self.model.preprocess(
                self.dataset.adj, features, adapt=self.args.m_p, degree=self.dataset.mining_list)
            # self.model.preprocess(
            #     self.dataset.adj, features,  adapt=self.args.m_p, degree=self.dataset.mining_list)
            pre_time_ed = time.time()
            print(
                f"Feature Propagate done in {(pre_time_ed - pre_time_st):.4f}s")
            # kmeans = KMeans(
            #     n_clusters=self.dataset.num_classes.item(), n_init=20)
            # y_pred = kmeans.fit_predict(
            #     self.model.processed_feat_list[-1].numpy())
            # cm = clustering_metrics(self.labels.cpu().numpy(), y_pred)
            # acc, nmi, adjscore = cm.evaluationClusterModelFromLabel()
            # print("acc:{}, nmi:{}, adjscore:{}".format(acc, nmi, adjscore))
            # exit(0)
        # self.model.load_state_dict(torch.load("best_model.pkl"))

        for epoch in range(self.epochs):

            # label reuse
            # small optimization: only utilize the predicted soft_labels in later epoches
            if self.label_iters > 0 and ((epoch+1) == 200):
                model = self.model
                full_idx = torch.arange(
                    self.dataset.num_node, dtype=torch.long)
                val_idx = np.array(self.dataset.val_idx)
                test_idx = np.array(self.dataset.test_idx)
                unlabeled_idx = np.concatenate(
                    [train_pred_idx, val_idx, test_idx])
                for i in range(self.label_iters):
                    print("label prop {}".format(i))
                    pred = None
                    if self.label_reuse_batch_size is not None:
                        label_reuse_loader = DataLoader(full_idx, batch_size=self.label_reuse_batch_size,
                                                        shuffle=False, drop_last=False)
                        for batch in label_reuse_loader:
                            tmp = model.model_forward(batch, self.device)
                            tmp = F.softmax(tmp, dim=1)
                            tmp = tmp.cpu().detach().numpy()
                            # pred.append(tmp)
                        # outputs = np.vstack((outputs, pred))
                        # pred = torch.cat(pred)
                            if pred is None:
                                pred = tmp
                            else:
                                pred = np.vstack((pred, tmp))
                    else:
                        pred = model.model_forward(full_idx, self.device)
                    # pred = pred.detach().cpu()
                    torch.cuda.empty_cache()
                    # features[unlabeled_idx, -self.dataset.num_classes:] = F.softmax(
                    #     pred[unlabeled_idx], dim=-1)
                    features[unlabeled_idx, -
                             self.dataset.num_classes:] = pred[unlabeled_idx]
                    # self.model.pre_graph_op = PprGraphOp(
                    #     self.args.prop_steps, 0.5, alpha=0.1)
                    self.model.preprocess(self.dataset.adj, features)

            t = time.time()
            if self.mini_batch is False:
                if self.dataset.name in ["ppi", "flickr", "reddit", "ppi_small"]:
                    loss_train, acc_train = node_cls_train_f1(self.model, self.dataset.train_idx, self.labels, self.device,
                                                              self.optimizer, self.loss_fn)
                    acc_val, acc_test = node_cls_evaluate_f1(self.model, self.dataset.val_idx, self.dataset.test_idx,
                                                             self.labels, self.device)
                else:
                    loss_train, acc_train = node_cls_train(self.model, train_pred_idx, self.labels, self.device,
                                                           self.optimizer, self.loss_fn)
                    acc_val, acc_test = node_cls_evaluate(self.model, self.dataset.val_idx, self.dataset.test_idx,
                                                          self.labels, self.device)
            else:
                if self.dataset.name in ["ppi", "flickr", "reddit", "ppi_small"]:
                    loss_train, acc_train = node_cls_mini_batch_train_f1(self.model, self.dataset.train_idx,
                                                                         self.train_loader,
                                                                         self.labels, self.device, self.optimizer,
                                                                         self.loss_fn)
                    acc_val, acc_test = node_cls_mini_batch_evaluate_f1(self.model, self.dataset.val_idx, self.val_loader,
                                                                        self.dataset.test_idx, self.test_loader,
                                                                        self.labels,
                                                                        self.device, self.loss_fn)
                else:
                    loss_train, acc_train = node_cls_mini_batch_train(self.model, train_pred_idx, self.train_loader,
                                                                      self.labels, self.device, self.optimizer, self.loss_fn)
                    acc_val, acc_test = node_cls_mini_batch_evaluate(self.model, self.dataset.val_idx, self.val_loader,
                                                                     self.dataset.test_idx, self.test_loader, self.labels,
                                                                     self.device)
            # self.scheduler.step()
            if self.normalize_times == 1:
                print('Epoch: {:03d}'.format(epoch + 1),
                      'loss_train: {:.4f}'.format(loss_train),
                      'acc_train: {:.4f}'.format(acc_train),
                      'acc_val: {:.4f}'.format(acc_val),
                      'acc_test: {:.4f}'.format(acc_test),
                      'time: {:.4f}s'.format(time.time() - t))
            if acc_val > best_val:
                best_val = acc_val
                best_test = acc_test
                torch.save(self.model.state_dict(), "best_model.pkl")

        self.model.load_state_dict(torch.load("best_model.pkl"))
        # print("Begin to postprocess")
        # acc_val, acc_test = self.postprocess()
        # print("the result of the postprocess is val:{:.4f}, test:{:.4f}".format(
        #     acc_val, acc_test))
        if acc_val > best_val:
            best_val = acc_val
            best_test = acc_test
            torch.save(self.model.state_dict(), "best_model.pkl")
        self.normalize_record["val_acc"].append(best_val)
        self.normalize_record["test_acc"].append(best_test)
        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        print(f'Best val: {best_val:.4f}, best test: {best_test:.4f}')
        if self.normalize_times == 1:

            with open('scalablegnn_6/output.txt', 'a') as f:

                f.write(f"{self.dataset.name}\n ")
                f.write(f"prop_steps : {self.args.prop_steps} model_name: {self.args.model_name}  num_layers: {self.args.num_layers} hidden_layer: {self.args.hidden_dim} simhomo_name: {self.args.data_name}   use_label: {self.args.use_label} m_r: {self.args.m_r} m_p:{self.args.m_p}  r_way: {self.args.r_way}  lr: {self.args.lr}  dropout: {self.args.dropout}  weight_decay: {self.args.weight_decay} num_epochs: {self.args.num_epochs} a: {self.args.a} b:  {self.args.b} c:  {self.args.c} \nval: {best_val}  test: {best_test}\n")
            f.close()
            print("finish task")
            print()

        return best_test

    def postprocess(self):
        self.model.eval()
        if self.mini_batch is False:
            outputs = self.model.model_forward(
                range(self.dataset.num_node), self.device).to(self.device)
        else:
            outputs = None
            for batch in self.all_eval_loader:
                output = self.model.model_forward(batch, self.device)
                output = F.softmax(output, dim=1)
                output = output.cpu().detach().numpy()
                if outputs is None:
                    outputs = output
                else:
                    outputs = np.vstack((outputs, output))

        final_output = self.model.postprocess(self.dataset.adj, outputs)
        acc_val = accuracy(
            final_output[self.dataset.val_idx], self.labels[self.dataset.val_idx])
        acc_test = accuracy(
            final_output[self.dataset.test_idx], self.labels[self.dataset.test_idx])
        return acc_val, acc_test
