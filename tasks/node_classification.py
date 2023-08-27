import time
import torch.nn.functional as F
from sklearn.metrics import f1_score
import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from tasks.base_task import BaseTask
from tasks.utils import accuracy, node_cls_mini_batch_evaluate_f1, node_cls_mini_batch_train_f1, node_cls_train, node_cls_mini_batch_train, node_cls_evaluate, node_cls_mini_batch_evaluate, node_cls_train_f1, node_cls_evaluate_f1, node_test_wrong
import os


class NodeClassification(BaseTask):
    def __init__(self, args, dataset, model, normalize_times,
                 lr, weight_decay, epochs, device, loss_fn=nn.CrossEntropyLoss(),
                 train_batch_size=None, eval_batch_size=None):
        super(NodeClassification, self).__init__()
        self.normalize_times = normalize_times
        self.normalize_record = {"val_acc": [], "test_acc": []}
        self.args = args
        self.dataset = dataset
        self.labels = self.dataset.y

        self.model = model
        self.optimizer = Adam(model.parameters(), lr=lr,
                              weight_decay=weight_decay)

        if self.dataset.name in ["ppi", "ppi_small"]:
            loss_fn = nn.BCEWithLogitsLoss()

        self.scheduler = StepLR(self.optimizer, step_size=200, gamma=0.5)
        self.epochs = epochs
        self.loss_fn = loss_fn
        self.device = device

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

        if args.m_r == True:
            self.emb_path = os.path.join("/mnt/ssd2/home/xkli/mjy", "embedding",
                                         f"{self.dataset.name}_smoothed_embs_K_{args.prop_steps}_use_label_{args.use_label}_modify_r_{args.m_r}_{args.r_way}_{args.a}_{args.b}_{args.c}.pt")
        else:
            self.emb_path = os.path.join("/mnt/ssd2/home/xkli/mjy", "embedding",
                                         f"{self.dataset.name}_smoothed_embs_K_{args.prop_steps}_use_label_{args.use_label}_modify_r_{args.m_r}_modify_p_{args.m_p}.pt")
        print(self.emb_path)

        for i in range(self.normalize_times):
            if i == 0:
                normalize_times_st = time.time()
            self.execute()
            print(f"finish training for {i+1} round!")

        if self.normalize_times > 1:
            print("Optimization Finished!")
            print("Total training time is: {:.4f}s".format(
                time.time() - normalize_times_st))
            print("Mean Val ± Std Val: {}±{}, Mean Test ± Std Test: {}±{}".format(round(np.mean(self.normalize_record["val_acc"]), 4), round(np.std(
                self.normalize_record["val_acc"], ddof=1), 4), round(np.mean(self.normalize_record["test_acc"]), 4), round(np.std(self.normalize_record["test_acc"], ddof=1), 4)))
            s = "Mean Val ± Std Val: {}±{}, Mean Test ± Std Test: {}±{}".format(round(np.mean(self.normalize_record["val_acc"]), 4), round(np.std(
                self.normalize_record["val_acc"], ddof=1), 4), round(np.mean(self.normalize_record["test_acc"]), 4), round(np.std(self.normalize_record["test_acc"], ddof=1), 4))
            print(self.normalize_record["test_acc"])
            with open('output.txt', 'a') as f:

                f.write(f"{self.dataset.name}\n ")
                f.write(f"prop_steps : {self.args.prop_steps} model_name: {self.args.model_name}  num_layers: {self.args.num_layers} hidden_layer: {self.args.hidden_dim} homo_name: {self.args.data_name}   use_label: {self.args.use_label} m_r: {self.args.m_r}  m_p:{self.args.m_p}  r_way: {self.args.r_way}  lr: {self.args.lr}  dropout: {self.args.dropout}  weight_decay: {self.args.weight_decay}  num_epochs: {self.args.num_epochs} a: {self.args.a} b: {self.args.b}  c:{self.args.c}\n")
                f.write(s)
                f.write(f"\n")

            print("finish task")
            print()

    def get_test_acc(self):
        return np.mean(self.normalize_record["test_acc"])

    def execute(self):
        pre_time_st = time.time()
        print("adj length: ", self.dataset.adj.getnnz())
        self.model.preprocess(
            self.dataset.adj, self.dataset.x, adapt=self.args.m_p, degree=self.dataset.mining_list)

        pre_time_ed = time.time()

        if self.normalize_times == 1:
            print(f"Preprocessing done in {(pre_time_ed - pre_time_st):.4f}s")
        if self.dataset.name == "ppi":
            self.labels = self.labels.float()
        self.model = self.model.to(self.device)
        self.labels = self.labels.to(self.device)

        t_total = time.time()
        best_val = 0.
        best_test = 0.
        if self.mini_batch is True:
            print("batch training")
        # self.model.load_state_dict(torch.load("best_model.pkl"))
        for epoch in range(self.epochs):
            t = time.time()
            if self.mini_batch is False:
                if self.dataset.name in ["ppi", "flickr", "reddit", "ppi_small"]:
                    loss_train, acc_train = node_cls_train_f1(self.model, self.dataset.train_idx, self.labels, self.device,
                                                              self.optimizer, self.loss_fn)
                    acc_val, acc_test = node_cls_evaluate_f1(self.model, self.dataset.val_idx, self.dataset.test_idx,
                                                             self.labels, self.device)
                else:

                    loss_train, acc_train = node_cls_train(self.model, self.dataset.train_idx, self.labels, self.device,
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
                    loss_train, acc_train = node_cls_mini_batch_train(self.model, self.dataset.train_idx, self.train_loader,
                                                                      self.labels, self.device, self.optimizer, self.loss_fn)
                    # if epoch % 50 == 0:
                    acc_val, acc_test = node_cls_mini_batch_evaluate(self.model, self.dataset.val_idx, self.val_loader,
                                                                     self.dataset.test_idx, self.test_loader, self.labels,
                                                                     self.device)
            # self.scheduler.step()
            if (epoch+1) % 1 == 0:
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

        print("Optimization Finished!")
        print("Total training time is: {:.4f}s".format(
            time.time() - t_total))
        print(f'Best val: {best_val:.4f}, best test: {best_test:.4f}')
        self.normalize_record["val_acc"].append(best_val)
        self.normalize_record["test_acc"].append(best_test)

        if self.normalize_times == 1:
            with open('output.txt', 'a') as f:

                f.write(f"{self.dataset.name}\n ")
                f.write(f"prop_steps : {self.args.prop_steps} model_name: {self.args.model_name}  num_layers: {self.args.num_layers} hidden_layer: {self.args.hidden_dim} homo_name: {self.args.data_name}   use_label: {self.args.use_label} m_r: {self.args.m_r} m_p:{self.args.m_p}  r_way: {self.args.r_way}  lr: {self.args.lr}  dropout: {self.args.dropout}  weight_decay: {self.args.weight_decay} num_epochs: {self.args.num_epochs} a: {self.args.a} b:  {self.args.b} c:  {self.args.c} \nval: {best_val}  test: {best_test}\n")

            print("finish task")
            print()

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
        # outputs = outputs.cpu().detach().numpy()
        final_output = self.model.postprocess(self.dataset.adj, outputs)
        # final_output = self.model.postprocess(self.dataset.adj, outputs, self.dataset.x, self.dataset.train_idx, self.labels, self.dataset.num_classes)

        final_output = torch.tensor(final_output)

        # final_output = outputs

        if self.dataset.name in ["ppi", "flickr", "reddit", "ppi_small"]:
            final_output = torch.argmax(final_output, dim=-1)
            acc_val = f1_score(
                self.labels[self.dataset.val_idx].cpu(), final_output[self.dataset.val_idx].cpu(), average="micro")
            acc_test = f1_score(
                self.labels[self.dataset.test_idx].cpu(), final_output[self.dataset.test_idx].cpu(), average="micro")
        else:
            acc_val = accuracy(
                final_output[self.dataset.val_idx], self.labels[self.dataset.val_idx])
            acc_test = accuracy(
                final_output[self.dataset.test_idx], self.labels[self.dataset.test_idx])
        return acc_val, acc_test
