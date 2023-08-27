import math
from typing import Optional
import torch
import torch.nn as nn
from torch.nn import Parameter




class IdenticalMapping(nn.Module):
    def __init__(self) -> None:
        super(IdenticalMapping, self).__init__()
        self.adj = None

    def forward(self, feature):
        return feature


class LogisticRegression(nn.Module):
    def __init__(self, feat_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.adj = None
        self.fc = nn.Linear(feat_dim, output_dim)

    def forward(self, feature):
        output = self.fc(feature)
        return output


class MultiLayerPerceptron(nn.Module):
    def __init__(self, feat_dim, hidden_dim, num_layers, output_dim, dropout, bn=False):
        super(MultiLayerPerceptron, self).__init__()
        self.adj = None
        if num_layers < 2:
            raise ValueError("MLP must have at least two layers!")
        self.num_layers = num_layers
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(feat_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.fcs.append(nn.Linear(hidden_dim, hidden_dim))
        self.fcs.append(nn.Linear(hidden_dim, output_dim))

        self.bn = bn
        if self.bn is True:
            self.bns = nn.ModuleList()
            for _ in range(num_layers - 1):
                self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.input_dropout = nn.Dropout(0.0)
        self.prelu = nn.PReLU()
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for fc in self.fcs:
            nn.init.xavier_uniform_(fc.weight, gain=gain)
            nn.init.zeros_(fc.bias)

    def forward(self, feature):
        for i in range(self.num_layers - 1):
            feature = self.fcs[i](self.input_dropout(feature))
            if self.bn is True:
                feature = self.bns[i](feature)
            feature = self.prelu(feature)
            feature = self.dropout(feature)

        output = self.fcs[-1](feature)
        return output


class ResMultiLayerPerceptron(nn.Module):
    def __init__(self, feat_dim, hidden_dim, num_layers, output_dim, dropout=0.8, bn=False):
        super(ResMultiLayerPerceptron, self).__init__()
        self.adj = None
        if num_layers < 2:
            raise ValueError("ResMLP must have at least two layers!")
        self.num_layers = num_layers
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(feat_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.fcs.append(nn.Linear(hidden_dim, hidden_dim))
        self.fcs.append(nn.Linear(hidden_dim, output_dim))

        self.bn = bn
        if self.bn is True:
            self.bns = nn.ModuleList()
            for _ in range(num_layers - 1):
                self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, feature):
        feature = self.dropout(feature)
        feature = self.fcs[0](feature)
        if self.bn is True:
            feature = self.bns[0](feature)
        feature = self.relu(feature)
        residual = feature

        for i in range(1, self.num_layers - 1):
            feature = self.dropout(feature)
            feature = self.fcs[i](feature)
            if self.bn is True:
                feature = self.bns[i](feature)
            feature_ = self.relu(feature)
            feature = feature_ + residual
            residual = feature_

        feature = self.dropout(feature)
        output = self.fcs[-1](feature)
        return output


class TwoLayerGraphConvolution(nn.Module):
    def __init__(self, feat_dim, hidden_dim, output_dim, dropout=0.5):
        super(TwoLayerGraphConvolution, self).__init__()
        self.adj = None
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(feat_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, feature):
        x = feature
        x = self.fc1(x)
        x = torch.mm(self.adj, x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.mm(self.adj, x)
        return x


class MultiHeadLinear(nn.Module):
    def __init__(self, in_feats, out_feats, n_heads, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(
            size=(n_heads, in_feats, out_feats)))
        if bias:
            self.bias = nn.Parameter(
                torch.FloatTensor(size=(n_heads, 1, out_feats)))
        else:
            self.bias = None

    def reset_parameters(self) -> None:
        for weight, bias in zip(self.weight, self.bias):
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            if bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(bias, -bound, bound)

    # def reset_parameters(self):
    #     gain = nn.init.calculate_gain("relu")
    #     for weight in self.weight:
    #         nn.init.xavier_uniform_(weight, gain=gain)
    #     if self.bias is not None:
    #         nn.init.zeros_(self.bias)

    def forward(self, x):
        # input size: [N, d_in] or [H, N, d_in]
        # output size: [H, N, d_out]
        if len(x.shape) == 3:
            x = x.transpose(0, 1)

        x = torch.matmul(x, self.weight)
        if self.bias is not None:
            x += self.bias
        return x.transpose(0, 1)


eps = 1e-5


class MultiHeadBatchNorm(nn.Module):
    def __init__(self, n_heads, in_feats, momentum=0.1, affine=True, device=None,
                 dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        assert in_feats % n_heads == 0
        self._in_feats = in_feats
        self._n_heads = n_heads
        self._momentum = momentum
        self._affine = affine
        if affine:
            self.weight = nn.Parameter(torch.empty(
                size=(n_heads, in_feats // n_heads)))
            self.bias = nn.Parameter(torch.empty(
                size=(n_heads, in_feats // n_heads)))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.register_buffer("running_mean", torch.zeros(
            size=(n_heads, in_feats // n_heads)))
        self.register_buffer("running_var", torch.ones(
            size=(n_heads, in_feats // n_heads)))
        self.running_mean: Optional[torch.Tensor]
        self.running_var: Optional[torch.Tensor]
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()  # type: ignore[union-attr]
        self.running_var.fill_(1)  # type: ignore[union-attr]
        if self._affine:
            nn.init.zeros_(self.bias)
            for weight in self.weight:
                nn.init.ones_(weight)

    def forward(self, x):
        assert x.shape[1] == self._in_feats
        x = x.view(-1, self._n_heads, self._in_feats // self._n_heads)

        self.running_mean = self.running_mean.to(x.device)
        self.running_var = self.running_var.to(x.device)
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (
                self.running_var is None)
        if bn_training:
            mean = x.mean(dim=0, keepdim=True)
            var = x.var(dim=0, unbiased=False, keepdim=True)
            out = (x-mean) * torch.rsqrt(var + eps)
            self.running_mean = (1 - self._momentum) * \
                self.running_mean + self._momentum * mean.detach()
            self.running_var = (1 - self._momentum) * \
                self.running_var + self._momentum * var.detach()
        else:
            out = (x - self.running_mean) * torch.rsqrt(self.running_var + eps)
        if self._affine:
            out = out * self.weight + self.bias
        return out


class GroupMLP(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, n_heads, n_layers, dropout, input_drop=0., residual=False, normalization="batch"):
        super(GroupMLP, self).__init__()
        self._residual = residual
        self.adj = None
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self._n_heads = n_heads
        self._n_layers = n_layers

        self.input_drop = nn.Dropout(input_drop)

        if self._n_layers == 1:
            self.layers.append(MultiHeadLinear(in_feats, out_feats, n_heads))
        else:
            self.layers.append(MultiHeadLinear(in_feats, hidden, n_heads))
            if normalization == "batch":
                self.norms.append(MultiHeadBatchNorm(
                    n_heads, hidden * n_heads))
                # self.norms.append(nn.BatchNorm1d(hidden * n_heads))
            if normalization == "layer":
                self.norms.append(nn.GroupNorm(n_heads, hidden * n_heads))
            if normalization == "none":
                self.norms.append(nn.Identity())
            for i in range(self._n_layers - 2):
                self.layers.append(MultiHeadLinear(hidden, hidden, n_heads))
                if normalization == "batch":
                    self.norms.append(MultiHeadBatchNorm(
                        n_heads, hidden * n_heads))
                    # self.norms.append(nn.BatchNorm1d(hidden * n_heads))
                if normalization == "layer":
                    self.norms.append(nn.GroupNorm(n_heads, hidden * n_heads))
                if normalization == "none":
                    self.norms.append(nn.Identity())
            self.layers.append(MultiHeadLinear(hidden, out_feats, n_heads))
        if self._n_layers > 1:
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)

        for head in range(self._n_heads):

            for layer in self.layers:

                nn.init.kaiming_uniform_(layer.weight[head], a=math.sqrt(5))
                if layer.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                        layer.weight[head])
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(layer.bias[head], -bound, bound)
        self.reset_parameters()

    def reset_parameters(self):

        gain = nn.init.calculate_gain("relu")

        for head in range(self._n_heads):
            for layer in self.layers:
                nn.init.xavier_uniform_(layer.weight[head], gain=gain)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias[head])
        for norm in self.norms:
            norm.reset_parameters()
            # for norm in self.norms:
            #     norm.moving_mean[head].zero_()
            #     norm.moving_var[head].fill_(1)
            #     if norm._affine:
            #         nn.init.ones_(norm.scale[head])
            #         nn.init.zeros_(norm.offset[head])
        # print(self.layers[0].weight[0])

    def forward(self, x):
        x = self.input_drop(x)
        if len(x.shape) == 2:
            x = x.view(-1, 1, x.shape[1])
        if self._residual:
            prev_x = x
        for layer_id, layer in enumerate(self.layers):
            x = layer(x)

            if layer_id < self._n_layers - 1:
                shape = x.shape
                x = x.flatten(1, -1)
                x = self.dropout(self.relu(self.norms[layer_id](x)))
                x = x.reshape(shape=shape)

            if self._residual:
                if x.shape[2] == prev_x.shape[2]:
                    x += prev_x
                prev_x = x

        x = x.squeeze(dim=1)
        return x


class Dense(nn.Module):
    def __init__(self, in_features, out_features, bias='bn'):
        super(Dense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.FloatTensor(in_features, out_features))
        if bias == 'bn':
            self.bias = nn.BatchNorm1d(out_features)
        else:
            self.bias = lambda x: x
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        output = torch.mm(input, self.weight)
        output = self.bias(output)
        if self.in_features == self.out_features:
            output = output + input
        return output


# MLP apply initial residual
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, alpha, bns=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(
            self.in_features, self.out_features))
        self.alpha = alpha
        self.reset_parameters()
        self.bns = bns
        self.bias = nn.BatchNorm1d(out_features)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, h0):
        support = (1-self.alpha)*input+self.alpha*h0
        output = torch.mm(support, self.weight)
        # if self.bns:
        output = self.bias(output)
        if self.in_features == self.out_features:
            output = output+input
        return output


# adapted from dgl sign
class FeedForwardNet(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, n_layers, dropout, bns=True):
        super(FeedForwardNet, self).__init__()
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.adj = None
        self.n_layers = n_layers
        if n_layers == 1:
            self.layers.append(nn.Linear(in_feats, out_feats))
        else:
            self.layers.append(nn.Linear(in_feats, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
            for i in range(n_layers - 2):
                self.layers.append(nn.Linear(hidden, hidden))
                self.bns.append(nn.BatchNorm1d(hidden))
            self.layers.append(nn.Linear(hidden, out_feats))
        if self.n_layers > 1:
            self.prelu = nn.PReLU()
            self.dropout = nn.Dropout(dropout)
        self.norm = bns
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer_id, layer in enumerate(self.layers):
            x = layer(x)
            if layer_id < self.n_layers - 1:
                if self.norm:
                    x = self.dropout(self.prelu(self.bns[layer_id](x)))
                else:
                    x = self.dropout(self.prelu(x))
        return x


class FeedForwardNetII(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, n_layers, dropout, alpha, bns=False):
        super(FeedForwardNetII, self).__init__()
        self.layers = nn.ModuleList()
        self.adj = None
        self.n_layers = n_layers
        self.in_feats = in_feats
        self.hidden = hidden
        self.out_feats = out_feats
        if n_layers == 1:
            self.layers.append(Dense(in_feats, out_feats))
        else:
            self.layers.append(Dense(in_feats, hidden))
            for i in range(n_layers - 2):
                self.layers.append(GraphConvolution(
                    hidden, hidden, alpha, bns))
            self.layers.append(Dense(hidden, out_feats))

        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x):
        x = self.layers[0](x)
        h0 = x
        for layer_id, layer in enumerate(self.layers):
            if layer_id == 0:
                continue
            elif layer_id == self.n_layers - 1:
                x = self.dropout(self.prelu(x))
                x = layer(x)
            else:
                x = self.dropout(self.prelu(x))
                x = layer(x, h0)
                #x = self.dropout(self.prelu(x))
        return x
