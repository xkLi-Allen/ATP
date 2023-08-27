import torch
import torch.nn.functional as F

from torch import nn
from torch.nn import Parameter, Linear
from operators.base_operator import MessageOp
from operators.utils import one_dim_weighted_add, two_dim_weighted_add
from models.base_scalable.simple_models import MultiLayerPerceptron


class LearnableWeightedMessageOp(MessageOp):

    # 'simple' needs one additional parameter 'prop_steps';
    # 'simple_weighted' allows negative weights, all else being the same as 'simple';
    # 'gate' needs one additional parameter 'feat_dim';
    # 'ori_ref' needs one additional parameter 'feat_dim';
    # 'jk' needs two additional parameter 'prop_steps' and 'feat_dim'
    def __init__(self, start, end, combination_type, *args):
        super(LearnableWeightedMessageOp, self).__init__(start, end)
        self.aggr_type = "learnable_weighted"

        if combination_type not in ["simple", "simple_allow_neg", "gate", "ori_ref", "jk", "sagn"]:
            raise ValueError(
                "Invalid weighted combination type! Type must be 'simple', 'simple_allow_neg', 'gate', 'ori_ref' or 'jk'.")
        self.combination_type = combination_type
        self.input_dropout = nn.Dropout(0.2)
        self.learnable_weight = None
        if combination_type == "simple" or combination_type == "simple_allow_neg":
            if len(args) != 1:
                raise ValueError(
                    "Invalid parameter numbers for the simple learnable weighted aggregator!")
            prop_steps = args[0]
            # a 2d tensor is required to use xavier_uniform_.
            tmp_2d_tensor = torch.FloatTensor(1, prop_steps + 1)
            nn.init.xavier_normal_(tmp_2d_tensor)
            self.learnable_weight = Parameter(tmp_2d_tensor.view(-1))

        elif combination_type == "gate":
            if len(args) != 1:
                raise ValueError(
                    "Invalid parameter numbers for the gate learnable weighted aggregator!")
            feat_dim = args[0]
            self.learnable_weight = Linear(feat_dim, 1)

        elif combination_type == "ori_ref":
            if len(args) != 1:
                raise ValueError(
                    "Invalid parameter numbers for the ori_ref learnable weighted aggregator!")
            feat_dim = args[0]
            self.learnable_weight = Linear(feat_dim + feat_dim, 1)

        elif combination_type == "jk":
            if len(args) != 2:
                raise ValueError(
                    "Invalid parameter numbers for the jk learnable weighted aggregator!")
            prop_steps, feat_dim = args[0], args[1]
            self.learnable_weight = Linear(
                feat_dim + (prop_steps + 1) * feat_dim, 1)

        elif combination_type == "sagn":
            if len(args) != 3:
                raise ValueError(
                    "Invalid parameter numbers for the sagn learnable weighted aggregator!")
            prop_steps,  feat_dim, negative_slope = args[0], args[1], args[2]
            self.__act = nn.LeakyReLU(negative_slope)
            self.__learnable_weight = nn.Parameter(
                torch.FloatTensor(size=(1, 1, feat_dim)))
            self.hop_attn_l = nn.Parameter(
                torch.FloatTensor(size=(1, 1, feat_dim)))
            self.hop_attn_r = nn.Parameter(
                torch.FloatTensor(size=(1, 1, feat_dim)))
            self.res_fc = nn.Linear(feat_dim, feat_dim, bias=False)
            self.multihop_encoders = nn.ModuleList([MultiLayerPerceptron(
                feat_dim, feat_dim, 3, feat_dim, dropout=0.4)for i in range(prop_steps+1)])
            self.attn_dropout = nn.Dropout(0.4)

    def combine(self, feat_list):
        weight_list = None
        weighted_feat = None
        if self.combination_type == "simple":
            weight_list = F.softmax(torch.sigmoid(
                self.learnable_weight[self.start:self.end]), dim=0)

        elif self.combination_type == "simple_allow_neg":
            weight_list = self.learnable_weight[self.start:self.end]

        elif self.combination_type == "gate":
            adopted_feat_list = torch.vstack(feat_list[self.start:self.end])
            weight_list = F.softmax(
                torch.sigmoid(self.learnable_weight(adopted_feat_list).view(self.end - self.start, -1).T), dim=1)

        elif self.combination_type == "ori_ref":
            reference_feat = feat_list[0].repeat(self.end - self.start, 1)
            adopted_feat_list = torch.hstack(
                (reference_feat, torch.vstack(feat_list[self.start:self.end])))
            weight_list = F.softmax(
                torch.sigmoid(self.learnable_weight(adopted_feat_list).view(-1, self.end - self.start)), dim=1)

        elif self.combination_type == "jk":
            reference_feat = torch.hstack(feat_list).repeat(
                self.end - self.start, 1)
            adopted_feat_list = torch.hstack(
                (reference_feat, torch.vstack(feat_list[self.start:self.end])))
            weight_list = F.softmax(
                torch.sigmoid(self.learnable_weight(adopted_feat_list).view(-1, self.end - self.start)), dim=1)

        elif self.combination_type == "sagn":
            out = 0
            hidden = []
            for i in range(len(feat_list)):
                hidden.append(self.multihop_encoders[i](
                    self.input_dropout(feat_list[i])).view(-1, 1, feat_list[i].size()[1]))

            # simple sagn
            # astack = [(feat * self.__learnable_weight).sum(-1)
            #           for feat in hidden]
            # astack = torch.stack([a for a in astack], dim=-1)

            # complete sagn
            # focal_feat = 0
            # for h in hidden:
            #     focal_feat += h
            # focal_feat /= len(hidden)

            focal_feat = hidden[0]

            astack_l = [(h * self.hop_attn_l).sum(dim=-1).unsqueeze(-1)
                        for h in hidden]
            a_r = (focal_feat * self.hop_attn_r).sum(dim=-1).unsqueeze(-1)
            astack = torch.stack([(a_l + a_r) for a_l in astack_l], dim=-1)

            a = self.__act(astack)
            a = F.softmax(a, dim=-1)
            a = a.squeeze(-2)
            a = self.attn_dropout(a)
            for i in range(a.shape[-1]):
                out += hidden[i] * a[:, :, [i]]
                # out += hidden[i] * a[:, :, :, [i]]

            out += self.res_fc(feat_list[0]).view(-1,
                                                  1, feat_list[i].size()[1])

            weighted_feat = out.flatten(1, -1)

        else:
            raise NotImplementedError

        if self.combination_type == "simple" or self.combination_type == "simple_allow_neg":
            weighted_feat = one_dim_weighted_add(
                feat_list[self.start:self.end], weight_list=weight_list)
        elif self.combination_type in ["gate", "ori_ref", "jk"]:
            weighted_feat = two_dim_weighted_add(
                feat_list[self.start:self.end], weight_list=weight_list)
        elif self.combination_type in ["sagn"]:
            pass
        else:
            raise NotImplementedError

        return weighted_feat
