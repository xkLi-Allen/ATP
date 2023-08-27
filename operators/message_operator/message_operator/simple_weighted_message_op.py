import torch

from torch import Tensor
from operators.base_operator import MessageOp
from operators.utils  import one_dim_weighted_add


class SimpleWeightedMessageOp(MessageOp):

    # 'alpha' needs one additional parameter 'alpha';
    # 'hand_crafted' needs one additional parameter 'weight_list'
    def __init__(self, start, end, combination_type, *args):
        super(SimpleWeightedMessageOp, self).__init__(start, end)
        self.aggr_type = "simple_weighted"

        if combination_type not in ["alpha", "hand_crafted"]:
            raise ValueError(
                "Invalid weighted combination type! Type must be 'alpha' or 'hand_crafted'.")
        self.combination_type = combination_type

        if len(args) != 1:
            raise ValueError(
                "Invalid parameter numbers for the simple weighted aggregator!")
        self.alpha, self.weight_list = None, None
        if combination_type == "alpha":
            self.alpha = args[0]
            if not isinstance(self.alpha, float):
                raise TypeError("The alpha must be a float!")
            elif self.alpha > 1 or self.alpha < 0:
                raise ValueError("The alpha must be a float in [0,1]!")

        elif combination_type == "hand_crafted":
            self.weight_list = args[0]
            if isinstance(self.weight_list, list):
                self.weight_list = torch.FloatTensor(self.weight_list)
            elif not isinstance(self.weight_list, (list, Tensor)):
                raise TypeError(
                    "The input weight list must be a list or a tensor!")

    def combine(self, feat_list):
        if self.combination_type == "alpha":
            self.weight_list = [self.alpha]
            for _ in range(len(feat_list) - 1):
                self.weight_list.append(
                    (1 - self.alpha) * self.weight_list[-1])
            self.weight_list = torch.FloatTensor(
                self.weight_list[self.start:self.end])

        elif self.combination_type == "hand_crafted":
            pass
        else:
            raise NotImplementedError

        weighted_feat = one_dim_weighted_add(
            feat_list[self.start:self.end], weight_list=self.weight_list)
        return weighted_feat
