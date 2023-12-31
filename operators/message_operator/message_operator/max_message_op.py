import torch

from operators.base_operator import MessageOp


class MaxMessageOp(MessageOp):
    def __init__(self, start, end):
        super(MaxMessageOp, self).__init__(start, end)
        self.aggr_type = "max"

    def combine(self, feat_list):
        return torch.stack(feat_list[self.start:self.end], dim=0).max(dim=0)[0]
