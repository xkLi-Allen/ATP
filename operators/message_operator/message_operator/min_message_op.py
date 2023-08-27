import torch

from operators.base_operator import MessageOp


class MinMessageOp(MessageOp):
    def __init__(self, start, end):
        super(MinMessageOp, self).__init__(start, end)
        self.aggr_type = "min"

    def combine(self, feat_list):
        return torch.stack(feat_list[self.start:self.end], dim=0).min(dim=0)[0]
