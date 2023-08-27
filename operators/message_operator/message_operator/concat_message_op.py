import torch
from operators.base_operator import MessageOp

class ConcatMessageOp(MessageOp):
    def __init__(self, start, end):
        super(ConcatMessageOp, self).__init__(start, end)
        self.aggr_type = "concat"

    def combine(self, feat_list):
        return torch.hstack(feat_list[self.start:self.end])
