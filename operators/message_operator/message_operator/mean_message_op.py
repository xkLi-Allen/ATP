from operators.base_operator import MessageOp


class MeanMessageOp(MessageOp):
    def __init__(self, start, end):
        super(MeanMessageOp, self).__init__(start, end)
        self.aggr_type = "mean"

    def combine(self, feat_list):
        return sum(feat_list[self.start:self.end]) / (self.end - self.start)
