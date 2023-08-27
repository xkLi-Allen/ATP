from operators.base_operator import MessageOp


class SumMessageOp(MessageOp):
    def __init__(self, start, end):
        super(SumMessageOp, self).__init__(start, end)
        self.aggr_type = "sum"

    def combine(self, feat_list):
        return sum(feat_list[self.start:self.end])
