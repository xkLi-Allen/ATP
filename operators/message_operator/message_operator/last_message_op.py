from operators.base_operator import MessageOp


class LastMessageOp(MessageOp):
    def __init__(self):
        super(LastMessageOp, self).__init__()
        self.aggr_type = "last"

    def combine(self, feat_list):
        return feat_list[-1]
