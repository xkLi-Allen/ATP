from models.base_scalable.base_model import BaseSGModel
from models.base_scalable.simple_models import FeedForwardNet, LogisticRegression, MultiLayerPerceptron
from operators.graph_operator.symmetrical_simgraph_laplacian_operator import SymLaplacianGraphOp
from operators.message_operator.message_operator.last_message_op import LastMessageOp
from operators.graph_operator.symmetrical_simgraph_ppr_operator import PprGraphOp
from operators.message_operator.message_operator.mean_message_op import MeanMessageOp


class SGC(BaseSGModel):
    def __init__(self, prop_steps, r, feat_dim, output_dim, hidden_dim, num_layers, dropout):
        super(SGC, self).__init__(prop_steps, feat_dim, output_dim)
        self.pre_graph_op = SymLaplacianGraphOp(prop_steps, r=r)
        # self.pre_graph_op = PprGraphOp(prop_steps, r, alpha=0.10)
        # self.pre_msg_op = LastMessageOp()
        self.pre_msg_op = MeanMessageOp(start=0, end=prop_steps + 1)
        self.base_model = LogisticRegression(feat_dim, output_dim)
        # self.base_model = SimMultiLayerPerceptron(
        #     feat_dim, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=output_dim, dropout=dropout)
        # self.base_model = FeedForwardNet(
        #     in_feats=feat_dim, hidden=hidden_dim, out_feats=output_dim, n_layers=num_layers, dropout=dropout)
        self.post_graph_op = SymLaplacianGraphOp(prop_steps, r=r)
        self.post_msg_op = LastMessageOp()
