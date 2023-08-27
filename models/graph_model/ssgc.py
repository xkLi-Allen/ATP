

from models.base_scalable.simple_models import LogisticRegression, MultiLayerPerceptron
from operators.graph_operator.symmetrical_simgraph_laplacian_operator import SymLaplacianGraphOp
from operators.message_operator.message_operator.mean_message_op import MeanMessageOp
from models.base_scalable.base_model import BaseSGModel
from operators.graph_operator.symmetrical_simgraph_ppr_operator import PprGraphOp
from operators.message_operator.message_operator.last_message_op import LastMessageOp


class SSGC(BaseSGModel):
    def __init__(self, prop_steps, feat_dim,hidden_dim, output_dim, num_layers, dropout ,r=0.5):
        super(SSGC, self).__init__(prop_steps, feat_dim, output_dim)

        # self.pre_graph_op = SymSimLaplacianGraphOp(prop_steps, r=r)
        self.pre_graph_op = PprGraphOp(prop_steps, r, alpha=0.05)
        self.pre_msg_op = MeanMessageOp(start=0, end=prop_steps + 1)

        # self.base_model = SimLogisticRegression(feat_dim, output_dim)
        self.base_model = MultiLayerPerceptron(
            feat_dim=feat_dim, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=output_dim, dropout=dropout)
        self.post_graph_op = SymLaplacianGraphOp(prop_steps, r=r)
        self.post_msg_op = MeanMessageOp(start=0, end=prop_steps + 1)
