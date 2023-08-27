

from models.base_scalable.base_model import BaseSGModel
from models.base_scalable.simple_models import MultiLayerPerceptron
from operators.graph_operator.symmetrical_simgraph_laplacian_operator import SymLaplacianGraphOp
from operators.message_operator.message_operator.simple_weighted_message_op import SimpleWeightedMessageOp
from operators.graph_operator.symmetrical_simgraph_ppr_operator import PprGraphOp


class GBP(BaseSGModel):
    def __init__(self, prop_steps, feat_dim, output_dim, hidden_dim, num_layers, r=0.5, alpha=0.85):
        super(GBP, self).__init__(prop_steps, feat_dim, output_dim)

        # self.pre_graph_op = SymSimLaplacianGraphOp(prop_steps, r=r)
        self.pre_graph_op = PprGraphOp(prop_steps, r, alpha=0.10)
        self.pre_msg_op = SimpleWeightedMessageOp(
            0, prop_steps + 1, "alpha", alpha)
        self.base_model = MultiLayerPerceptron(
            feat_dim, hidden_dim, num_layers, output_dim, dropout=0.5)
