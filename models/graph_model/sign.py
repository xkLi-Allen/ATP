
from models.base_scalable.base_model import BaseSGModel
from models.base_scalable.simple_models import MultiLayerPerceptron
from operators.graph_operator.symmetrical_simgraph_laplacian_operator import SymLaplacianGraphOp
from operators.message_operator.message_operator.concat_message_op import ConcatMessageOp
from operators.message_operator.message_operator.projected_concat_message_op import ProjectedConcatMessageOp
from operators.graph_operator.symmetrical_simgraph_ppr_operator import PprGraphOp
from operators.message_operator.message_operator.last_message_op import LastMessageOp


class SIGN(BaseSGModel):
    def __init__(self, prop_steps, r, feat_dim, incep_dim, output_dim, hidden_dim, num_layers, dropout):
        super(SIGN, self).__init__(prop_steps, feat_dim, output_dim)

        # self.pre_graph_op = SymSimLaplacianGraphOp(prop_steps, r=r)
        self.pre_graph_op = PprGraphOp(prop_steps, r, alpha=0.1)
        self.pre_msg_op = ProjectedConcatMessageOp(
            0, prop_steps + 1, feat_dim, incep_dim, num_layers)
        # self.pre_msg_op = SimConcatMessageOp(0, prop_steps + 1)
        # self.base_model = SimMultiLayerPerceptron(
        #     (prop_steps + 1) * feat_dim, hidden_dim, num_layers, output_dim, dropout=dropout)
        self.base_model = MultiLayerPerceptron(
            (prop_steps + 1) * incep_dim, hidden_dim, num_layers, output_dim, dropout=dropout)

        self.post_graph_op = PprGraphOp(prop_steps, r, alpha=0.1)

        self.post_msg_op = LastMessageOp()
