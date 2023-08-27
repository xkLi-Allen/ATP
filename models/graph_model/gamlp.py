from models.base_scalable.base_model import BaseSGModel
from models.base_scalable.simple_models import FeedForwardNetII, MultiLayerPerceptron
from operators.graph_operator.symmetrical_simgraph_laplacian_operator import SymLaplacianGraphOp
from operators.message_operator.message_operator.learnable_weighted_messahe_op import LearnableWeightedMessageOp
from operators.message_operator.message_operator.iterate_learnable_weighted_message_op import IterateLearnableWeightedMessageOp
from operators.graph_operator.symmetrical_simgraph_ppr_operator import PprGraphOp
from operators.message_operator.message_operator.last_message_op import LastMessageOp


class GAMLP(BaseSGModel):
    def __init__(self, prop_steps, r, feat_dim, output_dim, hidden_dim, num_layers, dropout):
        super(GAMLP, self).__init__(prop_steps, feat_dim, output_dim)

        # self.pre_graph_op = SymSimLaplacianGraphOp(prop_steps, r=r)
        self.pre_graph_op = PprGraphOp(prop_steps, r, alpha=0.1)
        # self.pre_msg_op = SimLearnableWeightedMessageOp(
        #     0, prop_steps + 1, "jk", prop_steps, feat_dim)
        self.pre_msg_op = IterateLearnableWeightedMessageOp(
            0, prop_steps, "recursive", feat_dim)
        self.base_model = MultiLayerPerceptron(
            feat_dim, hidden_dim, num_layers, output_dim, dropout)
        # self.base_model = FeedForwardNetII(
        #         feat_dim, hidden_dim, output_dim, num_layers, dropout, alpha = 0.5,bns = True)

        self.post_graph_op = PprGraphOp(prop_steps, r, alpha=0.1)
        self.post_msg_op = LastMessageOp()
    