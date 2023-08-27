

from models.base_scalable.base_model import BaseSGModel
from models.base_scalable.simple_models import GroupMLP, MultiLayerPerceptron
from operators.graph_operator.symmetrical_simgraph_laplacian_operator import SymLaplacianGraphOp
from operators.message_operator.message_operator.learnable_weighted_messahe_op import LearnableWeightedMessageOp
from operators.message_operator.message_operator.last_message_op import LastMessageOp
from operators.message_operator.message_operator.mean_message_op import MeanMessageOp


class SAGN(BaseSGModel):
    def __init__(self, prop_steps, feat_dim, output_dim, hidden_dim, num_layers, r=0.5):
        super(SAGN, self).__init__(prop_steps, feat_dim, output_dim)

        self.pre_graph_op = SymLaplacianGraphOp(prop_steps, r=r)
        self.pre_msg_op = LearnableWeightedMessageOp(
            0, prop_steps + 1, "sagn", prop_steps, feat_dim, 0.0)
        # self.base_model = SimMultiLayerPerceptron(
        #     feat_dim, hidden_dim, num_layers, output_dim, dropout=0.4)
        self.base_model = GroupMLP(
            feat_dim, hidden_dim, output_dim, 1, num_layers, dropout=0.7, input_drop=0.0)

        self.post_graph_op = SymLaplacianGraphOp(prop_steps, r=r)

        self.post_msg_op = LastMessageOp()
        # self.post_msg_op = SimMeanMessageOp(0, prop_steps+1)
