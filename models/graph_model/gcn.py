from models.base_scalable.base_model import BaseSGModel
from models.base_scalable.simple_models import TwoLayerGraphConvolution
from operators.graph_operator.symmetrical_simgraph_laplacian_operator import SymLaplacianGraphOp

class GCN(BaseSGModel):
    def __init__(self, r, feat_dim, hidden_dim, output_dim, dropout):
        super(GCN, self).__init__(prop_steps=None, feat_dim=feat_dim, output_dim=output_dim)
        self.naive_graph_op = SymLaplacianGraphOp(prop_steps=None, r=r)
        self.base_model = TwoLayerGraphConvolution(feat_dim=feat_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout)