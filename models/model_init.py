from models.graph_model.sgc import SGC
from models.graph_model.gamlp import GAMLP
from models.graph_model.gcn import GCN
from models.graph_model.sign import SIGN
from models.graph_model.gbp import GBP
from models.graph_model.ssgc import SSGC
from models.graph_model.sagn import SAGN



def load_model(feat_dim, output_dim, args, r=0.5):

    if args.model_name == "sgc":
        print(
            f"model: {args.model_name}, prop_steps: {args.prop_steps}, r: {r}, lr: {args.lr}")
        model = SGC(prop_steps=args.prop_steps, r=r,
                    feat_dim=feat_dim, output_dim=output_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout=args.dropout)

    elif args.model_name == "gamlp":
        print(f"model: {args.model_name}, prop_steps: {args.prop_steps}, r: {r}, hidden_dim: {args.hidden_dim}, num_layers: {args.num_layers}, dropout: {args.dropout}")
        model = GAMLP(prop_steps=args.prop_steps, r=r,
                      feat_dim=feat_dim+output_dim, output_dim=output_dim,
                      hidden_dim=args.hidden_dim,
                      num_layers=args.num_layers, dropout=args.dropout)

    elif args.model_name == "gcn":
        print(f"model: {args.model_name}, r: {r}, hidden_dim: {args.hidden_dim}, num_layers: {args.num_layers}, dropout: {args.dropout}")
        model = GCN(r=r, feat_dim=feat_dim, hidden_dim=args.hidden_dim,
                    output_dim=output_dim, dropout=args.dropout)

    elif args.model_name == "sign":
        print(f"model: {args.model_name}, r: {r}, hidden_dim: {args.hidden_dim}, num_layers: {args.num_layers}, dropout: {args.dropout}")
        model = SIGN(prop_steps=args.prop_steps, r=r, feat_dim=feat_dim, incep_dim=args.incep_dim, hidden_dim=args.hidden_dim,
                     output_dim=output_dim, num_layers=args.num_layers, dropout=args.dropout)

    elif args.model_name == "gbp":
        print(f"model: {args.model_name}, r: {r}, prop_steps: {args.prop_steps}, hidden_dim: {args.hidden_dim}, num_layers: {args.num_layers}, dropout: {args.dropout}")
        model = GBP(prop_steps=args.prop_steps, r=r, feat_dim=feat_dim, hidden_dim=args.hidden_dim,
                    output_dim=output_dim, num_layers=args.num_layers)

    elif args.model_name == "ssgc":
        print(
            f"model: {args.model_name}, prop_steps: {args.prop_steps}, r: {r}")
        model = SSGC(prop_steps=args.prop_steps, r=r,
                     feat_dim=feat_dim, hidden_dim=args.hidden_dim, output_dim=output_dim, num_layers=args.num_layers, dropout=args.dropout)

    elif args.model_name == "sagn":
        print(f"model: {args.model_name}, r: {r}, hidden_dim: {args.hidden_dim}, num_layers: {args.num_layers}, dropout: {args.dropout}")
        model = SAGN(prop_steps=args.prop_steps, r=r, feat_dim=feat_dim + output_dim, hidden_dim=args.hidden_dim,
                     output_dim=output_dim, num_layers=args.num_layers)



        return NotImplementedError
    return model

