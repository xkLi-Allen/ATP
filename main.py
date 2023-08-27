import os
import time
import torch
import datetime
import argparse
from models.model_init import load_model
from configs.data_config import add_data_config

from tasks.node_classification_with_label_use import NodeClassificationWithLabelUse
from utils import seed_everything, get_params
from configs.training_config import add_training_config
from configs.model_config import add_model_config
from tasks.node_classification import NodeClassification
from GraphM import GraphData


if __name__ == "__main__":
    print(f"program start: {datetime.datetime.now()}")
    parser = argparse.ArgumentParser(add_help=False)
    add_data_config(parser)
    add_model_config(parser)
    add_training_config(parser)
    args = parser.parse_args()
    run_id = f"lr={args.lr}, q={args.q}"

    # set up seed
    seed_everything(args.seed)
    device = torch.device('cuda:{}'.format(args.gpu_id) if (
        args.use_cuda and torch.cuda.is_available()) else 'cpu')
    print(device)
    # set up dataset
    print("Load network")
    set_up_datasets_start_time = time.time()

    Data = GraphData(root=args.root, name=args.data_name, args=args)
    # Data.drop_nodes(ratio=0.10)
    if args.r_way == "cluster":
        Data.mining(method="clustering_coefficients")
        # Data.mining(method="pagerank")
    if args.r_way == "degree":
        Data.mining(method="degree_centrality")

    if args.r_way == "Engienvector_centrality":
        Data.mining(method="Engienvector_centrality")

    if args.r_way == "together":
        Data.mining(method="together", a=args.a, b=args.b, c=args.c)

    dataset = Data.dataset

    set_up_datasets_end_time = time.time()
    print(f"datasets: {args.data_name}, root dir: {args.root}, node-level split method: {args.split}, the running time is: {round(set_up_datasets_end_time-set_up_datasets_start_time,4)}s")

    if args.m_r == True:
        print("modify r")
        model = load_model(feat_dim=dataset.num_features,
                           output_dim=dataset.num_classes, args=args, r=dataset.mining_list)
    else:
        model = load_model(feat_dim=dataset.num_features,
                           output_dim=dataset.num_classes, args=args)

    print("# Params:", get_params(model))
    if args.use_label == False:

        # NodeClassification(args, dataset, model, normalize_times=args.normalize_times, lr=args.lr,
        #                    weight_decay=args.weight_decay, epochs=args.num_epochs, device=device)
        NodeClassification(args, dataset, model, normalize_times=args.normalize_times, lr=args.lr,
                           weight_decay=args.weight_decay, epochs=args.num_epochs, device=device,
                           train_batch_size=args.train_batch_size, eval_batch_size=args.eval_batch_size)

    if args.use_label == True:
        # NodeClassificationWithLabelUse(
        #     args, dataset, model, args.lr, args.weight_decay, args.num_epochs, device=device, normalize_times=args.normalize_times, seed=args.seed)
        # NodeClassificationWithLabelUse(
        #     args,  dataset, model, args.lr, args.weight_decay, args.num_epochs,  device, args.normalize_times, seed=args.seed, reuse_start_epoch=200, label_iters=10)

        NodeClassificationWithLabelUse(args, dataset, model, args.lr, args.weight_decay, args.num_epochs,
                                       device,  normalize_times=args.normalize_times, seed=args.seed, train_batch_size=args.train_batch_size, eval_batch_size=args.eval_batch_size)

        # NodeClassificationWithLabelUse(args, dataset, model, args.lr, args.weight_decay, args.num_epochs,
        #                                device,  normalize_times=args.normalize_times, seed=args.seed, train_batch_size=args.train_batch_size, eval_batch_size=args.eval_batch_size, reuse_start_epoch=200, label_iters=5, label_reuse_batch_size=100000)

    print("# Params:", get_params(model))
    exit(0)
