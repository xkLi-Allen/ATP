import argparse


def add_data_config(parser):
    # parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_name', help='unsigned & undirected & unweighted data name', type=str, default="cora")
    parser.add_argument('--root', help='unsigned & undirected & unweighted data root',
                        type=str, default="dataset/homo_data/")
    parser.add_argument(
        '--split', help='unsigned & undirected & unweighted simplex homogeneous data split method', type=str, default="official")


    # data_args = parser.parse_args()
