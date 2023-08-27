import argparse


def add_model_config(parser):
    # parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', help='gnn model',
                        type=str, default="sgc")
    parser.add_argument(
        '--num_layers', help='number of gnn layers', type=int, default=2)
    parser.add_argument('--dropout', help='drop out of gnn model',
                        type=float, default=0.5)
    parser.add_argument(
        '--hidden_dim', help='hidden units of gnn model', type=int, default=256)
    # scalable gnn model
    parser.add_argument('--prop_steps', help='prop steps',
                        type=int, default=3)
    # adj normalize
    parser.add_argument('--r', help='symmetric normalized unit',
                        type=float, default=0.5)
    parser.add_argument(
        '--q', help='the imaginary part of the complex unit', type=float, default=0)

    parser.add_argument(
        '--incep_dim', type=int, default=512)

    parser.add_argument('--a', help='the ratio of degree',
                        type=float, default=0.5)
    parser.add_argument('--b', help='the ratio of cluster',
                        type=float, default=0.5)
    parser.add_argument('--c', help='the ratio of Engienvector',
                        type=float, default=0.5)

    parser.add_argument(
        '--use_label', type=bool, default=False)

    parser.add_argument(
        '--m_r', type=bool, default=True)
    parser.add_argument(
        '--m_p', type=bool, default=False)

    parser.add_argument(
        '--r_way', type=str, default="together")

    # model_args = parser.parse_args()
