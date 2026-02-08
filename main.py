from config import Config
import argparse
from pretrain import Castle
from finetune_SL import Finetune


def parse_args():
    # Set some key configs from screen

    parser = argparse.ArgumentParser(description="main.py")

    parser.add_argument("-dataset_name",  # ["porto", "geolife", "tdrive", "aisus"]
                        help="Name of dataset")

    parser.add_argument("-distance_type",  # ["TP", "LCRS", "NetERP"]
                        help="metric")

    parser.add_argument("-segment_embedding_type", default="node2vec")
    parser.add_argument("-edge_dim", type=int)
    parser.add_argument("-loss", default="pos-out-single")

    parser.add_argument("-num_companion", type=int)
    parser.add_argument("-SLloss", default="difficulty")
    parser.add_argument("-epochs", type=int, default=10)
    parser.add_argument("-finetune_epochs", type=int, default=50)
    parser.add_argument("-finetune_batch_size", type=int, default=128)
    parser.add_argument("-hidden_size", type=int, default=128)
    parser.add_argument("-embed_size", type=int, default=128)
    parser.add_argument("-output_size", type=int, default=128)
    parser.add_argument("-num_layers", type=int, default=1)
    parser.add_argument("-training_size", type=int, default=10000)

    args = parser.parse_args()

    params = {}
    for param, value in args._get_kwargs():
        if value is not None:
            params[param] = value

    return params


if __name__ == '__main__':
    configs = Config()
    configs.default_update(parse_args())

    task = Finetune(configs)
    task.train()