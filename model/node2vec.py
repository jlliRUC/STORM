import sys
sys.path.append("../")
from config import Config
import torch
import numpy as np
import pandas as pd
from torch_geometric.nn import Node2Vec
import random
import pickle
random.seed(1953)


def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def train_epoch(model, loader, optimizer, device):
    # Training with epoch iteration
    last_loss = 1
    print("Training node embedding with node2vec...")
    for i in range(100):
        loss = train(model, loader, optimizer, device)
        print('Epoch: {0} \tLoss: {1:.4f}'.format(i, loss))
        if abs(last_loss - loss) < 1e-5:
            break
        else:
            last_loss = loss

@torch.no_grad()
def save_embeddings(model, num_nodes, dataset, device):
    model.eval()
    node_features = model(torch.arange(num_nodes, device=device)).cpu().continuous()
    with open(f"../data/{dataset}/{dataset}_node2vec_rs_embeddings.pkl", "rb") as f:
        pickle.dump(node_features, f)


if __name__ == "__main__":
    configs = Config()
    configs.dataset_update({"dataset_name": "porto"})
    configs.config_device()

    edge_index = pickle.load(open(f"../data/{configs.dataset_name}/{configs.dataset_name}_edge_index.pkl", "rb"))
    num_node = pd.read_csv(f"../data/{configs.dataset_name}/{configs.dataset_name}_segment.csv").shape[0]
    print(num_node, len(edge_index))

    feature_size = configs.seg_size
    walk_length = 20
    context_size = 10
    walks_per_node = 10
    p = 1
    q = 1

    model = Node2Vec(
        edge_index,
        embedding_dim=feature_size,
        walk_length=walk_length,
        context_size=context_size,
        walks_per_node=walks_per_node,
        num_negative_samples=1,
        p=p,
        q=q,
        sparse=True,
        num_nodes=num_node
    ).to(configs.device)

    loader = model.loader(batch_size=128, shuffle=True)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.01)

    # Train until delta loss has been reached
    train_epoch(model, loader, optimizer, configs.device)

    save_embeddings(model, num_node, str(configs.dataset_name), configs.device)
