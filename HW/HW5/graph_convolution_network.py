from pathlib import Path
from typing import Callable, List

import torch
import torch.nn as nn
from torch.autograd import Variable

from graph_utils import get_karate_data, train_graph_network_using_karate_data, test_graph_network_using_karate_data


class GCNLayer(nn.Module):
    def __init__(self, adj_mat: torch.Tensor, in_feats: int, out_feats: int, activation: Callable = lambda x: x):
        super(GCNLayer, self).__init__()
        self.adj_mat = adj_mat  # A
        # identity matrix helps account for the node connection to itself; otherwise, we lose this information when
        # performing matrix multiplication
        self.identity = torch.eye(adj_mat.shape[0])  # I
        self.deg_mat = torch.diag(torch.pow(torch.sum(adj_mat, dim=0), -0.5))  # D
        self.norm_lap_mat = Variable(torch.add(torch.matmul(torch.matmul(self.deg_mat, self.adj_mat), self.deg_mat), self.identity))  # L

        self.fully_connected = nn.Linear(in_features=in_feats, out_features=out_feats)
        nn.init.uniform_(self.fully_connected.weight)
        self.activation = activation

    def forward(self, X: torch.Tensor):
        # Z = D^-0.5 @ A @ D^-0.5 @ X
        # Z = L @ X
        Z = torch.matmul(self.norm_lap_mat, X.to(torch.float64)).to(torch.float32)
        Z = self.fully_connected(Z)
        Z = self.activation(Z)

        return Z


class GCN(nn.Module):
    def __init__(
        self,
        adj_mat: torch.Tensor,
        in_feats: int = 36,
        out_feats: int = 2,
        n_layers: int = 2,
        n_hidden_neurons: int = 6,
    ):
        super(GCN, self).__init__()
        self.in_feats = in_feats
        all_in_feats = [in_feats]
        all_in_feats.extend([n_hidden_neurons] * (n_layers - 1))
        all_out_feats = [n_hidden_neurons] * (n_layers - 1)
        all_out_feats.append(out_feats)
        self.out_feats = out_feats
        self.n_layers = n_layers
        self.n_hidden_neurons = n_hidden_neurons

        all_acts: List[Callable] = [lambda x: torch.relu(x)] * (n_layers - 1)
        all_acts.append(lambda x: x)
        gcn_layers = []
        for in_feat, out_feat, act in zip(all_in_feats, all_out_feats, all_acts):
            gcn_layers.append(GCNLayer(adj_mat, in_feat, out_feat, act))

        gcn_layers.append(nn.Linear(in_features=out_feats, out_features=out_feats))
        self.gcn_seq = nn.Sequential(*gcn_layers)

    def forward(self, X):
        return self.gcn_seq(X)


def main():
    edgelist_path = Path("./graph-attrs/karate.edgelist.txt")
    attrs_path = Path("./graph-attrs/karate.attributes.csv")

    karate_data = get_karate_data(edgelist_path, attrs_path)

    n_layers: int = 2
    n_hidden_neurons: int = 6
    lr: float = 1e-3
    n_epochs: int = 5000
    model_name: str = "GCN"
    load_model: bool = True
    train: bool = False
    test: bool = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_output_path = Path(f"./gcn/{n_layers}-layers--{n_hidden_neurons}-hidden-neurons")
    base_output_path.mkdir(exist_ok=True, parents=True)

    gcn_model = GCN(
        karate_data.adjaceny_mat,
        in_feats=karate_data.n_in_features,
        out_feats=karate_data.n_out_features,
        n_layers=n_layers,
        n_hidden_neurons=n_hidden_neurons,
    ).to(device)
    if load_model:
        model_path = sorted(base_output_path.glob("**/*.pth"))
        if len(model_path) > 0:
            model_path = model_path[0]
            state_dict = torch.load(str(model_path)).get(model_name)
            gcn_model.load_state_dict(state_dict)
        else:
            print(f"Failed to find model in '{base_output_path}'. Not loading any model.")

    if train:
        print(f"*** TRAINING {model_name} ***")
        updated_gcn_model = train_graph_network_using_karate_data(
            model=gcn_model,
            karate_data=karate_data,
            output_path=base_output_path,
            model_name=model_name,
            n_epochs=n_epochs,
            lr=lr,
        )

    if test:
        print(f"*** TESTING {model_name} ***")
        test_graph_network_using_karate_data(
            model=gcn_model,
            karate_data=karate_data,
        )


if __name__ == '__main__':
    main()
