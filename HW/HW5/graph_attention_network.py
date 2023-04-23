from pathlib import Path
from typing import List, Callable

import torch.cuda
import torch.nn as nn

from graph_utils import get_karate_data, train_graph_network_using_karate_data, test_graph_network_using_karate_data


class GraphAttentionLayer(nn.Module):
    def __init__(self, adj_mat: torch.Tensor, in_feats: int, out_feats: int, act: Callable = lambda x: torch.relu(x)):
        super(GraphAttentionLayer, self).__init__()
        self.adj_mat = adj_mat
        self.in_feats = in_feats
        self.out_feats = out_feats

        self.W = nn.Parameter(torch.empty(size=(in_feats, out_feats)))
        # original paper by Xavier Glorot & Yoshua Bengio suggest initializing weights using a
        # uniform distribution between -r & +r with r = sqrt(6/(in_feats + out_feats)), in order to
        # ensure that the variance is equal to sigma^2 = 2/(in_feats + out_feats)
        nn.init.xavier_uniform_(self.W.data, gain=1)

        self.W2 = nn.Parameter(torch.empty(size=(2 * out_feats, 1)))
        nn.init.xavier_uniform_(self.W2.data)

        # 0.2 comes from https://arxiv.org/abs/1710.10903
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.act = act

    def forward(self, feature_mat):
        Wh = torch.matmul(feature_mat, self.W)
        attention = self.attention_mech(Wh)
        h_prime = torch.matmul(attention, Wh)

        return self.act(h_prime)

    def attention_mech(self, Wh):
        # e_est = LeakyReLU(W_2^[k,l] * Concat(W_1^[k,l] h_i^l, W_1^[k,l] h_j^l))
        Wh1 = torch.matmul(Wh, self.W2[:self.out_feats, :])
        Wh2 = torch.matmul(Wh, self.W2[self.out_feats:, :])

        e_est = torch.add(Wh1, Wh2.T)
        e_est = self.leaky_relu(e_est)

        # e = Softmax(e_est)
        zeros_vector = torch.zeros(e_est.size())
        # only get attention coefficients for nodes that have edges between them
        # choose e if condition is met & zero_vector if condition is not met
        attention = torch.where(self.adj_mat > 0, e_est, zeros_vector)
        attention = torch.softmax(attention, dim=1)

        return attention


class GraphAttentionNetwork(nn.Module):
    def __init__(
        self,
        adj_mat: torch.Tensor,
        in_feats: int = 36,
        out_feats: int = 2,
        n_layers: int = 2,
        n_hidden_neurons: int = 2,
    ):
        super(GraphAttentionNetwork, self).__init__()

        self.adj_mat = adj_mat
        self.in_feats = in_feats
        all_in_feats = [in_feats]
        all_in_feats.extend([n_hidden_neurons] * (n_layers - 1))
        all_out_feats = [n_hidden_neurons] * (n_layers - 1)
        all_out_feats.append(out_feats)
        self.out_feats = out_feats
        self.n_layers = n_layers
        self.n_hidden_neurons = n_hidden_neurons

        all_acts: List[Callable] = [lambda x: torch.relu(x)] * n_layers
        # all_acts: List[Callable] = [lambda x: torch.relu(x)] * (n_layers - 1)
        # all_acts.append(lambda x: x)
        gat_layers = []
        for in_feat, out_feat, act in zip(all_in_feats, all_out_feats, all_acts):
            gat_layers.append(GraphAttentionLayer(adj_mat, in_feat, out_feat, act))

        gat_layers.append(nn.Linear(in_features=out_feats, out_features=out_feats))
        self.gat_seq = nn.Sequential(*gat_layers)

    def forward(self, feature_mat):
        return self.gat_seq(feature_mat)


def main():
    edgelist_path = Path("./graph-attrs/karate.edgelist.txt")
    attrs_path = Path("./graph-attrs/karate.attributes.csv")

    karate_data = get_karate_data(edgelist_path, attrs_path)

    n_layers: int = 3
    n_hidden_neurons: int = 6
    n_epochs: int = 5000
    lr: float = 1e-3
    model_name: str = "GAT"
    load_model: bool = False
    train: bool = True
    test: bool = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_output_path = Path(f"./gat/{n_layers}-layers--{n_hidden_neurons}-hidden-neurons")
    base_output_path.mkdir(exist_ok=True, parents=True)

    gat_model = GraphAttentionNetwork(
        karate_data.adjaceny_mat,
        karate_data.n_in_features,
        karate_data.n_out_features,
        n_layers,
        n_hidden_neurons,
    ).to(device)
    karate_data.feature_mat = karate_data.feature_mat.to(torch.float32)
    if load_model:
        model_path = sorted(base_output_path.glob("**/*.pth"))
        if len(model_path) > 0:
            model_path = model_path[0]
            state_dict = torch.load(str(model_path)).get(model_name)
            gat_model.load_state_dict(state_dict)
        else:
            print(f"Failed to find model in '{base_output_path}'. Not loading any model.")

    if train:
        print(f"*** TRAINING {model_name} ***")
        gat_model = train_graph_network_using_karate_data(
            model=gat_model,
            karate_data=karate_data,
            output_path=base_output_path,
            model_name=model_name,
            n_epochs=n_epochs,
            lr=lr,
        )

    if test:
        print(f"*** TESTING {model_name} ***")
        test_graph_network_using_karate_data(
            model=gat_model,
            karate_data=karate_data,
        )


if __name__ == '__main__':
    main()
