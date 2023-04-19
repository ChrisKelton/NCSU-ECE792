from pathlib import Path
from typing import Callable, List, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from networkx import read_edgelist, set_node_attributes, shortest_path_length
from sklearn.preprocessing import OneHotEncoder
from torch.autograd import Variable
from tqdm import tqdm

from mlp_mnist.training_utils import Loss, save_loss_plot

epochs_to_generate_scatter_plots: List[int] = [1, 2, 5, 10, 20, 50, 75, 100, 200, 300, 500, 750, 1000, 1500, 2000, 5000]


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

        self.gcn_seq = nn.Sequential(*gcn_layers)

    def forward(self, X):
        return self.gcn_seq(X)


def scatter_plots(
    node_groupings: np.ndarray,
    labels: np.ndarray,
    base_output_path: Path,
    gen_scat_plots_for_epochs: Optional[List[int]] = None,
    title_prefix: Optional[str] = None,
):
    if gen_scat_plots_for_epochs is None:
        gen_scat_plots_for_epochs = epochs_to_generate_scatter_plots

    base_output_path.mkdir(exist_ok=True, parents=True)

    red_idx = np.where(labels)[0]  # True = Administrator
    blue_idx = np.where(1 - labels)[0]  # False = Instructor

    for epoch in gen_scat_plots_for_epochs:
        output_path = base_output_path / f"epoch-{epoch}.png"
        node_grouping = node_groupings[epoch-1]
        red_nodes = node_grouping[red_idx]
        red_x, red_y = zip(red_nodes.T)
        red_x = red_x[0]
        red_y = red_y[0]

        blue_nodes = node_grouping[blue_idx]
        blue_x, blue_y = zip(blue_nodes.T)
        blue_x = blue_x[0]
        blue_y = blue_y[0]

        plt.scatter(red_x, red_y, alpha=0.5, label="Administrator", c="red")
        plt.scatter(blue_x, blue_y, alpha=0.5, label="Instructor", c="blue")
        if title_prefix is not None:
            title = f"{title_prefix} epoch {epoch}"
        else:
            title = f"epoch {epoch}"
        plt.title(title)
        plt.legend(["Administrator", "Instructor"])
        plt.savefig(str(output_path))
        plt.close()


def main():
    edgelist_path = Path("./graph-attrs/karate.edgelist.txt")
    attrs_path = Path("./graph-attrs/karate.attributes.csv")
    network = read_edgelist(str(edgelist_path), nodetype=int)
    attributes = pd.read_csv(str(attrs_path), index_col=["node"])

    for attribute in attributes:
        set_node_attributes(network, values=pd.Series(attributes[attribute], index=attributes.index).to_dict(), name=attribute)

    # setting Administrator as 'True' for our binary cross entropy
    # only use Administrator & Instructor nodes for training set
    X_train, y_train = map(np.array, zip(*[([node], data['role'] == 'Administrator') for node, data in network.nodes(data=True) if data['role'] in {'Administrator', 'Instructor'}]))
    X_train = np.squeeze(X_train)
    # only use Member nodes for testing set
    X_test, y_test = map(np.array, zip(*[([node], data['community'] == 'Administrator') for node, data in network.nodes(data=True) if data['role'] == 'Member']))
    X_test = np.squeeze(X_test)

    node_distance_administrator = shortest_path_length(network, target=X_train[0])
    node_distance_instructor = shortest_path_length(network, target=X_train[-1])

    # the input features of the nodes are to be designed as a 36-dimensional vector. The first 34 dimensions would
    # represent the node itself uniquely as a one-hot encoding. The last two dimensions will represent the length of the
    # shortest path in the graph from the Administrator & the Instructor respectively.
    in_enc = OneHotEncoder()
    in_enc.fit([[i] for i in range(len(network.nodes()))])
    X = torch.tensor(in_enc.transform(np.expand_dims(np.array(network.nodes()), axis=-1)).toarray())

    out_enc = OneHotEncoder()
    out_enc.fit([[i] for i in range(len(set(y_train)))])
    y_train_enc = torch.tensor(out_enc.transform(np.expand_dims(y_train, axis=-1)).toarray()).to(torch.float32)
    y_test_enc = torch.tensor(out_enc.transform(np.expand_dims(y_test, axis=-1)).toarray()).to(torch.float32)
    # create last two dimensions of the input feature vector, with first-dimension representing the shortest distance
    # on the graph from the Administrator (X_train[0]), while the second dimension representing the shortest distance
    # on the graph from the Instructor (X_train[-1])
    X_2 = torch.zeros((34, 2))
    for node in network.nodes():
        X_2[node][0] = node_distance_administrator[node]
        X_2[node][0] = node_distance_instructor[node]

    # input feature vector to our GCN
    X = torch.cat([X, X_2], dim=-1)

    A = torch.tensor(nx.to_numpy_array(network))
    n_layers: int = 2
    n_hidden_neurons: int = 6
    out_feats: int = 2
    lr: float = 1e-3
    n_epochs: int = 5000
    base_output_path = Path(f"./gcn/{n_layers}-layers--{n_hidden_neurons}-hidden-neurons")
    base_output_path.mkdir(exist_ok=True, parents=True)
    base_loss_plot_path = base_output_path / "loss"
    base_loss_plot_path.mkdir(exist_ok=True, parents=True)
    base_scatter_plot_path = base_output_path / "scatter-plots"
    base_scatter_plot_path.mkdir(exist_ok=True, parents=True)
    model_base_path = base_output_path / "models"
    model_base_path.mkdir(exist_ok=True, parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gcn_model = GCN(A, in_feats=X.shape[1], out_feats=out_feats, n_layers=n_layers, n_hidden_neurons=n_hidden_neurons).to(device)

    optimizer = torch.optim.Adam(gcn_model.parameters(), lr=lr)
    train_loss = Loss.init()
    # (epoch, node, reduced feature vector + decision)
    train_node_groupings = np.zeros((n_epochs, len(X_train), out_feats))
    test_loss = Loss.init()
    test_node_groupings = np.zeros((n_epochs, len(X_test), out_feats))
    criterion = nn.BCELoss()
    for epoch in tqdm(range(n_epochs)):
        for node_idx, (x, y) in enumerate(zip(X_train, y_train_enc)):
            optimizer.zero_grad()

            pred = gcn_model(X)[x]
            train_node_groupings[epoch, node_idx, :] = pred.detach().cpu().numpy()
            loss = criterion(torch.softmax(pred, dim=-1), y)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        gcn_model.eval()
        for node_idx, (x, y) in enumerate(zip(X_test, y_test_enc)):
            pred = gcn_model(X)[x]
            test_node_groupings[epoch, node_idx, :] = pred.detach().cpu().numpy()
            loss = criterion(torch.softmax(pred, dim=-1), y)

            test_loss += loss.item()

        gcn_model.train()
        train_loss.update_for_epoch()
        test_loss.update_for_epoch()
        if (epoch + 1) % 10 == 0:
            print(f"Train Loss: {train_loss.current_loss}\tTest Loss: {test_loss.current_loss}")

    model_path = model_base_path / f"GCN--{n_epochs}.pth"
    torch.save(
        {
            "GCN": gcn_model.state_dict(),
        },
        str(model_path),
    )

    loss_plot_path = base_loss_plot_path / f"epoch-{n_epochs}.png"
    save_loss_plot(train_loss.loss_vals_per_epoch, loss_plot_path, test_loss.loss_vals_per_epoch, alpha=1)

    scatter_plot_path = base_scatter_plot_path / "train"
    scatter_plot_path.mkdir(exist_ok=True, parents=True)
    scatter_plots(train_node_groupings, y_train, scatter_plot_path, title_prefix="train")

    scatter_plot_path = base_scatter_plot_path / "test"
    scatter_plot_path.mkdir(exist_ok=True, parents=True)
    scatter_plots(test_node_groupings, y_test, scatter_plot_path, title_prefix="test")


if __name__ == '__main__':
    main()
