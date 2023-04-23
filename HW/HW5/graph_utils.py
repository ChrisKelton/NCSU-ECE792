__all__ = [
    "scatter_plots",
    "get_karate_data",
    "KarateData",
    "train_graph_network_using_karate_data",
    "test_graph_network_using_karate_data",
]
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from networkx import read_edgelist, set_node_attributes, shortest_path_length
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

from mlp_mnist.training_utils import Loss, save_loss_plot

epochs_to_generate_scatter_plots: List[int] = [1, 2, 5, 10, 20, 50, 75, 100, 200, 300, 500, 750, 1000, 1500, 2000, 5000]


def load_karate(edgelist_path: Path, attrs_path: Path) -> nx.classes.graph.Graph:
    network = read_edgelist(str(edgelist_path), nodetype=int)
    attributes = pd.read_csv(str(attrs_path), index_col=["node"])

    for attribute in attributes:
        set_node_attributes(network, values=pd.Series(attributes[attribute], index=attributes.index).to_dict(), name=attribute)

    return network


@dataclass
class KarateData:
    feature_mat: torch.Tensor
    adjaceny_mat: torch.Tensor
    train_data: np.ndarray
    train_labels: np.ndarray
    train_labels_enc: torch.Tensor
    test_data: np.ndarray
    test_labels: np.ndarray
    test_labels_enc: torch.Tensor
    in_enc: OneHotEncoder
    n_in_features: int
    out_enc: OneHotEncoder
    n_out_features: int


def get_karate_data(edgelist_path: Path, attrs_path: Path) -> KarateData:
    network = load_karate(edgelist_path, attrs_path)

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

    return KarateData(
        feature_mat=X,
        adjaceny_mat=A,
        train_data=X_train,
        train_labels=y_train,
        train_labels_enc=y_train_enc,
        test_data=X_test,
        test_labels=y_test,
        test_labels_enc=y_test_enc,
        in_enc=in_enc,
        n_in_features=X.shape[1],
        out_enc=out_enc,
        n_out_features=len(set(y_train)),
    )


def train_graph_network_using_karate_data(
    model: Any,
    karate_data: KarateData,
    output_path: Path,
    model_name: str,
    n_epochs: int = 5000,
    lr: float = 1e-3,
    optimizer: Optional[Any] = None,
    optimizer_type: str = "adam",
    criterion: Optional[Any] = None,
    criterion_type: str = "bce",
    epochs_to_print_loss: int = 10,
    epochs_to_save_model: Optional[int] = None,
) -> Any:

    print(f"Saving outputs to '{output_path}'")

    if optimizer is None and optimizer_type.lower() in ["adam"]:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise RuntimeError(f"Got unidentified optimizer type '{optimizer_type}'")

    if criterion is None and criterion_type.lower() in ["bce", "bceloss"]:
        criterion = nn.BCELoss()
    else:
        raise RuntimeError(f"Got unidentified criterion type '{criterion_type}'")

    base_loss_plot_path = output_path / "loss"
    base_loss_plot_path.mkdir(exist_ok=True, parents=True)
    base_scatter_plot_path = output_path / "scatter-plots"
    base_scatter_plot_path.mkdir(exist_ok=True, parents=True)
    model_base_path = output_path / "models"
    model_base_path.mkdir(exist_ok=True, parents=True)

    train_loss = Loss.init()
    train_node_groupings = np.zeros((n_epochs, len(karate_data.train_data), karate_data.n_out_features))

    test_loss = Loss.init()
    test_node_groupings = np.zeros((n_epochs, len(karate_data.test_data), karate_data.n_out_features))

    for epoch in tqdm(range(n_epochs)):
        for node_idx, (x, y) in enumerate(zip(karate_data.train_data, karate_data.train_labels_enc)):
            optimizer.zero_grad()

            pred = model(karate_data.feature_mat)[x]
            train_node_groupings[epoch, node_idx, :] = pred.detach().cpu().numpy()
            loss = criterion(torch.softmax(pred, dim=-1), y)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        for node_idx, (x, y) in enumerate(zip(karate_data.test_data, karate_data.test_labels_enc)):
            pred = model(karate_data.feature_mat)[x]
            test_node_groupings[epoch, node_idx, :] = pred.detach().cpu().numpy()
            loss = criterion(torch.softmax(pred, dim=-1), y)

            test_loss += loss.item()

        model.train()

        train_loss.update_for_epoch()
        test_loss.update_for_epoch()
        if (epoch + 1) % epochs_to_print_loss == 0:
            print(f"Train Loss: {train_loss.current_loss}\tTest Loss: {test_loss.current_loss}")

        if epochs_to_save_model is not None and (epoch + 1) % epochs_to_save_model == 0:
            model_path = model_base_path / f"GCN--{n_epochs}.pth"
            torch.save(
                {
                    model_name: model.state_dict(),
                },
                str(model_path),
            )

    model_path = model_base_path / f"GCN--{n_epochs}.pth"
    torch.save(
        {
            model_name: model.state_dict(),
        },
        str(model_path),
    )

    loss_plot_path = base_loss_plot_path / f"epoch-{n_epochs}.png"
    save_loss_plot(train_loss.loss_vals_per_epoch, loss_plot_path, test_loss.loss_vals_per_epoch, alpha=1)

    scatter_plot_path = base_scatter_plot_path / "train"
    scatter_plot_path.mkdir(exist_ok=True, parents=True)
    scatter_plots(train_node_groupings, karate_data.train_labels, scatter_plot_path, title_prefix="train")

    scatter_plot_path = base_scatter_plot_path / "test"
    scatter_plot_path.mkdir(exist_ok=True, parents=True)
    scatter_plots(test_node_groupings, karate_data.test_labels, scatter_plot_path, title_prefix="test")

    return model


def test_graph_network_using_karate_data(
    model: Any,
    karate_data: KarateData,
    criterion: Optional[Any] = None,
    criterion_type: str = "bce",
):
    model.eval()

    if criterion is None and criterion_type.lower() in ["bce", "bceloss"]:
        criterion = nn.BCELoss()
    else:
        raise RuntimeError(f"Got unidentified criterion type '{criterion_type}'")

    train_loss = Loss.init()
    train_node_groupings = np.zeros((len(karate_data.train_data), karate_data.n_out_features))
    train_node_decisions = np.zeros((len(karate_data.train_data), 1))
    for node_idx, (x, y) in enumerate(zip(karate_data.train_data, karate_data.train_labels_enc)):
        pred = model(karate_data.feature_mat)[x]
        train_node_groupings[node_idx, :] = pred.detach().cpu().numpy()

        loss = criterion(torch.softmax(pred, dim=-1), y)
        train_node_decisions[node_idx] = bool(torch.argmax(torch.softmax(pred, dim=-1)))
        train_loss += loss.item()

    train_loss.update_for_epoch()

    test_loss = Loss.init()
    test_node_groupings = np.zeros((len(karate_data.test_data), karate_data.n_out_features))
    test_node_decisions = np.zeros((len(karate_data.test_data), 1))
    for node_idx, (x, y) in enumerate(zip(karate_data.test_data, karate_data.test_labels_enc)):
        pred = model(karate_data.feature_mat)[x]
        test_node_groupings[node_idx, :] = pred.detach().cpu().numpy()

        loss = criterion(torch.softmax(pred, dim=-1), y)
        test_node_decisions[node_idx] = bool(torch.argmax(torch.softmax(pred, dim=-1)))
        test_loss += loss.item()

    test_loss.update_for_epoch()
    print(f"Train Data Loss: {train_loss.loss_vals_per_epoch[-1]}\nTest Data Loss: {test_loss.loss_vals_per_epoch[-1]}\n")
    test_node_decisions = np.squeeze(test_node_decisions)
    print(f"Train Data Accuracy = "
          f"{((train_node_decisions == karate_data.train_labels).sum() / len(karate_data.train_labels)) * 100:.2f}%\n"
          f"Test Data Accuracy = "
          f"{((test_node_decisions == karate_data.test_labels).sum() / len(karate_data.test_labels)) * 100:.2f}%")


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
