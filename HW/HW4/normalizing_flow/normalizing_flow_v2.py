from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from HW.HW1.mlp_mnist.training_utils import Loss, save_loss_plot
from HW.HW4.normalizing_flow.normalizing_flow import (
    NFDataset,
    get_circles_test_and_train_csv_files,
    get_moon_test_and_train_csv_files,
)


class NormalizingFlow(nn.Module):
    def __init__(self, n_layers: int, hidden_size: int, n_hidden_layers: int = 1):
        super(NormalizingFlow, self).__init__()
        self.n_layers = n_layers
        self.s = nn.ModuleList([self.affine_coupling_layer(1, hidden_size, n_hidden_layers) for _ in range(n_layers)])
        self.t = nn.ModuleList([self.affine_coupling_layer(1, hidden_size, n_hidden_layers) for _ in range(n_layers)])

        # have learnable scale factor for each affine coupling layer
        self.s_scale = torch.nn.Parameter(torch.randn(n_layers))
        self.s_scale.requires_grad = True

        # this normalizing flow class is only for our 2d dataset
        self.norm_dist = MultivariateNormal(torch.zeros(2), torch.eye(2))

    @staticmethod
    def affine_coupling_layer(input_size: int, hidden_size: int, n_hidden_layers: int = 1):
        seq_layers = [nn.Linear(in_features=input_size, out_features=hidden_size), nn.LeakyReLU()]
        for idx in range(n_hidden_layers):
            if idx == n_hidden_layers - 1:
                seq_layers.extend([nn.Linear(in_features=hidden_size, out_features=input_size), nn.Tanh()])
            else:
                seq_layers.extend([nn.Linear(in_features=hidden_size, out_features=hidden_size), nn.LeakyReLU()])
        return nn.Sequential(*seq_layers)

    def forward(self, x: Optional[torch.Tensor] = None, sample_size: Optional[Tuple[int, ...]] = None):
        if x is not None:
            s_vals = []
            for i in range(self.n_layers):
                y_1_d, y_d_1_D = x[:, :1], x[:, 1:]
                # Alternating channel that gets transformed
                s = self.s[i](y_1_d) * self.s_scale[i]
                x_d_1_D = y_d_1_D * torch.exp(s) + self.t[i](y_1_d)
                y_d_1_D = x_d_1_D

                x = torch.concat([y_1_d, y_d_1_D], dim=1)
                # flip channel via [1] and reverse order via [0]
                x = torch.flip(torch.flip(x, dims=[1]), dims=[0])
                s_vals.append(s)

            return torch.cat([y_1_d, y_d_1_D], dim=1), torch.cat(s_vals)
        elif sample_size is not None:
            x = self.norm_dist.sample(sample_size)
            s_vals = []
            for i in range(self.n_layers - 1, -1, -1):
                z_1_d, z_d_1_D = x[:, :1], x[:, 1:]
                s = self.s[i](z_1_d) * self.s_scale[i]
                y_d_1_D = (z_d_1_D - self.t[i](z_1_d)) * torch.exp(-s)
                z_d_1_D = y_d_1_D

                x = torch.concat([z_1_d, z_d_1_D], dim=1)
                # flip channel via [1] and reverse order via [0]
                x = torch.flip(torch.flip(x, dims=[1]), dims=[0])
                s_vals.append(s)

            return torch.cat([z_1_d, z_d_1_D], dim=1), torch.cat(s_vals)
        else:
            raise RuntimeError(f"Need x or sample_size defined.")


standard_normal_const = torch.log(torch.tensor(2 * torch.pi)) / 2


def normalizing_flows_loss(x, s, batch_size):
    """
    z = normal distribution
    x = complex distribution
    determinants are taken of the jacobian matrix

        p_x(x) = p_z(z)*|det(\partial z/\partial x)|
               = p_z(g^-1(x))*|det(\partial g^-1(x)/\partial x)|
               = p_z(f(x))*|det(\partial f(x)/\partial x)|

    f(.) = stacked affine coupling layers from complex dist -> norm dist
    g(.) = stacked affine coupling layers from norm dist -> complex dist [=f^-1(.)]

        log(p_x(x)) = log(p_z(f_theta(x)) * |det(\partial f_theta(x)/\partial x)|)
                    = log(p_x(f_theta(x)) + log(|det(\partial f_theta(x)/\partial x)|)

    f_theta(.) = f(.) parameterized by theta (essentially just transforming the normal distribution between affine coupling layers)

    Since we are assuming a standard independent Gaussian for p_z, we can model log(p_z(f_theta(x)) as the standard normal PDF
    Therefore,

        log(p_x(x)) = -log(2*pi)/2 - (f_theta(x))^2/2 + log(|det(\partial f_theta(x)/\partial x)|)

    When training, we will use the negative log likelihood of log(p_x(x)); therefore, our loss function is:

        Loss = -log(p_x(x))

    Additionally due to the nature of the jacobian matrix, the determinant just becomes:

        log(|det(\partial y/\partial x.T)|) = sum(s); where s is the scaling paramter in the exponent of our affine coupling layer

    All in all, the Loss function using the arguments to our function:

        Loss = log(2*pi)/2 + (x^2)/2 - s

    Extending this to where our batch size is > 1:

        Loss = sum(log(2*pi)/2 + (x^2)/2) - sum(s)

    :param x: transformed distribution during training (base dist -> trans dist)
    :param s: scale output of the affine coupling (log of the determinant of the jacobian)
    :param batch_size:
    :return:
    """
    log_px_ftheta_x = torch.sum(torch.add(standard_normal_const, torch.div(torch.pow(x, 2), 2)))
    log_det_jac = -torch.sum(s)

    neg_log_px = torch.add(log_px_ftheta_x, log_det_jac)
    return torch.div(neg_log_px, batch_size)


def train(
    model: NormalizingFlow,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
) -> Tuple[NormalizingFlow, Loss, Loss]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loss = Loss.init()
    test_loss = Loss.init()
    for epoch in tqdm(range(epochs)):
        for batch_idx, samples in enumerate(train_dataloader):
            optimizer.zero_grad()

            samples.to(device)
            x, log_prob = model(samples)
            loss = normalizing_flows_loss(x, log_prob, len(samples))

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # if batch_idx % 50 == 0:
            #     print(" [%d/%d] [%d/%d]\tLoss: %.8f" % (epoch, epochs, batch_idx, len(dataloader), losses.current_loss))

        model.eval()
        for batch_idx, samples in enumerate(test_dataloader):
            samples = samples.to(device)

            x, log_prob = model(samples)
            loss = normalizing_flows_loss(x, log_prob, len(samples))

            test_loss += loss.item()

        model.train()

        train_loss.update_for_epoch()
        test_loss.update_for_epoch()
        print(
            "\tTrain Loss: %.8f\tTest Loss: %.8f"
            % (train_loss.loss_vals_per_epoch[-1], test_loss.loss_vals_per_epoch[-1])
        )

    return model, train_loss, test_loss


def main(dataset_base_path: Path):
    csv_files_base_path = dataset_base_path / "csv-files"
    batch_size = 88
    n_epochs = 500
    # n_layers_ = [1, 3, 5]
    n_layers_ = [5]
    # hidden_size = [128]
    hidden_sizes = [64, 256]
    n_hidden_layers = 1
    # lrs = [1e-3, 2e-3]
    lrs = [1e-3]
    # dataset_names = ["Circles", "Moons"]
    dataset_names = ["Moons"]

    for dataset_name in dataset_names:
        if dataset_name in ["Circles"]:
            train_set, test_set = get_circles_test_and_train_csv_files(csv_files_base_path)
        elif dataset_name in ["Moons"]:
            train_set, test_set = get_moon_test_and_train_csv_files(csv_files_base_path)
        else:
            raise RuntimeError(f"Got unidentified dataset name '{dataset_name}'")

        train_df = pd.read_csv(train_set, index_col=0, header=0)
        test_df = pd.read_csv(test_set, index_col=0, header=0)

        train_dataset = NFDataset(train_df)
        test_dataset = NFDataset(test_df)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        for n_layers in n_layers_:
            for lr in lrs:
                for hidden_size in hidden_sizes:
                    print(f"{dataset_name} -- {n_layers} layers -- {lr} learning rate -- {hidden_size} hidden size")
                    outputs_base_path = (
                        dataset_base_path
                        / "plots"
                        / f"{dataset_name}--{n_layers}-layers--{hidden_size}-hidden_size--{batch_size}-batch_size--{n_epochs}-epochs--{lr}-lr"
                    )
                    outputs_base_path.mkdir(exist_ok=True, parents=True)

                    model = NormalizingFlow(n_layers=n_layers, hidden_size=hidden_size, n_hidden_layers=n_hidden_layers)
                    model, train_loss, test_loss = train(model, train_dataloader, test_dataloader, epochs=n_epochs)

                    model_output_path = outputs_base_path / f"model--{n_epochs}.pth"
                    torch.save({"NF": model.state_dict()}, str(model_output_path))

                    model.eval()
                    x, _ = model(torch.from_numpy(train_df.values).to(torch.float32))
                    x = x.detach().cpu().numpy()

                    x_test, _ = model(torch.from_numpy(test_df.values).to(torch.float32))
                    x_test = x_test.detach().cpu().numpy()

                    y, _ = model(sample_size=(len(train_df),))
                    y = y.detach().cpu().numpy()

                    fig, ax = plt.subplots(1, 3, figsize=(20, 7))
                    ax[0].scatter(x[:, 0], x[:, 1])
                    ax[0].set_title("Normal Distribution Train Set")
                    ax[1].scatter(x_test[:, 0], x_test[:, 1])
                    ax[1].set_title("Normal Distribution Test Set")
                    ax[2].scatter(y[:, 0], y[:, 1])
                    ax[2].set_title(dataset_name)
                    fig.tight_layout()
                    dist_output = outputs_base_path / "distributions.png"
                    fig.savefig(str(dist_output))
                    plt.show()

                    loss_path = outputs_base_path / "loss.png"
                    save_loss_plot(train_loss.loss_vals_per_epoch, loss_path, test_loss.loss_vals_per_epoch)


if __name__ == "__main__":
    dataset_base_path = Path("./csv-files")
    main(dataset_base_path)
