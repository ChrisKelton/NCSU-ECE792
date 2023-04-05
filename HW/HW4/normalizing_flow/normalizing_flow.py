from pathlib import Path
from typing import Tuple, List, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# matplotlib.use("TkAgg")


def test_train_set_from_list_of_paths(path_list: List[Path]) -> Tuple[Path, Path]:
    if "test" in str(path_list[0]):
        test_set = path_list[0]
        train_set = path_list[1]
    else:
        test_set = path_list[1]
        train_set = path_list[0]

    return train_set, test_set


def get_circles_test_and_train_csv_files(csv_base_path: Path) -> Tuple[Path, Path]:
    csv_files = sorted(csv_base_path.glob("*.csv"))
    circles_files = list(filter(lambda x: x if "circle" in str(x) else None, csv_files))

    return test_train_set_from_list_of_paths(circles_files)


def get_moon_test_and_train_csv_files(csv_base_path: Path) -> Tuple[Path, Path]:
    csv_files = sorted(csv_base_path.glob("*.csv"))
    moon_files = list(filter(lambda x: x if "moon" in str(x) else None, csv_files))

    return test_train_set_from_list_of_paths(moon_files)


def plot_moon_and_circls(csv_files_base_path: Path):
    circles_train_set, circles_test_set = get_circles_test_and_train_csv_files(csv_files_base_path)
    moon_train_set, moon_test_set = get_moon_test_and_train_csv_files(csv_files_base_path)
    circles_train_df = pd.read_csv(circles_train_set, index_col=0, header=0)
    moon_train_df = pd.read_csv(moon_train_set, index_col=0, header=0)
    x_circles = circles_train_df["x1"]
    y_circles = circles_train_df["x2"]
    x_moon = moon_train_df["x1"]
    y_moon = moon_train_df["x2"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].scatter(x_circles, y_circles)
    ax[1].scatter(x_moon, y_moon)
    fig.tight_layout()
    plt.show()


class NFDataset(Dataset):
    def __init__(self, train_set: pd.DataFrame):
        super(NFDataset, self).__init__()
        self.train_set = train_set

    def __getitem__(self, index):
        return torch.Tensor(self.train_set.iloc[index])

    def __len__(self):
        return len(self.train_set)


class AffineCoupling(nn.Module):
    # x_1_d: torch.Tensor = None

    def __init__(self, dim: int = 2, split_factor: float = 0.75):
        super(AffineCoupling, self).__init__()

        self.dim = dim
        self.split_factor = split_factor
        self.exp_scale = nn.Parameter(torch.zeros(dim))
        self.shift = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor):
        """
        y_1:d = x_1:d
        y_d+1:D = x_d+1:D * exp(self.exp_scale*x_1:d) + self.shift(x_1:d)

        :param x: will be x_1:D, must split up into x_1:d, x_d+1:D
        :return:
        """
        d1 = int(len(x) * self.split_factor)
        # self.x_1_d = x[:d1]
        x_1_d = x[:d1]
        y_1_d = torch.clone(x_1_d)

        x_d_1_D = x[d1:]
        # y_d_1_D = torch.add(torch.mul(x_d_1_D, torch.exp(torch.mul(self.exp_scale, self.x_1_d))), self.shift)
        y_d_1_D = torch.add(torch.mul(x_d_1_D, torch.exp(self.exp_scale)), self.shift)

        y = torch.concat([y_1_d, y_d_1_D], axis=0)
        # x = torch.concat([x_1_d, x_d_1_D], axis=0)
        # x = torch.flip(x, [0])

        return y

    def inv_log_det_jac(self, y: torch.Tensor):
        # x = torch.divide(torch.subtract(y, self.shift), torch.exp(torch.mul(self.exp_scale, self.x_1_d)))
        x = torch.divide(torch.subtract(y, self.shift), torch.exp(self.exp_scale))

        # det_jac = 1 / torch.exp(torch.mul(self.exp_scale, self.x_1_d).sum())
        det_jac = 1 / torch.exp(self.exp_scale).sum()
        inv_log_det_jac = torch.mul(torch.ones(y.shape), torch.log(det_jac))

        return x, inv_log_det_jac


def init_coupling_weights(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.zeros_(m.weight.data)
        nn.init.zeros_(m.bias.data)


class RealNVPLayer(nn.Module):
    def __init__(
        self,
        neurons_per_hidden_layer: List[int],
        d_ratio: float = 0.5,
        data_dim: int = 2,
        # exp_scale: Optional[float] = None,
    ):
        super(RealNVPLayer, self).__init__()
        self.neurons_per_hidden_layer = neurons_per_hidden_layer
        in_feats = [data_dim]
        in_feats.extend(neurons_per_hidden_layer.copy())
        out_feats = neurons_per_hidden_layer.copy()
        out_feats.append(data_dim)
        self.n_layers = len(neurons_per_hidden_layer)
        self.d_ratio = d_ratio
        self.data_dim = data_dim
        # if exp_scale is None:
        #     exp_scale = data_dim
        # self.exp_scale = exp_scale
        # self.s_scale = nn.Parameter(torch.randn(len(self.neurons_per_hidden_layer)))
        self.s_scale = nn.Parameter(torch.randn(1))
        self.s_scale.requires_grad = True

        s_exp = nn.ModuleList(
            [nn.Linear(in_features=in_feat, out_features=out_feat) for in_feat, out_feat in zip(in_feats, out_feats)]
        )
        t_shift = nn.ModuleList(
            [nn.Linear(in_features=in_feat, out_features=out_feat) for in_feat, out_feat in zip(in_feats, out_feats)]
        )
        idx_to_insert = [i for i in range((2 * len(s_exp)) - 1) if i % 2 == 1]
        for idx in idx_to_insert:
            s_exp.insert(idx, nn.LeakyReLU())
            t_shift.insert(idx, nn.LeakyReLU())

        self.s_exp = nn.Sequential(*s_exp)
        self.s_exp.append(nn.Tanh())
        self.t_shift = nn.Sequential(*t_shift)
        self.t_shift.append(nn.Tanh())

        # self.apply(init_coupling_weights)

    def forward(self, x):
        if self.training:
            y = x
            d_len = int(len(y) * self.d_ratio)
            # y_1_d = y[:d_len]
            # y_1_d_mask = torch.zeros(y.shape)
            # x_1_d = torch.clone(y_1_d)
            # y_1_d_mask[:d_len] = y_1_d
            y_1_d_mask, y_d_1_D_mask = y[:, :1], y[:, 1:]
            x_1_d = torch.clone(y_1_d_mask)
            s = torch.multiply(self.s_exp(y_1_d_mask), -self.s_scale)
            t = -self.t_shift(y_1_d_mask)

            # y_d_1_D = y[d_len:]
            # y_d_1_D_mask = torch.zeros(y.shape)
            # y_d_1_D_mask[d_len:] = torch.clone(y_d_1_D)
            # x_d_1_D = y_d_1_D_mask / exp(s(y_d_1)) - t(y_d_1)
            x_d_1_D = torch.multiply(torch.add(y_d_1_D_mask, t), torch.exp(s))

            # x_d_1_D[:d_len] = x_1_d
            # x = x_d_1_D
            x = torch.concat([x_1_d, x_d_1_D], dim=1)

            # inv_log_det_jac = s.view(s.T.shape).sum(-1)
            inv_log_det_jac = s.sum(-1)

            z = x
            det_jac = inv_log_det_jac
        else:
            d_len = int(len(x) * self.d_ratio)
            # x_1_d = x[:d_len]
            # x_1_d_mask = torch.zeros(x.shape)
            # y_1_d = torch.clone(x_1_d)
            # x_1_d_mask[:d_len] = x_1_d
            x_1_d_mask, x_d_1_D_mask = x[:, :1], x[:, 1:]
            y_1_d = torch.clone(x_1_d_mask)
            s = torch.multiply(self.s_exp(x_1_d_mask), self.s_scale)
            t = self.t_shift(x_1_d_mask)

            # x_d_1_D = x[d_len:]
            # x_d_1_D_mask = torch.zeros(x.shape)
            # x_d_1_D_mask[d_len:] = torch.clone(x_d_1_D)
            # y_d_1_D = x_d_1_D_mask * exp(s(x_1_d)) + t(x_1_d)
            y_d_1_D = torch.add(torch.multiply(x_d_1_D_mask, torch.exp(s)), t)

            # y_d_1_D[:d_len] = y_1_d
            # y = y_d_1_D
            y = torch.concat([y_1_d, y_d_1_D], dim=1)

            log_det_jac = s.sum(-1)

            z = y
            det_jac = log_det_jac

        return z, det_jac

    # def inverse(self, y):
    #     d_len = int(len(y) * self.d_ratio)
    #     y_1_d = y[:d_len]
    #     y_1_d_mask = torch.zeros(y.shape)
    #     x_1_d = torch.clone(y_1_d)
    #     y_1_d_mask[:d_len] = y_1_d
    #     s = self.s_exp(y_1_d_mask) * -self.exp_scale
    #     t = -self.t_shift(y_1_d_mask)
    #
    #     y_d_1_D = y[d_len:]
    #     y_d_1_D_mask = torch.zeros(y.shape)
    #     y_d_1_D_mask[d_len:] = torch.clone(y_d_1_D)
    #     # x_d_1_D = y_d_1_D_mask / exp(s(y_d_1)) - t(y_d_1)
    #     x_d_1_D = (y_d_1_D_mask + t) * torch.exp(s)
    #
    #     x_d_1_D[:d_len] = x_1_d
    #     x = x_d_1_D
    #
    #     # inv_log_det_jac = s.view(s.T.shape).sum(-1)
    #     inv_log_det_jac = s.sum(-1)  # only take sum of masked portion?
    #
    #     return x, inv_log_det_jac


class RealNVP(nn.Module):
    def __init__(
        self,
        neurons_per_hidden_layer: Union[List[List[int]], List[int], int],
        n_layers: Optional[int] = None,
        d_ratio: Union[List[float], float] = 0.5,
        data_dim: int = 1,
        # exp_scale: Optional[Union[List[float], float]] = None,
    ):
        super(RealNVP, self).__init__()
        if n_layers is None:
            if isinstance(neurons_per_hidden_layer, int):
                raise RuntimeError(
                    f"Got n_layers 'None' & neurons_per_hidden_layer as an int. Please specify n_layers "
                    f"or reformat neurons_per_hidden_layer into a list of lists where the top level "
                    f"list indicates the number of layers & the embedded list indicates the number of "
                    f"hidden neurons in the NVP layer(s), if the length of this embedded list is > 1, "
                    f"then you will be extending the number of hidden layers within each NVP layer."
                )
            elif not isinstance(neurons_per_hidden_layer[0], list):
                print(
                    f"Warning...neurons_per_hidden_layer is not a list of a list of ints.\nInterpreting example:\n"
                    f"[64, 64, 64] -> [[64], [64], [64]]. Therefore, three NVP layers with a hidden layer of size "
                    f"64 for each layer. If desired functionality is a single NVP layer with three hidden layers "
                    f"each of size 64, then convert input to [[64, 64, 64]]."
                )
                temp: List[List[int]] = []
                for neurons in neurons_per_hidden_layer:
                    temp.append([neurons])
                neurons_per_hidden_layer = temp.copy()
        else:
            if isinstance(neurons_per_hidden_layer, int):
                neurons_per_hidden_layer = [[neurons_per_hidden_layer]] * n_layers
            elif isinstance(neurons_per_hidden_layer, list) and isinstance(neurons_per_hidden_layer[0], int):
                neurons_per_hidden_layer = [neurons_per_hidden_layer] * n_layers
            else:
                raise RuntimeError(
                    f"Got n_layers '{n_layers}' & neurons_per_hidden_layer as a list of lists. When "
                    f"using n_layers please provide neurons_per_hidden_layer simply as a list of ints "
                    f"or a single int that will be used as the dimensions of the hidden layer(s) in the "
                    f"NVP layer(s)."
                )
        self.neurons_per_hidden_layer = neurons_per_hidden_layer

        if isinstance(d_ratio, int) or isinstance(d_ratio, float):
            d_ratio = [d_ratio] * self.n_layers
        else:
            assert len(d_ratio) == self.n_layers
        self.d_ratio = d_ratio

        self.data_dim = data_dim
        # if exp_scale is None:
        #     exp_scale = [data_dim] * self.n_layers
        # elif isinstance(exp_scale, int):
        #     exp_scale = [exp_scale] * self.n_layers
        # else:
        #     assert len(exp_scale) == self.n_layers
        # self.exp_scale = exp_scale

        self.layers = nn.ModuleList(
            [
                RealNVPLayer(
                    neurons_per_hidden_layer=neurons_per_hidden_layer_,
                    d_ratio=d_ratio_,
                    data_dim=self.data_dim,
                    # exp_scale=exp_scale_,
                )
                for neurons_per_hidden_layer_, d_ratio_ in zip(self.neurons_per_hidden_layer, self.d_ratio)
            ]
        )

        self.dist = MultivariateNormal(torch.zeros(2), torch.eye(2))

    @property
    def n_layers(self) -> int:
        return len(self.neurons_per_hidden_layer)

    def forward(self, x: Optional[torch.Tensor] = None, sample_size: Optional[Tuple[int, ...]] = None):
        if self.training:
            log_prob = torch.zeros(x.shape[0])
            for reverse_layer_idx in range(len(self.layers) - 1, -1, -1):
                # x, inv_log_det_jac = self.layers[reverse_layer_idx].inverse(x)
                x, inv_log_det_jac = self.layers[reverse_layer_idx](x)
                x = torch.flip(x, dims=[0])  # reverse x in order to help mix data
                log_prob += inv_log_det_jac
            log_prob += self.dist.log_prob(x)  # why adding to cumulative inv log determinant jacobian

            y = x
        else:
            z = self.dist.sample(sample_size)
            log_prob = self.dist.log_prob(z)

            for layer in self.layers:
                z, log_det_jac = layer.forward(z)
                z = torch.flip(z, dims=[0])
                log_prob += log_det_jac

            y = z

        return y, log_prob

    # def inverse(self, x):
    #     """
    #     defined from complex dist -> normal dist
    #     :return:
    #     """
    #     log_prob = torch.zeros(x.shape[0])
    #     for reverse_layer_idx in range(len(self.layers)-1, -1, -1):
    #         x, inv_log_det_jac = self.layers[reverse_layer_idx].inverse(x)
    #         x = torch.flip(x, dims=[0])  # reverse x in order to help mix data
    #         log_prob += inv_log_det_jac
    #     log_prob += self.dist.log_prob(x)  # why adding to cumulative inv log determinant jacobian
    #
    #     return x, log_prob


def compare_dicts(dict0, dict1) -> bool:
    results = []
    wrong_keys = []
    for key in list(dict0.keys()):
        val0 = dict0[key]
        val1 = dict1[key]
        if int(torch.sum(~torch.eq(val0, val1))) > 0:
            results.append(False)
            wrong_keys.append(key)

    if len(wrong_keys) > 0:
        for wrong_key, result in zip(wrong_keys, results):
            print(f"{wrong_key} = {result}")
        return False
    return True


def compare_list_of_dicts(list_) -> bool:
    results = []
    for idx in range(len(list_) - 1):
        list0 = list_[idx]
        list1 = list_[idx + 1]
        results.append(compare_dicts(list0, list1))

    if not all(results):
        return False
    return True


import copy


def train(
    model: RealNVP,
    train_set: pd.DataFrame,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nf_dataset = NFDataset(train_set)
    nf_dataloader = torch.utils.data.DataLoader(nf_dataset, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # norm_dist = MultivariateNormal(torch.zeros(batch_size), torch.eye(batch_size))

    losses = []
    epoch_losses = []
    # model_state_dicts = [copy.deepcopy(model.state_dict())]
    for epoch in tqdm(range(epochs)):
        for batch_idx, samples in enumerate(nf_dataloader):
            # if norm_dist.event_shape[0] != samples.shape[0]:
            #     norm_dist = MultivariateNormal(torch.zeros(samples.shape[0]), torch.eye(samples.shape[0]))
            optimizer.zero_grad()

            samples = samples.to(device)
            # y, x = affine_coupling(samples)
            # x, inv_log_det_jac = model.inverse(samples)
            # y, log_prob = model.inverse(samples)
            y, log_prob = model(samples)
            # log_prob = norm_dist.log_prob(inv_log_det_jac)
            loss = torch.div((-torch.sum(0.5 * y ** 2) - log_prob.sum()), len(samples))

            loss.backward()
            optimizer.step()

            # model_state_dicts.append(copy.deepcopy(model.state_dict()))

            losses.append(loss.item())
        epoch_losses.append(sum(losses[epoch * batch_size : batch_size * (1 + epoch)]) / batch_size)
        print(" [%d/%d]\tLoss: %.8f" % (epoch, epochs, epoch_losses[-1]))

    x, _ = model(torch.from_numpy(train_set.values).to(torch.float32).to(device))
    x = x.detach().cpu().numpy()
    model.eval()
    # dist_samples = norm_dist.sample((256, 2))
    y, _ = model(sample_size=(256,))
    y = y.detach().cpu().numpy()
    fig, ax = plt.subplots(1, 2, figsize=(12, 7))
    ax[0].scatter(x[:, 0], x[:, 1])
    ax[0].set_title("Normal dist")
    ax[1].scatter(y[:, 0], y[:, 1])
    ax[1].set_title("Circles")
    fig.tight_layout()
    fig.savefig("./outputs.png")
    a = 0

    return x, model


def main():
    # plt.close()
    csv_files_base_path = Path("./csv-files")
    # plot_moon_and_circls(csv_files_base_path)
    circles_train_set, _ = get_circles_test_and_train_csv_files(csv_files_base_path)
    circles_train_df = pd.read_csv(circles_train_set, index_col=0, header=0)
    # train(circles_train_df, epochs=34)
    # a = 0
    # RealNVPLayer([64], 0.5, 2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RealNVP(neurons_per_hidden_layer=[[64, 64, 64], [64, 64, 64], [64, 64, 64], [64, 64, 64], [64, 64, 64]]).to(
        device
    )
    x, trained_model = train(model, circles_train_df, epochs=1)
    a = 0


if __name__ == "__main__":
    main()
