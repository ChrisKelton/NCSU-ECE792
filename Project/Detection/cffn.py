__all__ = ["CFFN"]
from typing import Union, Tuple, List, Callable

import torch
import torch.nn as nn

TupleType = Union[Tuple[Union[int, float], Union[int, float]], List[Union[int, float]], torch.Size]


def check_tuples_for_shape_determination(
    padding_size: Union[int, TupleType],
    filter_size: Union[int, TupleType],
    stride_size: Union[int, TupleType],
) -> Tuple[TupleType, TupleType, TupleType]:
    if isinstance(padding_size, int):
        padding_size = [padding_size] * 2
    if len(padding_size) != 2:
        raise ValueError(f"padding_size length is not '2'. Got '{len(padding_size)}'")
    if isinstance(filter_size, int):
        filter_size = [filter_size] * 2
    if len(filter_size) != 2:
        raise ValueError(f"filter_size length is not '2'. Got '{len(filter_size)}'")
    if isinstance(stride_size, int):
        if stride_size == 0:
            stride_size = 1
        stride_size = [stride_size] * 2
    elif 0 in stride_size:
        stride_size = [1] * 2
    if len(stride_size) != 2:
        raise ValueError(f"stride_size length is not '2'. Got '{len(stride_size)}'")

    return padding_size, filter_size, stride_size


def conv2d_out_shape_(
    input_size: int,
    padding_size: int,
    filter_size: int,
    stride_size: int,
) -> int:
    return int(((input_size + (2 * padding_size) - filter_size) / stride_size) + 1)


def conv2d_out_shape(
    input_shape: Tuple[int, int],
    padding_size: Union[int, TupleType],
    filter_size: Union[int, TupleType],
    stride_size: Union[int, TupleType],
) -> Tuple[int, int]:
    padding_size, filter_size, stride_size = check_tuples_for_shape_determination(padding_size, filter_size, stride_size)
    width = conv2d_out_shape_(input_shape[0], padding_size[0], filter_size[0], stride_size[0])
    height = conv2d_out_shape_(input_shape[1], padding_size[1], filter_size[1], stride_size[1])

    return width, height


class DenseBlock(nn.Module):
    def __init__(
        self,
        n_conv: int,
        in_channels: int,
        out_channels: int,
        growth_rate: int,
        transition_layer_theta: float,
    ):
        super().__init__()
        self.modules = []
        self.batch_norms = []
        for idx in range(n_conv):
            in_channels_with_growth = in_channels + (idx * growth_rate)
            out_channels_with_growth = in_channels + ((idx + 1) * growth_rate)
            self.modules.append(
                [
                    nn.Conv2d(
                        in_channels=in_channels_with_growth,
                        out_channels=in_channels_with_growth * 2,
                        kernel_size=(1, 1),
                        padding=0,
                        stride=(1, 1),
                    ),
                    nn.Conv2d(
                        in_channels=in_channels_with_growth * 2,
                        out_channels=growth_rate,
                        kernel_size=(3, 3),
                        padding=1,
                        stride=(1, 1),
                    ),
                ]
            )
            self.batch_norms.append(nn.BatchNorm2d(out_channels_with_growth))
        trans_kernel_size = int(1/transition_layer_theta)
        self.trans_layer = nn.MaxPool3d(kernel_size=(trans_kernel_size, 1, 1))
        self.activation_func = nn.ReLU()

    def forward(self, x):
        layer_outputs = [x]
        for d_block, batch_norm in zip(self.modules, self.batch_norms):
            for module in d_block:
                x = module(x)
            x = torch.concat((x, layer_outputs[-1]), dim=1)
            x = batch_norm(x)
            x = self.activation_func(x)
            layer_outputs.append(x)

        x = self.trans_layer(x)
        return x


class CFFNEnergyFunction(nn.Module):
    def __init__(self, m_th: float = 0.5):
        super().__init__()
        self.m_th = m_th
        self.energy_function = torch.nn.MSELoss()

    def forward(self, img0_batch, img1_batch, pairs_indicator):
        E_w = self.energy_function(img0_batch, img1_batch)
        loss = (0.5 * (pairs_indicator * (E_w ** 2))) + ((1 - pairs_indicator) * torch.max(0, nn.functional.mse_loss(self.m_th, E_w)))

        return loss


class CFFN(nn.Module):
    dense_conv1_out = None
    dense_conv2_out = None
    dense_conv3_out = None
    dense_conv4_out = None
    conv5_out = None
    batch_norm5_out = None
    activation5_out = None

    def __init__(
        self,
        input_image_shape: Tuple[int, int],
        growth_rate: int = 24,
        transition_layer_theta: float = 0.5,
        m_th: float = 0.5,  # threshold for contrastive loss
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=3, out_channels=48, kernel_size=(7, 7), stride=(4, 4))
        self.batch_norm0 = nn.BatchNorm2d(48)
        self.activation0 = nn.ReLU()
        self.dense_conv1 = DenseBlock(
            n_conv=2,
            in_channels=48,
            out_channels=48,
            growth_rate=growth_rate,
            transition_layer_theta=transition_layer_theta,
        )
        self.dense_conv2 = DenseBlock(
            n_conv=3,
            in_channels=48,
            out_channels=60,
            growth_rate=24,
            transition_layer_theta=transition_layer_theta,
        )
        self.dense_conv3 = DenseBlock(
            n_conv=4,
            in_channels=60,
            out_channels=78,
            growth_rate=24,
            transition_layer_theta=transition_layer_theta,
        )
        self.dense_conv4 = DenseBlock(
            n_conv=2,
            in_channels=78,
            out_channels=126,
            growth_rate=24,
            transition_layer_theta=1,
        )
        self.conv5 = nn.Conv2d(in_channels=126, out_channels=128, kernel_size=(3, 3))
        self.batch_norm5 = nn.BatchNorm2d(128)
        self.activation5 = nn.ReLU()
        # conv_out_shape = conv2d_out_shape(
        #     input_shape=input_image_shape,
        #     padding_size=0,
        #     filter_size=7,
        #     stride_size=4,
        # )
        self.fully_connected: Callable = lambda in_conv_n_channels, conv_shape0, conv_shape1: nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_conv_n_channels * conv_shape0 * conv_shape1, 128),
                nn.ReLU(),
            )
        self.loss = CFFNEnergyFunction(m_th=m_th)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = self.conv0(x)
        x = self.batch_norm0(x)
        x = self.activation0(x)
        self.dense_conv1_out = self.dense_conv1(x)
        self.dense_conv2_out = self.dense_conv2(self.dense_conv1_out)
        self.dense_conv3_out = self.dense_conv3(self.dense_conv2_out)
        self.dense_conv4_out = self.dense_conv4(self.dense_conv3_out)
        self.conv5_out = self.conv5(self.dense_conv4_out)
        self.batch_norm5_out = self.batch_norm5(self.conv5_out)
        self.activation5_out = self.activation5(self.batch_norm5_out)

        fn5_module = self.fully_connected(*self.activation5_out.shape[1:])
        fn5 = fn5_module(self.activation5_out)

        fn4_module = self.fully_connected(*self.dense_conv4_out.shape[1:])
        fn4 = fn4_module(self.dense_conv4_out)

        fn3_module = self.fully_connected(*self.dense_conv3_out.shape[1:])
        fn3 = fn3_module(self.dense_conv3_out)

        # fn2_module = self.fully_connected(60)
        # fn2 = fn2_module(self.dense_conv2_out)
        #
        # fn1_module = self.fully_connected(48)
        # fn1 = fn1_module(self.dense_conv1_out)

        # x_out = torch.cat((fn4, fn3, fn2, fn1), dim=1)
        x_out = torch.cat((fn5, fn4, fn3), dim=1)

        return x_out

    def loss_back_grad(self, img0_batch, img1_batch, pairs_indicators):
        criterion = self.loss(img0_batch, img1_batch, pairs_indicators)
        criterion.backward()
        self.optimizer.step()