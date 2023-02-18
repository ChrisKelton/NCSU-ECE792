import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from base_mnist import BaseMnist
from files import load_mnist_dataset, get_real_path
from training_utils import Loss, Accuracy, save_accuracy_plot, save_loss_plot

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
    return int(np.round(((input_size + (2 * padding_size) - filter_size) / stride_size) + 0.4999) + 1)


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


def deconv2d_out_shape_(
    input_size: int,
    padding_size: int,
    filter_size: int,
    stride_size: int,
) -> int:
    return int((stride_size * (input_size - 1)) + filter_size - (2 * padding_size))


def deconv2d_out_shape(
    input_shape: Tuple[int, int],
    padding_size: Union[int, TupleType],
    filter_size: Union[int, TupleType],
    stride_size: Union[int, TupleType],
) -> Tuple[int, int]:
    padding_size, filter_size, stride_size = check_tuples_for_shape_determination(padding_size, filter_size, stride_size)
    width = deconv2d_out_shape_(input_shape[0], padding_size[0], filter_size[0], stride_size[0])
    height = deconv2d_out_shape_(input_shape[1], padding_size[1], filter_size[1], stride_size[1])

    return width, height


def deconvnet(
    activation_output: torch.Tensor,
    weights: Union[List[torch.Tensor], torch.Tensor],
    figures_output_path: Path,
    labels: List[Union[str, int]],
    maxunpool=None,
    maxpool_idx=None,
    activation_layer=None,
    upsample: bool = False,
    upsample_size: Optional[Tuple[int, int]] = None,
    only_upsample: bool = False,
    save_outputs: bool = True,
) -> torch.Tensor:
    if activation_output.ndim == 3:
        activation_output = activation_output.unsqueeze(0)
    elif activation_output.ndim != 4:
        raise ValueError(
            "Expecting activation output when reconstructing to have 3 or 4 channels: "
            "(batches, channels, width, height)"
        )
    if only_upsample and not upsample:
        raise RuntimeError(f"If desiring to only upsample outputs, you must also turn on the 'upsample' flag.")
    if upsample and upsample_size is None:
        raise RuntimeError(f"When upsampling, you must specify a desired size to upsample to.")

    if not isinstance(weights, list):
        weights = [weights]

    # determine output shape from deconvolutions
    out_shape = activation_output.shape[-2:]
    if maxunpool is not None:
        if not isinstance(maxunpool, list):
            maxunpool = [maxunpool]
        if not isinstance(maxpool_idx, list):
            maxpool_idx = [maxpool_idx]
    if maxunpool is not None:
        for idx, weight in enumerate(weights[:-1]):
            out_shape = deconv2d_out_shape(out_shape, 0, weight.shape[-2:], 1)[0] * maxunpool[idx].stride[0]
            out_shape = [out_shape] * 2

    deconv_out = torch.empty((activation_output.shape[0], activation_output.shape[1], out_shape[0], out_shape[1]), dtype=activation_output.dtype)
    for idx, act_out in enumerate(torch.swapaxes(activation_output, 0, 1)):
        act_out_zeros = torch.zeros(activation_output.shape, dtype=activation_output.dtype)
        act_out_zeros[:, idx, :, :] = act_out
        for weight_idx, weight in enumerate(weights):
            if weight_idx == len(weights) - 1 or maxunpool is None:
                padding = "same"
            elif maxunpool is not None:
                padding = maxunpool[weight_idx].stride[0] * 2
            else:
                padding = "same"
            deconv_ = F.conv2d(act_out_zeros, weight=torch.swapaxes(weight.transpose(dim0=2, dim1=3), 0, 1), padding=padding).squeeze()
            if weight_idx != len(weights) - 1 and maxunpool is not None:
                deconv_ = maxunpool[weight_idx](deconv_, maxpool_idx[weight_idx])
                deconv_ = activation_layer(deconv_)
            # if len(np.unique(np.ravel(deconv_.detach().numpy()))) < 100:
            #     print(f"number of unique vals in deconv_ = '{len(np.unique(np.ravel(deconv_.detach().numpy())))}'")
            act_out_zeros = deconv_
        deconv_out[:, idx, :, :] = deconv_

        if save_outputs:
            if upsample and only_upsample:  # want double protection for this case
                # want to use nearest neighbors in order to preserve high activation points
                deconv = F.interpolate(deconv_out[:, idx, :, :], size=upsample_size, mode="nearest")
            else:
                deconv = deconv_out[:, idx, :, :]
            if deconv.ndim == 3:
                deconv = deconv.unsqueeze(1)

            for batch_idx, (deconv_batch, label) in enumerate(zip(deconv, labels)):
                label_fig_path = figures_output_path / f"label-{label}--batch-idx-{batch_idx}"
                label_fig_path.mkdir(exist_ok=True, parents=True)
                # for channel, deconv_batch_channel in enumerate(deconv_batch):
                kernel_fig_name = f"kernel-{idx}.png"  #--to--channel-{channel}.png"
                deconv_batch = deconv_batch.squeeze()
                if not only_upsample:
                    kernel_fig_path_default_size = label_fig_path / "default_size"
                    kernel_fig_path_default_size.mkdir(exist_ok=True)
                    plt.imsave(
                        get_real_path(kernel_fig_path_default_size / kernel_fig_name),
                        deconv_batch.detach().numpy(),
                        # deconv_batch_channel.detach().numpy(),
                    )
                if upsample:
                    kernel_fig_path_upsampled = label_fig_path / "upsampled"
                    kernel_fig_path_upsampled.mkdir(exist_ok=True)
                    # deconv_batch_channel = F.interpolate(
                    deconv_batch = F.interpolate(
                        # deconv_batch_channel.unsqueeze(0).unsqueeze(0),
                        deconv_batch.unsqueeze(0).unsqueeze(0),
                        size=upsample_size,
                        mode="nearest",
                    ).squeeze(0).squeeze(0)
                    plt.imsave(
                        get_real_path(kernel_fig_path_upsampled / kernel_fig_name),
                        # deconv_batch_channel.detach().numpy(),
                        deconv_batch.detach().numpy(),
                    )

    return deconv_out


class BaseCnnMnist(nn.Module):

    @dataclass
    class IntermediateOutputs:
        conv1: torch.Tensor
        activation1: torch.Tensor
        maxpool1: torch.Tensor
        maxpool_idx1: torch.Tensor
        conv2: torch.Tensor
        activation2: torch.Tensor
        maxpool2: torch.Tensor
        maxpool_idx2: torch.Tensor
        linear3: torch.Tensor
        activation3: torch.Tensor
        linear4: torch.Tensor
        activation4: torch.Tensor
        linear5: torch.Tensor

    intermediate_outputs: IntermediateOutputs

    def __init__(self):
        super(BaseCnnMnist, self).__init__()
        # CNN Layers
        # first convolutional layers
        # size of output after convolution operation uses the formula
        # O = (I + 2*P - K)/S + 1, where O = output width/height
        #                                I = input width/height
        #                                P = width/height-wise padding size
        #                                K = width/height of the convolution filter
        #                                S = width/height-wise stride size
        # I = 1 * 28 * 28
        # P = 2
        # K = 5 * 5
        # S = 1
        # O = (784 + 2*2 - 25)/1 + 1 = 764
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.activation1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), return_indices=True)

        # second convolutional layers
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), stride=(1, 1))
        self.activation2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), return_indices=True)

        # first fully-connected network
        self.flatten = nn.Flatten()
        self.linear3 = nn.Linear(in_features=400, out_features=120)
        self.activation3 = nn.ReLU()

        # second fully-connected network
        self.linear4 = nn.Linear(in_features=120, out_features=84)
        self.activation4 = nn.ReLU()

        # third fully-connected network
        self.linear5 = nn.Linear(in_features=84, out_features=10)

        self.output6 = nn.Softmax(dim=1)

        self.reconstruction_maxunpool = nn.MaxUnpool2d(kernel_size=(2, 2), stride=(2, 2))

    def forward(self, x):
        conv1 = self.conv1(x)
        activation1 = self.activation1(conv1)
        maxpool1, maxpool_idx1 = self.maxpool1(activation1)

        conv2 = self.conv2(maxpool1)
        activation2 = self.activation2(conv2)
        maxpool2, maxpool_idx2 = self.maxpool2(activation2)

        flattened = self.flatten(maxpool2)
        linear3 = self.linear3(flattened)
        activation3 = self.activation3(linear3)

        linear4 = self.linear4(activation3)
        activation4 = self.activation4(linear4)

        linear5 = self.linear5(activation4)

        self.intermediate_outputs = self.IntermediateOutputs(
            conv1=conv1,
            activation1=activation1,
            maxpool1=maxpool1,
            maxpool_idx1=maxpool_idx1,
            conv2=conv2,
            activation2=activation2,
            maxpool2=maxpool2,
            maxpool_idx2=maxpool_idx2,
            linear3=linear3,
            activation3=activation3,
            linear4=linear4,
            activation4=activation4,
            linear5=linear5,
        )

        return linear5

    def class_probabilities(self, x):
        return self.output6(x)

    def reconstruction(
        self,
        output_path: Union[str, Path],
        original_inputs: torch.Tensor,
        labels_per_batch_idx: List[int],
        batch_idx: Optional[List[int]] = None,
        one_unique_label: bool = True,
        upsample: bool = True,
        upsample_size: Tuple[int, int] = (256, 256)
    ):
        if Path(output_path).suffix in [""]:
            Path(output_path).mkdir(exist_ok=True, parents=True)
        else:
            raise ValueError(f"output_path for reconstruction must be a directory. Got '{output_path}'")

        if one_unique_label:
            # only get one label each from outputs to reconstruct rather than reconstructing many similar outputs
            unique_vals = np.unique(labels_per_batch_idx)
            unique_idx = []
            for val in unique_vals:
                unique_idx.append(labels_per_batch_idx.index(val))
            batch_idx = [batch_idx[idx] for idx in unique_idx]
            labels_per_batch_idx = [labels_per_batch_idx[idx] for idx in unique_idx]

        if batch_idx is not None:
            maxpool2 = self.intermediate_outputs.maxpool2[batch_idx]
            maxpool_idx2 = self.intermediate_outputs.maxpool_idx2[batch_idx]
            maxpool1 = self.intermediate_outputs.maxpool1[batch_idx]
            maxpool_idx1 = self.intermediate_outputs.maxpool_idx1[batch_idx]
            original_inputs = original_inputs[batch_idx]
        else:
            maxpool2 = self.intermediate_outputs.maxpool2
            maxpool_idx2 = self.intermediate_outputs.maxpool_idx2
            maxpool1 = self.intermediate_outputs.maxpool1
            maxpool_idx1 = self.intermediate_outputs.maxpool_idx1

        for idx, (original_input, label) in enumerate(zip(original_inputs, labels_per_batch_idx)):
            fig_path = output_path / f"{label}.png"
            plt.imsave(get_real_path(fig_path), original_input.detach().numpy())
            if upsample:
                upsample_original_input = F.interpolate(original_input.unsqueeze(0).unsqueeze(0), size=upsample_size, mode="nearest").squeeze()
                fig_path = output_path / f"{label}-upsampled.png"
                plt.imsave(get_real_path(fig_path), upsample_original_input.detach().numpy())

        maxunpool1 = self.reconstruction_maxunpool(maxpool1, maxpool_idx1)
        activation1 = self.activation1(maxunpool1)
        layer_fig_path = Path(output_path) / "layer-1"
        layer_fig_path.mkdir(exist_ok=True, parents=True)
        deconvnet(
            activation_output=activation1,
            weights=self.conv1.weight,
            figures_output_path=layer_fig_path,
            labels=labels_per_batch_idx,
            upsample=upsample,
            upsample_size=upsample_size,
        )

        maxunpool2 = self.reconstruction_maxunpool(maxpool2, maxpool_idx2)
        activation2 = self.activation2(maxunpool2)
        layer_fig_path = Path(output_path) / "layer-2"
        layer_fig_path.mkdir(exist_ok=True, parents=True)
        deconv2_weights = [
            self.conv2.weight,
            self.conv1.weight,
        ]
        deconv2 = deconvnet(
            activation_output=activation2,
            weights=deconv2_weights,
            figures_output_path=layer_fig_path,
            maxunpool=self.reconstruction_maxunpool,
            maxpool_idx=maxpool_idx1,
            activation_layer=self.activation1,
            labels=labels_per_batch_idx,
            upsample=upsample,
            upsample_size=upsample_size,
        )


class CnnMnist(BaseMnist):
    early_stopping_th: float = 1e-4
    validation_th_to_decrease_learning_rate: float = 0.01

    def __init__(
        self,
        n_epochs: int = 10,
        epoch_to_save_model: int = 10,
        learning_rate: float = 0.01,
        verbose: bool = False,
    ):
        super(CnnMnist, self).__init__()
        self.n_epochs = n_epochs
        self.epoch_to_save_model = epoch_to_save_model
        self.verbose = verbose
        self.cnn_mnist = BaseCnnMnist()
        self.learning_rate = learning_rate

        self.loss_function = nn.CrossEntropyLoss()  # employs softmax
        self.optimizer = torch.optim.Adam(self.cnn_mnist.parameters(), lr=learning_rate)
        self.output_decisions = 10

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        output_path: Union[str, Path],
        test_loader: Optional[torch.utils.data.DataLoader] = None,
        validation_loader: Optional[torch.utils.data.DataLoader] = None,
    ):
        if Path(output_path).suffix not in [""]:
            Path(output_path).mkdir(exist_ok=True, parents=True)

        train_loss = Loss.init()
        val_loss = Loss.init()
        test_loss = Loss.init()
        train_accuracy = Accuracy.from_output_decisions(self.output_decisions)
        val_accuracy = Accuracy.from_output_decisions(self.output_decisions)
        test_accuracy = Accuracy.from_output_decisions(self.output_decisions)

        epoch_tqdm = tqdm(total=self.n_epochs, desc="Epoch", position=0)
        for epoch in range(self.n_epochs):
            if test_loader is not None:
                self.cnn_mnist.eval()

                for test_batch_idx, test_data in enumerate(test_loader):
                    test_inputs, test_targets = test_data
                    # unsqueeze so dimensions match what CNN anticipates
                    # Will become (batch_size, 1 [channels], 28, 28)
                    test_outputs = self.cnn_mnist.forward(torch.unsqueeze(test_inputs, dim=1))
                    loss_val = self.loss_function(test_outputs, test_targets)
                    test_loss += loss_val.item()
                    output_class_probabilities = self.cnn_mnist.class_probabilities(test_outputs)
                    test_accuracy.compare_batch(targets=test_targets, outputs=output_class_probabilities)
                test_accuracy.update_for_epoch()
                test_loss.update_for_epoch()
                self.cnn_mnist.train()

            for train_batch_idx, train_data in enumerate(train_loader):
                train_inputs, train_targets = train_data
                self.optimizer.zero_grad()
                train_outputs = self.cnn_mnist.forward(torch.unsqueeze(train_inputs, dim=1))
                loss_val = self.loss_function(train_outputs, train_targets)
                loss_val.backward()
                self.optimizer.step()
                train_loss += loss_val.item()
                output_class_probabilities = self.cnn_mnist.class_probabilities(train_outputs)
                train_accuracy.compare_batch(targets=train_targets, outputs=output_class_probabilities)
            train_accuracy.update_for_epoch()
            train_loss.update_for_epoch()

            if validation_loader is not None:
                self.cnn_mnist.eval()

                for val_batch_idx, val_data in enumerate(validation_loader):
                    val_inputs, val_targets = val_data
                    val_outputs = self.cnn_mnist.forward(torch.unsqueeze(val_inputs, dim=1))
                    loss_val = self.loss_function(val_outputs, val_targets)
                    val_loss += loss_val.item()
                    output_class_probabilities = self.cnn_mnist.class_probabilities(val_outputs)
                    val_accuracy.compare_batch(targets=val_targets, outputs=output_class_probabilities)
                val_accuracy.update_for_epoch()
                val_loss.update_for_epoch()
                self.cnn_mnist.train()
                if len(val_accuracy.acc_vals_per_epoch) > 3:
                    if (
                            (val_loss.loss_vals_per_epoch[epoch] > val_loss.loss_vals_per_epoch[epoch - 1]) and
                            (val_loss.loss_vals_per_epoch[epoch] > val_loss.loss_vals_per_epoch[epoch - 2]) and
                            (val_loss.loss_vals_per_epoch[epoch] > val_loss.loss_vals_per_epoch[epoch - 3])) or \
                            (abs(val_loss.loss_vals_per_epoch[epoch] - val_loss.loss_vals_per_epoch[epoch - 1]) < self.validation_th_to_decrease_learning_rate):
                        self.learning_rate /= 10
                        self.validation_th_to_decrease_learning_rate /= 10
                        epoch_tqdm.write(f"Learning rate update to '{self.learning_rate}'")
                        for g in self.optimizer.param_groups:
                            g["lr"] = self.learning_rate

            if self.verbose:
                epoch_tqdm.write(f"Train Loss = {train_loss.current_loss}")
                if validation_loader is not None:
                    epoch_tqdm.write(f"Validation Loss = {val_loss.current_loss}")
                if test_loader is not None:
                    epoch_tqdm.write(f"Test Loss = {test_loss.current_loss}")
                epoch_tqdm.write(f"Train Accuracy = {train_accuracy.acc_vals_per_epoch[epoch]}")
                if validation_loader is not None:
                    epoch_tqdm.write(f"Validation Accuracy = {val_accuracy.acc_vals_per_epoch[epoch]}")
                if test_loader is not None:
                    epoch_tqdm.write(f"Test Accuracy = {test_accuracy.acc_vals_per_epoch[epoch]}")
            if len(train_loss.loss_vals_per_epoch) > 2:
                if (abs(train_loss.loss_vals_per_epoch[epoch] - train_loss.loss_vals_per_epoch[epoch - 1]) < self.early_stopping_th) and (abs(train_loss.loss_vals_per_epoch[epoch] - train_loss.loss_vals_per_epoch[epoch - 2]) < self.early_stopping_th):
                    epoch_tqdm.write(f"Training Loss has reached a minimum at '{train_loss.loss_vals_per_epoch[epoch]}'"
                                     f". Ending Testing.")
                    break
            epoch_tqdm.update(1)

            if self.epoch_to_save_model is not None and (epoch % self.epoch_to_save_model == 0 and epoch > 0):
                (
                    epoch_output_path,
                    train_loss_json_path,
                    train_accuracy_json_path,
                    test_loss_json_path,
                    test_accuracy_json_path,
                    confusion_matrix_path,
                    accuracy_plot_path,
                    loss_plot_path,
                ) = self.get_output_paths_(output_path, epoch)
                torch.save(self.cnn_mnist.state_dict(), epoch_output_path)
                train_loss.save_json(train_loss_json_path)
                train_accuracy.save_json(train_accuracy_json_path)
                train_accuracy.save_confusion_matrix(confusion_matrix_path, "0123456789")
                if test_loader is not None:
                    test_loss.save_json(test_loss_json_path)
                    test_accuracy.save_json(test_accuracy_json_path)
                    test_acc_per_epoch = test_accuracy.acc_vals_per_epoch
                    test_loss_per_epoch = test_loss.loss_vals_per_epoch
                else:
                    test_acc_per_epoch = None
                    test_loss_per_epoch = None
                if validation_loader is not None:
                    val_acc_per_epoch = val_accuracy.acc_vals_per_epoch
                    val_loss_per_epoch = val_loss.loss_vals_per_epoch
                else:
                    val_acc_per_epoch = None
                    val_loss_per_epoch = None
                save_accuracy_plot(train_accuracy.acc_vals_per_epoch, accuracy_plot_path, test_acc_per_epoch, val_acc_per_epoch)
                save_loss_plot(train_loss.loss_vals_per_epoch, loss_plot_path, test_loss_per_epoch, val_loss_per_epoch)

        self.learning_rate = 0.01
        self.validation_th_to_decrease_learning_rate = 0.01
        print("Training has finished.")
        (
            output_path,
            train_loss_json_path,
            train_accuracy_json_path,
            test_loss_json_path,
            test_accuracy_json_path,
            confusion_matrix_path,
            accuracy_plot_path,
            loss_plot_path,
        ) = self.get_output_paths_(output_path, epoch)
        print(f"Saving model to '{output_path}'")
        torch.save(self.cnn_mnist.state_dict(), output_path)

        self.train_loss = train_loss
        self.train_accuracy = train_accuracy
        self.train_loss.save_json(train_loss_json_path)
        self.train_accuracy.save_json(train_accuracy_json_path)
        self.train_accuracy.save_confusion_matrix(confusion_matrix_path, "0123456789")
        if test_loader is not None:
            self.test_loss = test_loss
            self.test_accuracy = test_accuracy
            self.test_loss.save_json(test_loss_json_path)
            self.test_accuracy.save_json(test_accuracy_json_path)
        if validation_loader is None:
            val_loss = None
            val_accuracy = None
        else:
            val_loss = val_loss.loss_vals_per_epoch
            val_accuracy = val_accuracy.acc_vals_per_epoch
        save_accuracy_plot(
            self.train_accuracy.acc_vals_per_epoch, accuracy_plot_path, self.test_accuracy.acc_vals_per_epoch, val_accuracy
        )
        save_loss_plot(self.train_loss.loss_vals_per_epoch, loss_plot_path, self.test_loss.loss_vals_per_epoch, val_loss)

    def reconstruct(
        self,
        images,
        labels,
        reconstruction_output_path: Union[str, Path],
        labels_to_reconstruct: List[Union[int, str]],
        model_path: Optional[Union[str, Path]] = None,
        one_unique_label: bool = True,
        upsample: bool = True,
        upsample_size: Tuple[int, int] = (256, 256),
    ):
        print("****STARTING RECONSTRUCTION****")
        if reconstruction_output_path.exists():
            shutil.rmtree(get_real_path(reconstruction_output_path))
        Path(reconstruction_output_path).mkdir(exist_ok=True, parents=True)
        dataloader = self.preprocess_data(images, labels, batch_size=len(images))
        self.test(
            test_loader=dataloader,
            model_path=model_path,
            reconstruct=True,
            reconstruction_output_path=reconstruction_output_path,
            labels_to_reconstruct=labels_to_reconstruct,
            n_epoch=1,
            one_unique_label=one_unique_label,
            upsample_reconstruction_activations=upsample,
            upsample_reconstruction_size=upsample_size,
            no_accuracy_plot=True,
        )

    def test(
        self,
        test_loader: torch.utils.data.DataLoader,
        model_path: Optional[Union[str, Path]] = None,
        output_path: Optional[Path] = None,
        reconstruct: bool = False,
        reconstruction_output_path: Optional[Union[str, Path]] = None,
        labels_to_reconstruct: Optional[List[Union[int, str]]] = None,
        n_epoch: Optional[int] = None,
        one_unique_label: bool = True,
        upsample_reconstruction_activations: bool = False,
        upsample_reconstruction_size: Tuple[int, int] = (256, 256),
        no_accuracy_plot: bool = False,
    ):
        if model_path is not None:
            model = BaseCnnMnist()
            model.load_state_dict(torch.load(model_path))
        else:
            if reconstruction_output_path is None and output_path is None:
                raise RuntimeError(f"Expected either model_path, reconstruction_output_path or output_path to be passed through.")
            elif reconstruction_output_path is not None:
                model_path = reconstruction_output_path
            else:
                model_path = output_path
            model = self.cnn_mnist
        model.eval()
        if reconstruct:
            if reconstruction_output_path is None:
                raise ValueError(f"When reconstructing convolution outputs, 'reconstruction_output_path' must be set!")
        if labels_to_reconstruct is not None:
            labels_to_reconstruct = [int(x) for x in labels_to_reconstruct]

        accuracy = Accuracy.from_output_decisions(self.output_decisions)
        if n_epoch is None:
            n_epoch = self.n_epochs
        epoch_tqdm = tqdm(total=n_epoch, desc="Epoch", position=0)

        for epoch in range(n_epoch):
            for batch_idx, data in enumerate(test_loader):
                inputs, targets = data
                outputs = model.forward(torch.unsqueeze(inputs, dim=1))
                class_probabilities = model.class_probabilities(outputs)
                accurate_idx = accuracy.compare_batch(targets=targets, outputs=class_probabilities)
                if reconstruct:
                    if labels_to_reconstruct is not None:
                        accurate_batch_idx = [acc_idx[0] for acc_idx in accurate_idx if acc_idx[1] in labels_to_reconstruct]
                        accurate_labels = [acc_idx[1] for acc_idx in accurate_idx if acc_idx[1] in labels_to_reconstruct]
                    else:
                        accurate_batch_idx = [acc_idx[0] for acc_idx in accurate_idx]
                        accurate_labels = [acc_idx[1] for acc_idx in accurate_idx]
                    model.reconstruction(
                        reconstruction_output_path,
                        original_inputs=inputs,
                        labels_per_batch_idx=accurate_labels,
                        batch_idx=accurate_batch_idx,
                        one_unique_label=one_unique_label,
                        upsample=upsample_reconstruction_activations,
                        upsample_size=upsample_reconstruction_size,
                    )
            accuracy.update_for_epoch()
            if self.verbose:
                epoch_tqdm.write(f"Accuracy = '{accuracy.acc_vals_per_epoch[epoch]}'")
            epoch_tqdm.update(1)

        print("Testing has finished.")
        (
            _,
            _,
            _,
            _,
            test_accuracy_json_path,
            confusion_matrix_path,
            accuracy_plot_path,
            _,
        ) = self.get_output_paths_(Path(model_path).parent / "testing", epoch, no_loss=True, no_models=True)

        self.test_accuracy = accuracy
        self.test_accuracy.save_json(test_accuracy_json_path)
        self.test_accuracy.save_confusion_matrix(confusion_matrix_path, "0123456789")
        if not no_accuracy_plot:
            save_accuracy_plot(self.test_accuracy.acc_vals_per_epoch, accuracy_plot_path, plot_labels="test")


def cnn_mnist_run(
    mnist_data_sets_base_path: Path,
    model_output_base_path: Path,
    n_epochs: int = 100,
    epochs_to_save_model: int = 5,
    batch_size_train: int = 64,
    batch_size_test: int = 1000,
    learning_rate: float = 0.01,
    validation_split: float = 0.1,
    shuffle_dataset: bool = False,
    random_seed: Optional[int] = None,
    train: bool = False,
    test: bool = True,
    model_path_to_test: Optional[Path] = None,
    reconstruct: bool = False,
    reconstruction_output_path: Optional[Path] = None,
    labels_to_reconstruct: Optional[List[Union[str, int]]] = None,
    one_unique_label_for_reconstruction: bool = True,
    upsample_reconstruction_activations: bool = True,
    upsample_reconstruction_size: Tuple[int, int] = (256, 256),
    test_using_training_set: bool = False,
    reconstruct_using_training_set: bool = False,
    verbose: bool = False,
):
    if not Path(model_output_base_path).suffix in [""]:
        raise RuntimeError(f"model_output_base_path must be a directory. Not a specific file.")
    mndata = load_mnist_dataset(mnist_data_sets_base_path)
    if mndata is None:
        raise RuntimeError(f"Failed to load in mnist dataset")

    cnn_mnist = CnnMnist(
        n_epochs=n_epochs,
        epoch_to_save_model=epochs_to_save_model,
        learning_rate=learning_rate,
        verbose=verbose,
    )
    if train:
        if test:
            test_images, test_labels = mndata.load_testing()
            test_loader = cnn_mnist.preprocess_data(test_images, test_labels, batch_size_test)
        else:
            test_loader = None
        train_images, train_labels = mndata.load_training()
        indices = np.arange(0, len(train_images))
        split = int(np.floor(validation_split * len(train_images)))
        if shuffle_dataset:
            if random_seed is not None:
                np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, validation_indices = indices[split:], indices[:split]
        validation_images = [train_images[val_idx] for val_idx in validation_indices]
        validation_labels = [train_labels[val_idx] for val_idx in validation_indices]
        validation_loader = cnn_mnist.preprocess_data(validation_images, validation_labels, batch_size_train, shuffle_dataset)
        train_images = [train_images[train_idx] for train_idx in train_indices]
        train_labels = [train_labels[train_idx] for train_idx in train_indices]
        train_loader = cnn_mnist.preprocess_data(train_images, train_labels, batch_size_train, shuffle_dataset)
        cnn_mnist.train(train_loader, get_real_path(model_output_base_path), test_loader, validation_loader)

    if test and model_path_to_test is not None:
        if test_using_training_set:
            test_images, test_labels = mndata.load_training()
        else:
            test_images, test_labels = mndata.load_testing()
        test_loader = cnn_mnist.preprocess_data(test_images, test_labels, batch_size_test)
        cnn_mnist.test(
            test_loader,
            get_real_path(model_path_to_test),
        )

    if reconstruct:
        if reconstruct_using_training_set:
            images, labels = mndata.load_training()
        else:
            images, labels = mndata.load_testing()
        cnn_mnist.reconstruct(
            images=images,
            labels=labels,
            reconstruction_output_path=reconstruction_output_path,
            labels_to_reconstruct=labels_to_reconstruct,
            model_path=model_path_to_test,
            one_unique_label=one_unique_label_for_reconstruction,
            upsample=upsample_reconstruction_activations,
            upsample_size=upsample_reconstruction_size,
        )


def main():
    mnist_data_sets_base_path = Path("DATA/MNIST")
    model_output_base_path = Path("./model-0")
    model_path_to_test = Path("./model-1/models/model-2023-02-09--00-29-12--15.pt")
    # model_path_to_test = None
    n_epochs = 100
    epochs_to_save_model = 5
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    train: bool = False
    test: bool = False
    reconstruct: bool = True
    reconstruction_output_path: Path = model_output_base_path / "reconstruction"
    labels_to_reconstruct: List[str] = ["0", "1", "5", "8"]
    one_unique_label_for_reconstruction: bool = True
    upsample_reconstruction_activations: bool = True
    upsample_reconstruction_size: Tuple[int, int] = (256, 256)
    test_using_training_set: bool = False
    verbose: bool = True
    cnn_mnist_run(
        mnist_data_sets_base_path=mnist_data_sets_base_path,
        model_output_base_path=model_output_base_path,
        n_epochs=n_epochs,
        epochs_to_save_model=epochs_to_save_model,
        batch_size_train=batch_size_train,
        batch_size_test=batch_size_test,
        learning_rate=learning_rate,
        train=train,
        test=test,
        model_path_to_test=model_path_to_test,
        reconstruct=reconstruct,
        reconstruction_output_path=reconstruction_output_path,
        labels_to_reconstruct=labels_to_reconstruct,
        one_unique_label_for_reconstruction=one_unique_label_for_reconstruction,
        upsample_reconstruction_activations=upsample_reconstruction_activations,
        upsample_reconstruction_size=upsample_reconstruction_size,
        test_using_training_set=test_using_training_set,
        verbose=verbose,
    )


if __name__ == '__main__':
    main()
