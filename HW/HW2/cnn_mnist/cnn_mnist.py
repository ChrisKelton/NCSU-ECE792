from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from mlp_mnist.files import load_mnist_dataset, get_real_path
from mlp_mnist.mlp import BaseMnist
from mlp_mnist.training_utils import Loss, Accuracy, save_accuracy_plot, save_loss_plot


@dataclass
class CnnMnistReconstructionLayers:
    maxunpool: torch.nn.Module
    deconv2: torch.nn.Module
    deconv1: torch.nn.Module

    @classmethod
    def init(cls) -> "CnnMnistReconstructionLayers":
        maxunpool = nn.MaxUnpool2d(kernel_size=(2, 2), stride=(2, 2))
        deconv2 = nn.ConvTranspose2d(16, 16, kernel_size=(5, 5), stride=(1, 1))
        deconv2.bias = None
        deconv1 = nn.ConvTranspose2d(6, 6, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        deconv1.bias = None

        return cls(
            maxunpool=maxunpool,
            deconv2=deconv2,
            deconv1=deconv1,
        )


def deconvnet(
    activation_output: torch.Tensor,
    weights: torch.Tensor,
    figures_output_path: Path,
    labels: List[Union[str, int]],
    upsample: bool = False,
    upsample_size: Optional[Tuple[int, int]] = None,
    only_upsample: bool = False,
):
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

    for idx, weight in enumerate(weights):
        deconv = F.conv_transpose2d(activation_output[:, idx, :, :].unsqueeze(1), weight=weight.unsqueeze(0))

        if upsample and only_upsample:  # want double protection for this case
            # want to use nearest neighbors in order to preserve high activation points
            deconv = F.interpolate(deconv, size=upsample_size, mode="nearest")

        for batch_idx, (deconv_batch, label) in enumerate(zip(deconv, labels)):
            label_fig_path = figures_output_path / f"label-{label}--batch-idx-{batch_idx}"
            label_fig_path.mkdir(exist_ok=True, parents=True)
            for in_channel, deconv_batch_channel in enumerate(deconv_batch):
                kernel_fig_name = f"kernel-{idx}--to--in-channel-{in_channel}.png"
                if not only_upsample:
                    kernel_fig_path_default_size = label_fig_path / "default_size"
                    kernel_fig_path_default_size.mkdir(exist_ok=True)
                    plt.imsave(
                        get_real_path(kernel_fig_path_default_size / kernel_fig_name),
                        deconv_batch_channel.detach().numpy(),
                    )
                if upsample:
                    kernel_fig_path_upsampled = label_fig_path / "upsampled"
                    kernel_fig_path_upsampled.mkdir(exist_ok=True)
                    deconv_batch_channel = F.interpolate(
                        deconv_batch_channel.unsqueeze(0).unsqueeze(0),
                        size=upsample_size,
                        mode="nearest",
                    ).squeeze(0).squeeze(0)
                    plt.imsave(
                        get_real_path(kernel_fig_path_upsampled / kernel_fig_name),
                        deconv_batch_channel.detach().numpy(),
                    )


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

        self.reconstruction_layers = CnnMnistReconstructionLayers.init()

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
        labels_per_batch_idx: List[int],
        batch_idx: Optional[List[int]] = None,
        one_unique_label: bool = True,
    ):
        if Path(output_path).suffix in [""]:
            Path(output_path).mkdir(exist_ok=True, parents=True)
        else:
            raise ValueError(f"output_path for reconstruction must be a directory. Got '{output_path}'")

        if one_unique_label:
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
        else:
            maxpool2 = self.intermediate_outputs.maxpool2
            maxpool_idx2 = self.intermediate_outputs.maxpool_idx2
            maxpool1 = self.intermediate_outputs.maxpool1
            maxpool_idx1 = self.intermediate_outputs.maxpool_idx1

        maxunpool1 = self.reconstruction_layers.maxunpool(maxpool1, maxpool_idx1)
        activation1 = self.activation1(maxunpool1)
        layer_fig_path = Path(output_path) / "layer-1"
        layer_fig_path.mkdir(exist_ok=True, parents=True)
        deconvnet(
            activation_output=activation1,
            weights=self.conv1.weight,
            figures_output_path=layer_fig_path,
            labels=labels_per_batch_idx,
            upsample=True,
            upsample_size=(256, 256),
        )

        maxunpool2 = self.reconstruction_layers.maxunpool(maxpool2, maxpool_idx2)
        activation2 = self.activation2(maxunpool2)
        layer_fig_path = Path(output_path) / "layer-2"
        layer_fig_path.mkdir(exist_ok=True, parents=True)
        deconvnet(
            activation_output=activation2,
            weights=self.conv2.weight,
            figures_output_path=layer_fig_path,
            labels=labels_per_batch_idx,
            upsample=True,
            upsample_size=(256, 256),
        )


class CnnMnist(BaseMnist):

    def __init__(
        self,
        n_epochs: int = 10,
        epoch_to_save_model: int = 10,
        learning_rate: float = 1e-4,
        verbose: bool = False,
    ):
        super(CnnMnist, self).__init__()
        self.n_epochs = n_epochs
        self.epoch_to_save_model = epoch_to_save_model
        self.verbose = verbose
        self.cnn_mnist = BaseCnnMnist()

        self.loss_function = nn.CrossEntropyLoss()  # employs softmax
        self.optimizer = torch.optim.Adam(self.cnn_mnist.parameters(), lr=learning_rate)
        self.output_decisions = 10

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        output_path: Union[str, Path],
        test_loader: Optional[torch.utils.data.DataLoader] = None,
    ):
        if Path(output_path).suffix not in [""]:
            Path(output_path).mkdir(exist_ok=True, parents=True)

        train_loss = Loss.init()
        test_loss = Loss.init()
        train_accuracy = Accuracy.from_output_decisions(self.output_decisions)
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

            for batch_idx, train_data in enumerate(train_loader):
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

            if self.verbose:
                epoch_tqdm.write(f"Train Loss = {train_loss.current_loss}")
                if test_loader is not None:
                    epoch_tqdm.write(f"Test Loss = {test_loss.current_loss}")
                epoch_tqdm.write(f"Train Accuracy = {train_accuracy.acc_vals_per_epoch[epoch]}")
                if test_loader is not None:
                    epoch_tqdm.write(f"Test Accuracy = {test_accuracy.acc_vals_per_epoch[epoch]}")
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
                save_accuracy_plot(train_accuracy.acc_vals_per_epoch, accuracy_plot_path, test_acc_per_epoch)
                save_loss_plot(train_loss.loss_vals_per_epoch, loss_plot_path, test_loss_per_epoch)

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
        save_accuracy_plot(
            self.train_accuracy.acc_vals_per_epoch, accuracy_plot_path, self.test_accuracy.acc_vals_per_epoch
        )
        save_loss_plot(self.train_loss.loss_vals_per_epoch, loss_plot_path, self.test_loss.loss_vals_per_epoch)

    def reconstruct(
        self,
        images,
        labels,
        reconstruction_output_path: Union[str, Path],
        labels_to_reconstruct: List[Union[int, str]],
        model_path: Optional[Union[str, Path]] = None,
    ):
        dataloader = self.preprocess_data(images, labels, batch_size=len(images))
        self.test(
            dataloader,
            model_path,
            True,
            reconstruction_output_path,
            labels_to_reconstruct,
            n_epoch=1,
        )

    def test(
        self,
        test_loader: torch.utils.data.DataLoader,
        model_path: Optional[Union[str, Path]] = None,
        reconstruct: bool = False,
        reconstruction_output_path: Optional[Union[str, Path]] = None,
        labels_to_reconstruct: Optional[List[Union[int, str]]] = None,
        n_epoch: Optional[int] = None,
    ):
        if model_path is not None:
            model = BaseCnnMnist()
            model.load_state_dict(torch.load(model_path))
        else:
            model = self.cnn_mnist
        model.eval()
        if reconstruct:
            if reconstruction_output_path is None:
                raise ValueError(f"When reconstructing convolution outputs, 'reconstruction_output_path' must be set!")
        if labels_to_reconstruct is not None:
            labels_to_reconstruct = [int(x) for x in labels_to_reconstruct]

        accuracy = Accuracy.from_output_decisions(self.output_decisions)
        epoch_tqdm = tqdm(total=self.n_epochs, desc="Epoch", position=0)
        if n_epoch is None:
            n_epoch = self.n_epochs

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
                        labels_per_batch_idx=accurate_labels,
                        batch_idx=accurate_batch_idx,
                        one_unique_label=True,
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
        ) = self.get_output_paths_(Path(model_path).parent / "testing", epoch)

        self.test_accuracy = accuracy
        self.test_accuracy.save_json(test_accuracy_json_path)
        self.test_accuracy.save_confusion_matrix(confusion_matrix_path, "0123456789")
        save_accuracy_plot(self.test_accuracy.acc_vals_per_epoch, accuracy_plot_path, plot_labels="test")


def cnn_mnist_run(
    mnist_data_sets_base_path: Path,
    model_output_base_path: Path,
    n_epochs: int = 100,
    epochs_to_save_model: int = 5,
    batch_size_train: int = 64,
    batch_size_test: int = 1000,
    learning_rate: float = 1e-4,
    train: bool = False,
    test: bool = True,
    model_path_to_test: Optional[Path] = None,
    reconstruct: bool = False,
    reconstruction_output_path: Optional[Path] = None,
    labels_to_reconstruct: Optional[List[Union[str, int]]] = None,
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
        train_loader = cnn_mnist.preprocess_data(train_images, train_labels, batch_size_train)
        cnn_mnist.train(train_loader, get_real_path(model_output_base_path), test_loader)

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
        )


def main():
    mnist_data_sets_base_path = Path("../../HW1/DATA/MNIST")
    model_output_base_path = Path("./model-0")
    model_path_to_test = Path("./model-0/models/model-2023-02-06--15-29-57--99.pt")
    n_epochs = 100
    epochs_to_save_model = 5
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 1e-4
    train: bool = False
    test: bool = True
    reconstruct: bool = True
    reconstruction_output_path = Path("./model-0/reconstruction")
    labels_to_reconstruct = ["0", "1", "5", "8"]
    test_using_training_set: bool = True
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
        test_using_training_set=test_using_training_set,
        verbose=verbose,
    )


if __name__ == '__main__':
    main()
