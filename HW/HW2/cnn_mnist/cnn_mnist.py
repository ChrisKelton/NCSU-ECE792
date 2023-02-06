from pathlib import Path
from typing import Optional, Union

import torch
from sklearn.preprocessing import OneHotEncoder
from torch import nn
from tqdm import tqdm

from mlp_mnist.files import load_mnist_dataset, get_real_path
from mlp_mnist.training_utils import Loss, Accuracy, save_accuracy_plot, save_loss_plot
from mlp_mnist.mlp import BaseMnist


class BaseCnnMnist(nn.Module):
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
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # second convolutional layers
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), stride=(1, 1))
        self.activation2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

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

    def forward(self, x):
        conv1 = self.conv1(x)
        activation1 = self.activation1(conv1)
        maxpool1 = self.maxpool1(activation1)

        conv2 = self.conv2(maxpool1)
        activation2 = self.activation2(conv2)
        maxpool2 = self.maxpool2(activation2)

        flattened = self.flatten(maxpool2)
        linear3 = self.linear3(flattened)
        activation3 = self.activation3(linear3)

        linear4 = self.linear4(activation3)
        activation4 = self.activation4(linear4)

        linear5 = self.linear5(activation4)

        return linear5

    def class_probabilities(self, x):
        return self.output6(x)


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


def main():
    mnist_data_sets_base_path = Path("../../HW1/DATA/MNIST")
    model_output_base_path = Path("./model-0")
    n_epochs = 100
    epochs_to_save_model = 5
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 1e-4
    train: bool = True
    test: bool = True
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
        verbose=verbose,
    )


if __name__ == '__main__':
    main()
