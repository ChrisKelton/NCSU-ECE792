__all__ = ["BaseMnist"]
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, Tuple

import numpy as np
import torch
import torchvision
from sklearn.preprocessing import OneHotEncoder
from torch import nn
from tqdm import tqdm

from mlp_mnist.files import load_mnist_dataset, get_real_path
from mlp_mnist.training_utils import Loss, Accuracy, save_accuracy_plot, save_loss_plot


class MultiLayerPerceptron(nn.Module):
    def __init__(self, number_of_inputs: int, number_of_outputs: int):
        super().__init__()  # initialize parent class
        self.layers = nn.Sequential(
            nn.Flatten(),  # flatten input to match expected number of inputs
            nn.Linear(
                number_of_inputs, 256
            ),  # first layer of model that connects 'number_of_inputs' neurons to '256' neurons
            nn.ReLU(),  # activation function
            nn.Linear(256, 64),  # next layer of model that connects '256' neurons to '64' neurons
            nn.ReLU(),  # activation function
            nn.Linear(64, 32),  # next layer of model that connects '64' neurons to '32' neurons
            nn.ReLU(),  # activation function
            nn.Linear(
                32, number_of_outputs
            ),  # last layer of model that connects '32' neurons to 'number_of_outputs' neurons
        )

    def forward(self, x):
        return self.layers(x)  # run self.layers() that is defined in the __init__() in sequential order on our input x


class MultiLayerPerceptronDropout(nn.Module):
    def __init__(self, number_of_inputs: int, number_of_outputs: int):
        super().__init__()  # initialize parent class
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(number_of_inputs, 256),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(32, number_of_outputs),
        )

    def forward(self, x):
        return self.layers(x)


class BaseMnist:
    train_loss: Loss
    train_accuracy: Accuracy
    test_loss: Loss
    test_accuracy: Accuracy

    def __init__(self):
        super(BaseMnist, self).__init__()
        self.enc = OneHotEncoder()  # encoding scheme for labels
        self.enc.fit([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]])

    @classmethod
    def get_output_paths_(cls, output_path: Path, epoch: int) -> Tuple[Union[str, Path], ...]:
        # utility function to get output paths for all meaningful outputs when saving a model
        if Path(output_path).suffix in [""]:
            Path(output_path).mkdir(exist_ok=True, parents=True)
            temp_output_path = output_path
        else:
            if Path(output_path).exists():
                os.remove(output_path)
            temp_output_path = Path(output_path).parent

        time_now = datetime.now()
        model_pt_name = time_now.strftime("%Y-%m-%d--%H-%M-%S")
        (Path(temp_output_path) / "loss").mkdir(exist_ok=True, parents=True)
        (Path(temp_output_path) / "accuracy").mkdir(exist_ok=True, parents=True)
        (Path(temp_output_path) / "confusion").mkdir(exist_ok=True, parents=True)
        (Path(temp_output_path) / "models").mkdir(exist_ok=True, parents=True)
        if os.name == "nt":
            temp_output_path = str(temp_output_path)
            train_loss_json_path = temp_output_path + f"\\loss\\loss-train-{model_pt_name}--{epoch}.json"
            train_accuracy_json_path = temp_output_path + f"\\accuracy\\accuracy-train-{model_pt_name}--{epoch}.json"
            test_loss_json_path = temp_output_path + f"\\loss\\loss-test-{model_pt_name}--{epoch}.json"
            test_accuracy_json_path = temp_output_path + f"\\accuracy\\accuracy-test-{model_pt_name}--{epoch}.json"
            confusion_matrix_path = temp_output_path + f"\\confusion\\confusion-matrix-{model_pt_name}--{epoch}.png"
            accuracy_plot_path = temp_output_path + f"\\accuracy\\accuracy-{model_pt_name}--{epoch}.png"
            loss_plot_path = temp_output_path + f"\\loss\\loss-{model_pt_name}--{epoch}.png"
            if Path(output_path).suffix in [".pt"]:
                output_path = output_path
            else:
                output_path = temp_output_path + f"\\models\\model-{model_pt_name}--{epoch}.pt"
        else:
            temp_output_path = Path(temp_output_path)
            train_loss_json_path = temp_output_path / "loss" / f"loss-train-{model_pt_name}--{epoch}.json"
            train_accuracy_json_path = temp_output_path / "accuracy" / f"accuracy-train-{model_pt_name}--{epoch}.json"
            test_loss_json_path = temp_output_path / "loss" / f"loss-test-{model_pt_name}--{epoch}.json"
            test_accuracy_json_path = temp_output_path / "accuracy" / f"accuracy-test-{model_pt_name}--{epoch}.json"
            confusion_matrix_path = temp_output_path / "confusion" / f"confusion-matrix-{model_pt_name}--{epoch}.png"
            accuracy_plot_path = temp_output_path / "accuracy" / f"accuracy-{model_pt_name}--{epoch}.png"
            loss_plot_path = temp_output_path / "loss" / f"loss-{model_pt_name}--{epoch}.png"
            if Path(output_path).suffix in [".pt"]:
                output_path = output_path
            else:
                output_path = temp_output_path / "models" / f"model-{model_pt_name}--{epoch}.pt"

        return (
            output_path,
            train_loss_json_path,
            train_accuracy_json_path,
            test_loss_json_path,
            test_accuracy_json_path,
            confusion_matrix_path,
            accuracy_plot_path,
            loss_plot_path,
        )

    def preprocess_data(self, images, labels, batch_size: int, shuffle: bool = True) -> torch.utils.data.DataLoader:
        # perform one hot encoding.
        # E.g., if class labels are '0', '1', '2', '3', then each label will map to the following
        # '0' -> [1, 0, 0, 0]
        # '1' -> [0, 1, 0, 0]
        # '2' -> [0, 0, 1, 0]
        # '3' -> [0, 0, 0, 1]
        labels = self.enc.transform(np.column_stack(labels).reshape(-1, 1)).toarray()
        # find the mean and standard deviation of the images in order to properly normalize the data
        mean_ = np.mean(images)
        std_ = np.std(images)
        # transform data to Tensor using FloatTensor
        # reshape the data tensors so that each tensor is square using the dimensions of the first data (assuming all data is the same shape)
        # if we have 10 inputs that each have a length of 100, then we will get a tensor with the dimensions = (10, 10, 10)
        if np.array(images[0]).ndim == 1:
            test_images_tensors = (
                torch.FloatTensor(images)
                .reshape(len(images), int(np.sqrt(len(images[0]))), int(np.sqrt(len(images[0]))))
                .unsqueeze(0)
            )
        else:
            test_images_tensors = (
                torch.FloatTensor(images)
                .reshape(len(images), np.array(images[0]).shape[0], np.array(images[0]).shape[1])
                .unsqueeze(0)
            )
        # normalize data using mean and standard deviation
        preprocessing_transform = torchvision.transforms.Normalize([mean_], [std_])
        images_preprocessed = torch.squeeze(preprocessing_transform(test_images_tensors))
        # load images and labels into dataloader for compatability with pytorch
        data_loader = torch.utils.data.DataLoader(
            [[img, label] for img, label in zip(images_preprocessed, labels)], shuffle=shuffle, batch_size=batch_size
        )
        return data_loader


default_number_of_input_features_for_mnist_dataset: int = 28 * 28
default_number_of_output_decisions_for_mnist_dataset: int = 10


class MnistMLP(BaseMnist):
    def __init__(
        self,
        number_of_inputs_features: int = default_number_of_input_features_for_mnist_dataset,
        number_of_output_decisions: int = default_number_of_output_decisions_for_mnist_dataset,
        learning_rate: float = 1e-4,
        num_epochs: int = 100,
        early_stopping_th: Optional[float] = 1e-4,
        epoch_to_save_model: Optional[int] = None,
        use_dropout: bool = True,
        verbose: bool = True,
    ):
        super().__init__()
        if number_of_inputs_features != default_number_of_input_features_for_mnist_dataset:
            print(
                "You are changing the default number of input features for training on the mnist dataset. Are you "
                "sure you want to do that?"
            )
        if number_of_output_decisions != default_number_of_output_decisions_for_mnist_dataset:
            print(
                "You are changing the default number of output decisions for training on the mnist dataset. Are you "
                "sure you want to do that?"
            )
        self.input_features = number_of_inputs_features
        self.output_decisions = number_of_output_decisions
        if use_dropout:  # use MLP with dropout layers
            self.mlp = MultiLayerPerceptronDropout(number_of_inputs_features, number_of_output_decisions)
        else:
            self.mlp = MultiLayerPerceptron(number_of_inputs_features, number_of_output_decisions)
        self.loss_function = (
            nn.CrossEntropyLoss()
        )  # loss function, which will be used during backpropagation using our optimizer
        self.optimizer = torch.optim.Adam(
            self.mlp.parameters(), lr=learning_rate
        )  # adam optimizer to perform backpropagation of updating our weight vectors
        self.num_epochs = num_epochs  # number of times to run our train and test loader
        self.early_stopping_th = (
            early_stopping_th
        )  # threshold to stop training if our loss does not continue to decrease
        self.epoch_to_save_model = epoch_to_save_model  # save model every time the epoch reaches this value
        self.verbose = verbose  # turn on/off reporting train/test loss/accuracy during training and/or testing

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        output_path: Union[str, Path],
        test_loader: Optional[torch.utils.data.DataLoader] = None,
    ):
        if Path(output_path).suffix not in [""]:
            Path(output_path).mkdir(exist_ok=True, parents=True)

        # initialize 'Loss' class to keep track of loss values during training
        train_loss = Loss.init()
        test_loss = Loss.init()
        # initialize 'Accuracy' class to keep track of accuracy values during training
        train_accuracy = Accuracy.from_output_decisions(self.output_decisions)
        test_accuracy = Accuracy.from_output_decisions(self.output_decisions)
        # initialize progressbar to log what epoch we are on
        epoch_tqdm = tqdm(total=self.num_epochs, desc="Epoch", position=0)
        for epoch in range(self.num_epochs):
            # if test_loader is not none, then test current model on test data set
            if test_loader is not None:
                # turn our multilayer perceptron (MLP) model on to evaluation mode, so we don't update weights when testing
                # torch.nn.Module models are defaulted to train() mode when initialized.
                self.mlp.eval()
                # iterate over test data to test on our model
                for test_batch_idx, test_data in enumerate(test_loader):
                    test_inputs, test_targets = (
                        test_data
                    )  # test data is stored as (2, images [length = batch_size_test], labels [length = batch_size_test])
                    test_outputs = self.mlp(test_inputs)  # test inputs through current MLP model
                    loss_val = self.loss_function(
                        test_outputs, test_targets
                    )  # determine loss value for outputs vs targets
                    test_loss += loss_val.item()  # update our test_loss to keep track of our loss values
                    test_accuracy.compare_batch(
                        targets=test_targets, outputs=test_outputs
                    )  # determine accuracy of current batch for model
                test_accuracy.update_for_epoch()  # determine accuracy of current epoch for model
                test_loss.update_for_epoch()  # determine loss of current epoch for model
                self.mlp.train()  # turn our MLP model back into training mode

            # iterate over training data to train model on
            for batch_idx, train_data in enumerate(train_loader):
                train_inputs, train_targets = train_data
                # zero gradients of our optimizer, b/c the default functionality is for the optimizer to accumulate
                # the gradients of our runs. This means, that if we didn't zero out the gradient of our optimizer, we
                # would be updating the weight vectors of our model with the current gradient and gradients in the past.
                # This is useful for other applications, like using RNN, but we don't want to do that here.
                self.optimizer.zero_grad()
                train_outputs = self.mlp(train_inputs)  # train model on inputs
                loss_val = self.loss_function(
                    train_outputs, train_targets
                )  # get loss value between our outputs and targets
                loss_val.backward()  # calculate the gradients for each weight vector from output layer to input layer
                self.optimizer.step()  # update weight vectors (MLP model parameters) with non-accumulated gradients from our loss_val
                train_loss += loss_val.item()  # update our 'Loss' class variable to keep track of our training loss
                train_accuracy.compare_batch(
                    targets=train_targets, outputs=train_outputs
                )  # determine accuracy of our batch for the model
            train_accuracy.update_for_epoch()  # determine accuracy of our epoch for the model
            train_loss.update_for_epoch()  # determine loss of our epoch for the model

            if self.verbose:
                epoch_tqdm.write(f"Train Loss = {train_loss.current_loss}")
                if test_loader is not None:
                    epoch_tqdm.write(f"Test Loss = {test_loss.current_loss}")
                epoch_tqdm.write(f"Train Accuracy = {train_accuracy.acc_vals_per_epoch[epoch]}")
                if test_loader is not None:
                    epoch_tqdm.write(f"Test Accuracy = {test_accuracy.acc_vals_per_epoch[epoch]}")

            epoch_tqdm.update(1)
            if ((train_loss.previous_loss - train_loss.current_loss) < self.early_stopping_th) and (
                (train_loss.previous_loss - train_loss.current_loss) > 0
            ):
                print(
                    f"Change in loss is below threshold: "
                    f"{train_loss.previous_loss - train_loss.current_loss} < {self.early_stopping_th}\n"
                    f"Breaking and saving model."
                )
                break
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
                torch.save(self.mlp.state_dict(), epoch_output_path)
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
        torch.save(self.mlp.state_dict(), output_path)

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

    def test(
        self, test_loader, output_path: Optional[Union[str, Path]] = None, model_path: Optional[Union[str, Path]] = None
    ):
        # load in model given path, if model_path is None, then use default self.mlp model, which may or may not have just been trained
        if model_path is not None:
            try:
                model = MultiLayerPerceptron(self.input_features, self.output_decisions)
                model.load_state_dict(torch.load(model_path))
            except Exception:
                model = MultiLayerPerceptronDropout(self.input_features, self.output_decisions)
                model.load_state_dict(torch.load(model_path))
        else:
            model = self.mlp
        model.eval()
        accuracy = Accuracy.from_output_decisions(self.output_decisions)
        epoch_tqdm = tqdm(total=self.num_epochs, desc="Epoch", position=0)
        for epoch in range(self.num_epochs):
            for batch_idx, data in enumerate(test_loader):
                # test our hopefully trained model on our test data
                inputs, targets = data
                outputs = model(inputs)
                accuracy.compare_batch(targets=targets, outputs=outputs)
            accuracy.update_for_epoch()
            if self.verbose:
                epoch_tqdm.write(f"Accuracy = {accuracy.acc_vals_per_epoch[epoch]}")
            epoch_tqdm.update(1)

        print("Testing has finished.")
        (
            _,
            _,
            _,
            test_loss_json_path,
            test_accuracy_json_path,
            confusion_matrix_path,
            accuracy_plot_path,
            loss_plot_path,
        ) = self.get_output_paths_(output_path, epoch)

        self.test_accuracy = accuracy
        self.test_accuracy.save_json(test_accuracy_json_path)
        self.test_accuracy.save_confusion_matrix(confusion_matrix_path, "0123456789")
        save_accuracy_plot(self.test_accuracy.acc_vals_per_epoch, accuracy_plot_path, plot_labels="test")


def test_and_or_train_on_mnist_dataset(
    mnist_data_sets_base_path: Union[str, Path],
    model_output_base_path: Union[str, Path],  # should be directory
    model_test_path: Optional[Union[str, Path]] = None,  # can be directory of .pt files or a single .pt file
    n_epochs: int = 1000,
    epochs_to_save_model: Optional[int] = 5,
    batch_size_train: int = 64,
    batch_size_test: int = 1000,
    learning_rate: float = 1e-4,
    early_stopping_th: float = 1e-4,
    use_dropout: bool = False,
    train: bool = True,
    test: bool = True,
    verbose: bool = True,
):
    if not Path(model_output_base_path).suffix in [""]:
        raise RuntimeError(f"model_output_base_path must be a directory. Not a specific file.")
    mndata = load_mnist_dataset(mnist_data_sets_base_path)
    if mndata is None:
        raise RuntimeError(f"Failed to load in mnist dataset")

    # initialize class that wraps our MLP model
    minst_mlp = MnistMLP(
        number_of_inputs_features=28 * 28,
        number_of_output_decisions=10,
        learning_rate=learning_rate,
        num_epochs=n_epochs,
        early_stopping_th=early_stopping_th,
        epoch_to_save_model=epochs_to_save_model,
        use_dropout=use_dropout,
        verbose=verbose,
    )
    if train:
        if test:
            test_images, test_labels = mndata.load_testing()
            test_loader = minst_mlp.preprocess_data(test_images, test_labels, batch_size_test)
        else:
            test_loader = None
        train_images, train_labels = mndata.load_training()
        train_loader = minst_mlp.preprocess_data(train_images, train_labels, batch_size_train)
        minst_mlp.train(train_loader, get_real_path(model_output_base_path), test_loader)

    if test:
        if model_test_path is not None:
            if model_test_path.is_dir():
                model_input_pt_paths = sorted(model_test_path.glob("*.pt"))
            else:
                model_input_pt_paths = [model_test_path]

            test_images, test_labels = mndata.load_testing()
            test_loader = minst_mlp.preprocess_data(test_images, test_labels, batch_size_test)
            for model_pt_path in model_input_pt_paths:
                print(f"Testing '{model_pt_path.stem}'")
                minst_mlp.test(test_loader, get_real_path(model_pt_path))
        else:
            print(f"model_test_path is 'None'. Not testing any *.pt models")

# TODO: determine balance of classes and if inbalanced, attempt to fix training dataset accordingly
def main():
    mnist_data_sets_base_path = Path("../DATA/MNIST")
    model_output_base_path = Path("./model-0")
    n_epochs = 1000
    epochs_to_save_model = 5
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 1e-4
    early_stopping_th = 1e-4
    use_dropout: bool = False
    train: bool = True
    test: bool = True
    verbose: bool = True

    test_and_or_train_on_mnist_dataset(
        mnist_data_sets_base_path=mnist_data_sets_base_path,
        model_output_base_path=model_output_base_path,
        model_test_path=None,
        n_epochs=n_epochs,
        epochs_to_save_model=epochs_to_save_model,
        batch_size_train=batch_size_train,
        batch_size_test=batch_size_test,
        learning_rate=learning_rate,
        early_stopping_th=early_stopping_th,
        use_dropout=use_dropout,
        train=train,
        test=test,
        verbose=verbose,
    )


if __name__ == "__main__":
    main()
