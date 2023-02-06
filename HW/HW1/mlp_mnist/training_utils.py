__all__ = ["Loss", "Accuracy", "save_accuracy_plot", "save_loss_plot"]
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from mlp_mnist.files import get_real_path
from mlp_mnist.serial import JsonSerial


@dataclass
class Loss(JsonSerial):
    loss_vals_per_batch: List[float]
    loss_vals_per_epoch: List[float]
    batch_cnt: int = 0
    previous_batch_cnt: int = 0
    epoch_cnt: int = 0

    @classmethod
    def init(cls) -> "Loss":
        return cls(loss_vals_per_batch=[], loss_vals_per_epoch=[], batch_cnt=0, previous_batch_cnt=0, epoch_cnt=0)

    @classmethod
    def from_json_file(cls, json_file: Union[str, Path]) -> "Loss":
        # method to read in 'Loss' class from saved json file (help with picking up from some spot of training)
        return super().read_json(json_file)

    def __add__(self, other: Union[float, int]):
        # add in additional loss values when a batch is done being processed
        self.loss_vals_per_batch.append(other)
        self.batch_cnt += 1
        return self

    def __iadd__(self, other: Union[float, int]) -> "Loss":
        # helps make code more readable when adding an item via '+='
        return self.__add__(other)

    @property
    def current_loss(self) -> float:
        # get current total loss mean by averaging all loss values
        return np.sum(self.loss_vals_per_batch) / self.batch_cnt

    @property
    def previous_loss(self) -> float:
        # get previous loss value to help with updating the average epoch loss value
        if len(self.loss_vals_per_batch) > 1:
            return np.sum(self.loss_vals_per_batch[:-2]) / self.batch_cnt - 1
        else:
            return 0

    def update_for_epoch(self):
        # update loss value per epoch for whichever epoch we are on
        self.epoch_cnt += 1
        self.loss_vals_per_epoch.append(
            sum(self.loss_vals_per_batch[self.previous_batch_cnt : self.batch_cnt])
            / (self.batch_cnt - self.previous_batch_cnt)
        )
        self.previous_batch_cnt = self.batch_cnt


@dataclass
class Accuracy(JsonSerial):
    acc_vals_per_batch: List[float]
    acc_vals_per_epoch: List[float]
    correct_hits: np.ndarray
    incorrect_hits: np.ndarray
    output_decisions: int
    batch_cnt: int = 0
    previous_batch_cnt: int = 0
    epoch_cnt: int = 0

    @classmethod
    def from_output_decisions(cls, output_size: int) -> "Accuracy":
        return cls(
            acc_vals_per_batch=[],
            acc_vals_per_epoch=[],
            batch_cnt=0,
            previous_batch_cnt=0,
            epoch_cnt=0,
            correct_hits=np.zeros((output_size,)),
            incorrect_hits=np.zeros((output_size, output_size)),
            output_decisions=output_size,
        )

    @classmethod
    def from_json_file(cls, json_file: Union[str, Path]) -> "Accuracy":
        # method to read in 'Accuracy' class from saved json file (help with picking up from some spot of training)
        return super().read_json(json_file)

    def compare_batch(self, targets: torch.Tensor, outputs: torch.Tensor):
        # determine accuracy between a batch of targets and outputs to update accuracy
        hit: int = 0
        for target, output in zip(targets, outputs):
            max_idx = int(torch.argmax(output))
            if bool(
                target[max_idx]
            ):  # see if the one hot encoding scheme of our output neuron layer determined the highest probability to be the same as the true target label
                hit += 1
                self.correct_hits[max_idx] += 1
            else:
                self.incorrect_hits[int(torch.argmax(target)), max_idx] += 1
        self.acc_vals_per_batch.append(hit / len(targets))
        self.batch_cnt += 1

    def update_for_epoch(self):
        # update accuracy for epoch
        self.epoch_cnt += 1
        self.acc_vals_per_epoch.append(
            sum(self.acc_vals_per_batch[self.previous_batch_cnt : self.batch_cnt])
            / (self.batch_cnt - self.previous_batch_cnt)
        )
        self.previous_batch_cnt = self.batch_cnt

    @property
    def confusion_matrix(self) -> np.ndarray:
        # get confusion matrix to better visualize incorrect hits vs correct hits
        return self.incorrect_hits.copy() + np.diag(self.correct_hits)

    def save_confusion_matrix(self, output_path: Union[str, Path], categories: str):
        # save confusion matrix
        output_path = get_real_path(output_path)
        df = pd.DataFrame(self.confusion_matrix, index=[i for i in categories], columns=[i for i in categories])
        plt.figure(figsize=(10, 7))
        sns.heatmap(df, annot=True)
        plt.savefig(output_path)
        plt.close()


def plot_accuracy_or_loss(
    train_vals: List[float],
    output_path: Union[str, Path],
    test_vals: Optional[List[float]] = None,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlabel: Optional[str] = None,
    plot_labels: Optional[Union[str, List[str]]] = None,
):
    if plot_labels is None:
        plot_labels = ["train", "test"]
    elif not isinstance(plot_labels, list):
        plot_labels = [plot_labels] * 2

    x_epochs = np.arange(1, len(train_vals) + 1)
    plt.plot(x_epochs, train_vals, label=plot_labels[0])
    if test_vals is not None:
        x_epochs = np.arange(len(train_vals) - len(test_vals) + 1, len(train_vals) + 1)
        plt.plot(x_epochs, test_vals, label=plot_labels[1])
    if title is not None:
        plt.title(title)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if xlabel is not None:
        plt.xlabel(xlabel)
    plt.legend()
    plt.savefig(get_real_path(output_path))
    plt.close()


def save_accuracy_plot(
    train_acc: List[float],
    output_path: Union[str, Path],
    test_acc: Optional[List[float]] = None,
    plot_labels: Optional[Union[str, List[str]]] = None,
):
    plot_accuracy_or_loss(
        train_vals=train_acc,
        test_vals=test_acc,
        output_path=output_path,
        title="Accuracy",
        ylabel="Accuracy",
        xlabel="Epochs",
        plot_labels=plot_labels,
    )


def save_loss_plot(
    train_loss: List[float],
    output_path: Union[str, Path],
    test_loss: Optional[List[float]] = None,
    plot_labels: Optional[Union[str, List[str]]] = None,
):
    plot_accuracy_or_loss(
        train_vals=train_loss,
        test_vals=test_loss,
        output_path=output_path,
        title="Loss",
        ylabel="Loss",
        xlabel="Epochs",
        plot_labels=plot_labels,
    )