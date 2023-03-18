__all__ = ["Loss", "Accuracy", "save_accuracy_plot", "save_loss_plot"]
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Union, Optional, Tuple, Callable

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
    precision_per_epoch: List[float]
    recall_per_epoch: List[float]
    f1_score_per_epoch: List[float]
    correct_hits: np.ndarray
    correct_hits_per_epoch: List[np.ndarray]
    incorrect_hits: np.ndarray
    incorrect_hits_per_epoch: List[np.ndarray]
    output_decisions: int
    batch_cnt: int = 0
    previous_batch_cnt: int = 0
    epoch_cnt: int = 0
    onehotencoding: bool = True

    @classmethod
    def from_output_decisions(cls, output_size: int, onehotencoding: bool = True) -> "Accuracy":
        return cls(
            acc_vals_per_batch=[],
            acc_vals_per_epoch=[],
            precision_per_epoch=[],
            recall_per_epoch=[],
            f1_score_per_epoch=[],
            batch_cnt=0,
            previous_batch_cnt=0,
            epoch_cnt=0,
            correct_hits=np.zeros((output_size,)),
            correct_hits_per_epoch=[],
            incorrect_hits=np.zeros((output_size, output_size)),
            incorrect_hits_per_epoch=[],
            output_decisions=output_size,
            onehotencoding=onehotencoding,
        )

    @classmethod
    def from_json_file(cls, json_file: Union[str, Path]) -> "Accuracy":
        # method to read in 'Accuracy' class from saved json file (help with picking up from some spot of training)
        return super().read_json(json_file)

    def compare_batch(self, targets: torch.Tensor, outputs: torch.Tensor) -> Optional[List[Tuple[int, int]]]:
        # determine accuracy between a batch of targets and outputs to update accuracy
        hit: int = 0
        indices = None
        if self.onehotencoding:
            indices = []
            for batch_idx, (target, output) in enumerate(zip(targets, outputs)):
                max_idx = int(torch.argmax(output))
                if bool(
                    target[max_idx]
                ):  # see if the one hot encoding scheme of our output neuron layer determined the highest probability to be the same as the true target label
                    hit += 1
                    self.correct_hits[max_idx] += 1
                    indices.append((batch_idx, max_idx))
                else:
                    self.incorrect_hits[int(torch.argmax(target)), max_idx] += 1
        else:
            # assuming output is some value between 0 & 1 and that the targets are either 0 or 1
            outputs = torch.round(outputs)

            zero_targets_idx = torch.where(targets == 0)
            zero_equality = torch.eq(outputs[zero_targets_idx], targets[zero_targets_idx])
            try:
                self.correct_hits[0] += sum(zero_equality).item()
            except AttributeError:
                self.correct_hits[0] += sum(zero_equality)
            hit = sum(zero_equality)
            try:
                self.incorrect_hits[0] += sum(~zero_equality).item()
            except AttributeError:
                self.incorrect_hits[0] += sum(~zero_equality)

            ones_targets_idx = torch.where(targets == 1)
            ones_equality = torch.eq(outputs[ones_targets_idx], targets[ones_targets_idx])
            try:
                self.correct_hits[1] += sum(ones_equality).item()
            except AttributeError:
                self.correct_hits[1] += sum(ones_equality)
            hit += sum(ones_equality)
            try:
                self.incorrect_hits[1] += sum(~ones_equality).item()
            except AttributeError:
                self.incorrect_hits[1] += sum(~ones_equality)
        self.acc_vals_per_batch.append(hit / len(targets))
        self.batch_cnt += 1

        return indices

    def update_for_epoch(self):
        # update accuracy for epoch
        self.epoch_cnt += 1
        self.acc_vals_per_epoch.append(
            sum(self.acc_vals_per_batch[self.previous_batch_cnt : self.batch_cnt])
            / (self.batch_cnt - self.previous_batch_cnt)
        )
        if len(self.correct_hits_per_epoch) == 0:
            self.correct_hits_per_epoch.append(self.correct_hits.copy())
            self.incorrect_hits_per_epoch.append(self.incorrect_hits.copy())
        else:
            correct_hits_previous_epoch = self.correct_hits_per_epoch[-1].copy()
            self.correct_hits_per_epoch.append(self.correct_hits - correct_hits_previous_epoch)
            incorrect_hits_previous_epoch = self.incorrect_hits_per_epoch[-1].copy()
            self.incorrect_hits_per_epoch.append(self.incorrect_hits - incorrect_hits_previous_epoch)
        self.previous_batch_cnt = self.batch_cnt

    @property
    def confusion_matrix(self) -> np.ndarray:
        # get confusion matrix to better visualize incorrect hits vs correct hits
        return self.incorrect_hits.copy() + np.diag(self.correct_hits)

    def save_confusion_matrix(self, output_path: Union[str, Path], categories: Union[str, List[str]]):
        # save confusion matrix
        output_path = get_real_path(output_path)
        df = pd.DataFrame(self.confusion_matrix, index=[i for i in categories], columns=[i for i in categories])
        plt.figure(figsize=(10, 7))
        sns.heatmap(df, annot=True)
        plt.savefig(output_path)
        plt.close()

    @property
    def current_accuracy(self) -> float:
        return sum(self.acc_vals_per_batch) / self.batch_cnt

    def roc_curve(self, output_path: Union[str, Path]):
        tp = []
        fp = []
        for epoch in range(self.epoch_cnt):
            tp.append(self.true_positive(epoch))
            fp.append(self.false_positive(epoch))
        # tp /= max(tp)
        # fp /= max(fp)
        plt.plot(fp, tp)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC curve")
        plt.savefig(output_path)
        plt.close()

    def cum_stats_to_csv(self, output_path: Union[str, Path]):
        df = pd.DataFrame(
            columns=[
                "true_positive",
                "false_positive",
                "true_negative",
                "false_negative",
                "accuracy",
                "recall",
                "precision",
                "f1_score",
            ]
        )
        for epoch in range(self.epoch_cnt):
            tp = self.true_positive(epoch)
            fp = self.false_positive(epoch)
            tn = self.true_negative(epoch)
            fn = self.false_negative(epoch)
            accuracy = self.acc_vals_per_epoch[epoch]
            recall = self.recall(epoch)
            precision = self.precision(epoch)
            f1_score = self.f1_score(epoch)
            df.loc[epoch] = [tp, fp, tn, fn, accuracy, recall, precision, f1_score]
        cum_tp = self.cum_true_positive
        cum_fp = self.cum_false_positive
        cum_tn = self.cum_true_negative
        cum_fn = self.cum_false_negative
        cum_accuracy = np.mean(self.acc_vals_per_epoch)
        cum_recall = self.cum_recall
        cum_precision = self.cum_precision
        cum_f1_score = self.cum_f1_score
        df.loc["cumulative"] = [cum_tp, cum_fp, cum_tn, cum_fn, cum_accuracy, cum_recall, cum_precision, cum_f1_score]
        df.to_csv(output_path)

    def true_positive(self, epoch: int) -> int:
        if len(self.correct_hits) != 2:
            raise RuntimeError(f"True Positive only defined for Binary Classification")
        return self.correct_hits_per_epoch[epoch][1]  # 1 = true label, 0 = false label

    @property
    def cum_true_positive(self) -> int:
        tp = 0
        for epoch in range(self.epoch_cnt):
            tp += self.true_positive(epoch)
        return tp

    def true_negative(self, epoch: int) -> int:
        if len(self.correct_hits) != 2:
            raise RuntimeError(f"True Negative only defined for Binary Classification")
        return self.correct_hits_per_epoch[epoch][0]

    @property
    def cum_true_negative(self) -> int:
        tn = 0
        for epoch in range(self.epoch_cnt):
            tn += self.true_negative(epoch)
        return tn

    def false_positive(self, epoch: int) -> int:
        if len(self.correct_hits) != 2:
            raise RuntimeError(f"False Positive only defined for Binary Classification")
        return sum(self.incorrect_hits_per_epoch[epoch][1])

    @property
    def cum_false_positive(self) -> int:
        fp = 0
        for epoch in range(self.epoch_cnt):
            fp += self.false_positive(epoch)
        return fp

    def false_negative(self, epoch: int) -> int:
        if len(self.correct_hits) != 2:
            raise RuntimeError(f"False Negative only defined for Binary Classification")
        return sum(self.incorrect_hits_per_epoch[epoch][0])

    @property
    def cum_false_negative(self) -> int:
        fn = 0
        for epoch in range(self.epoch_cnt):
            fn += self.false_negative(epoch)
        return fn

    def precision(self, epoch: int) -> float:
        return self.true_positive(epoch) / (self.true_positive(epoch) + self.false_positive(epoch))

    @property
    def cum_precision(self) -> float:
        return self.cum_true_positive / (self.cum_true_positive + self.cum_false_positive)

    def recall(self, epoch: int) -> float:
        return self.true_positive(epoch) / (self.true_positive(epoch) + self.false_negative(epoch))

    @property
    def cum_recall(self) -> float:
        return self.cum_true_positive / (self.cum_true_positive + self.cum_false_negative)

    def f1_score(self, epoch: int) -> float:
        return 2 * ((self.precision(epoch) * self.recall(epoch)) / (self.precision(epoch) + self.recall(epoch)))

    @property
    def cum_f1_score(self) -> float:
        return 2 * ((self.cum_precision * self.cum_recall) / (self.cum_precision + self.cum_recall))


def plot_accuracy_or_loss(
    train_vals: List[float],
    output_path: Union[str, Path],
    validation_vals: Optional[List[float]] = None,
    test_vals: Optional[List[float]] = None,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlabel: Optional[str] = None,
    plot_labels: Optional[Union[str, List[str]]] = None,
):
    if plot_labels is None:
        plot_labels = ["train"]
        plot_labels.append("validation")
        plot_labels.append("test")
    elif not isinstance(plot_labels, list):
        plot_labels = [plot_labels] * 3

    x_epochs = np.arange(1, len(train_vals) + 1)
    plt.plot(x_epochs, train_vals, label=plot_labels[0])
    if validation_vals is not None:
        x_epochs = np.arange(len(train_vals) - len(validation_vals) + 1, len(train_vals) + 1)
        plt.plot(x_epochs, validation_vals, label=plot_labels[1])
    if test_vals is not None:
        x_epochs = np.arange(len(train_vals) - len(test_vals) + 1, len(train_vals) + 1)
        plt.plot(x_epochs, test_vals, label=plot_labels[2])
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
    val_acc: Optional[List[float]] = None,
    plot_labels: Optional[Union[str, List[str]]] = None,
):
    plot_accuracy_or_loss(
        train_vals=train_acc,
        output_path=output_path,
        validation_vals=val_acc,
        test_vals=test_acc,
        title="Accuracy",
        ylabel="Accuracy",
        xlabel="Epochs",
        plot_labels=plot_labels,
    )


def save_loss_plot(
    train_loss: List[float],
    output_path: Union[str, Path],
    test_loss: Optional[List[float]] = None,
    val_loss: Optional[List[float]] = None,
    plot_labels: Optional[Union[str, List[str]]] = None,
):
    plot_accuracy_or_loss(
        train_vals=train_loss,
        output_path=output_path,
        validation_vals=val_loss,
        test_vals=test_loss,
        title="Loss",
        ylabel="Loss",
        xlabel="Epochs",
        plot_labels=plot_labels,
    )


training_path_name: Callable = lambda file_name, model_name, epoch, ext: f"{file_name}-{model_name}--{epoch}{ext}"


def training_plot_paths(
    output_path: Path,
    epoch: int,
    loss: bool = True,
    accuracy: bool = True,
    confusion: bool = True,
    roc_curve: bool = True,
    cum_stats: bool = True,
    models: bool = True,
) -> Tuple[Path, ...]:
    if not output_path.exists():
        output_path.mkdir(exist_ok=True, parents=True)
    time_now = datetime.now()
    model_pt_name = time_now.strftime("%Y-%m-%d--%H-%M-%S")
    model_output_path = None
    if models:
        Path(output_path / "models").mkdir(exist_ok=True, parents=True)
        model_output_path = output_path / "models" / training_path_name("model", model_pt_name, epoch, ".pth")
    loss_output_path = None
    if loss:
        Path(output_path / "loss").mkdir(exist_ok=True, parents=True)
        loss_output_path = output_path / "loss" / training_path_name("loss", model_pt_name, epoch, ".png")
    accuracy_output_path = None
    if accuracy:
        Path(output_path / "accuracy").mkdir(exist_ok=True, parents=True)
        accuracy_output_path = output_path / "accuracy" / training_path_name("accuracy", model_pt_name, epoch, ".png")
    roc_curve_output_path = None
    if roc_curve:
        Path(output_path / "stats").mkdir(exist_ok=True, parents=True)
        roc_curve_output_path = output_path / "stats" / training_path_name("roc-curve", model_pt_name, epoch, ".png")
    confusion_matrix_output_path = None
    if confusion:
        Path(output_path / "confusion").mkdir(exist_ok=True, parents=True)
        confusion_matrix_output_path = output_path / "confusion" / training_path_name("confusion-matrix", model_pt_name, epoch, ".png")
    cum_stats_output_path = None
    if cum_stats:
        Path(output_path / "stats").mkdir(exist_ok=True, parents=True)
        cum_stats_output_path = output_path / "stats" / training_path_name("cumulative-stats", model_pt_name, epoch, ".csv")

    return (
        model_output_path,
        confusion_matrix_output_path,
        accuracy_output_path,
        roc_curve_output_path,
        cum_stats_output_path,
        loss_output_path,
    )