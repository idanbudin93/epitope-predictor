from typing import NamedTuple, List


class BatchResult(NamedTuple):
    """
    Represents the result of training for a single batch: the loss
    and number of correct classifications.
    """
    loss: float
    accuracy: float
    fp_rate: float
    fn_rate: float


class EpochResult(NamedTuple):
    """
    Represents the result of training for a single epoch: the loss per batch
    and accuracy on the dataset (train or test).
    """
    loss: float
    accuracy: float
    fp_rate: float
    fn_rate: float


class FitResult(NamedTuple):
    """
    Represents the result of fitting a model for multiple epochs given a
    training and test (or validation) set.
    The losses are for each batch and the accuracies are per epoch.
    """
    num_epochs: int
    train_loss: List[float]
    train_acc: List[float]
    train_fp: List[float]
    train_fn: List[float]
    test_loss: List[float]
    test_acc: List[float]
    test_fp: List[float]
    test_fn: List[float]
