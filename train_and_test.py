__author__ = 'Smadar Gazit'

import abc
import os
import sys
import tqdm
import torch
import numpy as np

from torch.utils.data import DataLoader
from typing import Callable, Any
from pathlib import Path
from training_helpers import BatchResult, EpochResult, FitResult


class Trainer(abc.ABC):
    """
    A class abstracting the various tasks of training models.

    Provides methods at multiple levels:
    - Multiple epochs (fit)
    - Single epoch (train_epoch/test_epoch)
    - Single batch (train_batch/test_batch)
    """

    def __init__(self, model, loss_fn, optimizer, device='cpu'):
        """
        Initialize the trainer.
        :param model: Instance of the model to train.
        :param loss_fn: The loss function to evaluate with.
        :param optimizer: The optimizer to train with.
        :param device: torch.device to run training on ('cpu' or 'cuda).
        """

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        model.to(self.device)

    def fit(self, dl_train: DataLoader, dl_test: DataLoader,
            num_epochs, checkpoints: str = None,
            early_stopping: int = None,
            print_every=1, post_epoch_fn=None, **kw) -> FitResult:
        """
        Trains the model for multiple epochs with a given training set,
        and calculates validation loss over a given validation set.
        :param dl_train: Dataloader for the training set.
        :param dl_test: Dataloader for the test set.
        :param num_epochs: Number of epochs to train for.
        :param checkpoints: Whether to save model to file every time the
            test set accuracy improves. Should be a string containing a
            filename without extension.
        :param early_stopping: Whether to stop training early if there is no
            test loss improvement for this number of epochs.
        :param print_every: Print progress every this number of epochs.
        :param post_epoch_fn: A function to call after each epoch completes.
        :return: A FitResult object containing train and test losses per epoch.
        """

        actual_num_epochs = 0
        train_loss, train_acc, test_loss, test_acc = [], [], [], []
        train_fp, train_fn, test_fp, test_fn = [], [], [], []

        best_acc = None
        epochs_without_improvement = 0
        try:
            checkpoint_filename = None
            if checkpoints is not None:
                checkpoint_filename = f'{checkpoints}.pt'
                Path(os.path.dirname(checkpoint_filename)).mkdir(exist_ok=True)
                if os.path.isfile(checkpoint_filename):
                    print(f'*** Loading checkpoint file {checkpoint_filename}')
                    saved_state = torch.load(checkpoint_filename,
                                             map_location=self.device)
                    best_acc = saved_state.get('best_acc', best_acc)
                    epochs_without_improvement =\
                        saved_state.get('ewi', epochs_without_improvement)
                    self.model.load_state_dict(saved_state['model_state'])

            for epoch in range(num_epochs):
                save_checkpoint = True
                verbose = False  # pass this to train/test_epoch.
                if epoch % print_every == 0 or epoch == num_epochs - 1:
                    verbose = True
                self._print(f'--- EPOCH {epoch+1}/{num_epochs} ---', verbose)

                actual_num_epochs = epoch

                train_result = self.train_epoch(dl_train, verbose=verbose)
                train_loss.append(train_result.loss)
                train_acc.append(train_result.accuracy)
                train_fp.append(train_result.fp_rate)
                train_fn.append(train_result.fn_rate)

                test_result = self.test_epoch(dl_test, verbose=verbose)
                test_loss.append(test_result.loss)
                test_acc.append(test_result.accuracy)
                test_fp.append(test_result.fp_rate)
                test_fn.append(test_result.fn_rate)
                if epoch >= 1:
                    if test_result.accuracy < best_acc + 0.1:
                        epochs_without_improvement += 1
                    else:
                        epochs_without_improvement = 0

                if best_acc is None or best_acc < test_result.accuracy:
                    best_acc = test_result.accuracy

                if early_stopping:
                    if epochs_without_improvement >= early_stopping:
                        break

                # Save model checkpoint if requested
                if save_checkpoint and checkpoint_filename is not None:
                    saved_state = dict(best_acc=best_acc,
                                       ewi=epochs_without_improvement,
                                       model_state=self.model.state_dict(),
                                       hidden_dim=self.model.hidden_dim,
                                       n_layers=self.model.num_layers,
                                       bidirectional=(
                                           self.model.multiply_bi == 2),
                                       dropout=self.model.dropout)
                    torch.save(saved_state, checkpoint_filename)
                    print(f'*** Saved checkpoint {checkpoint_filename} '
                          f'at epoch {epoch+1}')

                if post_epoch_fn:
                    post_epoch_fn(epoch, test_result, train_result, verbose)
        except KeyboardInterrupt:
            print('\n *** Training interrupted by user')
        finally:
            return FitResult(actual_num_epochs,
                             train_loss, train_acc, train_fp, train_fn, test_loss, test_acc, test_fp, test_fn)

    def train_epoch(self, dl_train: DataLoader, **kw) -> EpochResult:
        """
        Train once over a training set (single epoch).
        :param dl_train: DataLoader for the training set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(True)  # set train mode
        return self._foreach_batch(dl_train, self.train_batch, **kw)

    def test_epoch(self, dl_test: DataLoader, **kw) -> EpochResult:
        """
        Evaluate model once over a test set (single epoch).
        :param dl_test: DataLoader for the test set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(False)  # set evaluation (test) mode
        return self._foreach_batch(dl_test, self.test_batch, **kw)

    @abc.abstractmethod
    def train_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model, calculates loss,
        preforms back-propagation and uses the optimizer to update weights.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def test_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model and calculates loss.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @staticmethod
    def _print(message, verbose=True):
        """ Simple wrapper around print to make it conditional """
        if verbose:
            print(message)

    @staticmethod
    def _foreach_batch(dl: DataLoader,
                       forward_fn: Callable[[Any], BatchResult],
                       verbose=True, max_batches=None) -> EpochResult:
        """
        Evaluates the given forward-function on batches from the given
        dataloader, and prints progress along the way.
        """
        loss = 0
        accuracy = 0
        fp = 0
        fn = 0
        num_batches = len(dl.batch_sampler)

        if max_batches is not None:
            if max_batches < num_batches:
                num_batches = max_batches

        if verbose:
            pbar_file = sys.stdout
        else:
            pbar_file = open(os.devnull, 'w')

        pbar_name = forward_fn.__name__
        with tqdm.tqdm(desc=pbar_name, total=num_batches,
                       file=pbar_file) as pbar:
            dl_iter = iter(dl)
            for _ in range(num_batches):
                data = next(dl_iter)
                batch_res = forward_fn(data)

                pbar.set_description(f'{pbar_name} ({batch_res.loss:.3f})')
                pbar.update()

                loss += batch_res.loss
                accuracy += batch_res.accuracy
                fp += batch_res.fp_rate
                fn += batch_res.fn_rate

            loss = loss / num_batches
            accuracy = 100. * accuracy / num_batches
            fp = 100. * fp / num_batches
            fn = 100. * fn / num_batches
            pbar.set_description(f'{pbar_name} '
                                 f'(Avg. Loss {loss:.3f}, '
                                 f'Accuracy {accuracy:.1f}, '
                                 f'FP. Rate {fp:.1f}, '
                                 f'FN. Rate {fn:.1f})')

        return EpochResult(loss=loss, accuracy=accuracy, fp_rate=fp, fn_rate=fn)


class LSTMTrainer(Trainer):
    def __init__(self, model, loss_fn, optimizer, device=None):
        super().__init__(model, loss_fn, optimizer, device)

    def train_epoch(self, dl_train: DataLoader, **kw):
        """
         :param dl_train: Dataloader for the training set.
         """
        self.h = None
        return super().train_epoch(dl_train, **kw)

    def test_epoch(self, dl_test: DataLoader, **kw):
        """
        """
        self.h = None
        return super().test_epoch(dl_test, **kw)

    @staticmethod
    def calc_accuracy(predicted, y):
        """
        """
        max_obj = torch.max(predicted, 1)
        y_vec = y.view(-1)
        max_obj_vec = max_obj.indices.view(-1)
        class_1_idx = torch.where(y_vec == 1)[0]
        class_0_idx = torch.where(y_vec == 0)[0]
        fn = torch.zeros(1, 1)
        fp = torch.zeros(1, 1)
        if len(class_1_idx) == 0:
            # all elements are negative
            cur_accuracy = torch.true_divide(
                torch.eq(max_obj.indices, y).int().sum(), len(class_0_idx))
            fp = 1 - cur_accuracy
        elif len(class_0_idx) == 0:
            # all elements are positive
            cur_accuracy = torch.true_divide(
                torch.eq(max_obj.indices, y).int().sum(), len(class_1_idx))
            fn = 1 - cur_accuracy
        else:
            tp = torch.true_divide(torch.eq(
                max_obj_vec[class_1_idx], y_vec[class_1_idx]).int().sum(), 2.0 * len(class_1_idx))
            tn = torch.true_divide(torch.eq(
                max_obj_vec[class_0_idx], y_vec[class_0_idx]).int().sum(), 2.0 * len(class_0_idx))
            cur_accuracy = tp + tn
            fn = 1 - 2.0 * tp
            fp = 1 - 2.0 * tn
        return cur_accuracy, fp, fn

    @staticmethod
    def avg_binary_loss(loss_fn):
        """

        :return: function that averges the given loss function between 
                 in-epitopes letters and out
        """
        def avg_binary_cross_entropy(predicted, real):
        count_of_1 = real.sum()
        count_of_0 = (1-real).sum()
        if count_of_0 == 0 or count_of_1 == 0:
            return loss_fn()(predicted, real)
        factor = torch.true_divide(
            count_of_1*count_of_0, count_of_1+count_of_0)
        weight = torch.tensor([torch.true_divide(factor, count_of_0), torch.true_divide(
            factor, count_of_1)], device=real.device)
        return loss_fn(weight=weight)(predicted, real)
    return avg_binary_cross_entropy

    def train_batch(self, batch) -> BatchResult:
        """
         Train the LSTM model on one batch of data.
         :param batch: batch of samples
         :return:  A BatchResult object containing  A FitResult object containing train losses and accuracy
        """
        # TODO:
        x, y = batch
        x = x.to(self.device, dtype=torch.float)  # (B,S,V)
        y = y.to(self.device, dtype=torch.long)  # (B,S)

        # Forward pass
        self.optimizer.zero_grad()
        predicted, h = self.model(x, self.h)
        self.h = (torch.autograd.Variable(h[0]), torch.autograd.Variable(h[1]))
        predicted = predicted.permute(0, 2, 1)

        # Calculate total loss over sequence
        loss = self.loss_fn(predicted, y)

        # Backward pass (BPTT)
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)

        # Update params
        self.optimizer.step()

        # Calculate number of correct char predictions
        with torch.no_grad():
            cur_accuracy, fp, fn = calc_accuracy(predicted, y)
        self.h[0].detach()
        self.h[1].detach()

        return BatchResult(loss.item(), cur_accuracy.item(), fp.item(), fn.item())

    def test_batch(self, batch) -> BatchResult:
        """
        Evaluate the LSTM model on one a batch of data.
        :param Batch: batch of embedded samples
        :return:

        """
        # TODO
        x, y = batch
        x = x.to(self.device, dtype=torch.float)  # (B,S,V)
        y = y.to(self.device, dtype=torch.long)  # (B,S)

        with torch.no_grad():  # it is test, not training
            # Forward pass
            predicted, self.h = self.model(x, self.h)
            predicted = predicted.permute(0, 2, 1)
            # Loss calculation
            loss = self.loss_fn(predicted, y)
            # Calculate number of correct char predictions
            cur_accuracy, fp, fn = calc_accuracy(predicted, y)

        return BatchResult(loss.item(), cur_accuracy.item(), fp.item(), fn.item())
