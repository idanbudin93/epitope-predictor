import math
import itertools

import numpy as np
import matplotlib.pyplot as plt
from training_helpers import FitResult


def tensors_as_images(tensors, nrows=1, figsize=(8, 8), titles=[],
                      wspace=0.1, hspace=0.2, cmap=None):
    """
    Plots a sequence of pytorch tensors as images.

    :param tensors: A sequence of pytorch tensors, should have shape CxWxH
    """
    assert nrows > 0

    num_tensors = len(tensors)

    ncols = math.ceil(num_tensors / nrows)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize,
                             gridspec_kw=dict(wspace=wspace, hspace=hspace),
                             subplot_kw=dict(yticks=[], xticks=[]))
    axes_flat = axes.reshape(-1)

    # Plot each tensor
    for i in range(num_tensors):
        ax = axes_flat[i]

        image_tensor = tensors[i]
        assert image_tensor.dim() == 3  # Make sure shape is CxWxH

        image = image_tensor.numpy()
        image = image.transpose(1, 2, 0)
        image = image.squeeze()  # remove singleton dimensions if any exist

        # Scale to range 0..1
        min, max = np.min(image), np.max(image)
        image = (image-min) / (max-min)

        ax.imshow(image, cmap=cmap)

        if len(titles) > i and titles[i] is not None:
            ax.set_title(titles[i])

    # If there are more axes than tensors, remove their frames
    for j in range(num_tensors, len(axes_flat)):
        axes_flat[j].axis('off')

    return fig, axes


def dataset_first_n(dataset, n, show_classes=False, class_labels=None,
                    random_start=True, **kw):
    """
    Plots first n images of a dataset containing tensor images.
    """

    if random_start:
        start = np.random.randint(0, len(dataset) - n)
        stop = start + n
    else:
        start = 0
        stop = n

    # [(img0, cls0), ..., # (imgN, clsN)]
    first_n = list(itertools.islice(dataset, start, stop))

    # Split (image, class) tuples
    first_n_images, first_n_classes = zip(*first_n)

    if show_classes:
        titles = first_n_classes
        if class_labels:
            titles = [class_labels[cls] for cls in first_n_classes]
    else:
        titles = []

    return tensors_as_images(first_n_images, titles=titles, **kw)


def plot_fit(fit_res: FitResult, fig=None, log_loss=False, legend=None):
    """
    Plots a FitResult object.
    Creates four plots: train loss, test loss, train acc, test acc.
    :param fit_res: The fit result to plot.
    :param fig: A figure previously returned from this function. If not None,
        plots will the added to this figure.
    :param log_loss: Whether to plot the losses in log scale.
    :param legend: What to call this FitResult in the legend.
    :return: The figure.
    """
    if fig is None:
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 10),
                                 sharex='col', sharey=False)
        axes = axes.reshape(-1)
    else:
        axes = fig.axes

    for ax in axes:
        for line in ax.lines:
            if line.get_label() == legend:
                line.remove()

    p = itertools.product(['train', 'test'], [
                          'loss', 'acc', 'fp', 'fn'])
    for idx, (traintest, lossaccfpfn) in enumerate(p):
        ax = axes[idx]
        attr = f'{traintest}_{lossaccfpfn}'
        data = getattr(fit_res, attr)
        h = ax.plot(np.arange(1, len(data) + 1), data, label=legend)
        ax.set_title(attr)
        if lossaccfpfn == 'loss':
            ax.set_xlabel('Epoch #')
            ax.set_ylabel('Loss')
            if log_loss:
                ax.set_yscale('log')
                ax.set_ylabel('Loss (log)')
        elif lossaccfpfn == 'acc':
            ax.set_xlabel('Epoch #')
            ax.set_ylabel('Accuracy (%)')
        elif lossaccfpfn == 'fp':
            ax.set_xlabel('Epoch #')
            ax.set_ylabel('FP. Rate (%)')
        elif lossaccfpfn == 'fn':
            ax.set_xlabel('Epoch #')
            ax.set_ylabel('FN. Rate (%)')
        if legend:
            ax.legend()
    return fig, axes


# probability plotting addition

def prob_heat(prob):
    """
    Accessory function for plot_hot_cold. Assigns a color to a probability
    with higher probabilities given a 'hotter' color for higher probabilities.
    :param prob: probability on the interval [0, 1].
    :return: RGB tuple of probability heat.
    """
    red_heat = max(0, 2 * (prob - 0.5))
    green_heat = max(0, 2 * (0.5 - abs(prob - 0.5)))
    blue_heat = max(0, 2 * (0.5 - prob))
    return red_heat, green_heat, blue_heat


def plot_hot_cold(y_data, prob_color=None, threshold=0.9):
    """
    Plot a probability vector with additional optionality to color points and label stretches
    of high probability. The default coloring scheme gives hotter colors for higher probabilities,
    but custom color schemes are possible. the labeling scheme considers that epitopes come in 
    contiguous stretches, and labels the position of the first point with probability higher than
    :threshold: for any contiguous stretch.
    :param y_data: iterable of probabilities to plot.
    :param color_probabilities: specify a color for all probabilities in y_data.
        if no color is specified, probabilities are colored by prob_heat.
    :param threshold: minimal value above which points are labeled, can be set to
        a value greater than 1.0 so no point is labeled.
    """
    plt.ylim([-0.05, 1.05])
    x_data = [i + 1 for i in range(len(y_data))]
    colors = [prob_heat(p) if not prob_color else prob_color for p in y_data]

    _, ax = plt.subplots(figsize=(15, 5))
    ax.scatter(x_data, y_data, c=colors, marker='o')
    ax.plot(x_data, y_data, color='black')
    ax.xlabel('position')
    ax.ylabel('probability')

    if 0.5 < threshold < 1.0:
        ax.xticks(threshold)

    above_threshold = False
    for x, y in zip(x_data, y_data):
        if y >= threshold and not above_threshold:
            ax.annotate(str(x), (x, y))
            above_threshold = True
        if y < threshold:
            above_threshold = False
    return ax
