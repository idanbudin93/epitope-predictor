from sys import argv

import matplotlib.pyplot as plt

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


def plot_hot_cold(y_data, save_path=None, prob_color=None, threshold=0.9, plot_threshold=False):
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
    x_data = [i + 1 for i in range(len(y_data))]
    colors = [prob_heat(p) if not prob_color else prob_color for p in y_data]

    _, ax = plt.subplots(figsize=(25,5))
    ax.scatter(x_data, y_data, c=colors, marker='o')
    ax.plot(x_data, y_data, color='black')
    
    plt.ylim([-0.15, 1.15])
    plt.xlim([0, len(x_data) + 1])
    ax.set_xlabel('position')
    ax.set_ylabel('probability')
    
    if 0.0 < threshold < 1.0 and plot_threshold:
        ax.plot([0, len(x_data) + 1], [threshold, threshold])
    
    above_threshold = False
    for x, y in zip(x_data, y_data):
      if y >= threshold and not above_threshold:
        ax.annotate(str(x), (x, y))
        above_threshold = True
      if y < threshold:
        above_threshold = False
        
    if save_path:
    	plt.savefig(f'{save_path}.png')
    plt.close('all')


if __name__ == "__main__":
	plot_hot_cold([random.uniform(0, 1) for _ in range(int(argv[1]))])

