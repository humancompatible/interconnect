import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm


def get_system_output(sim_class, reference_signal, iterations):
    sim = sim_class(reference_signal)
    sim.system.run(iterations, show_trace=False, disable_tqdm=True)
    output = sim.system.checkpointNode.outputValue.item()
    return output


def generate_outputs(sim_class, reference_signals, iterations, samples):
    outputs = np.empty((len(reference_signals), samples))
    for i in range(len(reference_signals)):
        outputs[i] = np.array([get_system_output(sim_class, reference_signals[i], iterations) for _ in range(samples)])
    return outputs


def kernel_density_distribution(x, x_trn, h):
    def kernel(y):
        return norm.pdf(y, loc=0, scale=h)

    n = x_trn.shape[0]
    x = np.reshape(x, (x.shape[0], 1))
    p = (1 / n) * np.sum(kernel(x - x_trn), axis=1)
    return p


def plot_kernel_density_estimation(x_range, x, y, h, bins_n, label="", show_histograms=False):
    """
    Plots the kernel density estimation of the given data.
        
    :param x: x-axis values.
    :type x: np.array
    :param y: y-axis values.
    :type y: np.array
    :param hist: Histogram values.
    :type hist: np.array
    :param bins: Bin values.
    :type bins: np.array
    :param h: Bandwidth of the kernel.
    :type h: float
    :param ref_sig: Reference signal.
    :type ref_sig: float
    :param show_hist: Flag to show histogram. Default is False.
    :type show_hist: bool

    :return: None
    """

    if show_histograms:
        hist, bins = np.histogram(x, bins=bins_n, range=(min(x_range), max(x_range)), density=True)
        centers = (bins[:-1] + bins[1:]) / 2
        width = bins[:-1] - bins[1:]
        plt.bar(centers, hist, width=width, edgecolor='k')
    plt.plot(x_range.T, y.T, linewidth=2, label=f"{label}")
    plt.title('h = {:.2f}'.format(h))
    plt.grid(True)
    plt.legend()


def get_distributions(x, h, labels, bins_n, step=0.1, show_histograms=False):
    """
    Estimate distributions for each reference signal using kernel density estimation.
    This method generates output signals, computes their kernel density distribution,
    and plots the kernel density estimation for each reference signal.

    :param h: Bandwidth of the kernel.
    :type h: float

    :return: Estimated distributions for each reference signal.
    :rtype: np.array
    """
    plt.figure(figsize=(8, 6))
    n = x.shape[0]
    x_min = np.min(x) - 10
    x_max = np.max(x) + 10
    x_range = np.arange(x_min, x_max, step)
    distributions = np.zeros((n, x_range.shape[0]))
    for i in range(n):
        distributions[i] = kernel_density_distribution(x_range, x[i], h)
        plot_kernel_density_estimation(x_range=x_range, x=x[i], y=distributions[i], h=h,
                                       bins_n=bins_n, label=f"{labels[i]}",
                                       show_histograms=show_histograms)
    plt.show()
    return distributions
