import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm


def get_system_output(sim_class, weights, reference_signal, iterations, node=None):
    sim = sim_class(reference_signal, weights)
    sim.system.run(iterations, show_trace=False, disable_tqdm=True)
    output = None
    if node is None:
        output = sim.system.checkpointNode.outputValue.item()
    else:
        output = sim.system.get_node(node).outputValue.item()
    return output


def generate_outputs(sim_class, weights, reference_signals, iterations, samples, node=None):
    outputs = np.empty((len(reference_signals), samples))
    for i in range(len(reference_signals)):
        outputs[i] = np.array([get_system_output(sim_class, weights, reference_signals[i], iterations, node) for _ in range(samples)])
    return outputs


def kernel_def(y):
    return norm.pdf(y)


def kernel_density_distribution(x, x_trn, h, kernel=None):
    if kernel is None:
        kernel = kernel_def

    n = x_trn.shape[0]
    x = np.reshape(x, (x.shape[0], 1))
    p = (1 / (n*h)) * np.sum(kernel((x - x_trn) / h), axis=1)
    return p


def plot_kernel_density_estimation(x_range, x, y, h, bins_n, label="", node=None, show_histograms=False, fig=None, ax=None):
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
    if fig is None:
        fig, ax = plt.subplots()
    if show_histograms:
        hist, bins = np.histogram(x, bins=bins_n, range=(min(x_range), max(x_range)), density=True)
        centers = (bins[:-1] + bins[1:]) / 2
        width = bins[:-1] - bins[1:]
        ax.barh(centers, hist, height=np.abs(width), edgecolor='k')
    ax.plot(y.T, x_range.T, linewidth=2, label=f"{label}")
    node_name = node+" " if node is not None else ""
    ax.set_title(f"Estimated distributions of the output of the {node_name}node\n"
                 f"h = {h:.2f}")
    ax.grid(True)
    ax.legend()


def get_distributions(x, h, labels, step=0.1, kernel=None, node = None, show_plots=False, show_histograms=False, fig=None, ax=None):
    """
    Estimate distributions for each reference signal using kernel density estimation.
    This method generates output signals, computes their kernel density distribution,
    and plots the kernel density estimation for each reference signal.

    :param h: Bandwidth of the kernel.
    :type h: float

    :return: Estimated distributions for each reference signal.
    :rtype: np.array
    """
    n = x.shape[0]
    x_min = np.min(x) - 10
    x_max = np.max(x) + 10
    x_range = np.arange(x_min, x_max, step)
    distributions = np.zeros((n, x_range.shape[0]))
    for i in range(n):
        distributions[i] = kernel_density_distribution(x_range, x[i], h, kernel)
        if show_plots:
            plot_kernel_density_estimation(x_range=x_range, x=x[i], y=distributions[i], h=h,
                                           bins_n=np.arange(x_min, x_max+step, step), label=f"{labels[i]}", node=node,
                                           show_histograms=show_histograms, fig=fig, ax=ax)
    return distributions
