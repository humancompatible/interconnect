import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm


class Distribution:
    """
    Class to generate distributions of the output of a simulator
    """
    def __init__(self, sim_class, iterations=100, samples=100):
        """
        :param sim_class: simulator class
        :param iterations: number of iterations to run the simulator
        :param samples: number of samples to generate
        """
        self.sim_class = sim_class
        self.iterations = iterations
        self.samples = samples

    def get_system_output(self, reference_signal):
        sim = self.sim_class(reference_signal)
        sim.system.run(self.iterations, show_trace=False, disable_tqdm=True)
        output = sim.system.checkpointNode.outputValue.item()
        return output

    def generate_outputs(self):
        outputs = np.array([self.get_system_output(self.reference_signal) for _ in range(self.samples)])
        return outputs

    def kernel_density_distribution(self, x, x_trn, h):
        def kernel(y):
            return norm.pdf(y, loc=0, scale=h)

        n = x_trn.shape[0]
        x = np.reshape(x, (x.shape[0], 1))
        p = (1 / n) * np.sum(kernel(x - x_trn), axis=1)
        return p

    def plot_kernel_density_estimation(self, x, y, hist, bins, h, ref_sig, show_hist=False):
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

        if show_hist:
            centers = (bins[:-1] + bins[1:]) / 2
            width = bins[:-1] - bins[1:]
            plt.bar(centers, hist, width=width, edgecolor='k')
        plt.plot(x.T, y.T, linewidth=2, label=f"reference signal = {ref_sig}")
        plt.title('h = {:.2f}'.format(h))
        plt.legend()

    def get_distributions(self, h, reference_signals, x_min, x_max, bins):
        """
        Estimate distributions for each reference signal using kernel density estimation.
        This method generates output signals, computes their kernel density distribution,
        and plots the kernel density estimation for each reference signal.

        :param h: Bandwidth of the kernel.
        :type h: float
        :param reference_signals: List of reference signals.
        :type reference_signals: np.array
        :param x_min: Minimum x-axis.
        :type x_min: float
        :param x_max: Maximum x-axis.
        :type x_max: float
        :param bins: Number of bins.
        :type bins: integer

        :return: Estimated distributions for each reference signal.
        :rtype: np.array
        """
        plt.figure(figsize=(8, 6))
        x_range = np.arange(x_min, x_max, 0.1)
        distributions = np.zeros((reference_signals.shape[0], x_range.shape[0]))
        rn = 0
        for ref_sig in reference_signals:
            self.reference_signal = ref_sig
            x = self.generate_outputs()
            y = self.kernel_density_distribution(x_range, x, h)
            hist, bins = np.histogram(x, bins=bins, range=(x_min, x_max), density=True)
            # plots of the estimates
            print(f"reference signal: {ref_sig}")
            self.plot_kernel_density_estimation(x_range, y, hist, bins, h, ref_sig, show_hist=False)
            distributions[rn] = y
            rn += 1
        plt.show()
        return distributions
