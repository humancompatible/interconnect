import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm


class Distribution:
    def __init__(self, sim_class, reference_signal, iterations=100, samples=100):
        self.sim_class = sim_class
        self.reference_signal = reference_signal
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

    def plot_kernel_density_estimation(self, x, y, hist, bins, h):
        centers = (bins[:-1] + bins[1:]) / 2
        width = bins[:-1] - bins[1:]
        plt.bar(centers, hist, width=width, edgecolor='k')
        plt.plot(x.T, y.T, 'r', linewidth=2)
        plt.title('h = {:.2f}'.format(h))

    def get_distribution(self, h):
        x = self.generate_outputs()
        x_range = np.arange(np.min(x), np.max(x), 0.1)
        y = self.kernel_density_distribution(x_range, x, h)
        hist, bins = np.histogram(x, 20, density=True)
        # plots of the estimates
        plt.figure(figsize=(6, 4))
        self.plot_kernel_density_estimation(x_range, y, hist, bins, h)
        return y
