import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm


class Distribution:
    def __init__(self, sim_class, iterations=100, samples=100):
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

    def plot_kernel_density_estimation(self, x, y, hist, bins, h, ref_sig):
        centers = (bins[:-1] + bins[1:]) / 2
        width = bins[:-1] - bins[1:]
        # plt.bar(centers, hist, width=width, edgecolor='k')
        plt.plot(x.T, y.T, linewidth=2, label=f"reference signal = {ref_sig}")
        plt.title('h = {:.2f}'.format(h))
        plt.legend()

    def get_distribution(self, h, reference_signals):
        plt.figure(figsize=(6, 4))
        x_range = np.arange(-10.0, 10.0, 0.1)
        distributions = np.zeros((reference_signals.shape[0], x_range.shape[0]))
        rn = 0
        for ref_sig in reference_signals:
            self.reference_signal = ref_sig
            x = self.generate_outputs()
            y = self.kernel_density_distribution(x_range, x, h)
            hist, bins = np.histogram(x, 20, density=True)
            # plots of the estimates
            self.plot_kernel_density_estimation(x_range, y, hist, bins, h, ref_sig)
            distributions[rn] = y
            rn += 1
        return distributions
