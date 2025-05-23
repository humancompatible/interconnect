import torch
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

from humancompatible.interconnect.simulators.distribution import *


def draw_plots(make_plots, ref_sig, combinations, history, g_history, ax1, ax2, inputs_node, outputs_node):
    # plot node histories
    for i in range(len(history)):
        ax1.plot([h.detach().numpy() for h in history[i]], label=f"Sim: reference signal = {ref_sig}; p = {combinations[i]}")
    ax1.set_title(f"Node {make_plots} output values")
    # ax1.legend()
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Value")
    ax1.grid(True)

    # plot gradient histories
    for i in range(len(combinations)):
        ax2.plot(g_history[i], label=f"Grads in Sim: reference signal = {ref_sig}; p = {combinations[i]}")
    ax2.set_title(f"Gradients ({inputs_node} - {outputs_node})")
    # ax2.legend()
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Gradient Value")
    ax2.grid(True)
    return


def get_jacobian_norm(inp, out):
    grad_vectors = []
    for i in range(out.shape[0]):
        grad_vectors.append(torch.autograd.grad(out[i], inp, retain_graph=True)[0])
    J = torch.stack(grad_vectors, dim=0)
    return J.norm(p=1).detach()


def contraction_for_simulation(simulation, inputs_node, outputs_node, make_plots, it=100):
    simulation.system.run(it, show_trace=False, show_loss=False, disable_tqdm=True)

    inputs = simulation.system.get_node(inputs_node).history
    outputs = simulation.system.get_node(outputs_node).history
    n = min(len(inputs), len(outputs))
    g_history = np.array([(get_jacobian_norm(inputs[j], outputs[j])) for j in range(n)]).squeeze()
    max_g = np.max(g_history)
    history = None
    if make_plots is not None:
        history = simulation.system.get_node(make_plots).history
    return max_g, history, g_history


def contraction_single_reference(reference_signal, agent_probs, sim_class, inputs_node, outputs_node, weights=None, it=100, make_plots=None, ax1=None, ax2=None):
    combinations = [1]
    if agent_probs is not None:
        combinations = list(product(*agent_probs))  # combinations of population probs (a[0,0], a[1,0]); (a[0,0], a[1,1])...
    sim = [sim_class(reference_signal, weights) for _ in range(len(combinations))]

    max_grads = np.empty(len(combinations))
    history = [np.empty(1) for _ in range(len(combinations))]
    g_history = [np.empty(1) for _ in range(len(combinations))]

    for i in range(len(combinations)):
        if agent_probs is not None:
            p_num = agent_probs.shape[0]  # shape = (num of populations, list of probabilities for each population)
            for j in range(p_num):
                p_name = "P" + str(j + 1)
                sim[i].system.get_node(p_name).logic.set_probability(combinations[i][j])
        max_grads[i], history[i], g_history[i] = contraction_for_simulation(sim[i], inputs_node, outputs_node, make_plots, it)

    # print(np.mean(g_history, axis=1))

    if make_plots is not None:
        draw_plots(make_plots, reference_signal, combinations, history, g_history, ax1, ax2, inputs_node, outputs_node)

    # print(f"{max_grads} {1 / len(combinations)}")
    # r_factor = np.sum(max_grads * (1 / len(combinations)))
    # print(f"r_factor = {r_factor}")
    # return r_factor, np.mean(g_history, axis=1), history
    return max_grads, np.mean(g_history, axis=1), history


def get_factor_from_list(reference_signals, agent_probs, sim_class, it, trials,
                         weights=None, inputs_node="A1", outputs_node="F",
                         node_outputs_plot=None,
                         show_distributions_plot=True,
                         show_distributions_histograms_plot=True):
    """
    Computes the contraction factor for a set of reference signals and agent probabilities.

    :param reference_signals: List of reference signals.
    :type reference_signals: np.array
    :param agent_probs: Array of agent probabilities.
    :type agent_probs: np.array
    :param sim_class: The simulator class to be used.
    :type sim_class: class
    :param inputs_node: Name of the node outputs of which will be used as gradient inputs.
    :type inputs_node: str
    :param outputs_node: Name of the node outputs of which will be used as gradient outputs.
    :type outputs_node: str
    :param it: Number of iterations to run the simulator. Default is 100.
    :type it: int
    :param show_node_outputs_plot: Flag to indicate whether to generate plots. Default is False.
    :type show_node_outputs_plot: str

    :return: The maximum contraction factor.
    :rtype: float
    """

    fig, ax = None, [None] * 4
    if node_outputs_plot:
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 9))
        ax = ax.flatten()
        ax[3].axis('off')
    len_prod = 1
    if agent_probs is not None:
        len_prod = len(list(product(*agent_probs)))
    maximums = np.zeros((reference_signals.shape[0], len_prod, trials))
    means = np.zeros((reference_signals.shape[0], len_prod, trials))
    end_outputs = np.zeros((reference_signals.shape[0], len_prod, trials))
    for i in range(len(reference_signals)):
        ref_sig = reference_signals[i]
        for j in range(trials):
            temp_res, temp_mean, temp_history = contraction_single_reference(reference_signal=ref_sig,
                                                                             agent_probs=agent_probs, sim_class=sim_class,
                                                                             inputs_node=inputs_node, outputs_node=outputs_node,
                                                                             weights=weights, it=it,
                                                                             make_plots=node_outputs_plot, ax1=ax[0], ax2=ax[2])
            # maximums[i] = max(maximums[i], temp_res)
            # means[i] = temp_mean
            for k in range(len(temp_history)):
                maximums[i][k] = np.maximum(maximums[i][k], temp_res[k])
                means[i][k] = temp_mean[k]
            if node_outputs_plot is not None:
                for k in range(len(temp_history)):
                    end_outputs[i][k][j] = temp_history[k][-1]

    # print(f"maximums {maximums}")
    maximums = maximums.max(axis=2)
    means = means.mean(axis=2)
    combinations = ["Unmodified"]
    if agent_probs is not None:
        combinations = list(product(*agent_probs))  # combinations of population probs (a[0,0], a[1,0]); (a[0,0], a[1,1])...
    if node_outputs_plot is not None:
        if show_distributions_plot:
            for i in range(len(reference_signals)):
                labels = [(f"Reference signal={reference_signals[i]}; "
                           f"Population Probabilities = {combinations[j]}") for j in range(len(combinations))]
                _ = get_distributions(x=end_outputs[i], h=1.9, labels=labels, step=0.1,
                                      show_plots=show_distributions_plot,
                                      show_histograms=show_distributions_histograms_plot,
                                      fig=fig, ax=ax[1], node=node_outputs_plot)
        else:
            ax[1].axis('off')
    if node_outputs_plot:
        plt.show()
    # return maximums, means
    return means


def get_factor_from_space(input_space, agent_probs, sim_class, weights=None,
                          inputs_node="A1", outputs_node="F", sample_paths=1000, it=5):
    len_prod = 1
    if agent_probs is not None:
        len_prod = len(list(product(*agent_probs)))
    means = np.zeros((sample_paths, len_prod))
    for i in range(sample_paths):
        ref_sig = input_space.get_random_point()
        _, temp_mean, _ = contraction_single_reference(reference_signal=ref_sig, agent_probs=agent_probs,
                                                       sim_class=sim_class,
                                                       inputs_node=inputs_node, outputs_node=outputs_node,
                                                       weights=weights, it=it)
        means[i] = temp_mean
    return means.mean(axis=0)
