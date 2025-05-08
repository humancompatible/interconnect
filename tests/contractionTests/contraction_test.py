import torch
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

from humancompatible.interconnect.simulators.distribution import Distribution


def draw_plots(make_plots, ref_sig, combinations, history, g_history, ax1, ax2, inputs_node, outputs_node):
    # plot node histories
    for i in range(len(history)):
        ax1.plot([h.detach().numpy() for h in history[i]], label=f"Sim: reference signal = {ref_sig}; p = {combinations[i]}")
    ax1.set_title(f"Node {make_plots} output values")
    ax1.legend()
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Value")

    # plot gradient histories
    for i in range(len(combinations)):
        ax2.plot(g_history[i], label=f"Grads in Sim: reference signal = {ref_sig}; p = {combinations[i]}")
    ax2.set_title(f"Gradients ({inputs_node} - {outputs_node})")
    ax2.legend()
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Gradient Value")
    return


def get_jacobian_norm(inp, out):
    grad_vectors = []
    for i in range(out.shape[0]):
        grad_vectors.append(torch.autograd.grad(out[i], inp, retain_graph=True)[0])
    J = torch.stack(grad_vectors, dim=0)
    return J.norm(p=1).detach()


def contraction_single(reference_signal, agent_probs, sim_class, inputs_node, outputs_node, it=100, make_plots=None, ax1=None, ax2=None):
    p_num = agent_probs.shape[0]
    combinations = list(product(*agent_probs))
    sim = [sim_class(reference_signal) for _ in range(len(combinations))]

    for i in range(len(combinations)):
        for j in range(p_num):
            p_name = "P" + str(j + 1)
            sim[i].system.get_node(p_name).logic.set_probability(combinations[i][j])
        sim[i].system.run(it, show_trace=False, show_loss=False, disable_tqdm=True)

    g_history = [np.empty(1) for _ in range(len(combinations))]
    for i in range(len(combinations)):
        inputs = sim[i].system.get_node(inputs_node).history
        outputs = sim[i].system.get_node(outputs_node).history
        n = min(len(inputs), len(outputs))
        g_history[i] = np.array([(get_jacobian_norm(inputs[j], outputs[j])) for j in range(n)]).squeeze()
    max_g = np.max(np.abs(g_history), axis=1)

    if make_plots is not None:
        history = [sim[i].system.get_node(make_plots).history for i in range(len(combinations))]
        draw_plots(make_plots, reference_signal, combinations, history, g_history, ax1, ax2, inputs_node, outputs_node)

    r_factor = np.sum(max_g * (1 / len(combinations)))
    return r_factor


def get_factor_from_list(reference_signals, agent_probs, sim_class, inputs_node="A1", outputs_node="F", it=100, make_plots=None):
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
    :param make_plots: Flag to indicate whether to generate plots. Default is False.
    :type make_plots: str

    :return: The maximum contraction factor.
    :rtype: float
    """
    
    res = -np.inf
    fig, ax1, ax2 = None, None, None
    if make_plots:
        fig, (ax1, ax2) = plt.subplots(2, constrained_layout=True, figsize=(10, 8))
    for ref_sig in reference_signals:
        res = np.maximum(res, contraction_single(ref_sig, agent_probs, sim_class,
                                                 inputs_node=inputs_node,
                                                 outputs_node=outputs_node,
                                                 it=it, make_plots=make_plots, ax1=ax1, ax2=ax2))
    if make_plots:
        plt.show()
    return res

def get_factor_from_space(input_space, agent_probs, sim_class, inputs_node="A1", outputs_node="F", sample_paths = 1000, it=5):
    res = torch.tensor([0.0])
    for i in range(sample_paths):
        ref_sig = input_space.get_random_point()
        res = np.maximum(res, contraction_single(ref_sig, agent_probs, sim_class,
                                                 inputs_node=inputs_node, outputs_node=outputs_node, it=it))
    return res


if __name__ == '__main__':
    from examples.example_systems.example_ReLU_system import ExampleReLUSim
    # from example_sim_1 import ExampleSim
    # from example_sim_2 import ExampleReLUSim
    # from example_sim_3 import ExampleSimTwoP
    # example_sim_1 (Default)
    eps = 0.01
    r = get_factor_from_list(reference_signals=np.array([100.0, 300.0]), agent_probs=np.array([[eps, 1 - eps]]), it=100, make_plots="C", sim_class=ExampleReLUSim)
    print(f"Factor = {r}")
    # run_sim(sim_class=ExampleSim, reference_signal=100.0, it=300)

    # example_sim_2 (ReLU controller)
    # r = contraction(reference_signal=4.0, agent_probs=[0.0, 1.0], it=100, make_plots=True, sim_class=ExampleReLUSim)
    # print(f"Factor = {r}")
    # run_sim(sim_class=ExampleReLUSim, reference_signal=100.0, it=300)

    # example_sim_3 (2 populations)
    # r = get_factor(reference_signals=np.array([300.0, 100.0]), agent_probs=np.array([[0.0, 1.0], [0.0, 1.0]]), it=100, make_plots=True, sim_class=ExampleSimTwoP)
    # print(f"Factor = {r}")
    # run_sim(sim_class=ExampleSimTwoP, reference_signal=100.0, it=600)

    # dist = Distribution(ExampleSimTwoP, iterations=100)
    # distributions = dist.get_distributions(h=1.0, reference_signals=np.array([100.0, 115.0, 130.0]), x_min=-50.0, x_max=50.0)
