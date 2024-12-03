import torch
import matplotlib.pyplot as plt
import numpy as np
from itertools import product


def run_sim(sim_class, reference_signal, it=100):
    sim = sim_class(reference_signal)
    sim.system.run(it, show_trace=False, show_loss=False, disable_tqdm=True)
    sim.plot.node_outputs(sim.system.get_node("A1"))


def draw_plots(combinations, history, g_history):
    # plot node histories
    fig, (ax1, ax2) = plt.subplots(2, constrained_layout=True, figsize=(10, 8))
    for i in range(len(history)):
        ax1.plot([h.detach().numpy() for h in history[i]], label=f"Sim p = {combinations[i]}")
    ax1.set_title("Node output values")
    ax1.legend()
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Value")

    # plot gradient histories
    for i in range(len(combinations)):
        ax2.plot(g_history[i], label=f"Grads in Sim p = {combinations[i]}")
    ax2.set_title("Gradients")
    ax2.legend()
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Gradient Value")
    plt.show()
    return


def contraction(reference_signal, agent_probs, sim_class, it=100, make_plots=False):
    p_num = agent_probs.shape[0]
    combinations = list(product(*agent_probs))
    sim = [sim_class(reference_signal) for _ in range(len(combinations))]

    for i in range(len(combinations)):
        for j in range(p_num):
            p_name = "P" + str(j + 1)
            sim[i].system.get_node(p_name).logic.set_probability(combinations[i][j])
            sim[i].system.run(it, show_trace=False, show_loss=False, disable_tqdm=True)

    history = [sim[i].system.get_node("A1").history for i in range(len(combinations))]

    # get gradients
    g_history = [np.empty(1) for _ in range(len(combinations))]
    for i in range(len(combinations)):
        inputs = sim[i].system.get_node("A1").history
        outputs = sim[i].system.get_node("F").history
        n = min(len(inputs), len(outputs))
        g_history[i] = np.array([(torch.autograd.grad(outputs[j], inputs[j], retain_graph=True)[0]).detach().numpy() for j in range(n)]).squeeze()
    max_g = np.max(np.abs(g_history), axis=1)

    if make_plots:
        draw_plots(combinations, history, g_history)

    r_factor = np.sum(max_g * (1 / len(combinations)))
    return r_factor


if __name__ == '__main__':
    from example_sim_1 import ExampleSim
    from example_sim_2 import ExampleReLUSim
    from example_sim_3 import ExampleSimTwoP
    # example_sim_1 (Default)
    # r = contraction(reference_signal=100.0, agent_probs=np.array([[0.0, 1.0]]), it=100, make_plots=True, sim_class=ExampleSim)
    # print(f"Factor = {r}")
    # run_sim(sim_class=ExampleSim, reference_signal=100.0, it=300)

    # example_sim_2 (ReLU controller)
    # r = contraction(reference_signal=4.0, agent_probs=[0.0, 1.0], it=100, make_plots=True, sim_class=ExampleReLUSim)
    # print(f"Factor = {r}")
    # run_sim(sim_class=ExampleReLUSim, reference_signal=100.0, it=300)

    # example_sim_3 (2 populations)
    r = contraction(reference_signal=300.0, agent_probs=np.array([[0.0, 1.0], [0.0, 1.0]]), it=100, make_plots=True, sim_class=ExampleSimTwoP)
    print(f"Factor = {r}")
    run_sim(sim_class=ExampleSimTwoP, reference_signal=100.0, it=600)
