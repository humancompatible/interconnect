import torch
import matplotlib.pyplot as plt
import numpy as np
import example_sim_1


def draw_plots(agent_probs, history, g_history):
    # plot node histories
    fig, (ax1, ax2) = plt.subplots(2, constrained_layout=True, figsize=(10, 8))
    for i in range(len(history)):
        ax1.plot([h.detach().numpy() for h in history[i]], label=f"Sim p = {agent_probs[i]}")
    ax1.set_title("Node output values")
    ax1.legend()
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Value")

    # plot gradient histories
    for i in range(len(agent_probs)):
        ax2.plot(g_history[i], label=f"Grads in Sim p = {agent_probs[i]}")
    ax2.set_title("Gradients")
    ax2.legend()
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Gradient Value")
    plt.show()
    return


def contraction(reference_signal, agent_probs, it=100, make_plots=False):
    sim = [example_sim_1.ExampleSim(reference_signal) for _ in agent_probs]

    for i in range(len(agent_probs)):
        sim[i].system.get_node("P1").logic.set_probability(agent_probs[i])
        sim[i].system.run(it, show_trace=False, show_loss=False)

    history = [sim[i].system.get_node("A1").history for i in range(len(agent_probs))]

    # get gradients
    g_history = [np.empty(1) for _ in range(len(agent_probs))]
    for i in range(len(agent_probs)):
        inputs = sim[i].system.get_node("A1").history
        outputs = sim[i].system.get_node("F").history
        n = min(len(inputs), len(outputs))
        g_history[i] = np.array([(torch.autograd.grad(outputs[j], inputs[j], retain_graph=True)[0]).detach().numpy() for j in range(n)]).squeeze()
    max_g = np.max(np.abs(g_history), axis=1)

    if make_plots:
        draw_plots(agent_probs, history, g_history)

    r_factor = np.sum(max_g * 0.5)
    return r_factor


if __name__ == '__main__':
    r = contraction(reference_signal=300.0, agent_probs=[0.0, 1.0], it=100, make_plots=True)
    print(f"Factor = {r}")
