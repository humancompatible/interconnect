import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from humancompatible.interconnect.simulators.nodes.reference import ReferenceSignal
from humancompatible.interconnect.simulators.nodes.aggregator import Aggregator
from humancompatible.interconnect.simulators.nodes.controller import Controller
from humancompatible.interconnect.simulators.nodes.population import Population
from humancompatible.interconnect.simulators.nodes.delay import Delay
from humancompatible.interconnect.simulators.nodes.filterer import Filterer
from humancompatible.interconnect.simulators.simulation import Simulation


# Following classes are similar to ones created in stable_simulation.ipynb example notebook
class AggregatorLogic1:
    def __init__(self):
        self.tensors = []
        self.result = None

    def forward(self, values):
        if type(values) is not list:
            values = [values]
        self.tensors = values
        result = torch.sum(
            torch.stack([torch.sum(t.detach().clone().requires_grad_()) for t in self.tensors])).unsqueeze(dim=0)
        self.result = result
        return result


class AggregatorLogic2:
    def __init__(self):
        self.tensors = []

    def forward(self, values):
        if type(values) is not list:
            values = [values]
        self.tensors = values
        result = torch.sum(torch.stack([torch.sum(t) for t in self.tensors])).unsqueeze(dim=0)
        return result


class PiControllerLogic(torch.nn.Module):
    def __init__(self, kp, ki):
        super().__init__()
        self.tensors = {
            "e": torch.tensor([0.0], requires_grad=True),
            "x": torch.tensor([0.0], requires_grad=True)}
        self.kp = torch.nn.Parameter(torch.tensor([kp], dtype=torch.float32))
        self.ki = torch.nn.Parameter(torch.tensor([ki], dtype=torch.float32))
        self.variables = ["e"]

    def forward(self, values):
        self.tensors["e"] = values["e"]
        result = ((self.kp * self.tensors["e"]) +
                  (self.ki * (self.tensors["x"] + self.tensors["e"])))
        self.tensors["x"] = self.tensors["x"] + self.tensors["e"]
        return result


class AgentLogic:
    def __init__(self, probability=0.5):
        self.tensors = {"x": torch.tensor([0.0], requires_grad=True),
                        "pi": torch.tensor([0.0], requires_grad=True)}
        self.variables = ["p"]
        self.probability = probability

    def forward(self, values, number_of_agents):
        self.tensors["p"] = values["p"]
        f1_part = torch.bernoulli(torch.ones(1, 1) * self.probability) * 0.6
        # f1_part = torch.bernoulli(torch.ones(1, 1) * 0.5) / number_of_agents
        # result = f1_part * self.tensors["p"] + (1 - f1_part) * (2.2 * self.tensors["p"] - 10.0)
        result = f1_part * self.tensors["p"] + (1 - f1_part) * (self.tensors["p"] - 10.0)
        # result = f1_part * self.tensors["p"]
        return result


class FiltererLogic:
    def __init__(self):
        self.tensors = {"S": torch.tensor([0.0], requires_grad=True),
                        "K": torch.tensor([3.0], requires_grad=True)}
        self.variables = ["S"]
        self.result = None

    def forward(self, values):
        self.tensors["S"] = values["S"]
        result = - self.tensors["S"] / self.tensors["K"]
        self.result = result
        return result


def create_system(reference_signal, probability):
    # create nodes
    refsig = ReferenceSignal(name="r")
    refsig.set_reference_signal(reference_signal)
    agg1 = Aggregator(name="A1", logic=AggregatorLogic1())
    agg2 = Aggregator(name="A2", logic=AggregatorLogic2())
    cont = Controller(name="C", logic=PiControllerLogic(kp=0.5, ki=0.5))
    pop1 = Population(name="P1", logic=AgentLogic(probability=probability), number_of_agents=1000)
    pop2 = Population(name="P2", logic=AgentLogic(probability=probability), number_of_agents=1000)
    delay = Delay(name="Z", time=1)
    fil = Filterer(name="F", logic=FiltererLogic())

    # build system
    sim = Simulation()
    sim.system.add_nodes([refsig, agg1, agg2, cont, pop1, pop2, delay, fil])
    sim.system.connect_nodes(refsig, agg1)
    sim.system.connect_nodes(agg1, cont)
    sim.system.connect_nodes(cont, pop1)
    sim.system.connect_nodes(cont, pop2)
    sim.system.connect_nodes(pop1, agg2)
    sim.system.connect_nodes(pop2, agg2)
    sim.system.connect_nodes(agg2, delay)
    sim.system.connect_nodes(delay, fil)
    sim.system.connect_nodes(fil, agg1)
    sim.system.set_start_node(refsig)
    sim.system.set_checkpoint_node(agg1)

    # define learning node
    model = sim.system.get_node("C").logic
    # sim.system.set_learning_model(model)
    sim.system.set_loss_function(nn.L1Loss())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    sim.system.set_optimizer(optimizer)
    return sim


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
    sim = [create_system(reference_signal, p) for p in agent_probs]

    for i in range(len(agent_probs)):
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
