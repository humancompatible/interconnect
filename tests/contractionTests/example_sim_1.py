import torch
import torch.nn as nn

from humancompatible.interconnect.simulators.nodes.reference import ReferenceSignal
from humancompatible.interconnect.simulators.nodes.aggregator import Aggregator
from humancompatible.interconnect.simulators.nodes.controller import Controller
from humancompatible.interconnect.simulators.nodes.population import Population
from humancompatible.interconnect.simulators.nodes.delay import Delay
from humancompatible.interconnect.simulators.nodes.filterer import Filterer
from humancompatible.interconnect.simulators.simulation import Simulation


class ExampleSim(Simulation):
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
        def __init__(self):
            self.tensors = {"x": torch.tensor([0.0], requires_grad=True),
                            "pi": torch.tensor([0.0], requires_grad=True)}
            self.variables = ["p"]
            self.probability = 0.5

        def set_probability(self, probability):
            self.probability = probability

        def forward(self, values, number_of_agents):
            self.tensors["p"] = values["p"]
            f1_part = torch.bernoulli(torch.ones(1, 1) * self.probability) * 0.6
            # f1_part = torch.bernoulli(torch.ones(1, 1) * 0.5) / number_of_agents
            # result = f1_part * self.tensors["p"] + (1 - f1_part) * (4.2 * self.tensors["p"] - 10.0)
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

    def __init__(self, reference_signal):
        super().__init__()
        # create nodes
        refsig = ReferenceSignal(name="r")
        refsig.set_reference_signal(reference_signal)
        agg1 = Aggregator(name="A1", logic=self.AggregatorLogic1())
        agg2 = Aggregator(name="A2", logic=self.AggregatorLogic2())
        cont = Controller(name="C", logic=self.PiControllerLogic(kp=0.5, ki=0.5))
        pop1 = Population(name="P1", logic=self.AgentLogic(), number_of_agents=2000)
        delay = Delay(name="Z", time=1)
        fil = Filterer(name="F", logic=self.FiltererLogic())

        # build system
        self.system.add_nodes([refsig, agg1, agg2, cont, pop1, delay, fil])
        self.system.connect_nodes(refsig, agg1)
        self.system.connect_nodes(agg1, cont)
        self.system.connect_nodes(cont, pop1)
        self.system.connect_nodes(pop1, agg2)
        self.system.connect_nodes(agg2, delay)
        self.system.connect_nodes(delay, fil)
        self.system.connect_nodes(fil, agg1)
        self.system.set_start_node(refsig)
        self.system.set_checkpoint_node(agg1)
        # define learning node
        model = self.system.get_node("C").logic
        # sim.system.set_learning_model(model)
        self.system.set_loss_function(nn.L1Loss())
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        self.system.set_optimizer(optimizer)
