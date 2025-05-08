import torch
import torch.nn as nn

from humancompatible.interconnect.simulators.simulation import Simulation
from humancompatible.interconnect.simulators.nodes.reference import ReferenceSignal
from humancompatible.interconnect.simulators.nodes.aggregator import Aggregator
from humancompatible.interconnect.simulators.nodes.controller import Controller
from humancompatible.interconnect.simulators.nodes.population import Population
from humancompatible.interconnect.simulators.nodes.filter import Filter


class BlankSystem(Simulation):
    """
    Minimal system for test purposes where all nodes except controller pass the unchanged
    values from the previous node (not impacting Lipschitz constant of the system),
    this allows to check correctness of estimator when comparing to exact Lipschitz constant value
    of the controller's neural net
    """

    class AggregatorLogic1:
        def __init__(self):
            self.tensors = []
            self.result = None

        def forward(self, values):
            if type(values) is not list:
                values = [values]
            self.tensors = values
            result = self.tensors[0].detach().clone().requires_grad_()
            self.result = result
            return result

    class ReLUControllerLogic(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_features=2, out_features=128),
                nn.ReLU(),
                nn.Linear(in_features=128, out_features=128),
                nn.ReLU(),
                nn.Linear(in_features=128, out_features=1)
            )
            self.tensors = {"e": torch.tensor([0.0], requires_grad=True)}
            self.variables = ["e"]

        def forward(self, values):
            self.tensors["e"] = values["e"]
            result = self.net(self.tensors["e"])
            return result

    class AgentLogic:
        def __init__(self):
            self.tensors = {"p": torch.tensor([0.0], requires_grad=True)}
            self.variables = ["p"]
            self.probability = 0.5

        def set_probability(self, probability):
            self.probability = probability

        def forward(self, values, number_of_agents):
            self.tensors["p"] = values["p"]
            result = self.tensors["p"]
            return result

    class FilterLogic:
        def __init__(self):
            self.tensors = {"S": torch.tensor([0.0], requires_grad=True)}
            self.variables = ["S"]
            self.result = None

        def forward(self, values):
            self.tensors["S"] = values["S"]
            result = - self.tensors["S"]
            self.result = result
            return result

    def __init__(self, reference_signal=0.0):
        super().__init__()
        refsig = ReferenceSignal(name="r")
        refsig.set_reference_signal(torch.tensor(reference_signal, requires_grad=True, dtype=torch.float))
        agg1 = Aggregator(name="A1", logic=self.AggregatorLogic1())
        cont = Controller(name="C", logic=self.ReLUControllerLogic())
        cont.logic.load_state_dict(torch.load("./test_weights/weights2D.pth", weights_only=True))
        pop1 = Population(name="P1", logic=self.AgentLogic(), number_of_agents=200)
        fil = Filter(name="F", logic=self.FilterLogic())

        self.system.add_nodes([refsig, agg1, cont, pop1, fil])
        self.system.connect_nodes(refsig, agg1)
        self.system.connect_nodes(agg1, cont)
        self.system.connect_nodes(cont, pop1)
        self.system.connect_nodes(pop1, fil)
        self.system.connect_nodes(fil, agg1)

        self.system.set_start_node(refsig)
        self.system.set_checkpoint_node(agg1)
