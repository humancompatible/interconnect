import torch
import torch.nn as nn

from humancompatible.interconnect.simulators.nodes.reference import ReferenceSignal
from humancompatible.interconnect.simulators.nodes.aggregator import Aggregator
from humancompatible.interconnect.simulators.nodes.controller import Controller
from humancompatible.interconnect.simulators.nodes.population import Population
from humancompatible.interconnect.simulators.nodes.delay import Delay
from humancompatible.interconnect.simulators.nodes.filter import Filter
from humancompatible.interconnect.simulators.simulation import Simulation

from humancompatible.interconnect.simulators.logic.delays.exponentialSmoothingDelay import ESDelayLogic
from humancompatible.interconnect.simulators.logic.controllers.ReLU_controller import ReLUControllerLogic
from humancompatible.interconnect.simulators.logic.agents.non_trivial_agent import NonTrivialAgentLogic


class ExampleSim(Simulation):

    class AggregatorLogic1:
        def __init__(self):
            self.tensors = []

        def forward(self, tensors):
            if type(tensors) is not list:
                tensors = [tensors]
            self.tensors = tensors
            result = torch.sum(
                torch.stack([torch.sum(t.detach().clone().requires_grad_()) for t in self.tensors])).unsqueeze(dim=0)
            return result

    class AggregatorLogic2:
        def __init__(self):
            self.tensors = []

        def forward(self, tensors):
            if type(tensors) is not list:
                tensors = [tensors]
            self.tensors = tensors
            return torch.sum(torch.stack([torch.sum(t) for t in self.tensors])).unsqueeze(dim=0)

    class FilterLogic:
        def __init__(self):
            self.tensors = {"S": torch.tensor([0.0], requires_grad=True),
                            "K": torch.tensor([2.0], requires_grad=True)}
            self.variables = ["S"]

        def forward(self, values):
            self.tensors["S"] = values["S"]
            result = - (self.tensors["S"]) / self.tensors["K"]
            return result

    def __init__(self, reference_signal=8.0, weights=None):
        super().__init__()

        refsig = ReferenceSignal(name="r")
        refsig.set_reference_signal(torch.tensor(reference_signal, requires_grad=True, dtype=torch.float))
        agg1 = Aggregator(name="A1", logic=self.AggregatorLogic1())  # Error computation
        agg2 = Aggregator(name="A2", logic=self.AggregatorLogic2())  # Sums population outputs
        cont = Controller(name="C", logic=ReLUControllerLogic())
        pop1 = Population(name="P1", logic=NonTrivialAgentLogic(a=5, b=0), number_of_agents=20)
        pop2 = Population(name="P2", logic=NonTrivialAgentLogic(a=2, b=0.5, offset=-0.4), number_of_agents=20)
        delay = Delay(name="Z", logic=ESDelayLogic())
        fil = Filter(name="F", logic=self.FilterLogic())
        self.system.add_nodes([refsig, agg1, agg2, cont, pop1, pop2, delay, fil])
        self.system.connect_nodes(refsig, agg1)
        self.system.connect_nodes(agg1, cont)
        self.system.connect_nodes(cont, pop1)
        self.system.connect_nodes(cont, pop2)
        self.system.connect_nodes(pop1, agg2)
        self.system.connect_nodes(pop2, agg2)
        self.system.connect_nodes(agg2, delay)
        self.system.connect_nodes(delay, fil)
        self.system.connect_nodes(fil, agg1)

        self.system.set_start_node(refsig)
        self.system.set_checkpoint_node(agg1)

        if weights is not None:
            # Load controller logic
            cont.logic.load_state_dict(torch.load(weights))

        # Set learning node
        model = cont.logic
        self.system.set_learning_model(model)
        self.system.set_loss_function(nn.MSELoss())
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.5)
        self.system.set_optimizer(optimizer)
