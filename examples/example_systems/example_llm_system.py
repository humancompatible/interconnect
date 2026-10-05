"""Closed loop with an LLM agent and an if-else instruction-ladder controller.

Topology is deliberately identical to ``example_nontrivial_system.ExampleSim`` --
only the population logic and the controller change::

    r=0 -> A1 (e = r - y) -> C (level k) -> P1 (LLM ensemble) -> A2 -> Z -> F -> A1
            ^ checkpoint

Two design points that the ergodicity analysis depends on:

* The delay is configured with ``alpha = 1`` (pass-through) so the controller
  decides from the *current* observation only.  The loop state is then the
  instruction level alone and the closed loop is a finite Markov chain.  With
  ``alpha < 1`` the state becomes the hybrid pair ``(k, y)`` and the finite-state
  certificates in ``guarantees.markov`` no longer apply.
* ``number_of_agents`` is the number of LLM samples averaged per step.  Every run
  in ``examples/llm_poc`` uses the default of 1, so the sampling noise reaches the
  controller undiminished and the observation law is the pool itself rather than a
  mean over it.
"""

import torch

from humancompatible.interconnect.simulators.simulation import Simulation
from humancompatible.interconnect.simulators.nodes.reference import ReferenceSignal
from humancompatible.interconnect.simulators.nodes.aggregator import Aggregator
from humancompatible.interconnect.simulators.nodes.controller import Controller
from humancompatible.interconnect.simulators.nodes.population import Population
from humancompatible.interconnect.simulators.nodes.delay import Delay
from humancompatible.interconnect.simulators.nodes.filter import Filter

from humancompatible.interconnect.simulators.logic.delays.exponentialSmoothingDelay import ESDelayLogic
from humancompatible.interconnect.simulators.logic.controllers.instruction_ladder import IfElseLadderLogic
from humancompatible.interconnect.simulators.logic.agents.llm_agent import NUM_LEVELS


class LLMSim(Simulation):

    class AggregatorErrorLogic:
        """A1: the error node, ``e = r + F``.  Also the checkpoint."""

        def __init__(self):
            self.tensors = []

        def forward(self, tensors):
            if not isinstance(tensors, list):
                tensors = [tensors]
            self.tensors = tensors
            return torch.sum(torch.stack([torch.sum(t) for t in tensors])).unsqueeze(0).detach()

    class AggregatorPassLogic:
        """A2: collects the population output (a single ensemble mean here)."""

        def __init__(self):
            self.tensors = []

        def forward(self, tensors):
            if not isinstance(tensors, list):
                tensors = [tensors]
            self.tensors = tensors
            return torch.sum(torch.stack([torch.sum(t) for t in tensors])).unsqueeze(0).detach()

    class FilterLogic:
        """F: sign inversion and loop gain, ``F = -y / K``."""

        def __init__(self, K=1.0):
            self.tensors = {"S": torch.tensor([0.0]), "K": torch.tensor([float(K)])}
            self.variables = ["S"]

        def forward(self, values):
            self.tensors["S"] = values["S"]
            return -(self.tensors["S"]) / self.tensors["K"]

    def __init__(self, reference_signal=0.0, weights=None, agent_logic=None,
                 number_of_agents=1, num_levels=NUM_LEVELS, tau=0.25,
                 tau_offset=0.0, initial_level=None, alpha=1.0, loop_gain=1.0,
                 polarity=+1):
        super().__init__()
        if agent_logic is None:
            raise ValueError(
                "agent_logic is required: pass CachedLLMAgentLogic.from_npz(pools) "
                "for analysis runs, or LLMAgentLogic(responder, scorer) for live runs."
            )

        refsig = ReferenceSignal(name="r")
        refsig.set_reference_signal(torch.tensor(float(reference_signal)))
        agg1 = Aggregator(name="A1", logic=self.AggregatorErrorLogic())
        cont = Controller(name="C", logic=IfElseLadderLogic(
            num_levels=num_levels, tau=tau, tau_offset=tau_offset,
            initial_level=initial_level, loop_gain=loop_gain, polarity=polarity))
        pop1 = Population(name="P1", logic=agent_logic, number_of_agents=number_of_agents)
        agg2 = Aggregator(name="A2", logic=self.AggregatorPassLogic())
        delay = Delay(name="Z", logic=ESDelayLogic(alpha=alpha))
        fil = Filter(name="F", logic=self.FilterLogic(K=loop_gain))

        self.system.add_nodes([refsig, agg1, cont, pop1, agg2, delay, fil])
        self.system.connect_nodes(refsig, agg1)
        self.system.connect_nodes(agg1, cont)
        self.system.connect_nodes(cont, pop1)
        self.system.connect_nodes(pop1, agg2)
        self.system.connect_nodes(agg2, delay)
        self.system.connect_nodes(delay, fil)
        self.system.connect_nodes(fil, agg1)

        self.system.set_start_node(refsig)
        self.system.set_checkpoint_node(agg1)

        self.controller_logic = cont.logic
        self.agent_logic = agent_logic

    # -- convenience accessors -------------------------------------------------
    @property
    def level_history(self):
        """Instruction level visited at each iteration -- the Markov chain path."""
        return list(self.controller_logic.level_history)

    def sentiment_history(self):
        """Observed ensemble-mean sentiment at each iteration."""
        return [float(h.item()) for h in self.system.get_node("P1").history]
