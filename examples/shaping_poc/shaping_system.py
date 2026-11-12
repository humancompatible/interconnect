"""A small interconnect loop whose agent law is the design variable of Algorithm 2.

r -> A1 -> C -> P1 -> F -> A1.  The state x in [0, 1] is the agent's answer, the output of
P1; ControlSystem.run() carries every node's output to the next pass, so A1 always reads the
previous answer.  One agent picks one of m response modes at every iteration, independently,
with law mu.  The closed loop is then a probabilistic IFS x_{t+1} = w_i(x_t), i ~ mu, on [0, 1].
"""

import numpy as np
import scipy.sparse as sp
import torch

from humancompatible.interconnect.simulators.nodes.aggregator import Aggregator
from humancompatible.interconnect.simulators.nodes.controller import Controller
from humancompatible.interconnect.simulators.nodes.filter import Filter
from humancompatible.interconnect.simulators.nodes.population import Population
from humancompatible.interconnect.simulators.nodes.reference import ReferenceSignal
from humancompatible.interconnect.simulators.simulation import Simulation
from tests.contractionTests.contraction_test import get_jacobian_norm

from ims_core import GridChain

# name: (c, g), the response x_next = clamp(c + g * u, 0, 1) to the control signal u = r - x.
# With r = 0.5: low and high are contractions with slope 0.7 and fixed points 0.1 and 0.9;
# herding has slope 1.6 and pushes the state away from 0.5.
DEFAULT_MODES = {
    "low": (0.38, -0.7),
    "high": (0.62, -0.7),
    "herding": (0.5, -1.6),
}


class ModeAgentLogic:
    """A single agent that draws a response mode from the law mu at every step.

    forced_mode overrides the draw; mode_history records the random draws only."""

    def __init__(self, modes, mu=None):
        self.c = torch.tensor([c for c, _ in modes], dtype=torch.float64)
        self.g = torch.tensor([g for _, g in modes], dtype=torch.float64)
        self.m = len(modes)
        self.tensors = {"u": torch.zeros(1, dtype=torch.float64)}
        self.variables = ["u"]
        self.mu = torch.full((self.m,), 1.0 / self.m, dtype=torch.float64) if mu is None else torch.as_tensor(mu, dtype=torch.float64)
        self.forced_mode = None
        self.mode_history = []

    def set_probability(self, mu):
        self.mu = torch.as_tensor(mu, dtype=torch.float64)

    def forward(self, values, number_of_agents):
        self.tensors["u"] = values["u"]
        if self.forced_mode is None:
            i = int(torch.multinomial(self.mu, 1))
            self.mode_history.append(i)
        else:
            i = self.forced_mode
        return torch.clamp(self.c[i] + self.g[i] * values["u"], 0.0, 1.0)


class ProportionalLogic:
    """u = kappa * e."""

    def __init__(self, kappa=1.0):
        self.tensors = {"kappa": torch.tensor([kappa], dtype=torch.float64),
                        "e": torch.zeros(1, dtype=torch.float64)}
        self.variables = ["e"]

    def forward(self, values):
        self.tensors["e"] = values["e"]
        return self.tensors["kappa"] * values["e"]


class SumLogic:
    """Elementwise sum of the incoming signals.

    In the first pass of run() the feedback has no value yet, so only r arrives; ``initial``,
    if set, is used as the missing feedback.  That is how a run starts from a chosen state.
    """

    def __init__(self):
        self.tensors = []
        self.initial = None

    def forward(self, tensors):
        if len(tensors) == 1 and self.initial is not None:
            tensors = tensors + [self.initial]
        self.tensors = tensors
        return sum(tensors)


class NegateLogic:
    """F(x) = -x, so that A1 forms e = r - x."""

    def __init__(self):
        self.tensors = {"S": torch.zeros(1, dtype=torch.float64)}
        self.variables = ["S"]

    def forward(self, values):
        self.tensors["S"] = values["S"]
        return -values["S"]


class ShapingSim(Simulation):
    """r -> A1 -> C -> P1 -> F -> A1, with state x = output of the agent P1."""

    def __init__(self, reference_signal=0.5, modes=None, kappa=1.0):
        super().__init__()
        modes = DEFAULT_MODES if modes is None else modes
        self.mode_names = list(modes)

        self.refsig = ReferenceSignal(name="r")
        self.refsig.set_reference_signal(torch.tensor([reference_signal], dtype=torch.float64))
        self.agg = Aggregator(name="A1", logic=SumLogic())
        self.cont = Controller(name="C", logic=ProportionalLogic(kappa))
        self.pop = Population(name="P1", logic=ModeAgentLogic(list(modes.values())), number_of_agents=1)
        self.fil = Filter(name="F", logic=NegateLogic())
        self.system.add_nodes([self.refsig, self.agg, self.cont, self.pop, self.fil])
        for a, b in [(self.refsig, self.agg), (self.agg, self.cont), (self.cont, self.pop),
                     (self.pop, self.fil), (self.fil, self.agg)]:
            self.system.connect_nodes(a, b)
        self.system.set_start_node(self.refsig)
        self.system.set_checkpoint_node(self.agg)

    @property
    def m(self):
        return self.pop.logic.m

    def one_pass(self, x, i):
        """w_i(x): one pass F -> A1 -> C -> P1 from the state x, with the agent forced to mode i.

        Calls each node's logic in the order of the graph; works on a 1-D batch of states."""
        logic = self.pop.logic
        forced, logic.forced_mode = logic.forced_mode, i
        try:
            f = self.fil.logic.forward({"S": x})
            e = self.agg.logic.forward([self.refsig.ReferenceSignal, f])
            u = self.cont.logic.forward({"e": e})
            return logic.forward({"u": u}, self.pop.number_of_agents)
        finally:
            logic.forced_mode = forced

    def maps(self, x):
        """Array (m, N) of w_i(x_j)."""
        xt = torch.as_tensor(np.asarray(x, dtype=float))
        with torch.no_grad():
            return np.stack([self.one_pass(xt, i).numpy() for i in range(self.m)])

    def derivatives(self, x):
        """Array (N, m) of D[j, i] = |w_i'(x_j)|, from the toolkit's get_jacobian_norm.

        The state is a fresh leaf and the output is the state one pass later, so the
        Jacobian is that of the full step."""
        D = np.empty((len(x), self.m))
        for j, xj in enumerate(np.asarray(x, dtype=float)):
            y = torch.tensor([xj], dtype=torch.float64, requires_grad=True)
            for i in range(self.m):
                D[j, i] = float(get_jacobian_norm(y, self.one_pass(y, i)))
        return D

    def simulate(self, mu, iterations, x0=0.0, seed=None):
        """Run the graph under the law mu from x_0 = x0.

        Returns the state path [x_0, x_1, ...] and modes, where modes[k] took x_k to x_(k+1)."""
        if seed is not None:
            torch.manual_seed(seed)
        logic = self.pop.logic
        logic.set_probability(mu)
        logic.forced_mode = None
        logic.mode_history = []
        with torch.no_grad():
            self.agg.logic.initial = self.fil.logic.forward({"S": torch.tensor([x0], dtype=torch.float64)})
            try:
                self.system.run(iterations, disable_tqdm=True)
            finally:
                self.agg.logic.initial = None
        path = np.concatenate([[x0], torch.cat([h.reshape(1) for h in self.pop.history]).numpy()])
        return path, np.array(logic.mode_history)

    def grid_chain(self, n=801):
        """Finite-state model of the loop on [0, 1], with the mean-preserving
        rounding of GridChain.affine applied to the maps read off the graph."""
        x = np.linspace(0.0, 1.0, n)
        h = x[1] - x[0]
        W = []
        for wi in self.maps(x):
            t = np.clip(wi, 0.0, 1.0) / h
            j = np.minimum(np.floor(t).astype(int), n - 2)
            frac = np.clip(t - j, 0.0, 1.0)
            rows = np.concatenate([np.arange(n), np.arange(n)])
            cols = np.concatenate([j, j + 1])
            vals = np.concatenate([1.0 - frac, frac])
            W.append(sp.csr_matrix((vals, (rows, cols)), shape=(n, n)))
        return GridChain(W, x)


def empirical_measure(path, x):
    """Mean-preserving projection of samples onto the uniform grid x."""
    h = x[1] - x[0]
    t = np.clip(path, x[0], x[-1]) / h
    j = np.minimum(np.floor(t).astype(int), len(x) - 2)
    frac = t - j
    p = np.bincount(j, 1.0 - frac, len(x)) + np.bincount(j + 1, frac, len(x))
    return p / p.sum()
