"""A loop whose one-step maps have no hand-derivable Lipschitz constants.

r -> A1 -> F -> C -> P1 -> A1.  The state y in (0, 1) is the agent's answer, the output of P1.
A1 forms the error e = r - y, the filter F distorts it nonlinearly, the controller C is a small
network with fixed random weights, and the agent P1 draws one of three sigmoid response modes
with law mu at every step.

Everything is computed by running the graph with ControlSystem.run(): a run starts from a chosen
state through ``initial_outputs={"P1": y}``, and one pass from y is ``run(1)``.
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
from humancompatible.interconnect.guarantees.contraction import mode_maps, mode_slopes, one_pass

from ims_core import GridChain

# name: (c, s), the response y = sigmoid(c + s * u) to the control signal u.
# calm has an attracting fixed point at 0.5; each polariser has a repelling one, at 0.25 and 0.75.
MODES = {
    "calm": (-2.3, 4.2),
    "polarise low": (9.5, -16.5),
    "polarise high": (9.6, -18.6),
}


class DifferenceLogic:
    """e = r - y: the first input (the reference) minus the second (the fed-back answer)."""

    def __init__(self):
        self.tensors = []

    def forward(self, tensors):
        self.tensors = tensors
        reference, feedback = tensors
        return reference - feedback


class SineFilterLogic:
    """F(e) = e - 0.05 sin(2 pi e)."""

    def __init__(self):
        self.tensors = {"S": torch.zeros(1, dtype=torch.float64)}
        self.variables = ["S"]

    def forward(self, values):
        self.tensors["S"] = values["S"]
        return values["S"] - 0.05 * torch.sin(2 * np.pi * values["S"])


class NetworkControllerLogic:
    """u = W2 tanh(W1 e + b1), with fixed random weights."""

    def __init__(self, hidden=16, seed=2):
        g = torch.Generator().manual_seed(seed)
        self.W1 = torch.randn(hidden, generator=g, dtype=torch.float64)
        self.b1 = torch.randn(hidden, generator=g, dtype=torch.float64)
        self.W2 = torch.randn(hidden, generator=g, dtype=torch.float64) / 4
        self.tensors = {"e": torch.zeros(1, dtype=torch.float64)}
        self.variables = ["e"]

    def forward(self, values):
        self.tensors["e"] = values["e"]
        e = values["e"]
        return torch.tanh(e[..., None] * self.W1 + self.b1) @ self.W2


class SigmoidModeAgentLogic:
    """One agent that draws a response mode from mu at every step; forced_mode overrides the draw.

    mode_history records the random draws only."""

    def __init__(self, modes):
        self.c = torch.tensor([c for c, _ in modes], dtype=torch.float64)
        self.s = torch.tensor([s for _, s in modes], dtype=torch.float64)
        self.m = len(modes)
        self.mu = torch.full((self.m,), 1.0 / self.m, dtype=torch.float64)
        self.tensors = {"u": torch.zeros(1, dtype=torch.float64)}
        self.variables = ["u"]
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
        return torch.sigmoid(self.c[i] + self.s[i] * values["u"])


class OpaqueSim(Simulation):

    def __init__(self, reference_signal=0.5, modes=None):
        super().__init__()
        modes = MODES if modes is None else modes
        self.mode_names = list(modes)

        self.refsig = ReferenceSignal(name="r")
        self.refsig.set_reference_signal(torch.tensor([reference_signal], dtype=torch.float64))
        self.agg = Aggregator(name="A1", logic=DifferenceLogic())
        self.fil = Filter(name="F", logic=SineFilterLogic())
        self.cont = Controller(name="C", logic=NetworkControllerLogic())
        self.agent = Population(name="P1", logic=SigmoidModeAgentLogic(list(modes.values())), number_of_agents=1)
        self.system.add_nodes([self.refsig, self.agg, self.fil, self.cont, self.agent])
        # r is connected to A1 before P1, so A1 receives [r, y] and DifferenceLogic forms r - y.
        for a, b in [(self.refsig, self.agg), (self.agg, self.fil), (self.fil, self.cont),
                     (self.cont, self.agent), (self.agent, self.agg)]:
            self.system.connect_nodes(a, b)
        self.system.set_start_node(self.refsig)
        self.system.set_checkpoint_node(self.agg)

    @property
    def m(self):
        return self.agent.logic.m

    def step(self, y, mode):
        """w_mode(y): the agent's next answer after one run() pass from the state y, with the mode forced.

        y may be a 1-D batch of states; autograd connections to y are kept."""
        return one_pass(self.system, "P1", y, mode)

    def maps(self, x):
        """Array (m, N) of w_i(x_j), one batched run() pass per mode."""
        return mode_maps(self.system, "P1", x, self.m)

    def slopes(self, states):
        """Array (N, m) of D[j, i] = |w_i'(y_j)|: get_jacobian_norm of one run() pass, per state and mode."""
        return mode_slopes(self.system, "P1", states, self.m)

    def simulate(self, mu, iterations, x0=0.0, seed=None):
        """Run the graph under the law mu from y_0 = x0.

        Returns the state path [y_0, y_1, ...] and modes, where modes[k] took y_k to y_(k+1)."""
        if seed is not None:
            torch.manual_seed(seed)
        logic = self.agent.logic
        logic.set_probability(mu)
        logic.forced_mode = None
        logic.mode_history = []
        with torch.no_grad():
            self.system.run(iterations, disable_tqdm=True,
                            initial_outputs={"P1": torch.tensor([x0], dtype=torch.float64)})
        path = np.concatenate([[x0], torch.cat([h.reshape(1) for h in self.agent.history]).numpy()])
        return path, np.array(logic.mode_history)


def grid_chain(maps, x):
    """GridChain from maps (m, n) on the uniform grid x, with the mean-preserving rounding
    of GridChain.affine."""
    n, h = len(x), x[1] - x[0]
    W = []
    for wi in maps:
        t = (np.clip(wi, x[0], x[-1]) - x[0]) / h
        j = np.minimum(np.floor(t).astype(int), n - 2)
        frac = np.clip(t - j, 0.0, 1.0)
        rows = np.concatenate([np.arange(n), np.arange(n)])
        W.append(sp.csr_matrix((np.concatenate([1.0 - frac, frac]), (rows, np.concatenate([j, j + 1]))), shape=(n, n)))
    return GridChain(W, x)
