"""One-step maps of a closed loop and their slopes, measured by running the graph.

The loop is treated as a black box that ``ControlSystem.run()`` can execute.  Its state is the
output of one node (``state_node``), and a population node picks one of ``m`` response modes at
every pass.  One pass of the graph from a state ``y`` with the mode held at ``i`` is the map
``w_i(y)``.

    one_pass     one ``run(1)`` pass from a chosen state, optionally with the mode forced
    mode_maps    w_i(y_j) at chosen states
    mode_slopes  D[j, i] = |w_i'(y_j)|, how much mode i stretches distances at the state y_j

The table ``D`` does not depend on the law ``mu`` of the modes.  The contraction-on-average
constant of the Interconnect paper, eq. (12), restricted to the states in ``D``, is
``max_j sum_i mu_i D[j, i]`` for any ``mu``; it is an estimate from below, because the maximum
is over the states in ``D`` only.

The population's logic must have an attribute ``forced_mode``: ``None`` means "draw the mode at
random", an integer ``i`` means "use mode ``i``".
"""

import numpy as np
import torch


def get_jacobian_norm(inp, out):
    """Entrywise 1-norm of the Jacobian d out / d inp, by autograd."""
    grad_vectors = []
    for i in range(out.shape[0]):
        grad_vectors.append(torch.autograd.grad(out[i], inp, retain_graph=True)[0])
    J = torch.stack(grad_vectors, dim=0)
    return J.norm(p=1).detach()


def _node(system, node):
    return system.get_node(node) if isinstance(node, str) else node


def _population(system, population):
    if population is not None:
        return _node(system, population)
    populations = [n for n in system.nodes if n.type == "Population"]
    if len(populations) != 1:
        raise ValueError("The system has several Population nodes; pass the one whose modes are forced.")
    return populations[0]


def one_pass(system, state_node, state, mode=None, population=None):
    """Output of ``state_node`` after one ``run(1)`` pass that starts from ``state``.

    :param state: tensor that ``state_node`` holds before the pass; autograd connections are kept.
    :param mode: mode forced on the population for this pass; ``None`` leaves the random draw.
    """
    state_node = _node(system, state_node)
    logic = _population(system, population).logic
    if mode is not None and not hasattr(logic, "forced_mode"):
        raise ValueError("The population's logic has no 'forced_mode' attribute.")
    previous = getattr(logic, "forced_mode", None)
    if mode is not None:
        logic.forced_mode = mode
    try:
        system.run(1, disable_tqdm=True, initial_outputs={state_node: state})
    finally:
        if mode is not None:
            logic.forced_mode = previous
    return state_node.history[0]


def mode_maps(system, state_node, states, modes, population=None):
    """Array (m, N) of ``w_i(y_j)`` for scalar states: one batched ``run(1)`` pass per mode."""
    y = torch.as_tensor(np.asarray(states, dtype=float))
    with torch.no_grad():
        return np.stack([one_pass(system, state_node, y, i, population).numpy() for i in range(modes)])


def mode_slopes(system, state_node, states, modes, population=None):
    """Array (N, m) of ``D[j, i] = |w_i'(y_j)|``: ``get_jacobian_norm`` of one pass, per state and mode.

    :param states: N states; each is a scalar or a 1-D vector.
    :param modes: number of modes m of the population.
    """
    states = np.asarray(states, dtype=float)
    states = states.reshape(len(states), -1)
    D = np.empty((len(states), modes))
    for j, yj in enumerate(states):
        y = torch.tensor(yj, dtype=torch.float64, requires_grad=True)
        for i in range(modes):
            D[j, i] = float(get_jacobian_norm(y, one_pass(system, state_node, y, i, population)))
    return D
