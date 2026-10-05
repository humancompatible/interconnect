"""Signed if-else controller over a ladder of prompt instructions.

The controller's only actuator is an index ``k`` into
``logic.agents.llm_agent.NEUTRALITY_LADDER``, which runs from "be extremely
enthusiastic" (k = 0) through a neutrality directive to "be extremely critical"
(k = last).  The rule is the obvious one: if the response came out too positive,
push one rung toward critical; too negative, push one rung toward positive;
otherwise hold.

Why signed, and why a hold band
-------------------------------
* A one-sided "be more neutral" ladder has no authority to straddle the
  reference.  Measured on a 0.5B instruct model it moved sentiment by 0.014
  across six rungs while the baseline sat at +0.85, so the error never entered
  the deadband and the level pinned at the top rung forever.
* The hold band is what makes the chain aperiodic.  Without it the level is a
  pure +-1 walk, which has period 2: still a unique invariant measure, but no
  convergence in total variation and no Doeblin minorization.  Because the signed
  rule has a natural dead zone ``|s| <= tau``, one threshold suffices -- the
  magnitude formulation needed two.

Why the dead zone can be off-centre
-----------------------------------
``tau_offset`` shifts the whole hold band to ``[tau_offset - tau, tau_offset +
tau]``: the controller holds while ``|s - tau_offset| <= tau`` instead of while
``|s| <= tau``, so the tolerance for "too positive" and "too negative" stop being
equal.  It exists because the plant is *biased*.  This model writes positively
about almost anything and the classifier saturates near +1, so a band centred on
the reference is not centred on the observations, and the damage is visible in the
rates: at ``tau_offset = 0`` the back-off rate ``P(s < -tau | k)`` is exactly zero
on the four most positive rungs, which makes them transient and turns the lower
half of the ladder into a one-way funnel.

Two things to be clear about.  The offset is applied to the *observation*, which
is the same as regulating ``s - tau_offset`` to zero -- so a non-zero offset means
the loop is no longer aiming at the stated reference ``r = 0``, it is aiming at
``tau_offset``.  That is a legitimate move when the offset is identified as sensor
bias and a silent redefinition of the objective when it is not.  And it is not
free: recentring the band to balance the up and down rates buys a
better-conditioned chain with distance from the reference, which is why
``run_demo.py`` sweeps the offset and reports both quantities side by side rather
than optimising the conditioning alone.
"""

import numpy as np
import torch


class IfElseLadderLogic:
    """:param tau: dead zone half-width on the sentiment observation.
    :param tau_offset: centre of the dead zone.  The controller holds while
        ``|s - tau_offset| <= tau``, so the up threshold is ``tau_offset + tau``
        and the down threshold ``tau_offset - tau``.  Non-zero values make the
        two tolerances asymmetric about the reference; see the module docstring
        for why a biased plant wants that and what it costs.
    :param num_levels: size of the instruction ladder.
    :param initial_level: level at ``t = 0``; varying it is how the PoC probes
        whether the invariant measure is independent of the initial condition.
    :param loop_gain: the ``K`` used by the filter node, needed to recover the
        sentiment from the error signal.
    :param polarity: behaviour of the controller. Either +1 (corrective, default), or -1 (amplifying). 
    """

    def __init__(self, num_levels=7, tau=0.25, tau_offset=0, initial_level=None, loop_gain=1.0,
                 polarity=+1):
        if tau < 0:
            raise ValueError("tau must be non-negative")
        self.tensors = {"e": torch.tensor([0.0])}
        self.variables = ["e"]
        # num_levels=None is the naive one-sided design: "append one more copy of
        # the instruction" has no last rung, so the state space is unbounded and
        # the loop can escape to infinity instead of settling.
        self.num_levels = None if num_levels is None else int(num_levels)
        self.tau = float(tau)
        self.tau_offset = float(tau_offset)
        self.loop_gain = float(loop_gain)
        # polarity = +1 is negative feedback: too positive -> push critical.
        # polarity = -1 is the same if-else with the sign of its gain flipped, so
        # the controller amplifies what it observes instead of opposing it.
        if polarity not in (+1, -1):
            raise ValueError("polarity must be +1 (corrective) or -1 (amplifying)")
        self.polarity = int(polarity)
        default_start = 0 if self.num_levels is None else self.num_levels // 2
        self.initial_level = (default_start if initial_level is None
                              else int(initial_level))
        self.level = self.initial_level
        self.level_history = []

    def reset(self, level=None):
        self.level = self.initial_level if level is None else int(level)
        self.level_history = []

    def forward(self, values):
        self.tensors["e"] = values["e"]
        # Sign convention: the reference is 0 and the filter feeds back -y / K,
        # so the error node emits e = -s / K.  Recover the sentiment itself.
        sentiment = -float(values["e"].item()) * self.loop_gain
        step = 0
        if sentiment > self.tau + self.tau_offset:            # corrective: -> more critical
            step = self.polarity
        elif sentiment < -self.tau + self.tau_offset:         # corrective: -> more positive
            step = -self.polarity
        if step > 0:
            self.level = (self.level + 1 if self.num_levels is None
                          else min(self.level + 1, self.num_levels - 1))
        elif step < 0:
            self.level = max(self.level - 1, 0)
        # otherwise: hold, which is what makes the chain aperiodic
        self.level_history.append(self.level)
        return torch.tensor([float(self.level)])


def rates_from_pools(pools, tau, tau_offset=0, number_of_agents=1, rng=None, mc_samples=20000):
    """Per-rung escalate/back-off probabilities, ``(up, down)``.

    ``up[k] = P(s > tau_offset + tau | k)`` and
    ``down[k] = P(s < tau_offset - tau | k)`` are tail masses of the
    level-``k`` observation law, so the closed loop's transition law can be
    read straight off the pools with no simulation.  For ``number_of_agents > 1``
    the observation is a mean of that many draws and the tails are estimated by
    Monte Carlo over the pool instead of counted directly.

    Split out from :func:`transition_law_from_pools` because the unbounded ladder
    has no transition *matrix* to build -- the rates are all that exist.
    """
    pools = [np.asarray(p, dtype=np.float64) for p in pools]
    rng = rng if rng is not None else np.random.default_rng(0)
    up = np.empty(len(pools))
    down = np.empty(len(pools))
    for k, pool in enumerate(pools):
        if number_of_agents == 1:
            obs = pool
        else:
            obs = rng.choice(pool, size=(mc_samples, number_of_agents),
                             replace=True).mean(axis=1)
        up[k] = float((obs > tau + tau_offset).mean())
        down[k] = float((obs < -tau + tau_offset).mean())
    return up, down


def transition_law_from_pools(pools, tau, tau_offset=0, num_levels=None, number_of_agents=1,
                              rng=None, mc_samples=20000, polarity=+1):
    """Analytic transition matrix of the closed loop, straight from the pools.

    Given the per-level sentiment pools there is no need to simulate to learn the
    chain: the up/down/hold probabilities at level ``k`` are tail masses of the
    level-``k`` observation law.  For ``number_of_agents > 1`` the observation is a
    mean of that many draws, so the tails are estimated by Monte Carlo over the
    pool rather than counted directly.

    :return: ``(P, up, down)``, where ``up[k]`` is the probability of moving one
        rung toward critical and ``down[k]`` one rung toward positive.
    """
    n_levels = num_levels or len(pools)
    up, down = rates_from_pools(pools[:n_levels], tau, tau_offset,
                                number_of_agents=number_of_agents, rng=rng,
                                mc_samples=mc_samples)

    # `up`/`down` are properties of the observation law and do not depend on the
    # controller; only where the controller sends that observation does.
    hi, lo = (1, -1) if polarity > 0 else (-1, 1)
    P = np.zeros((n_levels, n_levels))
    for k in range(n_levels):
        P[k, min(max(k + hi, 0), n_levels - 1)] += up[k]
        P[k, min(max(k + lo, 0), n_levels - 1)] += down[k]
        P[k, k] += 1.0 - up[k] - down[k]
    return P, up, down
