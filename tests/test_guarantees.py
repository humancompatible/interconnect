"""Cross-checks for guarantees.markov and guarantees.transience.

The two modules answer opposite questions from the same cut-balance identity, so
most of what is worth testing is that they agree where they overlap, and that each
gives the right verdict on a chain whose answer is known by hand -- including the
negative controls, since a certificate that cannot say "no" is worthless.

Written for pytest but deliberately free of pytest-only constructs, so it also
runs under a bare interpreter (pytest is not installed in the ROCm environment
this PoC uses).
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from humancompatible.interconnect.guarantees import markov, transience
from humancompatible.interconnect.simulators.logic.controllers.instruction_ladder import (
    IfElseLadderLogic, rates_from_pools, transition_law_from_pools,
)


def birth_death_matrix(up, down):
    n = len(up)
    P = np.zeros((n, n))
    for k in range(n):
        P[k, min(k + 1, n - 1)] += up[k]
        P[k, max(k - 1, 0)] += down[k]
        P[k, k] += 1.0 - up[k] - down[k]
    return P


# -- markov ------------------------------------------------------------------

def test_birth_death_closed_form_matches_eigenvector():
    rng = np.random.default_rng(0)
    up = np.concatenate([rng.uniform(0.1, 0.4, 6), [0.0]])
    down = np.concatenate([[0.0], rng.uniform(0.1, 0.4, 6)])
    P = birth_death_matrix(up, down)
    assert markov.tv_distance(markov.stationary_distribution(P),
                              markov.birth_death_stationary(P)) < 1e-12


def test_birth_death_rejects_two_rung_jumps():
    P = np.array([[0.5, 0.0, 0.5], [0.3, 0.4, 0.3], [0.0, 0.5, 0.5]])
    assert markov.birth_death_stationary(P) is None


def test_identity_has_an_invariant_measure_but_not_a_unique_one():
    # The failure a finite state space permits: existence is automatic, so the
    # only thing that can go wrong is uniqueness.
    rep = markov.certify(np.eye(7))
    assert rep["unique_invariant_measure"] is False
    assert len(rep["closed_classes"]) == 7
    assert rep["doeblin_epsilon"] == 0


def test_mixing_bound_stays_positive_for_a_tiny_epsilon():
    # Regression: log(1 - eps) underflows to log(1.0) = 0 for eps below the
    # double-precision resolution of 1, which used to yield a *negative* bound.
    bound = markov.mixing_time_bound(1e-18, 5, target=0.01)
    assert np.isfinite(bound) and bound > 1e17


def test_doeblin_search_never_scores_a_degenerate_minorization_best():
    up = np.concatenate([np.full(20, 0.93), [0.0]])
    down = np.concatenate([[0.0], np.full(20, 0.019)])
    P = birth_death_matrix(up, down)
    eps, m, _ = markov.doeblin_epsilon(P, max_power=126)
    assert markov.mixing_time_bound(eps, m) > 0


def test_absorption_probabilities_are_harmonic_and_symmetric():
    # Simple random walk on 0..4 absorbed at both ends: P(hit 4 | start i) = i/4.
    n = 5
    P = np.zeros((n, n))
    P[0, 0] = P[n - 1, n - 1] = 1.0
    for k in range(1, n - 1):
        P[k, k - 1] = P[k, k + 1] = 0.5
    H, closed = markov.absorption_probabilities(P)
    assert closed == [[0], [4]]
    assert np.allclose(H[:, 1], np.arange(n) / (n - 1))
    assert np.all(H >= 0.0)


def test_invariant_measures_span_the_closed_classes():
    P = np.array([[1.0, 0, 0], [0.3, 0.4, 0.3], [0, 0, 1.0]])
    mus = markov.invariant_measures(P)
    assert len(mus) == 2
    for mu in mus:
        assert markov.tv_distance(mu @ P, mu) < 1e-12


def test_mixing_bound_includes_the_time_to_reach_the_recurrent_class():
    """A chain that mixes instantly once absorbed but takes ages to get absorbed
    must not be reported as mixing instantly."""
    n = 6
    P = np.zeros((n, n))
    P[0, 0] = 1.0                       # absorbing, so mixing within the class is 1
    for k in range(1, n):
        P[k, k - 1] = 0.001             # ... but the funnel is very slow
        P[k, k] = 0.999
    rep = markov.certify(P)
    assert rep["unique_invariant_measure"] is True
    assert rep["tv_mixing_bound"] <= 2
    assert rep["tv_mixing_bound_total"] > 1000
    assert rep["expected_hitting_times"][n - 1] > 1000


# -- transience --------------------------------------------------------------

def test_invariant_series_agrees_with_the_finite_closed_form():
    """Same identity, two modules: the series normalised is the stationary law."""
    rng = np.random.default_rng(1)
    up = np.concatenate([rng.uniform(0.1, 0.4, 6), [0.0]])
    down = np.concatenate([[0.0], rng.uniform(0.1, 0.4, 6)])
    v = transience.invariant_series(up, down)
    assert markov.tv_distance(v / v.sum(),
                              markov.birth_death_stationary(birth_death_matrix(up, down))) < 1e-12


def test_positive_drift_is_certified_as_having_no_invariant_measure():
    up = np.full(12, 0.9)
    down = np.concatenate([[0.0], np.full(11, 0.02)])
    rep = transience.certify_no_invariant_measure(up, down, n_samples=200, tail_from=3)
    assert rep["no_invariant_probability_measure"] is True
    assert rep["rho_conservative"] > 1
    assert rep["recurrence_series_tail_ratio"] < 1


def test_negative_drift_is_not_certified():
    """The certificate has to be able to say no: a chain that comes back down is
    positive recurrent and must not be reported as transient."""
    up = np.full(12, 0.1)
    down = np.concatenate([[0.0], np.full(11, 0.6)])
    rep = transience.certify_no_invariant_measure(up, down, n_samples=200, tail_from=3)
    assert rep["no_invariant_probability_measure"] is False
    assert np.isfinite(rep["invariant_series"].sum())


def test_zero_drift_is_not_certified():
    up = np.full(12, 0.3)
    down = np.concatenate([[0.0], np.full(11, 0.3)])
    rep = transience.certify_no_invariant_measure(up, down, n_samples=200, tail_from=3)
    assert rep["no_invariant_probability_measure"] is False


def test_truncated_mass_escapes_as_the_cap_is_lifted():
    up = np.full(12, 0.9)
    down = np.concatenate([[0.0], np.full(11, 0.02)])
    trunc = transience.truncation_sequence(0.9, 0.02, [10, 20, 40], up=up, down=down)
    below = [trunc[N][:6].sum() for N in (10, 20, 40)]
    assert below[0] > below[1] > below[2]
    assert below[-1] < 1e-20
    for N in (10, 20, 40):
        assert int(np.argmax(trunc[N])) == N


def test_truncation_with_no_observed_backoff_is_a_point_mass_at_the_cap():
    up = np.full(12, 0.9)
    down = np.zeros(12)
    trunc = transience.truncation_sequence(0.9, 0.0, [15], up=up, down=down)
    assert abs(trunc[15][-1] - 1.0) < 1e-12


def test_wilson_interval_brackets_and_never_returns_a_hard_zero():
    lo, hi = transience.wilson_interval(0, 200)
    assert lo == 0.0 and 0.0 < hi < 0.05
    lo, hi = transience.wilson_interval(100, 200)
    assert lo < 0.5 < hi


def test_ks_statistic_is_zero_for_identical_samples():
    x = np.linspace(-1, 1, 50)
    rows = transience.saturation_report([x, x], [0, 1], tau=0.5, tail_from=1)
    assert rows[0]["ks"] == 0.0


# -- the controller ----------------------------------------------------------

def test_unbounded_ladder_has_no_top_rung():
    c = IfElseLadderLogic(num_levels=None, tau=0.5, initial_level=0)
    for _ in range(50):
        c.forward({"e": torch.tensor([-0.9])})   # e = -s, so this is s = +0.9
    assert c.level == 50


def test_offset_shifts_the_hold_band_not_its_width():
    """The band is [tau_offset - tau, tau_offset + tau]: an observation inside it
    holds, one above escalates, one below backs off."""
    c = IfElseLadderLogic(num_levels=7, tau=0.2, tau_offset=0.5, initial_level=3)
    for sentiment, expected in ((0.9, 4), (0.71, 4), (0.6, 3), (0.3, 3),
                                (0.29, 2), (-0.9, 2)):
        c.reset(3)
        c.forward({"e": torch.tensor([-sentiment])})   # e = -s
        assert c.level == expected, (sentiment, c.level, expected)


def test_offset_reaches_rungs_a_symmetric_band_cannot():
    """The point of the offset on a positively biased plant: with the band on the
    reference, a rung whose whole output law sits above -tau can never back off."""
    # A rung whose whole output law sits in [+0.4, +1.0]: entirely positive, but
    # with real spread inside that range.
    pools = [np.linspace(0.4, 1.0, 200)] * 3
    up0, down0 = rates_from_pools(pools, tau=0.2, tau_offset=0.0)
    up1, down1 = rates_from_pools(pools, tau=0.2, tau_offset=0.7)
    # Band on the reference: the controller can only ever escalate.
    assert up0.min() == 1.0 and down0.max() == 0.0
    # Band recentred on the observations: both moves become available again.
    assert up1.min() > 0.1 and down1.min() > 0.1


def test_offset_is_forwarded_by_transition_law_from_pools():
    """Regression: the matrix builder accepted tau_offset and dropped it, so the
    certificate silently described a symmetric controller."""
    pools = [np.full(200, 0.7)] * 3
    P_sym, _, _ = transition_law_from_pools(pools, 0.45, 0.0)
    P_off, _, _ = transition_law_from_pools(pools, 0.45, 0.6)
    assert not np.allclose(P_sym, P_off)
    up, down = rates_from_pools(pools, 0.45, tau_offset=0.6)
    P_expected, up2, down2 = transition_law_from_pools(pools, 0.45, 0.6)
    assert np.allclose(up, up2) and np.allclose(down, down2)


def test_offset_and_polarity_are_independent():
    """Flipping the gain must move mass the other way without touching the rates."""
    pools = [np.full(50, 0.9), np.full(50, 0.0), np.full(50, -0.9)]
    P_corr, u_c, d_c = transition_law_from_pools(pools, 0.45, 0.1, polarity=+1)
    P_amp, u_a, d_a = transition_law_from_pools(pools, 0.45, 0.1, polarity=-1)
    assert np.allclose(u_c, u_a) and np.allclose(d_c, d_a)
    assert P_corr[0, 1] == u_c[0] and P_amp[0, 0] >= u_a[0]


def test_bounded_ladder_saturates():
    c = IfElseLadderLogic(num_levels=7, tau=0.5, initial_level=0)
    for _ in range(50):
        c.forward({"e": torch.tensor([-0.9])})
    assert c.level == 6


if __name__ == "__main__":
    # Runnable without pytest: the ROCm environment this PoC uses does not have it.
    import traceback

    failures = 0
    for name in sorted(n for n in dir() if n.startswith("test_")):
        try:
            globals()[name]()
            print(f"PASS {name}")
        except Exception:
            failures += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"\n{failures} failure(s)")
    raise SystemExit(1 if failures else 0)
