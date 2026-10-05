"""Certificates for the *absence* of an invariant probability measure.

:mod:`markov` answers "is there exactly one invariant measure, and how fast is it
reached".  This module answers the complementary question -- "is there none at
all" -- for the one structural situation in which a Markov chain can fail to have
an invariant probability measure: an unbounded state space out of which the mass
escapes.  On a finite state space that failure is impossible (Perron-Frobenius
guarantees at least one invariant measure), so the strongest finite-state failure
is *non-uniqueness*, which :func:`markov.certify` already reports through its
closed-class count.

The setting here is the closed loop whose controller has only one move -- append
one more copy of the instruction -- so the instruction level lives on
``{0, 1, 2, ...}`` with nearest-neighbour transitions::

    P(k, k+1) = p_k        (the controller escalates)
    P(k, k-1) = q_k        (the controller backs off; q_0 = 0, reflecting)
    P(k, k)   = 1 - p_k - q_k

Cut balance
-----------
For a nearest-neighbour chain, summing the invariance equations over
``{0, ..., k}`` cancels every internal flow and leaves the flow across the single
cut between ``k`` and ``k+1``:

.. math:: \\pi_k p_k = \\pi_{k+1} q_{k+1} \\qquad \\text{for all } k \\ge 0.

Everything below is a consequence of that one identity, so the certificate needs
no simulation -- only the measured tail masses ``p_k`` and ``q_k``.

* ``v_k = prod_{j<k} p_j / q_{j+1}`` is the unique invariant measure up to scale.
  It is a *probability* measure only if it is summable.
* If ``p_k / q_{k+1} >= rho > 1`` for all ``k >= K`` then ``v_k >= v_K rho^{k-K}``
  diverges geometrically, so no scaling of it is a probability measure, and
  (given ``p_k > 0`` below ``K``) the only non-negative solution of the balance
  equations with finite mass is the zero measure.  **No invariant probability
  measure exists.**
* Equivalently, in the classical birth-death criteria: the chain is recurrent iff
  ``sum_k prod_{j<=k} q_j/p_j`` diverges, and positive recurrent iff in addition
  ``sum_k v_k`` converges.  Both series are computed here from the same measured
  numbers, so the two routes cross-check each other.

Statistical honesty
-------------------
``p_k`` and ``q_k`` are tail masses estimated from finitely many LLM samples, so
the verdict is reported against Wilson score bounds: the drift uses the *lower*
bound on every up-rate and the *upper* bound on every down-rate, which is the
conservative direction for the conclusion.  The tail hypothesis ("the conditional
law has stopped moving beyond rung ``tail_from``") is an extrapolation and is
flagged as such -- :func:`saturation_report` is what tests it.
"""

import numpy as np

__all__ = [
    "wilson_interval", "cut_balance_ratios", "invariant_series",
    "recurrence_series", "truncation_sequence", "truncated_rates",
    "escape_probability",
    "saturation_report", "certify_no_invariant_measure", "format_report",
]


def wilson_interval(successes, trials, z=1.96):
    """Two-sided Wilson score interval for a binomial proportion.

    Chosen over the normal interval because the interesting rates here are near
    0 (a down-move that was never observed) and near 1, where the normal
    interval is useless or leaves the unit interval.
    """
    if trials == 0:
        return 0.0, 1.0
    p = successes / trials
    denom = 1.0 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denom
    margin = (z * np.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2))) / denom
    return float(max(0.0, centre - margin)), float(min(1.0, centre + margin))


def cut_balance_ratios(up, down):
    """``r_k = p_{k-1} / q_k``, the ratio ``pi_k / pi_{k-1}`` forced by cut balance.

    ``r_k = inf`` marks a cut the chain crosses upward but never downward: the
    measure below it is then forced to zero rather than merely small.
    """
    up = np.asarray(up, float)
    down = np.asarray(down, float)
    n = min(len(up), len(down))
    r = np.empty(n)
    r[0] = np.nan                      # no cut below state 0
    for k in range(1, n):
        r[k] = np.inf if down[k] <= 0 else up[k - 1] / down[k]
    return r


def invariant_series(up, down):
    """Unnormalised invariant measure ``v_k`` of the nearest-neighbour chain.

    Same closed form as :func:`markov.birth_death_stationary`, but returned
    *unnormalised* so that its divergence is visible: normalising would silently
    hide exactly the pathology we are trying to display.
    """
    r = cut_balance_ratios(up, down)
    v = np.ones(len(r))
    for k in range(1, len(r)):
        v[k] = v[k - 1] * r[k]
    return v


def recurrence_series(up, down):
    """Terms of ``sum_k prod_{j<=k} q_j / p_j``; the chain is recurrent iff it diverges."""
    up = np.asarray(up, float)
    down = np.asarray(down, float)
    n = min(len(up), len(down))
    terms = np.empty(n)
    terms[0] = np.nan
    acc = 1.0
    for k in range(1, n):
        acc = np.inf if up[k - 1] <= 0 else acc * down[k] / up[k - 1]
        terms[k] = acc
    return terms


def escape_probability(p, q):
    """Probability a homogeneous ``+-1``-with-hold walk never returns one rung down.

    For ``p > q`` the classical gambler's-ruin answer is ``q / p`` for the return
    probability, so the escape probability is ``1 - q/p``; hold moves do not
    change it, they only slow the walk down.
    """
    if p <= 0:
        return 0.0
    return float(max(0.0, 1.0 - q / p))


def truncated_rates(p_tail, q_tail, N, up=None, down=None):
    """Rates of the ladder capped at ``N``: measured where available, tail beyond."""
    u = np.full(N + 1, float(p_tail))
    d = np.full(N + 1, float(q_tail))
    if up is not None:
        m = min(len(up), N + 1)
        u[:m] = np.asarray(up, float)[:m]
        d[:m] = np.asarray(down, float)[:m]
    d[0] = 0.0                         # reflecting bottom
    u[N] = 0.0                         # reflecting top
    return u, d


def truncation_sequence(p_tail, q_tail, sizes, up=None, down=None):
    """Stationary law of the chain truncated (reflecting) at each size in ``sizes``.

    A finite truncation always *has* a stationary law -- that is the point.
    Watching where it puts its mass as the truncation is lifted is the numerical
    face of "the mass escapes": ``pi^{(N)}(k <= k0) -> 0`` for every fixed ``k0``.

    Cut balance ``pi_k p_k = pi_{k+1} q_{k+1}`` still determines the answer, with
    one wrinkle: a cut with ``q_{k+1} = 0`` that the chain crosses upward forces
    *all* the mass below it to zero, so the support starts at the highest such
    cut.  If no down-move was ever observed the stationary law is the point mass
    at the cap.  Computed in logs, since ``v_k`` overflows for large ``N``.
    """
    out = {}
    for N in sizes:
        u, d = truncated_rates(p_tail, q_tail, N, up=up, down=down)
        blocked = [k for k in range(1, N + 1) if d[k] <= 0 and u[k - 1] > 0]
        lo = max(blocked) if blocked else 0
        pi = np.zeros(N + 1)
        logv = np.concatenate([[0.0], np.cumsum(np.log(u[lo:N]) - np.log(d[lo + 1:N + 1]))])
        logv -= logv.max()
        w = np.exp(logv)
        pi[lo:] = w / w.sum()
        out[N] = pi
    return out


def saturation_report(pools, levels, tau, tail_from, z=1.96, tau_offset=0.0):
    """Does the conditional law stop moving beyond ``tail_from``?

    The certificate extrapolates the measured rates to every rung above the
    largest one probed.  That extrapolation is only as good as the evidence that
    the law has saturated, so this reports, for each probed rung in the tail, the
    two functionals the certificate actually depends on --
    ``P(s > tau_offset + tau)`` and ``P(s < tau_offset - tau)`` -- with confidence
    intervals, plus the Kolmogorov-Smirnov distance to the largest probed rung.

    ``tau_offset`` shifts the dead zone off the reference; it changes the two
    rates but not the KS column, which compares whole conditional laws and is
    therefore a property of the plant alone.
    """
    ref = np.asarray(pools[-1], float)
    rows = []
    for pool, k in zip(pools, levels):
        pool = np.asarray(pool, float)
        n = len(pool)
        nu = int((pool > tau_offset + tau).sum())
        nd = int((pool < tau_offset - tau).sum())
        rows.append(dict(
            level=int(k), n=n, mean=float(pool.mean()), sd=float(pool.std()),
            p=nu / n, q=nd / n,
            p_ci=wilson_interval(nu, n, z), q_ci=wilson_interval(nd, n, z),
            ks=_ks_statistic(pool, ref), in_tail=bool(k >= tail_from),
        ))
    return rows


def _ks_statistic(a, b):
    """Two-sample Kolmogorov-Smirnov statistic (no SciPy dependency)."""
    a = np.sort(np.asarray(a, float))
    b = np.sort(np.asarray(b, float))
    grid = np.concatenate([a, b])
    fa = np.searchsorted(a, grid, side="right") / len(a)
    fb = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.abs(fa - fb).max())


def certify_no_invariant_measure(up, down, n_samples, tail_from, z=1.96):
    """Certify that the unbounded-ladder chain has no invariant probability measure.

    :param up: ``p_k`` for each probed rung, ``P(escalate | k)``.
    :param down: ``q_k`` for each probed rung, ``P(back off | k)``.
    :param n_samples: samples behind each rate, for the Wilson bounds.
    :param tail_from: index (into ``up``/``down``) from which the law is taken to
        have saturated; rates above the last probed rung are extrapolated from
        the worst case over the tail.
    :return: report dict; ``no_invariant_probability_measure`` is the verdict.

    The verdict rests on two hypotheses, both reported so they can be checked:
    ``p_k > 0`` below the tail (the chain really does climb), and
    ``p_k / q_{k+1} >= rho > 1`` in the tail (the climb never turns around).
    """
    up = np.asarray(up, float)
    down = np.asarray(down, float)
    n = int(n_samples)

    p_ci = [wilson_interval(int(round(p * n)), n, z) for p in up]
    q_ci = [wilson_interval(int(round(q * n)), n, z) for q in down]

    tail = slice(tail_from, len(up))
    # Conservative direction: slowest credible climb against fastest credible fall.
    p_tail_lo = float(min(ci[0] for ci in p_ci[tail]))
    q_tail_hi = float(max(ci[1] for ci in q_ci[tail]))
    p_tail = float(up[tail].mean())
    q_tail = float(down[tail].mean())

    rho_point = np.inf if q_tail <= 0 else p_tail / q_tail
    rho_conservative = np.inf if q_tail_hi <= 0 else p_tail_lo / q_tail_hi

    climbs_below = bool(np.all(np.array([ci[0] for ci in p_ci[:tail_from]]) > 0)) \
        if tail_from > 0 else True

    v = invariant_series(up, down)
    # Same series built from the confidence bounds: slowest credible climb over
    # fastest credible fall.  It is a rigorous lower envelope for v_k, and unlike
    # v_k it stays finite when a down-move was never observed at all.
    p_lo = np.array([ci[0] for ci in p_ci])
    q_hi = np.array([ci[1] for ci in q_ci])
    v_conservative = invariant_series(p_lo, q_hi)
    rec = recurrence_series(up, down)

    verdict = bool(rho_conservative > 1.0 and p_tail_lo > 0.0 and climbs_below)
    return {
        "up": up, "down": down, "n_samples": n,
        "p_ci": p_ci, "q_ci": q_ci,
        "tail_from": int(tail_from),
        "drift": up - down,
        "drift_tail": p_tail - q_tail,
        "drift_tail_conservative": p_tail_lo - q_tail_hi,
        "p_tail": p_tail, "q_tail": q_tail,
        "p_tail_lcb": p_tail_lo, "q_tail_ucb": q_tail_hi,
        "rho": rho_point,
        "rho_conservative": rho_conservative,
        "climbs_below_tail": climbs_below,
        "invariant_series": v,
        "invariant_series_conservative": v_conservative,
        "invariant_series_diverges": bool(rho_conservative > 1.0),
        "recurrence_series": rec,
        "recurrence_series_tail_ratio": float(1.0 / rho_conservative)
        if np.isfinite(rho_conservative) and rho_conservative > 0 else 0.0,
        "escape_probability": escape_probability(p_tail_lo, q_tail_hi),
        "no_invariant_probability_measure": verdict,
        # A zero down-rate would kill even the sigma-finite invariant measure, but
        # "never observed in n samples" is not "impossible", so this is only the
        # point estimate and is reported as such.
        "sigma_finite_point_estimate": ("none" if np.any(down[tail] <= 0) else "exists"),
    }


def format_report(report, levels=None):
    """Human-readable summary of :func:`certify_no_invariant_measure`."""
    r = report
    k = np.arange(len(r["up"])) if levels is None else np.asarray(levels)
    lines = []
    lines.append(f"probed rungs          : {k.tolist()}")
    lines.append(f"P(escalate | k)   p_k : {np.round(r['up'], 4).tolist()}")
    lines.append(f"P(back off | k)   q_k : {np.round(r['down'], 4).tolist()}")
    lines.append(f"drift  p_k - q_k      : {np.round(r['drift'], 4).tolist()}")
    lines.append(f"tail (rungs >= {k[r['tail_from']]}) : p={r['p_tail']:.4f} "
                 f"(95% lcb {r['p_tail_lcb']:.4f}), q={r['q_tail']:.4f} "
                 f"(95% ucb {r['q_tail_ucb']:.4f})")
    lines.append(f"  drift               : {r['drift_tail']:+.4f}  "
                 f"(conservative {r['drift_tail_conservative']:+.4f})")
    rho = r["rho_conservative"]
    lines.append(f"  rho = p/q           : {r['rho']:.1f}  "
                 f"(conservative {rho:.1f})   need > 1 for divergence")
    lines.append(f"  climbs at every rung below the tail: {r['climbs_below_tail']}")
    v = r["invariant_series_conservative"]
    lines.append("invariant series v_k (from the confidence bounds, a lower envelope):")
    lines.append("  " + np.array2string(v, precision=2, max_line_width=100))
    lines.append(f"  sum v_k             : DIVERGES (v_k >= v_K rho^(k-K), rho>1)"
                 if r["invariant_series_diverges"] else
                 f"  sum v_k             : {np.nansum(v):.4g} (converges)")
    lines.append(f"recurrence series     : ratio {r['recurrence_series_tail_ratio']:.4f} < 1 "
                 f"=> geometric, sum finite => TRANSIENT"
                 if r["recurrence_series_tail_ratio"] < 1 else
                 "recurrence series     : does not converge => recurrent")
    lines.append(f"P(never returns one rung down) >= {r['escape_probability']:.4f}")
    lines.append("")
    if r["no_invariant_probability_measure"]:
        lines.append("VERDICT: NO INVARIANT PROBABILITY MEASURE EXISTS.")
        lines.append("  The cut-balance recursion admits only the geometrically growing")
        lines.append("  solution v_k, which is not summable, so no scaling of it is a")
        lines.append("  probability measure; the chain is transient and k_t -> infinity.")
        if r["sigma_finite_point_estimate"] == "exists":
            lines.append("  (A sigma-finite invariant measure does exist -- v_k itself --")
            lines.append("   with infinite total mass.  It cannot be normalised.)")
        else:
            lines.append("  (No down-move was observed anywhere in the tail.  Taken at face")
            lines.append("   value that kills even the sigma-finite invariant measure, since")
            lines.append("   cut balance then forces pi = 0 outright; but a rate estimated as")
            lines.append("   zero from finite samples is not a proven zero, so the certificate")
            lines.append("   above rests on the confidence bounds, not on that.)")
    else:
        lines.append("VERDICT: not certified transient on the evidence given.")
    return "\n".join(lines)
