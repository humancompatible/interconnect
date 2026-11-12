"""Unique-ergodicity certificates for finite-state closed loops.

This is the discrete-state counterpart to the Jacobian-based contraction estimate
in ``tests/contractionTests/contraction_test.py``.  When the controller is a
non-differentiable if-else rule, that estimator is vacuous -- a piecewise-constant
map has zero derivative almost everywhere, so it reports perfect contraction
having measured nothing.  On a finite state space the honest certificates are
irreducibility (uniqueness of the invariant measure) and a Doeblin minorization
(uniform ergodicity with an explicit rate), neither of which needs derivatives.
"""

import numpy as np


def estimate_transition_matrix(level_paths, num_states, prior=0.0):
    """Empirical transition matrix from one or more integer trajectories.

    :param level_paths: iterable of 1-D integer sequences of visited states.
    :param prior: pseudo-count added to every cell.  Keep at 0 for an honest
        estimate; a small positive value only smooths the *display*, and would
        destroy the reducibility of a genuinely reducible chain.
    :return: ``(P, counts)``; rows never visited are ``nan``.
    """
    counts = np.full((num_states, num_states), float(prior))
    for path in level_paths:
        a = np.asarray(path, dtype=int)
        if a.size < 2:
            continue
        np.add.at(counts, (a[:-1], a[1:]), 1.0)
    row = counts.sum(axis=1, keepdims=True)
    P = np.divide(counts, row, out=np.full_like(counts, np.nan), where=row > 0)
    return P, counts


def _support_graph(P, tol=0.0):
    A = np.nan_to_num(P, nan=0.0) > tol
    return A


def reachable_matrix(P, tol=0.0):
    """Boolean matrix: entry (i, j) True iff j is reachable from i in >=1 steps."""
    A = _support_graph(P, tol)
    n = A.shape[0]
    R = A.copy()
    for _ in range(n):                      # transitive closure
        R_new = R | (R @ A)
        if np.array_equal(R_new, R):
            break
        R = R_new
    return R


def communicating_classes(P, tol=0.0):
    """Partition states into communicating classes (i <-> j)."""
    R = reachable_matrix(P, tol)
    n = R.shape[0]
    mutual = R & R.T
    np.fill_diagonal(mutual, True)
    seen, classes = set(), []
    for i in range(n):
        if i in seen:
            continue
        cls = {j for j in range(n) if mutual[i, j]}
        classes.append(sorted(cls))
        seen |= cls
    return classes


def closed_classes(P, tol=0.0):
    """Communicating classes with no escape -- each carries its own invariant measure.

    More than one closed class means the invariant measure is **not** unique.
    """
    A = _support_graph(P, tol)
    out = []
    for cls in communicating_classes(P, tol):
        idx = np.array(cls)
        if not A[np.ix_(idx, np.setdiff1d(np.arange(A.shape[0]), idx))].any():
            out.append(cls)
    return out


def is_irreducible(P, tol=0.0):
    R = reachable_matrix(P, tol)
    return bool(R.all())


def period(P, tol=0.0, max_power=None, states=None):
    """Period of an irreducible chain: gcd of the lengths of returns to a state.

    :param states: restrict to this subset (e.g. the single closed class).  The
        reference state must be *recurrent*: measuring returns to a transient
        state gives a meaningless answer, which is why :func:`certify` always
        passes the closed class here.
    """
    from math import gcd
    A = _support_graph(P, tol)
    if states is not None:
        idx = np.array(sorted(states))
        A = A[np.ix_(idx, idx)]
    n = A.shape[0]
    max_power = max_power or 2 * n + 2
    g, M = 0, A.copy()
    for length in range(1, max_power + 1):
        if M[0, 0]:
            g = gcd(g, length)
            if g == 1:
                return 1
        M = M @ A
    return g if g else None


def is_aperiodic(P, tol=0.0):
    return period(P, tol) == 1


def stationary_distribution(P):
    """Invariant measure as the left eigenvector of ``P`` for eigenvalue 1."""
    Pc = np.nan_to_num(P, nan=0.0)
    vals, vecs = np.linalg.eig(Pc.T)
    k = int(np.argmin(np.abs(vals - 1.0)))
    pi = np.real(vecs[:, k])
    if pi.sum() < 0:
        pi = -pi
    pi = np.clip(pi, 0.0, None)
    return pi / pi.sum()


def birth_death_stationary(P, tol=1e-12, states=None):
    """Closed-form invariant measure for a birth-death(-with-hold) chain.

    Detailed balance gives ``pi_{k+1}/pi_k = P[k, k+1] / P[k+1, k]``.  Returns
    ``None`` if ``P`` has any jump of size >= 2, i.e. if the chain is not
    birth-death -- which doubles as a check that the controller really only ever
    moves one rung at a time.
    """
    Pc = np.nan_to_num(P, nan=0.0)
    full_n = Pc.shape[0]
    idx = np.arange(full_n) if states is None else np.array(sorted(states))
    Pc = Pc[np.ix_(idx, idx)]
    n = Pc.shape[0]
    for i in range(n):
        for j in range(n):
            if abs(i - j) >= 2 and Pc[i, j] > tol:
                return None
    pi = np.ones(n)
    for k in range(n - 1):
        down = Pc[k + 1, k]
        if down <= tol:
            return None
        pi[k + 1] = pi[k] * Pc[k, k + 1] / down
    pi = pi / pi.sum()
    if states is None:
        return pi
    out = np.zeros(full_n)
    out[idx] = pi
    return out


def invariant_measures(P, tol=0.0):
    """One invariant measure per closed class.

    When there are several, the set of invariant measures is exactly the convex
    hull of these: every ``sum_c a_c mu_c`` with ``a >= 0``, ``sum a_c = 1`` is
    invariant, so the chain has a whole simplex of them and no ergodic theorem.
    :func:`stationary_distribution` returns *an* invariant measure in that case --
    whichever eigenvector the solver happens to return -- which is why
    :func:`certify` checks the closed-class count before reporting one.
    """
    out = []
    for cls in closed_classes(P, tol):
        idx = np.array(sorted(cls))
        local = stationary_distribution(np.nan_to_num(P, nan=0.0)[np.ix_(idx, idx)])
        mu = np.zeros(P.shape[0])
        mu[idx] = local
        out.append(mu)
    return out


def absorption_probabilities(P, tol=0.0):
    """``H[i, c] = P(the chain ends up in closed class c | it starts at i)``.

    This is what "the limit depends on the initial condition" means quantitatively:
    from a transient state the chain converges to a *random* one of the invariant
    measures, and ``H[i]`` is the law of that choice.  Solves the harmonic system
    ``(I - P_TT) h_T = P_TC 1`` on the transient states.

    :return: ``(H, closed)`` with ``closed`` the list of closed classes, in the
        column order of ``H``.
    """
    Pc = np.nan_to_num(P, nan=0.0)
    closed = closed_classes(Pc, tol)
    n = Pc.shape[0]
    absorbed = set().union(*[set(c) for c in closed]) if closed else set()
    transient = np.array([i for i in range(n) if i not in absorbed], dtype=int)
    H = np.zeros((n, len(closed)))
    for c, cls in enumerate(closed):
        H[sorted(cls), c] = 1.0
    if transient.size:
        A = np.eye(transient.size) - Pc[np.ix_(transient, transient)]
        for c, cls in enumerate(closed):
            b = Pc[np.ix_(transient, np.array(sorted(cls)))].sum(axis=1)
            H[transient, c] = np.linalg.solve(A, b)
    # The solve can return -1e-17 for an unreachable class; clip, and squash the
    # resulting negative zero, so the report does not print "-0.0" for a
    # probability (np.clip leaves -0.0 alone -- it is not less than 0.0).
    H = np.clip(H, 0.0, 1.0)
    H[H == 0.0] = 0.0
    return H, closed


def doeblin_epsilon(P, max_power=12, criterion="mixing", target=0.01, form="sum"):
    """Best Doeblin minorization over powers of ``P``.

    Seeks ``m``, ``eps > 0`` and a probability measure ``nu`` with

        P^m(x, .) >= eps * nu(.)     for every state x,

    which by Doeblin's theorem implies a unique invariant ``pi`` and

        ||P^t(x, .) - pi||_TV <= (1 - eps)^{floor(t / m)}.

    Two choices of ``nu`` are supported:

    ``form="sum"`` (default)
        The largest admissible ``eps`` for a given ``m``:
        ``eps = sum_j min_i P^m(i, j)``, with ``nu`` the normalised column
        minima.  This is the standard general minorization.
    ``form="atom"``
        The special case ``nu = delta_j``, giving ``eps = max_j min_i P^m(i, j)``.
        Easier to state but strictly weaker -- on the PoC chain it reported
        eps=0.446 where the sum form gives 0.533.

    Larger ``m`` always raises ``eps``, but the rate is per ``m`` steps, so the
    largest ``eps`` is usually not the best bound.  ``criterion="mixing"``
    (default) returns the pair minimising the implied mixing bound;
    ``criterion="epsilon"`` maximises ``eps``.

    :return: ``(eps, m, nu)``; ``eps == 0`` means no minorization was found.
        ``nu`` is ``None`` when ``eps == 0``.
    """
    Pc = np.nan_to_num(P, nan=0.0)
    best, best_score = (0.0, None, None), np.inf
    M = Pc.copy()
    for m in range(1, max_power + 1):
        col_mins = M.min(axis=0)
        if form == "atom":
            j = int(np.argmax(col_mins))
            eps = float(col_mins[j])
            nu = np.zeros_like(col_mins); nu[j] = 1.0
        else:
            eps = float(col_mins.sum())
            nu = col_mins / eps if eps > 0 else None
        if eps > 0:
            if criterion == "epsilon":
                score = -eps
            else:
                # log1p, not log(1 - eps): for a tiny eps -- which is exactly
                # what a near-reducible chain produces -- 1.0 - eps rounds to 1.0
                # and log gives 0, so the score becomes -inf and the search picks
                # the most degenerate minorization on offer.
                score = m * np.log(target) / np.log1p(-min(eps, 1 - 1e-15))
            if score < best_score:
                best, best_score = (eps, m, nu), score
        M = M @ Pc
    return best


def doeblin_curve(P, max_power=12, form="sum"):
    """``eps(m)`` for each power, for transparency about the minorization search."""
    Pc = np.nan_to_num(P, nan=0.0)
    out, M = [], Pc.copy()
    for m in range(1, max_power + 1):
        col_mins = M.min(axis=0)
        eps = float(col_mins.max() if form == "atom" else col_mins.sum())
        out.append((m, eps))
        M = M @ Pc
    return out


def dobrushin_coefficient(P):
    """Dobrushin's ergodicity coefficient
    ``delta(P) = max_{i,i'} ||P(i,.) - P(i',.)||_TV``.

    The Markov operator contracts total variation by exactly this factor:
    ``||mu P - mu' P||_TV <= delta(P) ||mu - mu'||_TV``.  It is the sharpest of the
    three bounds here, because it compares rows *pairwise* rather than against a
    global minimum, and ``1 - delta(P) >= eps`` for any Doeblin ``eps`` at m = 1.

    This is the derivative-free counterpart of the Lipschitz-style contraction
    factor the toolkit estimates by autograd: a contraction coefficient on the
    space of probability measures under the TV metric, rather than on the state
    space under a norm.
    """
    Pc = np.nan_to_num(P, nan=0.0)
    n = Pc.shape[0]
    worst = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            worst = max(worst, 0.5 * float(np.abs(Pc[i] - Pc[j]).sum()))
    return worst


def dobrushin_mixing(P, max_power=12, target=0.01):
    """Best mixing bound from ``delta(P^m)`` over powers ``m``.

    :return: ``(delta, m, iterations)`` for the tightest bound found.
    """
    Pc = np.nan_to_num(P, nan=0.0)
    best, M = (1.0, None, np.inf), Pc.copy()
    for m in range(1, max_power + 1):
        d = dobrushin_coefficient(M)
        if d < 1.0:
            its = m * np.ceil(np.log(target) / np.log(d)) if d > 0 else m
            if its < best[2]:
                best = (d, m, float(its))
        M = M @ Pc
    return best


def mixing_time_bound(eps, m, target=0.01):
    """Iterations to get within ``target`` in total variation, from Doeblin data."""
    if eps <= 0:
        return np.inf
    if eps >= 1:
        return m
    return m * np.ceil(np.log(target) / np.log1p(-eps))


def expected_hitting_times(P, recurrent):
    """Expected steps to reach the recurrent class, from each transient state.

    ``(I - P_TT)^-1 1``.  Returns a full-length vector with zeros on the recurrent
    class.
    """
    Pc = np.nan_to_num(P, nan=0.0)
    n = Pc.shape[0]
    T = np.array([i for i in range(n) if i not in set(recurrent)], dtype=int)
    out = np.zeros(n)
    if T.size:
        out[T] = np.linalg.solve(np.eye(T.size) - Pc[np.ix_(T, T)], np.ones(T.size))
    return out


def absorption_time_bound(P, recurrent, target=0.005, max_steps=100000):
    """Smallest ``t`` with ``P_x(not yet in the recurrent class after t) <= target``
    for every ``x``.

    Computed by iterating the sub-stochastic block directly rather than through a
    spectral bound, which keeps it exact and cheap for these sizes.  Returns
    ``inf`` if the tail has not fallen below ``target`` within ``max_steps``.
    """
    Pc = np.nan_to_num(P, nan=0.0)
    n = Pc.shape[0]
    T = np.array([i for i in range(n) if i not in set(recurrent)], dtype=int)
    if T.size == 0:
        return 0
    Q = Pc[np.ix_(T, T)]
    v = np.ones(T.size)
    t = 0
    while v.max() > target and t < max_steps:
        v = Q @ v
        t += 1
    return t if v.max() <= target else np.inf


def wilson_lower_bound(successes, trials, z=1.96):
    """Wilson score lower confidence bound -- used to report Doeblin eps honestly."""
    if trials == 0:
        return 0.0
    p = successes / trials
    denom = 1.0 + z**2 / trials
    centre = p + z**2 / (2 * trials)
    margin = z * np.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2))
    return float(max(0.0, (centre - margin) / denom))


def tv_distance(p, q):
    """Total variation distance between two discrete distributions."""
    p = np.asarray(p, float); q = np.asarray(q, float)
    return 0.5 * float(np.abs(p / p.sum() - q / q.sum()).sum())


def occupation_measure(level_paths, num_states, burn_in=0):
    """Empirical fraction of time spent in each state."""
    counts = np.zeros(num_states)
    for path in level_paths:
        a = np.asarray(path, dtype=int)[burn_in:]
        if a.size:
            counts += np.bincount(a, minlength=num_states)
    return counts / counts.sum()


def certify(P, counts=None, max_power=12, tv_target=0.01):
    """Run the full battery and return a report dict."""
    classes = communicating_classes(P)
    closed = closed_classes(P)
    irreducible = is_irreducible(P)
    # A single closed class that is not the whole space is still uniquely
    # ergodic: the invariant measure lives on that class and the remaining
    # states are transient.  Restrict the period computation to it.
    recurrent = closed[0] if len(closed) == 1 else None
    per = period(P, states=recurrent) if recurrent is not None else None
    if recurrent is not None and len(recurrent) < P.shape[0]:
        idx = np.array(sorted(recurrent))
        sub = P[np.ix_(idx, idx)]
    else:
        idx, sub = None, P
    eps, m, nu_local = doeblin_epsilon(sub, max_power=max_power)
    d_delta, d_m, d_its = dobrushin_mixing(sub, max_power=max_power)
    if nu_local is None:
        nu = j = None
    elif idx is None:
        nu = nu_local; j = int(np.argmax(nu_local))
    else:
        nu = np.zeros(P.shape[0]); nu[idx] = nu_local; j = int(idx[np.argmax(nu_local)])
    pi = stationary_distribution(P) if closed else None
    measures = invariant_measures(P) if len(closed) > 1 else None
    absorption = absorption_probabilities(P)[0] if len(closed) > 1 else None
    report = {
        "irreducible": irreducible,
        "communicating_classes": classes,
        "closed_classes": closed,
        "unique_invariant_measure": len(closed) == 1,
        "recurrent_class": recurrent,
        "transient_states": (sorted(set(range(P.shape[0])) - set(recurrent))
                             if recurrent is not None else None),
        "period": per,
        "aperiodic": (per == 1) if per is not None else None,
        "doeblin_epsilon": eps,
        "doeblin_power": m,
        "doeblin_nu": nu,
        "doeblin_state": j,
        "dobrushin_delta": d_delta,
        "dobrushin_power": d_m,
        "dobrushin_mixing_bound": d_its,
        "uniformly_ergodic": eps > 0,
        "tv_mixing_bound": mixing_time_bound(eps, m, tv_target) if eps > 0 else np.inf,
        "stationary": pi,
        "stationary_birth_death": birth_death_stationary(P, states=recurrent),
        # Only populated when uniqueness fails; the invariant set is then the
        # convex hull of these, and `absorption[i, c]` says which one a
        # trajectory started at `i` ends up under.
        "invariant_measures": measures,
        "absorption": absorption,
    }
    # The Doeblin bound above is a bound on mixing *within* the recurrent class.
    # A trajectory started outside it has to get there first, and that leg can be
    # far slower than the mixing -- decisively so on a near-reducible chain.  The
    # honest total is the two legs at half the target each:
    #   ||P^t(x,.) - pi|| <= P_x(tau_C > t_hit) + sup_{y in C} ||P^u(y,.) - pi||
    if recurrent is not None and report["transient_states"]:
        t_hit = absorption_time_bound(P, recurrent, target=tv_target / 2)
        report["expected_hitting_times"] = expected_hitting_times(P, recurrent)
        report["absorption_time_bound"] = t_hit
        inner = mixing_time_bound(eps, m, tv_target / 2) if eps > 0 else np.inf
        report["tv_mixing_bound_total"] = t_hit + inner
        report["dobrushin_mixing_bound_total"] = t_hit + dobrushin_mixing(
            sub, max_power=max_power, target=tv_target / 2)[2]
    else:
        report["expected_hitting_times"] = None
        report["absorption_time_bound"] = 0
        report["tv_mixing_bound_total"] = report["tv_mixing_bound"]
        report["dobrushin_mixing_bound_total"] = report["dobrushin_mixing_bound"]
    if counts is not None and eps > 0:
        # Honest uncertainty on the minorization: the binding row is the one that
        # attains the column minimum at m = 1.
        if m == 1 and j is not None:
            i_star = int(np.nanargmin(np.nan_to_num(P[:, j], nan=np.inf)))
            report["doeblin_epsilon_lcb"] = wilson_lower_bound(
                counts[i_star, j], counts[i_star].sum())
            report["doeblin_binding_state"] = i_star
    return report


def format_report(report, num_states=None):
    """Human-readable summary of :func:`certify`."""
    r = report
    lines = []
    lines.append(f"communicating classes : {r['communicating_classes']}")
    lines.append(f"closed classes        : {r['closed_classes']}")
    lines.append(f"irreducible           : {r['irreducible']}")
    if r.get("transient_states"):
        lines.append(f"recurrent class       : {r['recurrent_class']}   "
                     f"(transient: {r['transient_states']})")
    lines.append(f"period                : {r['period']}   (aperiodic: {r['aperiodic']})")
    verdict = "UNIQUE" if r["unique_invariant_measure"] else "NOT UNIQUE"
    lines.append(f"invariant measure     : {verdict} "
                 f"({len(r['closed_classes'])} closed class(es))")
    if r.get("invariant_measures") is not None:
        lines.append(f"  the invariant set is the convex hull of "
                     f"{len(r['invariant_measures'])} extreme measures:")
        for c, mu in enumerate(r["invariant_measures"]):
            lines.append(f"    mu_{c} on {r['closed_classes'][c]}: "
                         f"{np.round(mu, 4).tolist()}")
        lines.append("  absorption probabilities P(class | start):")
        for i, row in enumerate(r["absorption"]):
            lines.append(f"    from state {i}: {np.round(row, 4).tolist()}")
    if r["doeblin_epsilon"] > 0:
        lines.append(f"Doeblin minorization  : eps={r['doeblin_epsilon']:.4f} at m={r['doeblin_power']}"
                     f"  (nu peaks at state {r['doeblin_state']})")
        if "doeblin_epsilon_lcb" in r:
            lines.append(f"  eps 95% lower bound : {r['doeblin_epsilon_lcb']:.4f} "
                         f"(binding state i={r['doeblin_binding_state']})")
        lines.append(f"  => uniformly ergodic, TV<0.01 within ~{r['tv_mixing_bound']:.0f} "
                     f"iterations once in the recurrent class")
        if r.get("absorption_time_bound"):
            hit = r["expected_hitting_times"]
            lines.append(f"  reaching the recurrent class: <= {r['absorption_time_bound']:.0f} "
                         f"iterations to 0.005 (expected {np.nanmax(hit):.1f} from the "
                         f"worst transient state)")
            lines.append(f"  => TV<0.01 from ANY start within ~{r['tv_mixing_bound_total']:.0f} "
                         f"iterations")
        lines.append(f"Dobrushin coefficient : delta(P^{r['dobrushin_power']})="
                     f"{r['dobrushin_delta']:.4f}  => TV<0.01 within "
                     f"~{r['dobrushin_mixing_bound']:.0f} iterations in the class, "
                     f"~{r['dobrushin_mixing_bound_total']:.0f} from any start (sharpest)")
    else:
        lines.append("Doeblin minorization  : NONE FOUND (no uniform ergodicity)")
    if r["stationary"] is not None:
        label = ("stationary pi         : " if r["unique_invariant_measure"]
                 else "an invariant pi       : ")
        lines.append(f"{label}{np.round(r['stationary'], 4).tolist()}"
                     + ("" if r["unique_invariant_measure"]
                        else "   <- one of many; see the extreme measures above"))
        bd = r["stationary_birth_death"]
        if bd is not None:
            lines.append(f"  birth-death closed form: {np.round(bd, 4).tolist()}")
            lines.append(f"  agreement (TV)         : {tv_distance(r['stationary'], bd):.2e}")
        else:
            lines.append("  birth-death closed form: n/a (chain is not birth-death)")
    return "\n".join(lines)
