"""Feasible regions for Algorithm 2 and the algorithm with a pluggable projection.

A_r^L = {mu : L(mu) <= r} is the region of the paper (one linear constraint).
A_r^C = {mu : C(mu) <= r} = {mu : D mu <= r} uses the contraction-on-average constant
instead; it is a convex polytope with one constraint per state in D and contains A_r^L.

Both start from the slope table D[j, i] = |w_i'(x_j)| at a finite set of states x_j, for
example the states visited by runs of the graph:

    L(mu) = sum_i mu_i * max_j D[j, i]     per-mode worst case, then average
    C(mu) = max_j sum_i mu_i * D[j, i]     average at each state, then worst state

For a one-dimensional state and C^1 maps, C(mu) is eq. (12) of the Interconnect paper and
C(mu) <= L(mu).  Both are maxima over the states in D, so they are estimates from below.
"""

import numpy as np
from scipy.optimize import minimize

from ims_core import proj_simplex


def lipschitz_constants(D):
    """ell_i = max_j D[j, i], the Lipschitz constant of each mode over the states in D."""
    return D.max(axis=0)


def average_lipschitz(mu, D):
    """L(mu) = sum_i mu_i ell_i."""
    return float(lipschitz_constants(D) @ np.asarray(mu, dtype=float))


def contraction_on_average(mu, D):
    """C(mu) = max_j sum_i mu_i D[j, i]."""
    return float((D @ np.asarray(mu, dtype=float)).max())


def nondominated_rows(D):
    """Rows of D that are not componentwise <= another row; the others are never active in D mu <= r."""
    U = np.unique(D, axis=0)
    dominated = ((U[None, :, :] >= U[:, None, :]).all(axis=2) & (U[None, :, :] > U[:, None, :]).any(axis=2)).any(axis=1)
    return U[~dominated]


def proj_Cr(y, D, r, tol=1e-9):
    """Euclidean projection of y onto {mu in simplex : D mu <= r}, as a small QP."""
    z = proj_simplex(y)
    if (D @ z).max() <= r:
        return z
    res = minimize(lambda m: 0.5 * np.sum((m - y) ** 2), z, jac=lambda m: m - y, method="SLSQP",
                   bounds=[(0.0, 1.0)] * len(y),
                   constraints=[{"type": "eq", "fun": lambda m: m.sum() - 1.0, "jac": lambda m: np.ones_like(m)},
                                {"type": "ineq", "fun": lambda m: r - D @ m, "jac": lambda m: -D}],
                   options={"ftol": 1e-15, "maxiter": 500})
    mu = np.clip(res.x, 0.0, None)
    mu /= mu.sum()
    if not res.success or (D @ mu).max() > r + tol:
        raise RuntimeError(f"projection failed: {res.message}, C = {(D @ mu).max():.6f} > r = {r}")
    return mu


def projected_subgradient(chain, nu, mu0, T, project, gamma=0.5, step="diminishing", Jstar=0.0):
    """Algorithm 2 as in ims_core.projected_subgradient, with the projection passed in."""
    mu = np.asarray(mu0, dtype=float).copy()
    p = chain.stationary(mu)
    traj, Js = [mu.copy()], []
    best = (np.inf, mu.copy())
    G0 = None
    for t in range(T):
        J = chain.w1(p, nu)
        Js.append(J)
        if J < best[0]:
            best = (J, mu.copy())
        g = chain.sensitivity(mu, p, chain.potential(p, nu))
        g = g - g.mean()
        gn = np.linalg.norm(g)
        if gn == 0.0:
            break
        if G0 is None:
            G0 = gn
        alpha = max(J - Jstar, 0.0) / gn ** 2 if step == "polyak" else gamma / (G0 * np.sqrt(t + 1.0))
        mu = project(mu - alpha * g)
        p = chain.stationary(mu, p0=p)
        traj.append(mu.copy())
    J = chain.w1(p, nu)
    Js.append(J)
    if J < best[0]:
        best = (J, mu.copy())
    return dict(traj=np.array(traj), J=np.array(Js), best_J=best[0], best_mu=best[1])
