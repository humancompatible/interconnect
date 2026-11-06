"""Core routines for the invariant-measure shaping experiments.

A probabilistic IFS on an interval is represented by its finite-state
(Ulam-type) model: row-stochastic matrices W_1..W_m on a uniform grid, and
P(mu) = sum_i mu_i W_i (Section "The discretized model").  On a uniform 1-D
grid, W_1 distances, optimal Kantorovich potentials, solutions of the Poisson
equation and the sensitivity vectors s_mu(f) of Section "Sensitivity" are all
explicit, which is what this module implements.
"""
import numpy as np
import scipy.sparse as sp


class GridChain:
    """Finite-state model of a probabilistic IFS on a uniform 1-D grid."""

    def __init__(self, W, x):
        self.W = [sp.csr_matrix(Wi) for Wi in W]
        self.WT = [Wi.T.tocsr() for Wi in self.W]
        self.x = np.asarray(x, dtype=float)
        self.h = float(self.x[1] - self.x[0])
        self.n = len(self.x)
        self.m = len(self.W)

    # ------------------------------------------------------------------ build
    @classmethod
    def affine(cls, a, b, lo, hi, n):
        """Maps w_i(x) = a_i x + b_i on [lo, hi]; images are assigned to the two
        neighbouring grid nodes with mean-preserving (randomized) rounding."""
        x = np.linspace(lo, hi, n)
        h = x[1] - x[0]
        W = []
        for ai, bi in zip(a, b):
            y = np.clip(ai * x + bi, lo, hi)
            t = (y - lo) / h
            j = np.minimum(np.floor(t).astype(int), n - 2)
            frac = np.clip(t - j, 0.0, 1.0)
            rows = np.concatenate([np.arange(n), np.arange(n)])
            cols = np.concatenate([j, j + 1])
            vals = np.concatenate([1.0 - frac, frac])
            W.append(sp.csr_matrix((vals, (rows, cols)), shape=(n, n)))
        return cls(W, x)

    @classmethod
    def m_adic(cls, m, K):
        """w_i(x) = (x + i)/m on [0,1), i = 0..m-1, on the level-K m-adic grid.
        The state is the string of the first K digits; w_i prepends digit i and
        drops the last one, so the stationary law is exactly mu^{(x)K}, i.e. the
        cell masses of the exact invariant measure (no discretization error
        beyond the projection onto cells of width m^-K)."""
        n = m ** K
        j = np.arange(n)
        W = []
        for i in range(m):
            cols = i * m ** (K - 1) + j // m
            W.append(sp.csr_matrix((np.ones(n), (j, cols)), shape=(n, n)))
        return cls(W, j / n)

    # ------------------------------------------------------------- operators
    def push(self, mu, p):
        """P(mu)^T p, i.e. the Markov operator P*_mu on measures.
        p: (n,) with mu: (m,), or p: (n,B) with mu: (B,m)."""
        mu = np.asarray(mu, dtype=float)
        if p.ndim == 1:
            out = np.zeros_like(p)
            for i in range(self.m):
                if mu[i] != 0.0:
                    out += mu[i] * (self.WT[i] @ p)
            return out
        out = np.zeros_like(p)
        for i in range(self.m):
            out += (self.WT[i] @ p) * mu[:, i][None, :]
        return out

    def apply(self, mu, f):
        """P(mu) f, i.e. the transition operator on functions."""
        out = np.zeros_like(f)
        for i in range(self.m):
            if mu[i] != 0.0:
                out += mu[i] * (self.W[i] @ f)
        return out

    def stationary(self, mu, p0=None, tol=1e-13, maxit=200000):
        """Invariant distribution by fixed-point iteration of P*_mu."""
        if p0 is None:
            p = np.full(self.n, 1.0 / self.n)
        else:
            p = p0.copy()
        for _ in range(maxit):
            q = self.push(mu, p)
            if self.w1(p, q) < tol:
                return q
            p = q
        return p

    def stationary_batch(self, mus, iters, p0=None):
        """k-fold iterate (P*_mu)^k p0 for a batch of sampling laws (rows of mus)."""
        B = mus.shape[0]
        if p0 is None:
            P = np.full((self.n, B), 1.0 / self.n)
        else:
            P = np.repeat(p0[:, None], B, axis=1)
        for _ in range(iters):
            P = self.push(mus, P)
        return P

    # -------------------------------------------------- transport quantities
    def w1(self, p, q):
        """W_1 between two measures on the grid (also for columns of arrays)."""
        return self.h * np.abs(np.cumsum(p - q, axis=0)[:-1]).sum(axis=0)

    def potential(self, p, q):
        """An optimal Kantorovich potential f (1-Lipschitz, f(x_0)=0) with
        sum_j f_j (p_j - q_j) = W_1(p, q)."""
        G = np.cumsum(p - q)[:-1]
        return np.concatenate([[0.0], -self.h * np.cumsum(np.sign(G))])

    # ----------------------------------------------------------- sensitivity
    def poisson(self, mu, pi, f, tol=1e-11, maxit=200000):
        """u = sum_k (P^k f - pi(f)), the solution of u - P u = f - pi(f), pi(u)=0.
        The series is truncated once P^k f is constant up to tol; the remaining
        terms only shift u by a constant, which the final centring removes."""
        c = pi @ f
        g = f.copy()
        u = np.zeros_like(f)
        for _ in range(maxit):
            u += g - c
            if np.ptp(g) < tol:
                break
            g = self.apply(mu, g)
        return u - pi @ u

    def sensitivity(self, mu, pi, f, **kw):
        """s_mu(f)_i = ((w_i)_# pi)(u_f), i = 1..m."""
        u = self.poisson(mu, pi, f, **kw)
        return np.array([pi @ (Wi @ u) for Wi in self.W])


# ------------------------------------------------------------------ geometry
def proj_simplex(y):
    """Euclidean projection onto the probability simplex (sorting method)."""
    u = np.sort(y)[::-1]
    css = np.cumsum(u) - 1.0
    ind = np.arange(1, len(y) + 1)
    cond = u - css / ind > 0
    rho = ind[cond][-1]
    theta = css[cond][-1] / rho
    return np.maximum(y - theta, 0.0)


def proj_Ar(y, ell, r, tol=1e-14):
    """Projection onto A_r = {mu in simplex : <ell, mu> <= r}: Proj_simplex(y - lam*ell)
    with lam >= 0 found by bisection (the map lam -> <ell, Proj(y - lam ell)> is
    nonincreasing)."""
    if r < ell.min():
        raise ValueError("A_r is empty: r < min_i L_i")
    z = proj_simplex(y)
    if ell @ z <= r:
        return z
    lo, hi = 0.0, 1.0
    while ell @ proj_simplex(y - hi * ell) > r:
        hi *= 2.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if ell @ proj_simplex(y - mid * ell) > r:
            lo = mid
        else:
            hi = mid
    return proj_simplex(y - hi * ell)


def simplex_grid(m, q):
    """All points (k_1/q, ..., k_m/q) with k_i >= 0 integers summing to q."""
    if m == 1:
        return np.array([[1.0]])
    pts = []

    def rec(prefix, remaining, slots):
        if slots == 1:
            pts.append(prefix + [remaining])
            return
        for k in range(remaining + 1):
            rec(prefix + [k], remaining - k, slots - 1)

    rec([], q, m)
    return np.array(pts, dtype=float) / q


# -------------------------------------------------------- subgradient method
def projected_subgradient(chain, nu, mu0, T, gamma=0.5, step="diminishing",
                          Jstar=0.0, ell=None, r=None, record_every=1):
    """Algorithm 2: projected subgradient method for J(mu) = W_1(nu, pi_mu).

    step = "diminishing": alpha_t = gamma / (G0 * sqrt(t+1)), G0 = |g_0|;
    step = "polyak":      alpha_t = (J(mu_t) - Jstar) / |g_t|^2.
    Returns a dict with the trajectory, objective values and best iterate."""
    if ell is None:
        ell = np.zeros(chain.m)
        r = 1.0
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
        f = chain.potential(p, nu)
        g = chain.sensitivity(mu, p, f)
        g = g - g.mean()  # projection is invariant to shifts along (1,...,1)
        gn = np.linalg.norm(g)
        if gn == 0.0:
            break
        if G0 is None:
            G0 = gn
        if step == "polyak":
            alpha = max(J - Jstar, 0.0) / gn ** 2
        else:
            alpha = gamma / (G0 * np.sqrt(t + 1.0))
        mu = proj_Ar(mu - alpha * g, ell, r)
        p = chain.stationary(mu, p0=p)
        if (t + 1) % record_every == 0:
            traj.append(mu.copy())
    J = chain.w1(p, nu)
    Js.append(J)
    if J < best[0]:
        best = (J, mu.copy())
    return dict(traj=np.array(traj), J=np.array(Js), best_J=best[0], best_mu=best[1])
