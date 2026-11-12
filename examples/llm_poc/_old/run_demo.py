"""Phase B of the LLM ergodicity PoC: certify and estimate the invariant measure.

Consumes the per-level sentiment pools written by ``build_pools.py`` and does no
LLM work at all, so the whole analysis runs in seconds.

1. Sweep the controller dead zone ``tau`` and pick one, from the pools alone.
2. Simulate the interconnect loop from several initial levels.
3. Estimate the transition matrix and cross-check it against the analytic one.
4. Certify: communicating classes, recurrent class, period, Doeblin minorization.
5. Estimate the invariant measure of the observed sentiment by KDE, overlaid
   across initial conditions.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.getcwd())

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from humancompatible.interconnect.guarantees import markov
from humancompatible.interconnect.simulators.distribution import kernel_density_distribution
from humancompatible.interconnect.simulators.logic.agents.llm_agent import CachedLLMAgentLogic
from humancompatible.interconnect.simulators.logic.controllers.instruction_ladder import (
    transition_law_from_pools,
)

sys.path.insert(0, os.path.join(os.getcwd(), "examples"))
from example_systems.example_llm_system import LLMSim


def load_pools(path):
    data = np.load(path)
    n = int(data["num_levels"])
    pools = [data[f"level_{k}"] for k in range(n)]
    meta = json.loads(str(data["meta"])) if "meta" in data else {}
    return pools, meta


def scan_tau(pools, number_of_agents, grid, min_mass=0.02):
    """Pick the dead zone from the pools alone -- no further LLM calls.

    Selection minimises the Doeblin mixing bound subject to the invariant measure
    putting real mass on more than one rung.  Selecting for spread alone is a
    trap: it drives tau to 0, removing the hold band and leaving a period-2 chain
    with no minorization at all.

    The dead zone here is symmetric about the reference by design.  The
    controller also supports an off-centre band via ``tau_offset``, but shifting
    it means regulating to the centre of the band rather than to ``r = 0``, so
    the main experiment does not use it; the counterexample runs do.
    """
    print(f"\n{'tau':>6} {'uniq':>5} {'aper':>5} {'eps':>7} {'m':>3} {'in-class':>8} "
          f"{'total':>6} {'H(pi)':>6} {'#rungs':>7}  stationary pi")
    print("-" * 104)
    rows = []
    for tau in grid:
        P, up, dn = transition_law_from_pools(pools, tau,
                                              number_of_agents=number_of_agents)
        rep = markov.certify(P)
        pi = rep["stationary"]
        H = -float(np.sum(pi[pi > 0] * np.log(pi[pi > 0])))
        # Select on the bound that actually applies to a run started anywhere,
        # i.e. including the time to reach the recurrent class -- not just the
        # mixing time once inside it.
        mix = rep["tv_mixing_bound_total"]
        rungs = int((pi >= min_mass).sum())
        ok = (rep["unique_invariant_measure"] and rep["aperiodic"]
              and rungs >= 2 and np.isfinite(mix))
        rows.append((tau, rep, H, mix, pi, ok, rungs))
        print(f"{tau:6.2f} {str(rep['unique_invariant_measure']):>5} "
              f"{str(rep['aperiodic']):>5} {rep['doeblin_epsilon']:7.4f} "
              f"{str(rep['doeblin_power']):>3} {rep['tv_mixing_bound']:8.0f} "
              f"{mix:6.0f} {H:6.3f} {rungs:7d}  "
              f"{np.round(pi, 3).tolist()}{'' if ok else '  (rejected)'}")

    viable = [r for r in rows if r[5]]
    if not viable:
        print("\nNo tau satisfies the criteria; relaxing to 'unique + aperiodic'.")
        viable = [r for r in rows if r[1]["unique_invariant_measure"] and r[1]["aperiodic"]]
    if not viable:
        raise SystemExit("No uniquely-ergodic aperiodic regime found in the grid.")
    viable.sort(key=lambda r: (r[3], -r[2]))
    best = viable[0]
    print(f"\nchosen tau = {best[0]:.2f}  (mixing bound {best[3]:.0f} its, "
          f"entropy {best[2]:.3f}, {best[6]} rungs with mass >= {min_mass})")
    return float(best[0])


def simulate(pools, args, initial_level, seed):
    logic = CachedLLMAgentLogic(pools, rng=np.random.default_rng(seed))
    sim = LLMSim(reference_signal=0.0, agent_logic=logic,
                 number_of_agents=args.agents, tau=args.tau,
                 initial_level=initial_level, alpha=args.alpha,
                 loop_gain=args.loop_gain)
    sim.system.run(args.iterations, disable_tqdm=True)
    return sim.level_history, sim.sentiment_history()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default="examples/llm_poc/pools.npz")
    ap.add_argument("--iterations", type=int, default=120)
    ap.add_argument("--runs", type=int, default=600)
    ap.add_argument("--agents", type=int, default=1)
    ap.add_argument("--burn-in", type=int, default=40)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--loop-gain", type=float, default=1.0)
    ap.add_argument("--initial-levels", type=int, nargs="+", default=None)
    ap.add_argument("--bandwidth", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fig", default="examples/llm_poc/invariant_measure.png")
    args = ap.parse_args()

    pools, meta = load_pools(args.pools)
    n_levels = len(pools)
    print(f"pools: {n_levels} levels x {len(pools[0])} samples   model={meta.get('model')}")
    print(f"topic: {meta.get('base_prompt')!r}")
    for k, p in enumerate(pools):
        print(f"  k={k}: mean={p.mean():+.4f} sd={p.std():.4f} "
              f"P(s>0)={float((p > 0).mean()):.3f}")

    if args.initial_levels is None:
        args.initial_levels = [0, n_levels - 1]

    if args.tau is None:
        args.tau = scan_tau(pools, args.agents, np.round(np.arange(0.05, 0.85, 0.05), 2))
    print(f"\nusing tau={args.tau:.2f}  agents={args.agents}  alpha={args.alpha}")

    P_analytic, up, dn = transition_law_from_pools(
        pools, args.tau, number_of_agents=args.agents)
    print(f"P(-> more critical | k) = {np.round(up, 4).tolist()}")
    print(f"P(-> more positive | k) = {np.round(dn, 4).tolist()}")

    paths, sentiments, terminal = {}, {}, {}
    for start_level in args.initial_levels:
        lp, sp = [], []
        for r in range(args.runs):
            levels, sents = simulate(pools, args, start_level, seed=args.seed + 10_000 * start_level + r)
            lp.append(levels); sp.append(sents)
        paths[start_level] = lp; sentiments[start_level] = sp
        terminal[start_level] = np.array([s[-1] for s in sp])
        occ = markov.occupation_measure(lp, n_levels, burn_in=args.burn_in)
        print(f"\ninitial level {start_level}: occupation {np.round(occ, 4).tolist()}")

    all_paths = [p for lvl in args.initial_levels for p in paths[lvl]]
    P_hat, counts = markov.estimate_transition_matrix(all_paths, n_levels)
    print(f"\ntransition counts (total {int(counts.sum())}):")
    print(counts.astype(int))
    print(f"max |P_hat - P_analytic| = {np.nanmax(np.abs(P_hat - P_analytic)):.4f}")

    print("\n" + "=" * 70)
    print("UNIQUE ERGODICITY CERTIFICATE (empirical transition matrix)")
    print("=" * 70)
    report = markov.certify(P_hat, counts=counts)
    print(markov.format_report(report))

    pi = report["stationary"]
    for start_level in args.initial_levels:
        occ = markov.occupation_measure(paths[start_level], n_levels, burn_in=args.burn_in)
        print(f"TV(occupation | k0={start_level}, pi) = {markov.tv_distance(occ, pi):.4f}")
    if len(args.initial_levels) >= 2:
        a, b = args.initial_levels[0], args.initial_levels[-1]
        occ_a = markov.occupation_measure(paths[a], n_levels, burn_in=args.burn_in)
        occ_b = markov.occupation_measure(paths[b], n_levels, burn_in=args.burn_in)
        print(f"TV(occupation k0={a}, k0={b}) = {markov.tv_distance(occ_a, occ_b):.4f}"
              f"   <- initial-condition independence")

    # ---- figure ---------------------------------------------------------------
    # NOTE: distribution.get_distributions() pads its evaluation range by +-10,
    # which swamps a sentiment score bounded in [-1, 1], so we call the underlying
    # estimator kernel_density_distribution() directly.
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8),
                             gridspec_kw={"width_ratios": [2.1, 1.0, 1.5]})
    ax_path, ax_kde, ax_pi = axes
    ax_kde.sharey(ax_path)

    for start_level in args.initial_levels[:2]:
        ax_path.plot(sentiments[start_level][0], lw=1.2, alpha=0.9, label=f"$k_0$={start_level}")
    ax_path.axhline(0.0, color="k", lw=0.9)
    ax_path.axhspan(-args.tau, args.tau, color="tab:green", alpha=0.13)
    ax_path.set_ylim(-1.05, 1.05)
    ax_path.set_title(r"Observed sentiment $s_t$ (green: hold band $|s|\leq\tau$)")
    ax_path.set_xlabel("iteration"); ax_path.set_ylabel("sentiment  $s_t$")
    ax_path.legend(fontsize=8, loc="lower right"); ax_path.grid(alpha=0.3)

    grid = np.arange(-1.05, 1.05 + 1e-9, 0.005)
    dens = {}
    for start_level in args.initial_levels:
        dens[start_level] = kernel_density_distribution(grid, terminal[start_level], args.bandwidth)
        ax_kde.plot(dens[start_level], grid, lw=1.8, label=f"$k_0$ = {start_level}")
    ax_kde.axhline(0.0, color="k", lw=0.9, ls=":")
    ax_kde.set_title(f"Invariant measure\n(KDE, h={args.bandwidth})")
    ax_kde.set_xlabel("density"); ax_kde.legend(fontsize=8); ax_kde.grid(alpha=0.3)

    width = 0.38
    idx = np.arange(n_levels)
    occ_all = markov.occupation_measure(all_paths, n_levels, burn_in=args.burn_in)
    ax_pi.bar(idx - width / 2, pi, width, label=r"stationary $\pi$ from $\hat P$")
    ax_pi.bar(idx + width / 2, occ_all, width, label="empirical occupation")
    ax_pi.set_title(f"Invariant measure over instruction level\n"
                    f"TV = {markov.tv_distance(pi, occ_all):.4f}, "
                    f"Doeblin $\\epsilon$={report['doeblin_epsilon']:.3f} "
                    f"at $m$={report['doeblin_power']}")
    ax_pi.set_xlabel("instruction level $k$"); ax_pi.set_xticks(idx)
    ax_pi.legend(fontsize=8); ax_pi.grid(alpha=0.3, axis="y")

    fig.tight_layout(); fig.savefig(args.fig, dpi=140)
    print(f"\nwrote {args.fig}")

    if len(args.initial_levels) >= 2:
        a, b = args.initial_levels[0], args.initial_levels[-1]
        step = grid[1] - grid[0]
        print(f"TV between sentiment KDEs (k0={a} vs {b}) = "
              f"{0.5 * float(np.abs(dens[a] - dens[b]).sum() * step):.4f}")
    for start_level in args.initial_levels:
        t = terminal[start_level]
        print(f"terminal sentiment | k0={start_level}: mean={t.mean():+.4f} sd={t.std():.4f}")


if __name__ == "__main__":
    main()
