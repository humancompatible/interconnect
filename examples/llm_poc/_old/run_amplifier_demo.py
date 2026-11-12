"""Third run: flip the sign of the controller's gain and uniqueness fails.

The first counterexample (``run_divergence_demo.py``) broke *existence* by giving
the controller an unbounded actuator.  This one keeps the seven-rung ladder, so by
Perron-Frobenius an invariant measure certainly exists, and breaks *uniqueness*
instead -- the failure a finite state space does permit.

The only change is one sign.  ``IfElseLadderLogic(polarity=-1)`` is the same
if-else statement with the sign of its gain flipped, so a response that comes out
too positive is answered with a *more* enthusiastic instruction rather than a more
critical one.  The loop amplifies what it observes.  No new LLM sampling is needed:
the conditional law of the sentiment given the rung is a property of the model and
the prompt, not of the controller, so the pools of the main experiment are reused
exactly as they are.

What comes out is more interesting than "positive feedback splits the ladder in
two", which is the naive expectation:

* With the dead zone centred on the reference the loop is still **uniquely
  ergodic** -- and its unique invariant measure is a point mass on the most
  enthusiastic rung.  Uniqueness survives because the model is positively biased,
  so the critical end leaks: even at "be extremely critical and negative" a
  response scores above the band a few percent of the time, which is enough to
  funnel every trajectory to the other end.  A certificate of unique ergodicity is
  not a certificate of good behaviour.
* Recentring the dead zone on where the observations actually live -- the
  textbook response to a biased sensor, ``tau_offset`` -- shuts that leak.  Both
  ends become absorbing and the invariant measure genuinely stops being unique:
  the loop is bistable and the limit depends on where it started.

So the offset does not repair the positive feedback.  It removes the accident that
was masking it: the plant's bias was the only thing keeping the critical end leaky,
and therefore the only thing propping uniqueness up.

Sections 4 and 5 then separate what is certified from what is only estimated,
because the difference between "reducible" and "escapes once in 10^4 steps" is not
visible in a few hundred samples -- and operationally the two are the same thing.
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

from humancompatible.interconnect.guarantees import markov, transience
from humancompatible.interconnect.simulators.distribution import kernel_density_distribution
from humancompatible.interconnect.simulators.logic.agents.llm_agent import CachedLLMAgentLogic
from humancompatible.interconnect.simulators.logic.controllers.instruction_ladder import (
    transition_law_from_pools,
)

sys.path.insert(0, os.path.join(os.getcwd(), "examples"))
from example_systems.example_llm_system import LLMSim

RULE = "=" * 78


def load_pools(path):
    data = np.load(path)
    n = int(data["num_levels"])
    pools = [data[f"level_{i}"] for i in range(n)]
    meta = json.loads(str(data["meta"])) if "meta" in data.files else {}
    return pools, meta


def simulate(pools, tau, tau_offset, polarity, k0, iterations, seed):
    logic = CachedLLMAgentLogic(pools, rng=np.random.default_rng(seed))
    sim = LLMSim(reference_signal=0.0, agent_logic=logic, number_of_agents=1,
                 tau=tau, tau_offset=tau_offset, initial_level=k0, alpha=1.0,
                 loop_gain=1.0, polarity=polarity)
    sim.system.run(iterations, disable_tqdm=True)
    return np.array(sim.level_history), np.array(sim.sentiment_history())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default="examples/llm_poc/pools.npz")
    ap.add_argument("--tau", type=float, default=0.45,
                    help="same dead zone the main experiment selected")
    ap.add_argument("--tau-offset", type=float, default=None,
                    help="centre of the dead zone: the band is "
                         "[tau_offset - tau, tau_offset + tau].  Default is the "
                         "smallest offset in the scan at which both ends of the "
                         "ladder close")
    ap.add_argument("--max-offset", type=float, default=0.6,
                    help="top of the tau_offset scan")
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--burn-in", type=int, default=20)
    ap.add_argument("--escape-rates", type=float, nargs="+",
                    default=[1e-2, 1e-3, 1e-4, 1e-5])
    ap.add_argument("--bandwidth", type=float, default=0.10)
    ap.add_argument("--reference-tv", type=float, default=0.0099,
                    help="TV(occupation | k0=0, k0=6) measured for the "
                         "corrective loop in the main experiment")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fig", default="examples/llm_poc/non_unique_measure.png")
    args = ap.parse_args()

    pools, meta = load_pools(args.pools)
    n_levels = len(pools)

    print(f"\n{RULE}")
    means = []
    for lvl in range(n_levels):
        means.append(pools[lvl].mean())
        print(f"Mean sentiment for k0 = {lvl}: {means[-1]}")
    print(f"Mean sentiment overall: {np.mean(means)}")
    print(RULE)
    # ---- 0. the change ------------------------------------------------------
    print(RULE)
    print("0. ONE SIGN")
    print(RULE)
    print(f"model={meta.get('model')}  pools reused unchanged from the main experiment")
    print("corrective  (polarity=+1): above the band -> more critical, below -> more positive")
    print("amplifying  (polarity=-1): above the band -> more positive, below -> more critical")
    P_amp, up, down = transition_law_from_pools(pools, args.tau, 0.0, polarity=-1)
    print(f"\nobservation law at tau={args.tau}, tau_offset=+0.00 "
          f"(hold band [{-args.tau:+.2f}, {args.tau:+.2f}]; identical for both "
          f"controllers):")
    print(f"  P(above the band | k) = {np.round(up, 4).tolist()}")
    print(f"  P(below the band | k) = {np.round(down, 4).tolist()}")
    print("\namplifying transition matrix:")
    print(np.round(P_amp, 3))

    # ---- 1. the symmetric band: unique -------------------------
    print("\n" + RULE)
    print("1. WITH THE DEAD ZONE ON THE REFERENCE: STILL UNIQUELY ERGODIC")
    print(RULE)
    rep1 = markov.certify(P_amp)
    print(markov.format_report(rep1))
    print(f"\nThe critical end leaks: even at rung {n_levels-1} a response scores above the")
    print(f"band with probability {up[-1]:.3f}, and rungs 0-3 never score below it at all")
    print(f"({np.round(down[:4], 4).tolist()}), so the ladder is a one-way funnel toward")
    print("rung 0.  The measure is unique -- and it is a point mass on 'be extremely")
    print("enthusiastic'.  Unique ergodicity says the loop forgets where it started,")
    print("not that where it ends up is anywhere anyone wanted.")

    # ---- 2. the bifurcation --------------------------------------------------

    print("\n" + RULE)
    print("2. RECENTRING THE DEAD ZONE CLOSES THE LEAK")
    print(RULE)
    print("The leak exists because the plant is biased positive while the dead zone is")
    print("centred on the reference, so the band sits below where the observations")
    print("actually live.  tau_offset moves it: the hold band becomes")
    print("[tau_offset - tau, tau_offset + tau], and the tolerance for 'too positive'")
    print("and 'too negative' stop being equal.")
    print(f"\n{'offset':>7} {'hold band':>16} {'leak k=%d' % (n_levels - 1):>10} "
          f"{'closed classes':>28} {'unique':>7}")
    offset_rows = []
    
    for off in np.round(np.arange(-0.2, args.max_offset + 1e-9, 0.1), 2):
        P_o, u_o, d_o = transition_law_from_pools(pools, args.tau, off, polarity=-1)
        rep_o = markov.certify(P_o)
        offset_rows.append((float(off), float(u_o[-1]),
                            len(rep_o["closed_classes"]),
                            rep_o["unique_invariant_measure"]))
        print(f"{off:>+7.2f} [{off - args.tau:+.2f},{off + args.tau:+.2f}] "
              f"{u_o[-1]:10.4f} {str(rep_o['closed_classes']):>28} "
              f"{str(rep_o['unique_invariant_measure']):>7}")

    broke = [r[0] for r in offset_rows if not r[3]]
    if args.tau_offset is None:
        if not broke:
            raise SystemExit(
                f"uniqueness survives every offset up to {args.max_offset:+.2f}; "
                f"raise --max-offset or --tau")
        args.tau_offset = min(broke)
        print(f"\nsmallest offset at which uniqueness fails: {args.tau_offset:+.2f} "
              f"-- using that below")

    # ---- 3. the bistable regime ---------------------------------------------
    print("\n" + RULE)
    print(f"3. AT tau_offset = {args.tau_offset:+.2f}: THE INVARIANT MEASURE IS NOT UNIQUE")
    print(RULE)
    P_bi, up_bi, dn_bi = transition_law_from_pools(
        pools, args.tau, args.tau_offset, polarity=-1)
    rep = markov.certify(P_bi)
    print(markov.format_report(rep))

    closed = rep["closed_classes"]
    if rep["absorption"] is None:
        raise SystemExit(
            f"at tau_offset={args.tau_offset:+.2f} the chain still has a single "
            f"closed class {closed}; nothing to demonstrate here -- pick a larger "
            f"--tau-offset")
    H = rep["absorption"]
    print("\nSimulating the real interconnect graph from every rung:")
    print(f"{'k0':>4} {'theory P(-> %s)' % closed[-1]:>18} {'empirical':>10} "
          f"{'mean terminal s':>16} {'mean level':>11}")
    paths, sentiments, terminal = {}, {}, {}
    empirical = []
    for start_level in range(n_levels):
        lp, sp = [], []
        for r in range(args.runs):
            lv, sv = simulate(pools, args.tau, args.tau_offset, -1, start_level,
                              args.iterations, args.seed + 7919 * start_level + r)
            lp.append(lv); sp.append(sv)
        paths[start_level] = np.array(lp); sentiments[start_level] = np.array(sp)
        terminal[start_level] = np.array([s[-1] for s in sp])
        emp = float(np.mean([lv[-1] in closed[-1] for lv in lp]))
        empirical.append(emp)
        print(f"{start_level:4d} {H[start_level, -1]:18.4f} {emp:10.4f} "
              f"{terminal[start_level].mean():+16.4f} {paths[start_level][:, -1].mean():11.2f}")

    occ0 = markov.occupation_measure(list(paths[0]), n_levels, burn_in=args.burn_in)
    occ6 = markov.occupation_measure(list(paths[n_levels - 1]), n_levels, burn_in=args.burn_in)
    print(f"\noccupation | k0=0     : {np.round(occ0, 4).tolist()}")
    print(f"occupation | k0={n_levels-1}     : {np.round(occ6, 4).tolist()}")
    print(f"TV(occupation k0=0, k0={n_levels-1}) = {markov.tv_distance(occ0, occ6):.4f}")
    print(f"  the corrective loop of the main experiment reports "
          f"{args.reference_tv:.4f} here.")
    print("  1.0 is the sharpest statement of non-uniqueness there is: the loop")
    print("  remembers its initial condition forever, and the time average along a")
    print("  trajectory tells you about that trajectory, not about the system.")


    # ---- 4. the same dead zone, corrective sign -----------------------------
    print("\n" + RULE)
    print("4. CONTROL: SAME DEAD ZONE, CORRECTIVE SIGN")
    print(RULE)
    P_corr, _, _ = transition_law_from_pools(
        pools, args.tau, args.tau_offset, polarity=+1)
    rep_corr = markov.certify(P_corr)
    print(markov.format_report(rep_corr))
    corr_terminal = {}
    for start_level in (0, n_levels - 1):
        sp = [simulate(pools, args.tau, args.tau_offset, +1, start_level,
                       args.iterations, args.seed + 3571 * start_level + r)[1]
              for r in range(args.runs)]
        corr_terminal[start_level] = np.array([s[-1] for s in sp])
    print(f"\nterminal sentiment | k0=0   : mean {corr_terminal[0].mean():+.4f}")
    print(f"terminal sentiment | k0={n_levels-1}   : mean {corr_terminal[n_levels-1].mean():+.4f}")
    print("Same plant, same observable, same dead zone -- offset and all.  Only the")
    print("sign of the gain differs, and it decides whether the loop has one invariant")
    print("measure or a simplex of them.")

    # ---- 5. reducible, or merely metastable? --------------------------------
    print("\n" + RULE)
    print("5. WHAT IS CERTIFIED, AND WHAT IS ONLY ESTIMATED")
    print(RULE)
    n_s = len(pools[0])
    ucb = transience.wilson_interval(0, n_s)[1]
    print("Two closed classes require the escape rate out of *each* end to be exactly")
    print(f"zero, and no finite sample can establish that: a rate counted as 0 out of")
    print(f"{n_s} samples has a 95% upper bound of {ucb:.4f}.  So suppose both ends leak at")
    print("some small eps > 0 instead: the chain is then irreducible, the invariant")
    print("measure is unique again, and the question becomes how long uniqueness takes")
    print("to assert itself.")
    print(f"\n{'assumed eps':>12} {'unique':>7} {'Doeblin eps':>12} "
          f"{'Doeblin bound':>15} {'Dobrushin':>12}   (from any starting rung)")
    meta_rows = []
    for eps in args.escape_rates:
        # Every rate the bootstrap reported as exactly zero is set to eps and the
        # hold probability absorbs the difference.  Perturbing only the two
        # extreme rungs is not enough: rungs 0-3 never scored below -tau either,
        # so the ladder stays a one-way funnel and the chain stays reducible.
        u_eps = np.where(up_bi <= 0, eps, up_bi)
        d_eps = np.where(dn_bi <= 0, eps, dn_bi)
        P_eps = np.zeros((n_levels, n_levels))
        for k in range(n_levels):
            P_eps[k, max(k - 1, 0)] += u_eps[k]
            P_eps[k, min(k + 1, n_levels - 1)] += d_eps[k]
            P_eps[k, k] += 1.0 - u_eps[k] - d_eps[k]
        rep_eps = markov.certify(P_eps, max_power=800)
        meta_rows.append((eps, rep_eps["tv_mixing_bound_total"]))
        print(f"{eps:12.0e} {str(rep_eps['unique_invariant_measure']):>7} "
              f"{rep_eps['doeblin_epsilon']:12.3g} {rep_eps['tv_mixing_bound_total']:15.3g} "
              f"{rep_eps['dobrushin_mixing_bound']:12.3g}")
    ratios = [meta_rows[i + 1][1] / meta_rows[i][1] for i in range(len(meta_rows) - 1)
              if np.isfinite(meta_rows[i][1]) and meta_rows[i][1] > 0]
    print(f"\nBoth are upper bounds, and loose ones: the minorization search is capped at")
    print(f"m=400, far below where this chain actually starts to mix, so neither route")
    print(f"finds a tight certificate.  The scaling is what carries the argument --")
    print(f"successive bounds grow by {np.round(ratios, 1).tolist()} as eps falls by 10x.")
    print("The mixing time scales as 1/eps, as it must -- the chain has to wait for the")
    print("rare escape.  The loop is uniquely ergodic at every eps in this table and")
    print("needs more iterations to show it than the whole experiment used LLM calls.  For any")
    print("horizon this system will ever be run on, 'not unique' and 'unique but")
    print("metastable' are the same statement, and the honest report is both numbers.")

    # ---- figure --------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    ax_bif, ax_path, ax_abs, ax_occ, ax_kde, ax_meta = axes.ravel()

    for start_level in [0, n_levels-1]:
        ax_path.plot(sentiments[start_level][0], lw=1.2, alpha=0.9, label=f"$k_0$={start_level}")

    ax_path.axhline(0.0, color="k", lw=0.9)
    ax_path.axhspan(-args.tau + args.tau_offset, args.tau + args.tau_offset, color="tab:green", alpha=0.13)
    ax_path.set_ylim(-1.05, 1.05)
    ax_path.set_title(r"Observed sentiment $s_t$ (green: hold band $|s|\leq\tau$)")
    ax_path.set_xlabel("iteration"); ax_path.set_ylabel("sentiment  $s_t$")
    ax_path.legend(fontsize=8, loc="lower right"); ax_path.grid(alpha=0.3)

    offs = [r[0] for r in offset_rows]
    floor = 0.5 / len(pools[0])          # a count of 0 is "below one sample"
    ax_bif.semilogy(offs, [max(r[1], floor) for r in offset_rows], "o-", lw=1.8,
                    color="tab:red", label=f"leak out of rung {n_levels - 1}")
    ax_bif.axhline(floor, color="tab:red", lw=0.8, ls=":")
    ax_bif.text(offs[0], floor * 1.15, "below one sample in the pool",
                fontsize=7, color="tab:red")
    ax_bif.axvline(args.tau_offset, color="k", lw=0.9, ls="--")
    ax_bif.set_xlabel(r"dead-zone centre $\tau_0$"); ax_bif.set_ylabel("leak rate")
    ax_bif.grid(alpha=0.3, which="both")
    ax2 = ax_bif.twinx()
    ax2.step(offs, [r[2] for r in offset_rows], where="mid", color="tab:brown",
             lw=1.6, label="closed classes")
    ax2.set_ylabel("number of closed classes")
    ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax_bif.set_title("(a) Recentring the dead zone closes the\nleak, and uniqueness dies with it")
    h1, l1 = ax_bif.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax_bif.legend(h1 + h2, l1 + l2, fontsize=8, loc="center left")

    # # A few runs from every rung, coloured by which end captured them.  The split
    # # is a watershed rather than a coin flip: rungs 0-3 never produce a response
    # # below -tau, so nothing starting below the divide can ever cross it.
    # for start_level in range(n_levels):
    #     for r in range(min(8, args.runs)):
    #         ax_path.plot(paths[start_level][r], lw=0.9, alpha=0.6,
    #                      color="tab:blue" if paths[start_level][r][-1] == 0 else "tab:red")
    # ax_path.set_xlim(0, min(30, args.iterations))
    # ax_path.set_xlabel("iteration"); ax_path.set_ylabel("instruction level $k_t$")
    # ax_path.set_yticks(range(n_levels))
    # ax_path.set_title("(b) A watershed, not a coin flip: eight runs\n"
    #                   "from every rung, coloured by where they end")
    # ax_path.grid(alpha=0.3)

    ax_abs.plot(range(n_levels), H[:, -1], "o-", lw=2.0, label="theory (absorption solve)")
    ax_abs.plot(range(n_levels), empirical, "s--", lw=1.6, label=f"empirical ({args.runs} runs)")
    ax_abs.set_xlabel("initial level $k_0$")
    ax_abs.set_ylabel(r"$P(\to$ critical end$)$")
    ax_abs.set_ylim(-0.05, 1.05)
    ax_abs.set_title("(c) The limit is a function of the\ninitial condition")
    ax_abs.legend(fontsize=8); ax_abs.grid(alpha=0.3)

    width = 0.38
    idx = np.arange(n_levels)
    ax_occ.bar(idx - width / 2, occ0, width, label="occupation | $k_0$=0")
    ax_occ.bar(idx + width / 2, occ6, width, label=f"occupation | $k_0$={n_levels-1}")
    ax_occ.set_xlabel("instruction level $k$"); ax_occ.set_xticks(idx)
    ax_occ.set_title(f"(d) Two initial conditions, two measures\n"
                     f"TV = {markov.tv_distance(occ0, occ6):.3f} "
                     f"(corrective loop: {args.reference_tv:.3f})")
    ax_occ.legend(fontsize=8); ax_occ.grid(alpha=0.3, axis="y")

    grid = np.arange(-1.05, 1.05 + 1e-9, 0.005)
    for start_level, c in ((0, "tab:blue"), (n_levels - 1, "tab:red")):
        ax_kde.plot(grid, kernel_density_distribution(grid, terminal[start_level], args.bandwidth),
                    lw=2.0, color=c, label=f"amplifying, $k_0$={start_level}")
        ax_kde.plot(grid, kernel_density_distribution(grid, corr_terminal[start_level],
                                                      args.bandwidth),
                    lw=1.4, ls="--", color=c, alpha=0.7,
                    label=f"corrective, $k_0$={start_level}")
    ax_kde.axvline(0.0, color="k", lw=0.9, ls=":")
    ax_kde.set_xlabel("terminal sentiment $s$"); ax_kde.set_ylabel("density")
    ax_kde.set_title("(e) Amplifying: two limits.\nCorrective: one, from either end")
    ax_kde.legend(fontsize=7); ax_kde.grid(alpha=0.3)

    ax_meta.loglog([m[0] for m in meta_rows], [m[1] for m in meta_rows], "o-",
                   lw=1.8, color="tab:purple")
    ax_meta.set_xlabel(r"assumed escape rate $\varepsilon$ out of the critical end")
    ax_meta.set_ylabel("iterations to TV < 0.01")
    ax_meta.set_title("(f) Reducible, or merely metastable?\n"
                      r"mixing time $\sim 1/\varepsilon$, and $\varepsilon$ is unmeasurable")
    ax_meta.grid(alpha=0.3, which="both")

    fig.tight_layout(); fig.savefig(args.fig, dpi=140)
    print(f"\nwrote {args.fig}")


if __name__ == "__main__":
    main()
