"""Counterexample run: the same loop, the naive actuator, and no invariant measure.

``run_demo.py`` certifies that the closed loop of the main experiment has exactly
one invariant measure and reaches it in about a dozen iterations.  This script is
its negative: the plant, the observable, the controller law and the threshold are
all unchanged, and only the *actuator* is swapped for the one the PoC originally
proposed -- "if the response is too positive, append one more copy of 'Be more
neutral.'".  That single change destroys ergodicity, and the point of the run is
that the destruction is certifiable rather than merely visible.

Two distinct failures are demonstrated, because they are genuinely different:

1. **No invariant probability measure at all** (sections 1-5).  The naive
   actuator has no last rung, so the state space is ``{0, 1, 2, ...}``; the
   instruction saturates, the loop keeps escalating, and the mass escapes to
   infinity.  Certified from cut balance by ``guarantees.transience``.

2. **An invariant measure that is not unique** (section 6).  On the *bounded*
   ladder, widening the controller's dead zone past the observable's range leaves
   every state absorbing.  An invariant measure then exists -- a whole simplex of
   them -- and the long-run behaviour is decided by the initial condition.
   Certified from the closed-class count by ``guarantees.markov``.

Neither section calls the LLM: section 1-5 consume ``pools_repeat.npz`` and
section 6 the ``pools.npz`` of the main experiment.
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
from humancompatible.interconnect.simulators.logic.agents.llm_agent import CachedLLMAgentLogic
from humancompatible.interconnect.simulators.logic.controllers.instruction_ladder import (
    rates_from_pools, transition_law_from_pools,
)

sys.path.insert(0, os.path.join(os.getcwd(), "examples"))
from example_systems.example_llm_system import LLMSim

RULE = "=" * 78


def load_pools(path):
    data = np.load(path)
    n = int(data["num_levels"])
    pools = [data[f"level_{i}"] for i in range(n)]
    levels = ([int(v) for v in data["levels"]] if "levels" in data.files
              else list(range(n)))
    meta = json.loads(str(data["meta"])) if "meta" in data.files else {}
    return pools, levels, meta


def simulate_unbounded(pools, levels, tau, tau_offset, initial_level, iterations,
                       seed):
    """Run the real interconnect graph with an uncapped instruction ladder."""
    logic = CachedLLMAgentLogic(pools, rng=np.random.default_rng(seed), levels=levels)
    sim = LLMSim(reference_signal=0.0, agent_logic=logic, number_of_agents=1,
                 num_levels=None, tau=tau, tau_offset=tau_offset,
                 initial_level=initial_level, alpha=1.0, loop_gain=1.0)
    sim.system.run(iterations, disable_tqdm=True)
    return np.array(sim.level_history), np.array(sim.sentiment_history())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools-repeat", default="examples/llm_poc/pools_repeat.npz")
    ap.add_argument("--pools", default="examples/llm_poc/pools.npz")
    ap.add_argument("--tau", type=float, default=0.45,
                    help="same dead zone as the main experiment")
    ap.add_argument("--tau-offset", type=float, default=0.0,
                    help="centre of the dead zone; the counterexample is "
                         "insensitive to it, which is the point -- see below")
    ap.add_argument("--tail-from", type=int, default=5,
                    help="ladder rung from which the conditional law is taken as saturated")
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--initial-levels", type=int, nargs="+", default=[0, 5])
    ap.add_argument("--truncations", type=int, nargs="+", default=[10, 20, 50, 100])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fig", default="examples/llm_poc/no_invariant_measure.png")
    args = ap.parse_args()

    pools_r, levels_r, meta_r = load_pools(args.pools_repeat)
    pools_s, levels_s, _ = load_pools(args.pools)
    tail_idx = int(np.searchsorted(levels_r, args.tail_from))

    # ---- 0. the two actuators ------------------------------------------------
    print(RULE)
    print("0. THE TWO ACTUATORS, MEASURED")
    print(RULE)
    mean_s = np.array([p.mean() for p in pools_s])
    mean_r = np.array([p.mean() for p in pools_r])
    print(f"model={meta_r.get('model')}  topic={meta_r.get('base_prompt')!r}  "
          f"n={len(pools_r[0])}/rung")
    print(f"signed ladder  rungs {levels_s}")
    print(f"  mean sentiment {np.round(mean_s, 3).tolist()}")
    print(f"  span {mean_s.max() - mean_s.min():+.3f}   crosses the reference: "
          f"{bool(mean_s.min() < 0 < mean_s.max())}")
    print(f"repeat ladder  rungs {levels_r}")
    print(f"  mean sentiment {np.round(mean_r, 3).tolist()}")
    print(f"  span {mean_r.max() - mean_r.min():+.3f}   crosses the reference: "
          f"{bool(mean_r.min() < 0 < mean_r.max())}")
    print("\nRepeating an instruction does not compound its effect: the ladder has")
    print("no authority, so the error never enters the dead zone and the controller")
    print("never stops escalating.")

    # ---- 1. saturation -------------------------------------------------------
    print("\n" + RULE)
    print("1. HAS THE CONDITIONAL LAW SATURATED?  (the one extrapolation we make)")
    print(RULE)
    rows = transience.saturation_report(pools_r, levels_r, args.tau,
                                        args.tail_from, tau_offset=args.tau_offset)
    print(f"{'rung':>5} {'mean':>8} {'sd':>6} {'p=P(s>tau)':>12} {'95% CI':>16} "
          f"{'q=P(s<-tau)':>12} {'95% CI':>16} {'KS to top':>10}")
    for r in rows:
        star = "*" if r["in_tail"] else " "
        print(f"{r['level']:5d}{star}{r['mean']:+8.3f} {r['sd']:6.3f} {r['p']:12.3f} "
              f"[{r['p_ci'][0]:.3f},{r['p_ci'][1]:.3f}] {r['q']:12.3f} "
              f"[{r['q_ci'][0]:.3f},{r['q_ci'][1]:.3f}] {r['ks']:10.3f}")
    tail_ks = [r["ks"] for r in rows if r["in_tail"]]
    print(f"\n* = treated as tail (rung >= {args.tail_from}).  Max KS within the tail: "
          f"{max(tail_ks):.3f}")
    print(f"Rungs 20 and 50 exist only to test this: if the law at rung 50 still")
    print(f"matched rung 5, extrapolating past rung {levels_r[-1]} is the mild step it looks like.")

    # ---- 2. rates ------------------------------------------------------------
    print("\n" + RULE)
    print(f"2. CLOSED-LOOP TRANSITION RATES AT tau={args.tau}, "
          f"tau_offset={args.tau_offset:+.2f}")
    print(RULE)
    up, down = rates_from_pools(pools_r, args.tau, tau_offset=args.tau_offset)
    print(f"rung        {levels_r}")
    print(f"p_k (up)    {np.round(up, 4).tolist()}")
    print(f"q_k (down)  {np.round(down, 4).tolist()}")

    # ---- 3. certificate ------------------------------------------------------
    print("\n" + RULE)
    print("3. CERTIFICATE: NO INVARIANT PROBABILITY MEASURE")
    print(RULE)
    cert = transience.certify_no_invariant_measure(
        up, down, n_samples=len(pools_r[0]), tail_from=tail_idx)
    print(transience.format_report(cert, levels=levels_r))

    # ---- 4. closed-loop simulation ------------------------------------------
    print("\n" + RULE)
    print("4. CLOSED-LOOP SIMULATION THROUGH THE INTERCONNECT GRAPH")
    print(RULE)
    paths, sents = {}, {}
    for k0 in args.initial_levels:
        lp, sp = [], []
        for run in range(args.runs):
            lv, sv = simulate_unbounded(pools_r, levels_r, args.tau,
                                        args.tau_offset, k0, args.iterations,
                                        args.seed + 9973 * k0 + run)
            lp.append(lv); sp.append(sv)
        paths[k0] = np.array(lp); sents[k0] = np.array(sp)
        final = paths[k0][:, -1]
        t = np.arange(args.iterations)
        half = args.iterations // 2
        slope = float(np.polyfit(t[half:], paths[k0].mean(axis=0)[half:], 1)[0])
        returned = float(np.mean([(lv[20:] <= args.tail_from).any() for lv in lp]))
        print(f"k0={k0}: mean level after {args.iterations} its = {final.mean():.1f} "
              f"(min {final.min()}, max {final.max()})")
        print(f"       empirical drift {slope:+.4f}/it vs predicted "
              f"{cert['drift_tail']:+.4f}/it")
        print(f"       runs ever returning to rung <= {args.tail_from} after t=20: "
              f"{returned:.3f}   (bound q/p <= {1 - cert['escape_probability']:.3f})")
        print(f"       mean sentiment over the run {sents[k0].mean():+.4f} "
              f"-- reference is 0, dead zone is "
              f"[{args.tau_offset - args.tau:+.2f}, {args.tau_offset + args.tau:+.2f}]")
    print("\nThe level does not fluctuate around anything: it grows linearly, forever.")
    print("There is no occupation measure to converge to -- the fraction of time")
    print("spent in any fixed rung tends to zero as the horizon grows.")
    win = args.tail_from
    for k0 in args.initial_levels:
        horizons = sorted({h for h in (25, 50, 100, args.iterations)
                           if h <= args.iterations})
        occ = [(paths[k0][:, :T] <= win).mean() for T in horizons]
        print(f"  k0={k0}: fraction of time in rungs <= {win} over horizons "
              f"{'/'.join(map(str, horizons))}: {np.round(occ, 3).tolist()}")

    # ---- 5. truncation is not a fix -----------------------------------------
    print("\n" + RULE)
    print("5. TRUNCATING THE LADDER MANUFACTURES A CERTIFICATE")
    print(RULE)
    # Use the confidence bounds rather than the point estimates: they are the
    # most recurrence-friendly rates the data still allows, so the truncated
    # chain is genuinely irreducible and the certificate it produces is a real
    # one -- which is the whole point of the section.
    p_lo = np.array([ci[0] for ci in cert["p_ci"]])
    q_hi = np.array([ci[1] for ci in cert["q_ci"]])
    trunc = transience.truncation_sequence(cert["p_tail_lcb"], cert["q_tail_ucb"],
                                           args.truncations, up=p_lo, down=q_hi)
    print("Cap the ladder at N and the state space is finite again, so an invariant")
    print("measure exists and -- the chain being irreducible -- is unique.  But it is")
    print("a property of the cap, not of the loop: it slides to the top rung and the")
    print("mass below any fixed rung goes to zero.  Rates below are the 95% bounds,")
    print("i.e. the most recurrence-friendly ones consistent with the measurements.")
    print(f"\n{'cap N':>6} {'pi(k<=%d)' % win:>12} {'pi(top)':>9} {'mode':>6} "
          f"{'unique?':>8} {'TV<0.01 in':>12}")
    for N in args.truncations:
        pi = trunc[N]
        u, d = transience.truncated_rates(cert["p_tail_lcb"], cert["q_tail_ucb"], N,
                                          up=p_lo, down=q_hi)
        P_tr = np.zeros((N + 1, N + 1))
        for k in range(N + 1):
            P_tr[k, min(k + 1, N)] += u[k]
            P_tr[k, max(k - 1, 0)] += d[k]
            P_tr[k, k] += 1 - u[k] - d[k]
        rep_tr = markov.certify(P_tr, max_power=min(6 * (N + 1), 400))
        print(f"{N:6d} {pi[:win + 1].sum():12.3e} {pi[-1]:9.4f} "
              f"{int(np.argmax(pi)):6d} {str(rep_tr['unique_invariant_measure']):>8} "
              f"{rep_tr['tv_mixing_bound']:12.0f}")
    print("\nEvery cap yields a unique invariant measure and a finite mixing time --")
    print("a clean certificate for a loop that has none.  Two things give the game")
    print("away.  The measure is a property of the cap: lift the cap and it moves")
    print("with it, converging to nothing.  And the mixing bound grows without bound")
    print("as the cap is lifted, which is the certificate itself reporting that the")
    print("cap is doing all the work.  Truncating an unbounded actuator is an")
    print("assumption about the system, not a modelling convenience, and it has to be")
    print("earned -- here it would have to be earned by a rung at which the model")
    print("stops accepting further instructions, and there is no such rung.")

    # ---- 6. the other failure: non-uniqueness -------------------------------
    print("\n" + RULE)
    print("6. THE OTHER FAILURE: A BOUNDED LADDER WITH NO *UNIQUE* INVARIANT MEASURE")
    print(RULE)
    print("On a finite state space Perron-Frobenius guarantees an invariant measure")
    print("exists, so the only available failure is non-uniqueness.  Widen the dead")
    print("zone past the range of the observable and every state becomes absorbing.")
    print(f"\n{'tau':>6} {'closed':>7} {'unique':>7} {'eps':>9} {'m':>3} {'TV<0.01 in':>12} "
          f"{'min rate':>10}")
    sweep = []
    for tau in [0.45, 0.60, 0.70, 0.80, 0.85, 0.90, 0.92, 0.94, 0.95, 0.96, 0.98, 0.99, 1.00]:
        P, u2, d2 = transition_law_from_pools(pools_s, tau, args.tau_offset)
        rep = markov.certify(P)
        moves = np.concatenate([u2, d2])
        nz = moves[moves > 0]
        sweep.append((tau, len(rep["closed_classes"]), rep["tv_mixing_bound_total"]))
        print(f"{tau:6.2f} {len(rep['closed_classes']):7d} "
              f"{str(rep['unique_invariant_measure']):>7} {rep['doeblin_epsilon']:9.4g} "
              f"{str(rep['doeblin_power']):>3} {rep['tv_mixing_bound_total']:12.0f} "
              f"{(nz.min() if nz.size else 0.0):10.4g}")
    n_s = len(pools_s[0])
    ucb = transience.wilson_interval(0, n_s)[1]
    print(f"\nAt tau >= 1 the fracture is exact, not sampled: the classifier readout is")
    print(f"p_pos - p_neg in (-1, 1), so no observation can leave the dead zone, P = I,")
    print(f"and every one of the {len(pools_s)} states is absorbing -- a whole simplex of")
    print(f"invariant measures, and the loop's limit is whatever it was initialised to.")
    print(f"Below tau = 1 the fracture is only *estimated*: a rate observed as 0 in "
          f"{n_s} samples")
    print(f"has 95% upper bound {ucb:.4f}, so a chain that mixes in ~{np.ceil(1/ucb):.0f} "
          f"iterations is")
    print("indistinguishable from a reducible one.  A minorization estimated from n")
    print("samples cannot certify anything about events rarer than about 1/n.")

    # ---- figure --------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    ax_prof, ax_lvl, ax_sent, ax_v, ax_tr, ax_tau = axes.ravel()

    # (a) the actuators
    xs = np.arange(len(levels_s))
    xr = np.arange(len(levels_r))
    sd_s = np.array([p.std() for p in pools_s])
    sd_r = np.array([p.std() for p in pools_r])
    ax_prof.axhspan(args.tau_offset - args.tau, args.tau_offset + args.tau,
                    color="tab:green", alpha=0.13)
    ax_prof.axhline(0, color="k", lw=0.9)
    ax_prof.errorbar(xs, mean_s, yerr=sd_s, marker="o", lw=1.8, capsize=3,
                     label="signed ladder (works)")
    ax_prof.errorbar(xr, mean_r, yerr=sd_r, marker="s", lw=1.8, capsize=3,
                     color="tab:red", label="repeat ladder (fails)")
    ax_prof.set_xticks(np.arange(max(len(levels_s), len(levels_r))))
    ax_prof.set_xticklabels([str(levels_r[i]) if i < len(levels_r) else ""
                             for i in range(max(len(levels_s), len(levels_r)))])
    ax_prof.set_xlabel("rung (repeat-ladder labels; signed ladder is 0..6)")
    ax_prof.set_ylabel("sentiment  $s$")
    ax_prof.set_title("(a) Actuator authority: only one ladder\nreaches the reference")
    ax_prof.legend(fontsize=8, loc="lower left"); ax_prof.grid(alpha=0.3)

    # (b) the level, bounded vs unbounded
    for k0, c in zip(args.initial_levels, ["tab:red", "tab:orange"]):
        for j in range(min(12, args.runs)):
            ax_lvl.plot(paths[k0][j], lw=0.7, alpha=0.45, color=c)
        ax_lvl.plot(paths[k0].mean(axis=0), lw=2.4, color=c, label=f"repeat, $k_0$={k0}")
    t = np.arange(args.iterations)
    ax_lvl.plot(t, args.initial_levels[0] + cert["drift_tail"] * t, "k--", lw=1.4,
                label=r"drift $k_0+(p-q)t$")
    ax_lvl.axhspan(0, len(levels_s) - 1, color="tab:blue", alpha=0.18)
    ax_lvl.text(args.iterations * 0.02, len(levels_s) + 1,
                "signed ladder lives in here", fontsize=8, color="tab:blue")
    ax_lvl.set_xlabel("iteration"); ax_lvl.set_ylabel("instruction level $k_t$")
    ax_lvl.set_title("(b) The state does not return: $k_t\\to\\infty$")
    ax_lvl.legend(fontsize=8, loc="upper left"); ax_lvl.grid(alpha=0.3)

    # (c) sentiment never enters the dead zone
    ax_sent.axhspan(args.tau_offset - args.tau, args.tau_offset + args.tau,
                     color="tab:green", alpha=0.13)
    ax_sent.axhline(0, color="k", lw=0.9)
    k0 = args.initial_levels[0]
    ax_sent.plot(sents[k0][0], lw=1.0, color="tab:red", alpha=0.85, label="one run")
    ax_sent.plot(sents[k0].mean(axis=0), lw=2.2, color="darkred",
                 label=f"mean over {args.runs} runs")
    ax_sent.set_ylim(-1.05, 1.05)
    ax_sent.set_xlabel("iteration"); ax_sent.set_ylabel("sentiment  $s_t$")
    ax_sent.set_title(r"(c) The error never enters the dead zone $|s|\leq\tau$")
    ax_sent.legend(fontsize=8, loc="lower right"); ax_sent.grid(alpha=0.3)

    # (d) the invariant series, both loops, same formula
    Ps, us, ds = transition_law_from_pools(pools_s, args.tau, args.tau_offset)
    rec = markov.certify(Ps)["recurrent_class"]
    v_s = transience.invariant_series(us[rec[0]:], ds[rec[0]:])
    v_r = cert["invariant_series_conservative"]
    ax_v.semilogy(np.array(rec), v_s / v_s[0], "o-", lw=1.8,
                  label=f"signed ladder, $\\sum v_k$={v_s.sum() / v_s[0]:.2f} (finite)")
    ax_v.semilogy(levels_r[:len(v_r)], np.where(np.isfinite(v_r), v_r, np.nan), "s-",
                  lw=1.8, color="tab:red",
                  label=r"repeat ladder (95% envelope), $\sum v_k=\infty$")
    ax_v.set_xlabel("instruction level $k$")
    ax_v.set_ylabel(r"unnormalised invariant measure $v_k$")
    ax_v.set_title("(d) Same cut-balance formula, opposite verdict\n"
                   r"$v_k=\prod_{j<k} p_j/q_{j+1}$")
    ax_v.legend(fontsize=8); ax_v.grid(alpha=0.3, which="both")

    # (e) truncation slides the measure to the cap
    for N in args.truncations:
        ax_tr.plot(np.arange(N + 1), trunc[N], lw=1.6, label=f"cap $N$={N}")
    ax_tr.set_yscale("log")
    # Clip the floor: each cap's measure only becomes visible near its own cap,
    # which is the point -- the mass below any fixed rung is beyond negligible.
    ax_tr.set_ylim(1e-30, 2.0)
    ax_tr.set_xlabel("instruction level $k$"); ax_tr.set_ylabel(r"$\pi^{(N)}(k)$")
    ax_tr.set_title("(e) Truncating the ladder invents a measure\nthat lives at the cap")
    ax_tr.legend(fontsize=8); ax_tr.grid(alpha=0.3)

    # (f) the second failure mode
    taus = [s[0] for s in sweep]
    closed = [s[1] for s in sweep]
    mix = [s[2] for s in sweep]
    finite = [(t, m) for t, m in zip(taus, mix) if np.isfinite(m)]
    ax_tau.plot([t for t, _ in finite], [m for _, m in finite], "o-", lw=1.8,
                color="tab:purple", label="TV<0.01 bound")
    broken = [t for t, c in zip(taus, closed) if c > 1]
    if broken:
        ax_tau.axvspan(min(broken), 1.0, color="tab:red", alpha=0.10)
        ax_tau.text(min(broken), max(m for _, m in finite), "  no unique\n  invariant measure",
                    fontsize=8, color="tab:red", va="top")
    ax_tau.set_yscale("log")
    ax_tau.set_xlabel(r"dead zone $\tau$ (signed ladder)")
    ax_tau.set_ylabel("iterations to TV < 0.01")
    ax_tau.grid(alpha=0.3)
    ax2 = ax_tau.twinx()
    ax2.step(taus, closed, where="mid", color="tab:brown", lw=1.6,
             label="closed classes")
    ax2.set_ylabel("number of closed classes")
    ax_tau.set_title("(f) The bounded failure: widen the dead zone\n"
                     "and uniqueness dies before mixing does")
    h1, l1 = ax_tau.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax_tau.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")

    fig.tight_layout(); fig.savefig(args.fig, dpi=140)
    print(f"\nwrote {args.fig}")


if __name__ == "__main__":
    main()
