"""Phase A of the LLM ergodicity PoC: precompute the conditional sentiment law.

Because the closed loop is Markov in the instruction level ``k`` and responses are
i.i.d. given ``k``, the *only* thing the loop dynamics depend on is the
conditional law of the sentiment observation given ``k``.  Sampling that law once
per level and reusing it makes every subsequent trajectory essentially free, so
the total LLM cost is ``levels x samples_per_level`` regardless of how long the
trajectories are or how many restarts the analysis needs.

Two ladders are supported:

``--ladder signed`` (default)
    The seven-rung enthusiastic..neutral..critical ladder of the main
    experiment.  Bounded, straddles the reference, uniquely ergodic.

``--ladder repeat``
    The naive one-sided ladder: append one more verbatim copy of "Be more
    neutral." per rung, unbounded above.  Used by the counterexample run, which
    needs a few far-out rungs (20, 50) to check that the conditional law has
    saturated -- so the probed levels need not be contiguous.

Usage::

    python examples/llm_poc/build_pools.py --samples-per-level 200 \
        --out examples/llm_poc/pools.npz

    python examples/llm_poc/build_pools.py --ladder repeat \
        --samples-per-level 200 --out examples/llm_poc/pools_repeat.npz

Output format (npz): ``levels`` is the array of ladder indices actually probed
and ``level_i`` is the score pool for ``levels[i]``.  For the signed ladder the
two coincide; for a sparse probe they do not.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())

import numpy as np
import torch

from humancompatible.interconnect.simulators.logic.agents.llm_agent import (
    BASE_PROMPT, LADDERS, NUM_LEVELS, LLMResponder, SentimentScorer,
)

#: Default probe points for the unbounded ladder: a contiguous run to see the
#: near field, plus two far rungs to test whether the law has stopped moving.
REPEAT_LEVELS = list(range(10)) + [20, 50]


def default_levels(ladder):
    return list(range(NUM_LEVELS)) if ladder == "signed" else list(REPEAT_LEVELS)


def build(args):
    torch.set_num_threads(args.threads)
    render = LADDERS[args.ladder]
    levels = args.levels if args.levels else default_levels(args.ladder)

    responder = LLMResponder(model_name=args.model, device=args.device,
                             temperature=args.temperature, top_p=args.top_p,
                             max_new_tokens=args.max_new_tokens, seed=args.seed,
                             render=render)
    scorer = SentimentScorer(device=args.device)

    pools, examples, wall = {}, {}, {}
    targets = args.only_levels if args.only_levels else levels
    if args.only_levels:
        # Merge into an existing file so a single reworded rung can be redone
        # without paying for the whole ladder again.
        prev = np.load(args.out)
        levels = [int(v) for v in prev["levels"]] if "levels" in prev.files \
            else list(range(int(prev["num_levels"])))
        for i in range(len(levels)):
            pools[f"level_{i}"] = prev[f"level_{i}"]
        examples = json.loads(str(prev["examples"]))
        print(f"merging into existing {args.out}; regenerating levels {targets}")
    unknown = [t for t in targets if t not in levels]
    if unknown:
        raise SystemExit(f"--only-levels {unknown} are not in the probed set {levels}")

    for level in targets:
        i = levels.index(level)
        t0 = time.time()
        texts = []
        while len(texts) < args.samples_per_level:
            n = min(args.batch_size, args.samples_per_level - len(texts))
            texts.extend(responder.sample(level, n, args.base_prompt))
        scores = scorer.score(texts)
        wall[level] = time.time() - t0
        pools[f"level_{i}"] = scores
        examples[str(level)] = texts[:3]
        print(f"level {level:3d}  n={len(scores):4d}  mean={scores.mean():+.4f}  "
              f"sd={scores.std():.4f}  median={np.median(scores):+.4f}  "
              f"[{wall[level]:.1f}s, {wall[level]/len(scores):.2f}s/sample]", flush=True)
        print(f"          prompt: {render(level, args.base_prompt)[:160]!r}", flush=True)

    meta = dict(model=args.model, temperature=args.temperature, top_p=args.top_p,
                max_new_tokens=args.max_new_tokens, base_prompt=args.base_prompt,
                samples_per_level=args.samples_per_level, seed=args.seed,
                ladder=args.ladder, device=str(responder.device),
                total_seconds=sum(wall.values()))
    np.savez(args.out, num_levels=len(levels), levels=np.array(levels, dtype=int),
             ladder=args.ladder, meta=json.dumps(meta),
             examples=json.dumps(examples), **pools)
    print(f"\nwrote {args.out}  ({sum(wall.values()):.1f}s total, "
          f"{len(targets) * args.samples_per_level} generations)")

    means = np.array([pools[f"level_{i}"].mean() for i in range(len(levels))])
    monotone = bool(np.all(np.diff(means) <= 1e-9))
    print(f"probed levels:   {levels}")
    print(f"per-level means: {np.round(means, 4).tolist()}")
    print(f"monotone non-increasing: {monotone}   "
          f"control authority (level {levels[0]} - level {levels[-1]}): "
          f"{means[0] - means[-1]:+.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--ladder", choices=sorted(LADDERS), default="signed")
    ap.add_argument("--levels", type=int, nargs="+", default=None,
                    help="ladder rungs to probe (default: 0..6 signed, 0..9,20,50 repeat)")
    ap.add_argument("--samples-per-level", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=28)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--device", default=None)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base-prompt", default=BASE_PROMPT)
    ap.add_argument("--only-levels", type=int, nargs="+", default=None,
                    help="regenerate only these rungs, merging into --out")
    ap.add_argument("--out", default="examples/llm_poc/pools.npz")
    build(ap.parse_args())


if __name__ == "__main__":
    main()
