# Ergodicity analysis of a feedback loop containing an LLM

A proof of concept for the toolkit's central question, applied to a loop whose
"plant" is a language model. The controller reads the sentiment of the model's
response and adjusts the prompt; the reference signal is 0 ("neutral").

The question asked is **not** whether the loop converges. It is whether the loop
has a **unique invariant measure** — whether the long-run distribution of
sentiment exists and is independent of where the loop started. That is the
prerequisite for the repeated-interaction fairness notions this toolkit targets:
if two invariant measures exist, "the typical outcome of interacting with this
system" is not well defined, and no fairness statement about it can be.

The demonstration is that a loop containing an LLM can be certified as uniquely
ergodic, with an explicit convergence rate, using only the ability to *sample* the
loop — **and that the same machinery certifies the failures**, which is what makes
it an analysis rather than a hope. Three runs, one positive and two negative:

| run | what changes | verdict |
| --- | --- | --- |
| `run_demo.py` (§9) | — | unique invariant measure, TV < 0.01 within ~14 iterations |
| `run_divergence_demo.py` (§10) | the actuator is unbounded: append one more copy of the instruction | **no invariant probability measure at all** — the state escapes to infinity |
| `run_amplifier_demo.py` (§11) | the controller's gain changes sign | **the invariant measure is not unique** — the limit depends on the initial condition |

The two counterexamples share the plant, the observable, the threshold and the
pools of the positive run. Only the actuator and the sign of the gain change, so
the difference in verdict is attributable to the control design and not to the
language model.

---

## 1. Experimental setup

### The loop

```text
r = 0 ──▶ A1 ──▶ C ──▶ P1 ──▶ A2 ──▶ Z ──▶ F ──┐
          ▲     level k   LLM                   │
          └───────────────────────────────────────┘
       (checkpoint)
```

| node | type | role |
| --- | --- | --- |
| `r` | `ReferenceSignal` | constant 0 — "perfectly neutral" |
| `A1` | `Aggregator` | error node, `e = r + F = −s/K`; the checkpoint |
| `C` | `Controller` | `IfElseLadderLogic` — holds the instruction level `k` |
| `P1` | `Population` | the LLM ensemble; emits mean sentiment `s` |
| `A2` | `Aggregator` | collects the population output |
| `Z` | `Delay` | exponential smoothing, `alpha = 1` (pass-through) |
| `F` | `Filter` | sign inversion and loop gain, `F = −y/K` |

Topology is deliberately identical to the shipped
`examples/example_systems/example_nontrivial_system.ExampleSim`; only the
population logic and the controller differ.

### The agent

A small instruction-tuned LLM answering a fixed base prompt, with the controller's
instruction appended:

- **model**: `Qwen/Qwen2.5-0.5B-Instruct`, sampled at `temperature = 1.0`,
  `top_p = 0.95`, `max_new_tokens = 32`
- **base prompt**: `"Write one short sentence about the public transport system."`
- **instruction ladder** (`NEUTRALITY_LADDER`, 7 rungs, signed):

  | `k` | instruction |
  | --- | --- |
  | 0 | Be extremely enthusiastic and positive. |
  | 1 | Be positive. |
  | 2 | Be mildly positive. |
  | 3 | Be strictly neutral, impersonal and factual. Use no evaluative or emotional language. |
  | 4 | Be mildly critical. |
  | 5 | Be critical and disapproving. |
  | 6 | Be extremely critical and negative. |

Rung 5 says "critical **and disapproving**" for a measured reason — see §12.

Temperature must stay strictly positive: **the LLM's sampling is the noise source
that drives the loop.** At temperature 0 the loop is deterministic and its
invariant measure collapses to a point mass.

`number_of_agents` is the number of LLM samples averaged per step, and every run
in this document uses the default of 1: one response per iteration, so the
sampling noise reaches the controller undiminished.

### The observable

`cardiffnlp/twitter-roberta-base-sentiment-latest` — RoBERTa-base, 12 layers,
3-class `{0: negative, 1: neutral, 2: positive}`.

The observation is the **probability difference** `s = P(positive) − P(negative)`
∈ [−1, 1], not the argmax label: estimating a density needs a continuous
observable. Label indices are read from `config.id2label` at load time rather than
hardcoded — the natural guess is wrong here, since index 1 is *neutral*.

A `readout="logit"` option returns the unbounded log-odds difference instead,
which does not saturate near the ends of the scale.

Two caveats on the observable:

- `s = 0` is reached both by a confident "neutral" and by a balanced
  positive/negative split. These are different states of the world mapped to the
  same reference point. It does not affect the certificates, which depend only on
  the level chain, but "sentiment 0" in the figures is not purely "neutral text".
- **The classifier saturates.** On real generations it is usually confident, so
  `P(pos) − P(neg)` piles up near ±1 and the observable is closer to binary than
  to a spread-out score. Within-rung distributions are themselves bimodal — rung 4
  has mean −0.199 but sd 0.592 with `P(s>0) = 0.365`. This is a substantial part of
  why the estimated invariant measure is bimodal (§9), so the bimodality should not
  be read as purely a property of the control dynamics. The `readout="logit"`
  option exists for this reason.

### The controller

The obvious if-else rule, with one threshold:

```text
s >  tau  →  k + 1   (too positive → one rung more critical)
s < −tau  →  k − 1   (too negative → one rung more positive)
otherwise →  hold
```

`tau` is chosen automatically from the pools (§6). The controller recovers `s`
from the error signal as `s = −e·K`, since the reference is 0 and the filter feeds
back `−y/K`.

The dead zone is load-bearing, not cosmetic: **the hold branch is what makes the
chain aperiodic.** Without it the level is a pure ±1 walk on a path graph, which
has period 2 — still a unique invariant measure, but no convergence in total
variation and no Doeblin minorization. Because the signed rule has a natural dead
zone, one threshold suffices; a magnitude-based rule ("if not neutral enough, be
more neutral") needs two, one to tighten and one to relax.

---

## 2. Why this analysis route, and not the toolkit's contraction estimator

`tests/contractionTests/contraction_test.py` estimates a contraction factor as the
L1 norm of the autograd Jacobian between two nodes. That is unusable here, because
the loop is non-differentiable in four places:

1. **categorical token sampling** in the LLM — no gradient through a discrete draw
2. **the tokenizer** re-encoding generated text for the classifier — even a smooth
   classifier has a discrete input
3. **the if-else controller** — piecewise constant, zero derivative a.e.
4. **level quantization**, `int(round(·))`

Any one of these is fatal. Verified: every node's history carries
`requires_grad=False, grad_fn=None`, and `get_jacobian_norm` raises
`RuntimeError: element 0 of tensors does not require grad`.

The loud failure is deliberate. The dangerous alternative is a **silent zero**:
`torch.round`, `torch.floor` and `torch.sign` all keep the autograd graph alive and
return a gradient of exactly `0.0`. A controller written as
`level = torch.round(gain * e)` would hand `get_factor_from_list` a contraction
factor of 0 — comfortably below 1, reported as a clean certificate of unique
ergodicity, with no error and no warning. This PoC therefore detaches in the
aggregators and emits fresh non-grad tensors, so that pointing the autograd
estimator at it crashes rather than lies.

What replaces the derivative is the transition kernel itself, in
`humancompatible/interconnect/guarantees/markov.py`: irreducibility gives
uniqueness of the invariant measure, and a Doeblin minorization gives uniform
ergodicity with an explicit rate. Both need only samples.

---

## 3. The theory: unique ergodicity and the Doeblin condition

A short, self-contained account of what is actually being certified. Everything
here is standard finite-state Markov chain theory; the point is which pieces the
PoC leans on and why.

### 3.1 The chain

The loop state is the instruction level `k_t ∈ S = {0, …, K}`. Because the prompt
is rebuilt from `k` alone (§4) and responses are i.i.d. given `k`, the closed loop
is a time-homogeneous Markov chain with transition matrix

    P(i, j) = Pr( k_{t+1} = j | k_t = i )

`P` is not known in closed form — it is a functional of the LLM's response
distribution — but it can be *sampled*, and that is all the theory below needs.

### 3.2 Invariant measures, and when there is only one

A probability measure `π` on `S` is **invariant** if `πP = π`, i.e.
`Σ_i π_i P(i,j) = π_j`. It is the candidate "long-run distribution of the loop".

On a finite state space at least one invariant measure always exists
(Perron–Frobenius), so **existence is free and uniqueness is the whole question.**

The structure theorem is the useful part: the set of invariant probability measures
is a simplex whose extreme points correspond one-to-one with the **closed
communicating classes** of the chain. Hence

> the invariant measure is unique  ⟺  the chain has exactly one closed class.

Irreducibility (one class covering all of `S`) is sufficient but *not* necessary. A
single closed class with transient states outside it is equally fine — and is
exactly what this PoC has: `{3,4,5,6}` closed, `{0,1,2}` transient. This is why
`certify` reports closed classes rather than merely testing irreducibility.

If instead there were two closed classes, each would carry its own invariant
measure, the long-run behaviour would depend on where the loop started, and "the
typical outcome of interacting with this system" would not be well defined — so no
fairness statement about it could be either. That is the failure mode this whole
exercise is designed to detect.

### 3.3 The metric: total variation

For probability measures `μ, ν` on a finite `S`,

    ‖μ − ν‖_TV  =  ½ Σ_j |μ_j − ν_j|  =  sup_{A ⊆ S} |μ(A) − ν(A)|

the largest disagreement about the probability of any event. This is the metric in
which everything below contracts. Implemented as `markov.tv_distance`.

### 3.4 The Doeblin condition

The chain satisfies a **Doeblin condition** (or *minorization*) if there exist an
integer `m ≥ 1`, a constant `ε ∈ (0, 1]`, and a probability measure `ν` on `S` with

    P^m(x, ·)  ≥  ε · ν(·)      for **every** state x.

In words: from *any* starting state, after `m` steps the chain has probability at
least `ε` of being distributed according to one **common** measure `ν`, shared by
all starting points. `ν` is the *Doeblin measure*; `ε` is how much mass every
starting point is forced to hold in common.

**Doeblin's theorem.** Under that condition the chain has a unique invariant `π`,
and for every starting state `x`

    ‖P^t(x, ·) − π‖_TV  ≤  (1 − ε)^{⌊t/m⌋}

Convergence is geometric, the bound is *uniform* in the starting state (hence
"uniform ergodicity"), and the rate is fully explicit — which is what makes it a
usable certificate rather than an asymptotic statement.

### 3.5 Why it is a contraction

The proof is a coupling argument, and it explains the connection to the rest of the
toolkit. The condition lets you **split** the kernel:

    P^m  =  ε · ν  +  (1 − ε) · Q

for some transition kernel `Q`. So each block of `m` steps is a coin flip: with
probability `ε` the chain *regenerates* — it redraws its state from `ν`, forgetting
its entire past — and with probability `1 − ε` it moves according to `Q`.

Run two copies of the chain from different initial states, driven by the same
randomness. At every block they regenerate together with probability `≥ ε`, after
which their laws are identical forever. After `n` blocks they have failed to couple
with probability at most `(1 − ε)^n`, and TV distance is bounded by the probability
of not having coupled. Equivalently: **the Markov operator contracts total
variation by a factor `(1 − ε)` every `m` steps.**

This is the bridge. The papers behind this toolkit certify unique ergodicity by
showing an iterated function system is *contracting on average* — a Lipschitz
condition on the state space, which is what `contraction_test.py` estimates with
autograd Jacobians. Doeblin reaches the same conclusion through a contraction on
the **space of probability measures** under TV, rather than on the state space
under a norm. When the loop has no derivatives to take (§2), the measure-space
route is still open and the state-space route is not.

### 3.6 Three bounds, in increasing sharpness

For a fixed `m` there is a hierarchy of admissible `ε`:

| | choice of `ν` | `ε` | in code |
| --- | --- | --- | --- |
| atom | `ν = δ_j` | `max_j min_i P^m(i,j)` | `doeblin_epsilon(form="atom")` |
| general | `ν ∝` column minima | `Σ_j min_i P^m(i,j)` | `doeblin_epsilon` (default) |
| Dobrushin | — | `1 − δ(P^m)` | `dobrushin_coefficient` |

The **atom** form is the easiest to state (one state that every start can reach in
`m` steps) but is strictly the weakest. The **general** form is the largest
admissible `ε` for that `m`. **Dobrushin's ergodicity coefficient**

    δ(P) = max_{i,i'} ‖P(i,·) − P(i',·)‖_TV,        ‖μP − μ'P‖_TV ≤ δ(P)·‖μ − μ'‖_TV

is sharpest of all, because it compares rows *pairwise* instead of against a global
minimum. Measured on this PoC's chain:

| `m` | atom | general | `1 − δ(P^m)` |
| --- | --- | --- | --- |
| 1 | 0.000 | 0.000 | 0.000 |
| 2 | 0.324 | 0.430 | 0.573 |
| 3 | 0.350 | 0.687 | 0.741 |
| 4 | 0.460 | 0.839 | 0.873 |
| 5 | 0.469 | 0.912 | 0.925 |

(measured on the recurrent class at the operating point `tau = 0.45`), giving
mixing bounds of ~24, ~12 and ~9 iterations respectively for `TV < 0.01`. The
choice of formulation is worth a factor of nearly three in the reported rate.

Note the trade-off in `m`: larger `m` always raises `ε` — for an aperiodic
irreducible chain it tends to 1 — but the rate applies *per `m` steps*, so the
largest `ε` is usually not the best bound. Both `doeblin_epsilon` and
`dobrushin_mixing` search over `m` and select by the implied bound, not by `ε`. On
this chain the atom form illustrates the difference starkly: selecting by `ε` picks
`m = 12, ε = 0.516` for a bound of **84** iterations, while selecting by the bound
picks `m = 2, ε = 0.324` for **24**.

### 3.7 Aperiodicity is necessary, which is why the hold band exists

If the chain has period `d > 1`, the state space splits into `d` cyclic classes and
`P^m(x, ·)` is supported on whichever class is `m` steps ahead of `x`'s. For any
target `j`, only starting states in one particular cyclic class can reach `j` in
exactly `m` steps, so `min_i P^m(i,j) = 0` — for every `j` and every `m`. **A
periodic chain admits no Doeblin minorization at all.**

Uniqueness of `π` survives periodicity; the *rate* does not, and neither does
convergence in total variation. A pure ±1 walk over the instruction ladder has
period 2, so the controller's hold branch (§1) is not a stylistic choice — it is
what makes the certificate exist.

### 3.8 What `certify` actually checks

| step | function | establishes |
| --- | --- | --- |
| communicating / closed classes | `closed_classes` | **uniqueness** of `π` (§3.2) |
| period | `period` | aperiodicity, i.e. a minorization can exist (§3.7) |
| minorization search over `m` | `doeblin_epsilon` | `ε`, and the geometric rate |
| pairwise row overlap | `dobrushin_mixing` | the sharpest rate |
| left eigenvector for eigenvalue 1 | `stationary_distribution` | `π` itself |
| detailed balance | `birth_death_stationary` | independent check on `π` |

The last row is a genuine cross-validation rather than a formality. Because the
controller moves at most one rung per step, the chain is birth–death (with holds),
hence reversible, so `π_k P(k,k+1) = π_{k+1} P(k+1,k)` and

    π_k  ∝  ∏_{j<k}  P(j, j+1) / P(j+1, j)

This closed form is derived by a completely different route from the eigenvector,
and the two agree to 2.6e-16 here. It also fails loudly — returning `None` — if `P`
ever contains a jump of two or more rungs, which would mean the controller is not
behaving as specified.

Finally, the occupation measures converging to `π` from different initial levels
(§9) is the **ergodic theorem** in action: for a uniquely ergodic chain the time
average along almost every trajectory converges to the space average `π`,
regardless of where the trajectory started. That is what makes a single long run of
the system informative about the system rather than about its initial condition.

### 3.9 The other side: when there is no invariant measure at all

Everything above is about *uniqueness*. Existence is the easier half, and it is
worth being precise about when it can fail, because the answer depends entirely on
whether the state space is bounded.

- **Finite state space.** `P` maps the probability simplex into itself and is
  continuous, so Brouwer — or Perron–Frobenius applied to `Pᵀ` — gives a fixed
  point. An invariant measure *always exists*. On the seven-rung ladder the only
  available failure is therefore **non-uniqueness**: two or more closed
  communicating classes, one invariant measure on each, and a limit decided by the
  initial condition. `certify` reports it as a closed-class count above one.
- **Unbounded state space.** Existence can fail outright. The chain may be
  *transient* (the mass escapes to infinity) or *null recurrent* (it returns, but
  with infinite expected return time). In both cases there is no invariant
  *probability* measure at all.

The naive instruction ladder of §10 is the second case, and for a nearest-neighbour
chain on `{0, 1, 2, …}` the whole question is settled by one identity. Sum the
invariance equations `π = πP` over `{0, …, k}`: every flow internal to that set
appears once in each direction and cancels, leaving only the flow across the single
cut between `k` and `k+1`,

    π_k · p_k  =  π_{k+1} · q_{k+1}                     (cut balance)

with `p_k = P(k, k+1)` the escalate rate and `q_k = P(k, k−1)` the back-off rate.
This is the same relation `birth_death_stationary` uses on the finite ladder (§3.8);
the identity does not care whether the state space is finite. It pins the invariant
measure down to a single constant,

    v_0 = 1,      v_k = ∏_{j<k}  p_j / q_{j+1}

and reduces existence to one question — is `v` summable?

- `Σ_k v_k < ∞` — normalise it and you have the unique invariant probability
  measure. This is positive recurrence, and it is the case §9 certifies.
- `Σ_k v_k = ∞` — no scaling of `v` is a probability measure, and since `v` is the
  only solution of cut balance up to scale, **there is no invariant probability
  measure**. A σ-finite invariant measure still exists — `v` itself — but it has
  infinite total mass and cannot be normalised, so there is no long-run
  distribution to converge to and no ergodic theorem to invoke.

The sufficient condition the certificate actually checks is geometric: if
`p_k / q_{k+1} ≥ ρ > 1` for every `k ≥ K`, then `v_k ≥ v_K · ρ^(k−K)` and the sum
diverges. In control terms that is a persistent positive drift, `p_k − q_k > 0` at
every rung: the controller escalates more often than it backs off, forever, and
never finds a rung where the two balance.

The classical birth–death criteria say the same thing in two other currencies, and
`transience.py` computes both as cross-checks:

| series | converges | diverges |
| --- | --- | --- |
| `Σ_k ∏_{j≤k} q_j / p_j` | chain is **transient** | chain is recurrent |
| `Σ_k v_k = Σ_k ∏_{j<k} p_j / q_{j+1}` | **positive** recurrent | not positive recurrent |

together with the escape probability: for a homogeneous walk with `p > q`,
gambler's ruin gives a return probability of `q/p` per rung, so the state never
comes back down one rung with probability `1 − q/p`. Hold moves change how fast the
walk climbs, not whether it returns.

Two things follow for the machinery of §3.4–§3.6. First, Doeblin's condition is not
merely violated in this regime, it is *unavailable*: for `k′ ≫ k` the laws
`P^m(k, ·)` and `P^m(k′, ·)` are supported on disjoint sets of rungs, so
`δ(P^m) = 1` and `ε = 0` for every `m`, and there is nothing left to certify.
Second, the verdict has to be reached from the rates directly, which is what makes
it checkable at all: `p_k` and `q_k` are tail masses of the same per-rung sentiment
pools that Phase A already measures, so the counterexample costs exactly one more
Phase A run and no new machinery.

### 3.10 Non-uniqueness: the invariant set is a simplex

The other failure the theory permits — and the only one available on a finite
ladder — is that the invariant measure exists but is not unique. It is worth
spelling out what "not unique" means concretely, because it is not "the analysis
is inconclusive": the structure is completely determined.

Every closed communicating class `C` carries exactly one invariant measure `μ_C`,
supported on `C` and obtained by restricting `P` to it. If there are `m` closed
classes, the set of invariant measures is exactly the simplex

    { Σ_c a_c μ_c  :  a_c ≥ 0,  Σ_c a_c = 1 }

so it is an `(m−1)`-dimensional family, and `μ_1, …, μ_m` are its extreme points.
`invariant_measures(P)` returns them. Note that `stationary_distribution(P)` still
happily returns *something* in this case — whichever eigenvector for eigenvalue 1
the solver picks — which is exactly why `certify` checks the closed-class count
before reporting a `π`, and why the report labels it "one of many" when it is.

Which one a trajectory converges to is decided by where it starts. From a
transient state `i`, the chain is eventually absorbed into one of the closed
classes, and

    H[i, c] = P(absorbed into class c | started at i)

solves the harmonic system `(I − P_TT) h_T = P_TC 1` on the transient states.
`absorption_probabilities(P)` returns it. So the limit of `P^t(i, ·)` is
`Σ_c H[i,c] μ_c` — a *different* measure for different `i`. The ergodic theorem
fails in the way that matters most in practice: the time average along a single
long trajectory converges to `μ_c` for whichever class captured it, so a long run
tells you about that run and not about the system.

The diagnostic that reads this off an experiment needs no matrix at all. Run the
loop from two initial conditions, take the occupation measures, and compute their
total variation distance. Under unique ergodicity it goes to zero — §9 measures
0.0099. Under non-uniqueness with disjoint attractors it stays at 1, which is
what §11 measures.

**One honest caveat, which §11 makes quantitative.** Reducibility is a statement
that certain transition probabilities are *exactly* zero, and no finite sample can
establish that. If the rate of escape from an attractor is some small `ε > 0`
rather than zero, the chain is irreducible and uniquely ergodic — but its mixing
time is `Θ(1/ε)`. So "two invariant measures" and "one invariant measure reached
after 10^7 iterations" are not distinguishable from data, and for any horizon the
system will actually be run on they are not distinguishable in consequence either.
The right report is both numbers: the classification under the estimated rates,
and the mixing time as a function of the escape rate you are unwilling to rule out.

---

## 4. Two design constraints that make the loop analysable

**The instruction is replaced, never appended.** `render_prompt` rebuilds the
prompt from scratch each iteration as a function of `k` alone. If instructions
accumulated, the state would be the entire transcript — infinite dimensional and
non-Markov, with no invariant measure on any tractable space.

**The delay is pass-through (`alpha = 1`).** The controller then decides from the
current observation only, so the loop state is the level alone and the closed loop
is a finite Markov chain. With `alpha < 1` the state becomes the hybrid pair
`(k, y)`, and the finite-state certificates no longer apply — that is the
continuous-state extension, not this demo.

---

## 5. Cost model

The observation law depends only on `k`, and responses are i.i.d. given `k`. So the
conditional law is sampled **once per rung** and reused. Total LLM cost is
`num_levels × samples_per_level` — independent of trajectory length and of how many
restarts the analysis needs. Analysis runs then cost nothing, which is what makes
the threshold sweep and the many-restart figures affordable.

At the measured 0.6–0.7 s/sample on 16 CPU threads, the default 7 × 200 = 1400
generations take about 15 minutes.

The counterexamples cost almost nothing on top of that:

| run | LLM cost | wall clock |
| --- | --- | --- |
| §9, signed ladder | 7 × 200 = 1400 generations | ~15 min |
| §10, unbounded ladder | 12 × 200 = 2400 generations | 26 min |
| §11, amplifying controller | **none** — `pools.npz` reused | seconds |

§11 is free because the conditional law of the observation given the rung belongs
to the model and the prompt, not to the controller. Any change confined to the
control law — the sign of the gain, the width or centre of the dead zone, the
initial condition — is a change to how the same pools are *assembled* into a transition
law, so it costs a matrix build and no generation at all. That is the practical
payoff of putting the LLM behind a conditional-law interface: the expensive object
is measured once and every control design is then a few milliseconds.

---

## 6. Code

### Added to the package

| file | contents |
| --- | --- |
| `simulators/logic/agents/llm_agent.py` | `NEUTRALITY_LADDER`, `BASE_PROMPT`, `render_prompt`; the unbounded counterexample ladder `REPEAT_INSTRUCTION` / `render_repeat_prompt`, and the `LADDERS` registry that names them; `SentimentScorer` (`.score(texts, readout)`); `LLMResponder` (`.sample(level, n)`, `render=` selects the ladder); `LLMAgentLogic` (live) and `CachedLLMAgentLogic` (pool-backed, `.from_npz`, `levels=` for a sparsely probed ladder) — both satisfying the `Population` contract |
| `simulators/logic/controllers/instruction_ladder.py` | `IfElseLadderLogic` (`num_levels=None` removes the top rung, giving the unbounded ladder; `polarity=-1` flips the gain; `tau_offset` moves the dead zone off the reference); `rates_from_pools(pools, tau, tau_offset, …)` → `(up, down)`; `transition_law_from_pools(pools, tau, tau_offset, …)` → `(P, up, down)`, the analytic transition matrix as a functional of the pools |
| `guarantees/markov.py` | the finite-state certificates — fills a subpackage that shipped empty |
| `guarantees/transience.py` | the complementary certificates: cut balance, the invariant series and its divergence, the two birth–death series, escape probability, truncation sequences, and `certify_no_invariant_measure` |
| `examples/example_systems/example_llm_system.py` | `LLMSim(Simulation)`; exposes `.level_history` (the Markov path) and `.sentiment_history()`; `polarity` and `num_levels=None` reach the controller from here |
| `tests/test_guarantees.py` | 22 cross-checks on both guarantee modules and the controller, including the negative controls — no model, no pools, no pytest required |

`markov.py` public API: `estimate_transition_matrix`, `reachable_matrix`,
`communicating_classes`, `closed_classes`, `is_irreducible`, `period`,
`is_aperiodic`, `stationary_distribution`, `birth_death_stationary`,
`doeblin_epsilon`, `doeblin_curve`, `dobrushin_coefficient`, `dobrushin_mixing`,
`mixing_time_bound`, `wilson_lower_bound`, `tv_distance`, `occupation_measure`,
`certify`, `format_report`.

Three details in there are worth knowing:

- `doeblin_epsilon` defaults to the **general** minorization `eps = Σ_j min_i
  P^m(i,j)` rather than the single-atom special case, and selects `m` by
  **minimising the implied mixing bound** rather than by maximising `eps`. Both
  choices matter: on this chain the atom form selected by `eps` gives a bound of 84
  iterations, the atom form selected by bound gives 24, and the general form gives
  12. `dobrushin_mixing` gives 9. See §3.6 for the theory.
- `certify` restricts `period`, `doeblin_epsilon` and `birth_death_stationary` to
  the **single closed class** when one exists. A closed class that is a strict
  subset of the state space is still uniquely ergodic; measuring the period of a
  transient state is meaningless. It then adds the second leg back: a run started
  on a transient rung has to *reach* the class, so `certify` also reports
  `absorption_time_bound`, `expected_hitting_times` and a
  `tv_mixing_bound_total` that budgets half the target to each leg. Without it the
  amplifying loop of §11 reports a mixing time of 1 iteration when the true answer
  is 2141.
- When there is more than one closed class `certify` stops reporting a single `π`
  and returns `invariant_measures` (one extreme measure per class, the invariant
  set being their convex hull) and `absorption` (which class each starting state
  ends up in). `stationary_distribution` on its own will still cheerfully return
  an arbitrary eigenvector in that situation, which is the trap those two
  functions exist to close.
- `mixing_time_bound` and the `m`-selection inside `doeblin_epsilon` use
  `log1p(-eps)`, not `log(1 - eps)`. For the tiny `eps` a near-reducible chain
  produces (1e-18 and below — see §11.5), `1.0 - eps` rounds to exactly `1.0`,
  `log` returns `0`, and the score becomes `-inf`, so the search silently selects
  the most degenerate minorization on offer and reports a *negative* mixing time.
  The counterexample runs are what surfaced this.
- `birth_death_stationary` returns `None` if `P` has any jump of size ≥ 2. This
  doubles as a check that the controller really only ever moves one rung at a time,
  and its agreement with the eigenvector solution (to ~1e-16) cross-validates the
  whole estimation path.

`markov.py` also gained, for the counterexamples: `invariant_measures` (one
extreme measure per closed class), `absorption_probabilities` (which class each
start ends up in), `expected_hitting_times` and `absorption_time_bound` (the second
leg of the mixing bound).

`transience.py` public API: `wilson_interval`, `cut_balance_ratios`,
`invariant_series`, `recurrence_series`, `truncated_rates`, `truncation_sequence`,
`escape_probability`, `saturation_report`, `certify_no_invariant_measure`,
`format_report`.

It is deliberately a separate module from `markov.py`: everything in `markov.py`
takes a transition *matrix*, and the unbounded ladder has none. `transience.py`
works from the rate sequences `p_k`, `q_k` alone, which is all that cut balance
(§3.9) needs. Two conventions in it are worth knowing:

- The verdict is computed from **Wilson score bounds in the conservative
  direction** — the lower bound on every escalate rate against the upper bound on
  every back-off rate — so a rate that happened to be observed as zero cannot by
  itself carry the conclusion. `invariant_series_conservative` is the matching
  lower envelope for `v_k`, and it is what the figure plots.
- `saturation_report` exists to test the one extrapolation the certificate makes:
  that the conditional law has stopped moving above the highest rung sampled. It
  reports `P(s > τ)` and `P(s < −τ)` with intervals, plus a two-sample KS distance
  to the top rung, per rung.

### Scripts

`build_pools.py` — **Phase A, the only LLM work.** Generates
`samples_per_level` responses per rung, scores them, and saves pools plus metadata
to `.npz`. Reports per-rung mean/sd/median, timing, and whether the ladder gives
monotone control authority.

Flags: `--model --ladder --levels --samples-per-level --batch-size
--max-new-tokens --temperature --top-p --device --threads --seed --base-prompt
--only-levels --out`

`--ladder signed` (default) builds the seven rungs of the main experiment.
`--ladder repeat` builds the counterexample's unbounded ladder, and defaults to
probing rungs `0…9, 20, 50` — sparse on purpose: the far rungs are there to test
whether the conditional law has saturated, which is the assumption the
non-existence certificate extrapolates on. The `.npz` records the probed rungs in
`levels`, so `level_i` is the pool for `levels[i]` and the two coincide only for a
contiguous ladder.

`--only-levels 5` regenerates just those rungs and merges them into the existing
`--out` file. Rewording one rung and remeasuring it costs ~2 minutes instead of
re-paying for the whole ladder; §12 explains why that workflow came up.

`run_demo.py` — **Phase B, no LLM work.** Sweeps `tau`; simulates the loop from
several initial levels; estimates `P̂` and cross-checks it against the analytic `P`;
runs the certificates; writes the figure.

Flags: `--pools --iterations --runs --agents --burn-in --tau --alpha --loop-gain
--initial-levels --bandwidth --seed --fig`

`tau` selection minimises the Doeblin mixing bound subject to the invariant measure
putting mass ≥ 0.02 on at least two rungs. Selecting for spread alone is a trap: it
drives `tau` to 0, which removes the hold band and leaves a period-2 chain with no
minorization at all. The sweep is not a formality — on synthetic pools the mixing
bound ranged from 123 iterations at `tau = 0.05` to 16 at `tau = 0.45`, all with the
same certified uniqueness. Selection is on the bound that applies to a run started
*anywhere*, including the time to reach the recurrent class — not the within-class
bound; using the latter picks `tau = 0.40` and a genuinely worse loop.

`run_divergence_demo.py` — **the counterexample, also no LLM work.** Consumes
`pools_repeat.npz` (and `pools.npz`, for the comparison) and prints six sections:
the two actuators side by side, the saturation test, the closed-loop rates, the
non-existence certificate, a closed-loop simulation through the real interconnect
graph with the cap removed, what truncating the ladder does to the certificate, and
the second failure mode on the bounded ladder. Writes `no_invariant_measure.png`.

Flags: `--pools-repeat --pools --tau --tau-offset --tail-from --iterations
--runs --initial-levels --truncations --seed --fig`

`--tau` defaults to `0.45`, the value the main experiment selected, so that the
only thing changing between the two runs is the actuator.

`run_amplifier_demo.py` — **the second counterexample, no LLM work at all.**
Reuses `pools.npz`. Prints the certificate for a dead zone centred on the
reference, scans the dead-zone centre `tau_offset` to find where uniqueness fails,
certifies non-uniqueness with extreme measures and absorption probabilities at
that offset, runs the corrective sign as a control, and closes with the
reducible-versus-metastable sweep. Writes `non_unique_measure.png`.

Flags: `--pools --tau --tau-offset --max-offset --iterations --runs --burn-in
--escape-rates --reference-tv --bandwidth --seed --fig`

`--tau-offset` defaults to the smallest offset in the scan at which the chain
acquires a second closed class, found by the scan rather than hard-coded, and the
script exits with a clear message if uniqueness survives the whole scan.

---

## 7. Running it

```bash
# Phase A — ~15 min for the default 1400 generations
python examples/llm_poc/build_pools.py --samples-per-level 200 \
    --batch-size 40 --threads 16 --device cpu --out examples/llm_poc/pools.npz

# Phase B — seconds.  This is the run of §9.
python examples/llm_poc/run_demo.py --pools examples/llm_poc/pools.npz \
    --runs 800 --iterations 150
```

The counterexamples:

```bash
# §10 needs its own pools: 12 rungs of the unbounded ladder, ~26 min
python examples/llm_poc/build_pools.py --ladder repeat --device cpu \
    --samples-per-level 200 --out examples/llm_poc/pools_repeat.npz
python examples/llm_poc/run_divergence_demo.py

# §11 needs no new sampling at all — it reuses pools.npz
python examples/llm_poc/run_amplifier_demo.py
```

The certificates themselves have tests that need no model and no pools:

```bash
python -m pytest tests/test_guarantees.py -q      # or just: python tests/test_guarantees.py
```

## 8. What the run reports

- per-rung sentiment mean/sd and `P(s > 0)`, and whether control authority is monotone
- the `tau` sweep, with certificates and mixing bound for each candidate
- the analytic transition law, cross-checked against `P̂` from simulated runs
- communicating classes, the recurrent class and which states are transient,
  irreducibility, period, Doeblin `eps` with the power `m` and the state carrying
  most of `nu`'s mass, the implied mixing bound, the time to reach the recurrent
  class and the resulting bound from any start, and the sharper Dobrushin bound
- stationary `pi`, cross-checked against the birth–death closed form
- occupation measures from different initial levels, and TV distances between them —
  the empirical signature of a *unique* invariant measure
- a three-panel figure: sample paths with the hold band; the invariant sentiment
  measure by KDE, overlaid across initial conditions and sharing the sentiment axis
  with the paths; and `pi` versus empirical occupation over rungs

The two counterexample scripts report the corresponding things for their own
failure: for §10 the saturation table, the escalate/back-off rates, the divergent
invariant series with its conservative envelope, the two birth–death series, the
simulated escape, and the truncation table; for §11 the dead-zone scan, the extreme
invariant measures with absorption probabilities, the corrective-sign control run,
and the escape-rate sweep.

---

## 9. Results

Measured profile, `Qwen2.5-0.5B-Instruct`, 200 samples per rung (1400 generations,
721 s on 16 CPU threads):

| `k` | instruction | mean `s` | sd | `P(s>0)` |
| --- | --- | --- | --- | --- |
| 0 | extremely enthusiastic | +0.910 | 0.105 | 1.000 |
| 1 | positive | +0.808 | 0.128 | 1.000 |
| 2 | mildly positive | +0.876 | 0.079 | 1.000 |
| 3 | strictly neutral | +0.638 | 0.194 | 1.000 |
| 4 | mildly critical | −0.199 | 0.592 | 0.365 |
| 5 | critical and disapproving | −0.421 | 0.583 | 0.215 |
| 6 | extremely critical | −0.733 | 0.359 | 0.065 |

Control authority 1.64, monotone across the whole critical side. The positive rungs
0–2 are not ordered correctly among themselves, but they are saturated and turn out
to be transient, so it does not matter.

Certificate at the auto-selected `tau = 0.45`, from 238,400 observed transitions
(800 runs × 150 iterations from each of `k₀ = 0` and `k₀ = 6`):

```text
communicating classes : [[0], [1], [2], [3, 4, 5, 6]]
closed classes        : [[3, 4, 5, 6]]
recurrent class       : [3, 4, 5, 6]   (transient: [0, 1, 2])
period                : 1   (aperiodic: True)
invariant measure     : UNIQUE (1 closed class)
Doeblin minorization  : eps=0.9989 at m=12  (nu peaks at state 4)
  => uniformly ergodic, TV<0.01 within ~12 iterations once in the recurrent class
  reaching the recurrent class: <= 4 iterations to 0.005 (expected 3.0 from the worst)
  => TV<0.01 from ANY start within ~16 iterations
Dobrushin coefficient : delta(P^9)=0.0067  => ~9 iterations in the class,
                                              ~14 from any start (sharpest)
stationary pi         : [0, 0, 0, 0.2879, 0.5163, 0.1676, 0.0282]
  birth-death closed form: identical to 2.64e-16
TV(occupation k0=0, k0=6) = 0.0099
max |P_hat - P_analytic| = 0.0052
terminal sentiment | k0=0: mean=+0.0021 sd=0.6602
terminal sentiment | k0=6: mean=-0.0456 sd=0.6627
```

**The loop is certified uniquely ergodic and mixes within ~14 iterations from any
starting rung.** The trajectories confirm it visually: starting from the most
enthusiastic and the most critical rung, both enter the same regime within about
ten iterations.

Two notes on reading that block. The Doeblin and Dobrushin bounds are bounds on
mixing *inside* the recurrent class `{3,4,5,6}`; a run started on a transient rung
has to reach that class first, and the honest total splits the error budget
between the two legs (§3.8). Here the second leg is short — rung 0 leaves within a
few steps — but it is not always, and §11 has a case where it is 2141 iterations
against a within-class bound of 1. And `tau` is chosen by the sweep to minimise
that *total*, which is why it lands at 0.45 rather than at a larger value with a
better `eps`.

The substantive finding is in the terminal-sentiment lines together with the KDE
panel. The controller drives *mean* sentiment to the reference — +0.002 and −0.046
from opposite ends of the ladder — but the invariant measure is **bimodal**, with
modes near +0.75 and −0.80 and a **trough at 0**. The loop hits its neutrality
target on average while almost never actually emitting neutral text; it alternates
between enthusiastic and disapproving responses. Sentiment sd stays at ~0.66
indefinitely.

This is exactly why "does the loop converge to the reference?" is the wrong
question, and what the ergodic framing buys: a stability analysis reporting
"error → 0" would be true of the mean and badly misleading about what any
individual user of this system experiences. Note the saturation caveat in §1 — part
of the bimodality comes from the classifier, not the dynamics.

---

## 10. Counterexample 1: an unbounded actuator, and no invariant measure

`run_divergence_demo.py` · `pools_repeat.npz` · `no_invariant_measure.png`

### 10.1 The change

The PoC was originally specified with the obvious controller: *if the response is
too positive, append "be more neutral" to the prompt*. Taken literally that is a
single move rather than a ladder, and rung `k` is the base prompt followed by `k`
verbatim copies of the instruction:

    render_repeat_prompt(3) ->
      "Write one short sentence about the public transport system.
       Be more neutral. Be more neutral. Be more neutral."

Nothing bounds `k`. `IfElseLadderLogic(num_levels=None)` removes the top rung, so
the loop state lives on `{0, 1, 2, …}` instead of on seven rungs. Everything
else — model, topic, temperature, classifier, dead zone `tau = 0.45`, graph
topology, `alpha = 1` — is exactly as in §9.

### 10.2 The measurement: repeating an instruction does not compound it

12 rungs × 200 samples = 2400 generations, 1574 s. Rungs 0–9 contiguously, plus 20
and 50 (§10.3 explains why those two):

| rung | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 20 | 50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mean `s` | +0.676 | +0.690 | +0.665 | +0.704 | +0.669 | +0.665 | +0.667 | +0.656 | +0.627 | +0.657 | +0.662 | +0.630 |
| sd | 0.252 | 0.167 | 0.176 | 0.145 | 0.171 | 0.182 | 0.200 | 0.204 | 0.208 | 0.200 | 0.176 | 0.249 |

- Span across **50 rungs: 0.077.** The signed ladder of §9 spans **1.643** across
  six.
- The sign never changes. The reference `r = 0` is outside the actuator's reach at
  every rung, so the error never enters the dead zone and the controller never
  stops escalating.
- The profile is not monotone; the movement is smaller than the standard error
  (0.25/√200 ≈ 0.018) and is consistent with noise.

Fifty verbatim repetitions of "Be more neutral." move the sentiment by less than
the sampling noise of a single rung. That is a fact about instruction-tuned models
worth stating on its own: **an instruction's effect does not accumulate with
repetition**, so a controller whose only actuator is "say it again, louder" has no
gain to work with. It is the same failure the signed ladder was built to avoid
(§12), measured properly this time.

### 10.3 The extrapolation, and how it is tested

The certificate is a statement about a chain on an infinite state space, and only
twelve of its rungs were ever sampled. Exactly one assumption bridges the gap: that
the conditional law has **saturated** — that above the highest rung measured,
nothing further changes. Rungs 20 and 50 exist to test it, and `saturation_report`
reports the two functionals the certificate actually depends on, with intervals:

```text
 rung     mean     sd   p=P(s>tau)           95% CI  q=P(s<-tau)           95% CI  KS to top
    0   +0.676  0.252        0.880 [0.828,0.918]        0.015 [0.005,0.043]      0.150
    5*  +0.665  0.182        0.885 [0.833,0.922]        0.000 [0.000,0.019]      0.100
    9*  +0.657  0.200        0.835 [0.777,0.880]        0.000 [0.000,0.019]      0.090
   20*  +0.662  0.176        0.870 [0.816,0.910]        0.000 [0.000,0.019]      0.075
   50*  +0.630  0.249        0.825 [0.766,0.871]        0.005 [0.001,0.028]      0.000
```

The escalate rate sits in `[0.815, 0.940]` at every rung from 0 to 50 with no
trend, and the two-sample KS distance between rung 50 and any tail rung is at most
0.115 on 200 samples each. That is evidence, not proof, and §13 keeps it on the
limitations list — but it is the kind of evidence that costs 400 extra generations
to obtain, which is why the probe is sparse rather than contiguous.

### 10.4 The certificate

The rates at `tau = 0.45`, straight off the pools:

```text
p_k (escalate)  [0.88, 0.93, 0.895, 0.94, 0.9, 0.885, 0.895, 0.865, 0.815, 0.835, 0.87, 0.825]
q_k (back off)  [0.015, 0.0,  0.0,   0.0,  0.0, 0.0,   0.0,   0.0,   0.0,   0.0,   0.0,  0.005]
```

and the verdict, computed from cut balance (§3.9) with every rate replaced by the
conservative end of its Wilson interval — lowest credible climb against highest
credible fall:

```text
tail (rungs >= 5) : p=0.8557 (95% lcb 0.7554), q=0.0007 (95% ucb 0.0278)
  drift               : +0.8550  (conservative +0.7277)
  rho = p/q           : 1198.0  (conservative 27.2)   need > 1 for divergence
  climbs at every rung below the tail: True
invariant series v_k (from the confidence bounds, a lower envelope):
  [1.00e+00 4.39e+01 2.06e+03 9.25e+04 4.41e+06 1.99e+08 8.80e+09 3.95e+11 1.70e+13
   6.80e+14 2.81e+16 8.25e+17]
  sum v_k             : DIVERGES (v_k >= v_K rho^(k-K), rho>1)
recurrence series     : ratio 0.0368 < 1 => geometric, sum finite => TRANSIENT
P(never returns one rung down) >= 0.9632

VERDICT: NO INVARIANT PROBABILITY MEASURE EXISTS.
```

Both classical criteria agree, as they must: the recurrence series converges
geometrically (transient) and the invariant series diverges geometrically (not
positive recurrent). Note that the *same closed form* that produced the stationary
distribution of §9 — `v_k = ∏ p_j/q_{j+1}`, cross-validated there against the
eigenvector to 2.6e-16 — is what proves non-existence here. Panel (d) of the figure
plots both on one axis: one decays and sums to 3.0, the other grows by a factor of
27 per rung.

`run_divergence_demo.py` also takes `--tau-offset`, and the verdict does not move:
recentring the band on `+0.40` drops the escalate rate from 0.86 to ~0.12 at every
rung, but the back-off rate stays at ~0, so `ρ` remains far above 1 and the chain
is still transient. That is the right answer and worth stating — no choice of dead
zone rescues this loop, because the failure is an actuator with no gain, not a
threshold in the wrong place. A dead zone can only decide *when* to act on an
error; it cannot manufacture the authority to correct one.

### 10.5 The loop, simulated

200 runs × 200 iterations from `k₀ = 0` and `k₀ = 5`, through the real
`ControlSystem` graph with the cap removed:

| | `k₀ = 0` | `k₀ = 5` |
| --- | --- | --- |
| mean level after 200 iterations | 165.0 (range 151–178) | 169.8 (range 154–183) |
| empirical drift per iteration | +0.818 | +0.819 |
| predicted drift `p − q` | +0.855 | +0.855 |
| runs ever returning to rung ≤ 5 after `t = 20` | 0 / 200 | 0 / 200 |
| mean sentiment over the run | +0.637 | +0.637 |

and the fraction of time spent in rungs ≤ 5, over horizons 25 / 50 / 100 / 200:
`0.270 / 0.135 / 0.067 / 0.034` from `k₀ = 0`. It halves with every doubling of
the horizon, which is what "there is no occupation measure to converge to" looks
like from inside a finite run: every fixed window of the state space is vacated,
and the empirical distribution of `k` has no limit to be estimated.

The control reading is simpler still. The mean sentiment is +0.637 at iteration 1
and +0.637 at iteration 200; the loop never regulates, and the only thing it
accomplishes is an ever-longer prompt. This is the LLM analogue of integrator
windup — the controller keeps integrating error into an actuator that does not
respond.

### 10.6 Truncating the ladder manufactures a certificate

The obvious repair is to cap the ladder at some `N`. The state space is finite
again, so an invariant measure exists, and the chain is irreducible, so it is
unique. Every one of those certificates is real, and every one of them is about the
cap:

| cap `N` | `π(k ≤ 5)` | `π(top rung)` | mode | unique? | TV<0.01 within |
| --- | --- | --- | --- | --- | --- |
| 10 | 7.1e-09 | 0.976 | 10 | yes | 50 |
| 20 | 2.9e-23 | 0.963 | 20 | yes | 80 |
| 50 | 2.6e-66 | 0.963 | 50 | yes | 138 |
| 100 | 5.0e-138 | 0.963 | 100 | yes | 226 |

(computed with the *most recurrence-friendly* rates the data allows, so the
truncated chain really is irreducible and the certificate really is earned).

Two things give the game away. The measure is a property of the cap — lift the cap
and it moves with it, converging to nothing. And the mixing bound grows without
bound as the cap is lifted, which is the certificate itself reporting that the cap
is carrying the argument. **Truncating an unbounded actuator is an assumption about
the system, not a modelling convenience.** Here it would have to be earned by
exhibiting a rung at which the model stops accepting further instructions, and the
rung-50 measurement is evidence that there is no such rung within reach.

### 10.7 A preview of the other failure: widen the dead zone

The same script closes by sweeping `tau` on the *bounded* signed ladder, where
existence is guaranteed and only uniqueness can fail:

| `tau` | closed classes | unique? | Doeblin `ε` | TV<0.01 within |
| --- | --- | --- | --- | --- |
| 0.45 | 1 | yes | 0.9988 | 16 |
| 0.60 | 1 | yes | 0.9994 | 17 |
| 0.80 | 1 | yes | 0.9769 | 32 |
| 0.90 | 1 | yes | 0.2629 | 238 |
| 0.92 | 1 | yes | 0.0250 | 729 |
| 0.94 | 3 | **no** | 0 | ∞ |
| 0.99 | 7 | **no** | 0 | ∞ |
| 1.00 | 7 | **no** | 0 | ∞ |

At `tau ≥ 1` the fracture is *exact* rather than sampled: the classifier readout is
`p_pos − p_neg ∈ (−1, 1)`, so no observation can ever leave the dead zone, `P = I`,
and all seven states are absorbing — the invariant set is the whole 6-simplex and
the loop's limit is whatever it was initialised to. Below `tau = 1` the fracture is
only *estimated*: a rate observed as 0 in 200 samples has a 95% upper bound of
0.019, so a chain that mixes in ~54 iterations is indistinguishable from a
reducible one. **A minorization estimated from `n` samples cannot certify anything
about events rarer than about `1/n`.** §11.5 turns that observation into a
quantitative statement.

---

## 11. Counterexample 2: the gain has the wrong sign

`run_amplifier_demo.py` · reuses `pools.npz` · `non_unique_measure.png`

### 11.1 The change

`IfElseLadderLogic(polarity=-1)` is the same if-else statement with the sign of its
gain flipped:

| | `s > tau` | `s < −tau` |
| --- | --- | --- |
| corrective (`polarity=+1`) | one rung more critical | one rung more positive |
| amplifying (`polarity=-1`) | one rung more positive | one rung more critical |

The loop now reinforces what it observes instead of opposing it. **No new sampling
is needed.** The conditional law of the sentiment given the rung is a property of
the model and the prompt, not of the controller, so `pools.npz` is reused byte for
byte; only where the controller *sends* each observation changes. That is also what
makes the comparison clean — the plant, the observable and the dead zone are all
held fixed, and every run here uses one response per iteration.

The expectation going in is that positive feedback splits the ladder into two
attractors — enthusiastic and critical — with the limit decided by the initial
condition. That turns out to be right, but only under a condition the naive
picture omits, and finding the condition is the interesting part.

### 11.2 With the dead zone on the reference: uniquely ergodic, and useless

```text
communicating classes : [[0, 1], [2], [3], [4, 5, 6]]
closed classes        : [[0, 1]]
recurrent class       : [0, 1]   (transient: [2, 3, 4, 5, 6])
invariant measure     : UNIQUE (1 closed class)
Doeblin minorization  : eps=1.0000 at m=10
  => TV<0.01 within ~10 iterations once in the recurrent class
  reaching the recurrent class: <= 3683 iterations to 0.005
                                (expected 699.7 from the worst transient state)
  => TV<0.01 from ANY start within ~3693 iterations
stationary pi         : [0.9655, 0.0345, 0, 0, 0, 0, 0]
```

Bistability requires **both** ends of the ladder to be absorbing, and only one of
them is. The critical end leaks: even at "be extremely critical and negative" a
response scores above the band a few percent of the time, and rungs 0–3 essentially
never score below it, so the ladder is a one-way funnel toward the enthusiastic
end. The model's positive bias — the same bias that made the topic choice matter in
§12 — is what keeps the chain irreducible.

Two things are worth extracting from that block.

**Unique ergodicity is not a certificate of good behaviour.** The measure is unique
and it sits, with 97% of its mass, on "be extremely enthusiastic". The loop was
asked to produce neutral text and it converges, with certainty, on the most
enthusiastic corner of the ladder. "The long-run distribution exists and is independent of the
initial condition" is a well-posedness statement about the analysis, not an
approval of the outcome.

**The two legs of the mixing bound can differ by three orders of magnitude.** Once
in the recurrent class the chain mixes in ten iterations. Getting there takes up to
3683. A certificate that reported only the within-class number would say "10" where
the answer is "3693"; this is the case that forced `tv_mixing_bound_total` into
`certify` (§3.8).

### 11.3 Recentring the dead zone closes the leak, and uniqueness dies with it

The leak exists because the *plant is biased* — the model writes positively about
almost anything — while the dead zone is centred on the reference, so the band sits
below where the observations actually live. `IfElseLadderLogic(tau_offset=…)` moves
it: the hold band becomes `[tau_offset − tau, tau_offset + tau]`, and the tolerance
for "too positive" and "too negative" stop being equal. Correcting for a biased
sensor is the textbook-correct thing to do, and it is the whole intervention:

| `tau_offset` | hold band | leak at `k=6` | closed classes | unique? |
| --- | --- | --- | --- | --- |
| −0.20 | [−0.65, +0.25] | 0.033 | `[[0,1]]` | yes |
| 0.00 | [−0.45, +0.45] | 0.033 | `[[0,1]]` | yes |
| +0.20 | [−0.25, +0.65] | 0.033 | `[[0,1,2]]` | yes |
| +0.30 | [−0.15, +0.75] | 0.033 | `[[0,1,2]]` | yes |
| +0.40 | [−0.05, +0.85] | 0.000 | `[[0,1,2], [6]]` | **no** |
| +0.50 | [+0.05, +0.95] | 0.000 | `[[0,1,2], [3], [6]]` | **no** |
| +0.60 | [+0.15, +1.05] | 0.000 | `[[2], [6]]` | **no** |

At `tau_offset = +0.40` both ends close and the invariant measure stops being
unique. The script picks that crossing point automatically rather than hard-coding
it, and exits with a clear message if uniqueness survives the whole scan.

**The mechanism inverts the usual intuition about bias.** A biased sensor is
normally a defect to be corrected, and the correction here is textbook. But the
plant's positive bias was the only thing keeping the critical end leaky, and
therefore the only thing propping uniqueness up. The offset does not repair the
positive feedback — nothing about the sign of the gain changes — it removes the
accident that was masking it. The bistability was there all along, put there by the
sign error, and hidden by a defect in the plant.

And it is not a free correction. §11.4's control run shows the price: with the band
recentred on `+0.40`, the *corrective* loop settles at a mean sentiment of `+0.54`
rather than near zero, because a loop with an off-centre dead zone regulates to the
centre of the band and not to the reference. That is why the sweep in `run_demo.py`
is deliberately symmetric — on the corrective loop an off-centre band buys nothing
a symmetric one does not already have, and it costs steady-state error.

### 11.4 What non-uniqueness looks like

At `tau_offset = +0.40`:

```text
closed classes        : [[0, 1, 2], [6]]
invariant measure     : NOT UNIQUE (2 closed classes)
  the invariant set is the convex hull of 2 extreme measures:
    mu_0 on [0, 1, 2]: [0.9194, 0.0766, 0.0040, 0, 0, 0, 0]
    mu_1 on [6]:       [0, 0, 0, 0, 0, 0, 1]
```

so the invariant measures are the whole segment `{a·μ₀ + (1−a)·δ₆ : a ∈ [0,1]}`,
and which one a run converges to is given by the absorption probabilities:

| `k₀` | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `P(→ critical end)`, theory | 0 | 0 | 0 | 0 | 0.9485 | 0.9959 | 1 |
| same, empirical (200 runs) | 0 | 0 | 0 | 0 | 0.9350 | 1.000 | 1 |
| mean terminal sentiment | +0.841 | +0.806 | +0.804 | +0.817 | −0.677 | −0.701 | −0.743 |

Note that the enthusiastic attractor is a genuine three-rung class with a spread
invariant measure, not a point mass — the loop wanders between "extremely
enthusiastic" and "positive" forever without ever leaving.

The dependence on the initial condition is a **watershed rather than a coin flip**.
Rungs 0–3 never produce a response below the band, so nothing starting below the
divide can ever cross it; only rungs 4 and 5 are undecided at all, and even there
the odds are 95:5 and 99.6:0.4. The limit is a deterministic function of the
initial condition almost everywhere on the ladder.

The single cleanest number:

```text
occupation | k0=0 : [0.9149, 0.0803, 0.0048, 0, 0, 0, 0]
occupation | k0=6 : [0, 0, 0, 0, 0, 0, 1]
TV(occupation k0=0, k0=6) = 1.0000       (corrective loop, §9: 0.0099)
```

A total variation distance of 1 between the occupation measures of two initial
conditions is the sharpest statement of non-uniqueness available: the loop
remembers where it started, forever, and the time average along a single long
trajectory tells you about *that trajectory* and not about the system. Every
fairness or "typical outcome" claim built on a long run of this loop would be a
claim about its initialisation.

**Control run.** Same `tau`, same `tau_offset = +0.40`, corrective sign:

```text
closed classes        : [[3, 4, 5, 6]]
invariant measure     : UNIQUE (1 closed class)
  => TV<0.01 from ANY start within ~27 iterations (~24 by Dobrushin)
stationary pi         : [0, 0, 0, 0.8643, 0.1296, 0.0056, 0.0004]
terminal sentiment | k0=0: +0.5368     | k0=6: +0.5738
```

Same plant, same observable, same dead zone — offset and all. One sign decides
whether the loop has a single invariant measure reached in a couple of dozen
iterations or a simplex of them that it never leaves. The `+0.54` is the offset's
bill coming due: a loop with an off-centre dead zone regulates to the centre of the
band, so this control run is well-conditioned and aimed at the wrong place.

### 11.5 Reducible, or merely metastable?

Two closed classes require the escape rate out of *each* end to be exactly zero, and
no finite sample can establish that. A rate counted as 0 out of `n` samples has a
95% Wilson upper bound of roughly `4/n` — the reported `0.000` means "not seen in
this pool", not "impossible".

So take the other reading. Suppose every rate counted as zero is actually some
small `ε > 0`. The chain is then irreducible and uniquely ergodic
again — and:

| assumed `ε` | unique? | Doeblin `ε` | TV<0.01 within |
| --- | --- | --- | --- |
| 1e-2 | yes | 1.6e-2 | 1.1e+05 |
| 1e-3 | yes | 1.6e-3 | 1.1e+06 |
| 1e-4 | yes | 1.6e-4 | 1.1e+07 |
| 1e-5 | yes | 1.6e-5 | 1.1e+08 |

Both routes give the same numbers here and both are loose — the minorization search
is capped at `m = 400`, far below where this chain begins to mix. The scaling is
what carries the argument: the bound grows by exactly 10× for every 10× drop in
`ε`, because the chain has to wait for the rare escape, so the mixing time is
`Θ(1/ε)`.

The conclusion is not that the classification is unreliable. It is that **"two
invariant measures" and "one invariant measure reached after 10^5 iterations" are
the same statement in practice**, and that neither is distinguishable from a few
hundred samples per rung. The honest report is both: the classification under the estimated
rates, and the mixing time as a function of the escape rate you are not willing to
rule out. `certify` gives the first; the `--escape-rates` sweep gives the second.

### 11.6 What the three runs establish together

The positive result (§9) is that a loop containing a language model can be
certified uniquely ergodic with an explicit rate, from sampling alone. The two
counterexamples are what make that a measurement rather than a formality:

- The machinery **returns a different verdict when the loop deserves one**, and the
  verdict is specific — non-existence via a divergent invariant series (§10),
  non-uniqueness via a closed-class count with absorption probabilities (§11).
- Both failures are **properties of the control design**, not of the language
  model. The same model, prompt, classifier and threshold appear in all three runs.
  An unbounded actuator with no gain destroys existence; a sign error destroys
  uniqueness; the corrective bounded ladder gets a certificate.
- Both failures are **invisible to a convergence-style analysis**. The divergent
  loop has a perfectly steady mean sentiment (+0.637, unchanging) and would look
  like a system in steady state to anything that only watches the output. The
  bistable loop settles down beautifully — to a different place each time.

---

## 12. Findings that shaped the design

Recorded because each one cost a measurement, and each would otherwise be
re-encountered.

**A one-sided "be more neutral" ladder does not work.** Measured across six rungs
on the 0.5B model, it moved mean sentiment by **0.014** — against a within-rung sd
of ~0.18 — and non-monotonically, with rung 2 the *most* positive of all. With a
baseline of +0.85 and an actuator range of 0.014, the reference r = 0 is simply
outside the controller's reach: the error never enters the deadband and the level
pins at the top rung forever. This is a control-design flaw, not an LLM limitation.
Hence the signed ladder.

**The topic matters more than the model size.** Under the signed ladder:

| instruction | "winter holidays" | "the public transport system" |
| --- | --- | --- |
| ++ enthusiastic | +0.979 | +0.860 |
| + mildly positive | +0.970 | +0.878 |
| 0 neutral | +0.731 | +0.599 |
| − mildly critical | +0.524 | **−0.370** |
| −− very critical | +0.486 | **−0.748** |

"Winter holidays" spans 0.49 but **never crosses zero** — the model will not speak
negatively about an intrinsically pleasant subject, so the reference stays
unreachable. A mixed-valence topic straddles it, and the full 200-sample profile in
§9 confirms the probe: authority 1.64 on transport. A 0.5B model has ample
authority once the topic permits it.

**Beware first-person preference prompts.** `"Write one short sentence about your
favourite holiday."` makes small instruct models emit *"As an AI language model, I
don't have personal preferences"* — off-topic text that still scores +0.72 and
pollutes the observable. Worse, those disclaimers *fake* control authority: they
score less positively than genuine prose, so removing them removed an apparent
0.32 of ladder authority that was never real. The base prompt is third-person for
this reason.

**Single words in the ladder can invert a rung.** `"Be critical."` measured
**+0.649** — more positive than the "mildly critical" rung below it and on a par
with the neutral rung — because the 0.5B model reads *critical* as
"essential/crucial" and writes *"provides an essential mode of urban mobility"*.
Rung 6 escaped the trap only because the word "negative" disambiguates it.
Rewording to `"Be critical and disapproving."` moved the rung to **−0.421** and
made the critical side monotone. Two lessons: an instruction ladder needs its
rungs *measured*, not assumed, and lexical ambiguity is a live failure mode for
small models. This is what `--only-levels` is for.

**Expect a strict recurrent class.** With a signed ladder the chain settles around
the zero crossing and the extreme rungs become transient — on synthetic pools
matching the measured profile, `{0,1,2}` were transient and `{3,4,5,6}` recurrent.
Uniqueness is unaffected, but the unreachable rungs are wasted LLM budget, and a
tighter ladder centred on the crossing would spend it better.

---

## 13. Limitations

- **The §10 certificate extrapolates.** It concerns a chain on `{0,1,2,…}` and
  twelve of its rungs were measured. Rungs 20 and 50 make the saturation
  assumption an evidenced one (§10.3) rather than an assumed one, but they do not
  prove it: a model that abruptly starts obeying its 200th repeated instruction
  would overturn the verdict, and nothing here rules that out. What does *not*
  depend on the extrapolation is the negative result about the actuator — the loop
  demonstrably fails to regulate at every rung anyone can afford to measure.
- **The unbounded ladder is an idealisation of a system that breaks differently.**
  A real deployment would hit the model's context window somewhere past rung 500
  and start truncating, so the state space is physically finite. That is not a
  rescue: it replaces "no invariant measure" with "an invariant measure determined
  by the context limit", which is §10.6's manufactured certificate wearing a
  different hat. Either way the loop never regulates.
- **The §11 classification rests on rates estimated as zero.** Reducibility is an
  exact-zero statement and 200 samples per rung cannot supply one. §11.5 reports
  the alternative reading — unique but mixing in `Θ(1/ε)` — but does not eliminate
  it, and nothing short of an analytic bound on the model's output distribution
  could.
- **The §11.3 crossing point is resolution-limited.** The offset at which the leak
  "closes" is the offset at which no sample in the pool exceeds the band, so it is
  bounded below by the pool size: with `n` samples per rung nothing rarer than
  about `1/n` is visible, and the crossing point would move with a larger pool.
  §11.5 is the quantitative version of this caveat.
- **One model, one topic, one classifier.** Every number here is
  `Qwen2.5-0.5B-Instruct` writing about public transport, scored by
  `twitter-roberta-base-sentiment-latest`. The *method* is model-agnostic; the
  measured profiles, and therefore every certificate, are not.
- **KDE boundary bias.** The sentiment score has compact support [−1, 1] and the
  estimator uses a Gaussian kernel, so density leaks past the boundary. A reflected
  or boundary-corrected kernel would fix it.
- **The KDE panel needs more runs than the certificates do.** On synthetic pools, TV
  between the two sentiment KDEs was 0.090 while TV between the level occupation
  measures was 0.0054 — the gap is KDE sampling noise, not a real discrepancy.
  Phase B is cache-served, so raise `--runs` for the figure.
- **`distribution.get_distributions` is unusable here.** It pads its evaluation
  range by ±10, which is reasonable for the order-10 node outputs of the shipped
  examples but swamps a score bounded in [−1, 1]. `run_demo.py` calls the
  underlying `kernel_density_distribution` directly.
- **Doeblin `eps` is a point estimate** from transition counts. A Wilson lower bound
  is reported when the minorization binds at `m = 1`; for `m > 1` the uncertainty is
  not propagated. `transience.py` does propagate it, because its verdict is a
  yes/no that hinges on rates near zero; `markov.py` does not.
- **The two-leg mixing bound splits the error budget evenly.** `certify` allocates
  `target/2` to reaching the recurrent class and `target/2` to mixing inside it.
  That is valid but not optimal; a tighter report would optimise the split.

## 14. Environment notes

- **The Radeon 780M iGPU on this machine cannot run this.** It exposes only a 1 GiB
  VRAM carve-out, and a 0.5B model in fp16 OOMs at ~855 MiB allocated alongside the
  classifier. Use `--device cpu`, or a discrete GPU. ROCm additionally needs
  `HSA_OVERRIDE_GFX_VERSION=11.0.0` for gfx1103 or every kernel fails with
  `HIP error: invalid device function`; `llm_agent.py` sets this via `setdefault`.
  With the override the iGPU does compute correctly (718 GFLOP/s on 2048² matmul) —
  memory, not kernels, is the blocker.
- Requires `transformers`, and `scipy` — which the repo's `requirements.txt` omits
  even though `simulators/distribution.py` imports it. On this machine only the
  `llm-cuda` conda env has both; the system interpreter and the `ic` venv do not,
  so Phase A must be run as
  `~/anaconda3/envs/llm-cuda/bin/python examples/llm_poc/build_pools.py …`.
  Phase B and `tests/test_guarantees.py` need only NumPy and Matplotlib.
- Throughput is sensitive to CPU contention: the same workload measured
  0.66–0.72 s/sample idle and 4.7 s/sample against one competing training job. The
  1400-generation signed build took 721 s and the 2400-generation repeat build
  1574 s, both on 16 idle threads.
