"""LLM-in-the-loop agent logic for ergodicity analysis.

The agent is a small instruction-tuned LLM answering a *fixed* base prompt.  The
controller's only actuator is a discrete "neutrality instruction level" ``k``,
which is rendered into a **freshly built** prompt on every iteration.  Nothing is
appended across iterations: the prompt is a pure function of ``k``, so the loop
state stays finite and the closed loop is a genuine Markov chain on ``k``.

Two logic classes are provided, both satisfying the ``Population`` contract
(``tensors``, ``variables``, ``forward(values, number_of_agents)``):

``LLMAgentLogic``
    Calls the LLM live at every step.  Use for validation runs.
``CachedLLMAgentLogic``
    Draws from per-level pools of sentiment scores precomputed by
    ``examples/llm_poc/build_pools.py``.  Because responses are i.i.d. given
    ``k``, this is distributionally exact and makes the loop essentially free,
    which is what allows long trajectories and many restarts.
"""

import os

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")  # ROCm iGPU kernels

import numpy as np
import torch

# Signed instruction ladder, ordered from most positive to most critical.
#
# A one-sided "be more neutral" ladder does not work: measured on a 0.5B instruct
# model it moved sentiment by 0.014 across six rungs, against a within-rung sd of
# ~0.18, and the baseline sat at +0.85 -- so the reference r = 0 was outside the
# actuator's reach and the level simply pinned at the top rung.  A signed ladder
# spans the reference, which is what makes regulation (and a non-degenerate
# invariant measure) possible at all.
NEUTRALITY_LADDER = [
    "Be extremely enthusiastic and positive. Embody a real enthusiast of the topic.",
    "Be mildly positive.", # switched places cause of the mean scores
    "Be causually positive.",
    "Be strictly neutral, impersonal and factual. Use no evaluative or emotional language.",
    "Be mildly critical and disapproving.",
    "Be critical and disapproving. Embody someone who dislikes the topic.",
    "Be extremely critical and negative. Embody a real hater of the topic.",
]

NUM_LEVELS = len(NEUTRALITY_LADDER)

# The topic matters more than the model.  "winter holidays" spans only +0.98 down
# to +0.49 under the ladder above -- it never crosses zero, because the model will
# not speak negatively about an intrinsically pleasant subject.  A mixed-valence
# topic spans +0.88 to -0.75 and straddles the reference.
BASE_PROMPT = "Write one short sentence about the public transport system."
# BASE_PROMPT = "Write one short sentence about the color white."


def render_prompt(level, base_prompt=BASE_PROMPT):
    """Build the full user prompt for instruction ``level``.

    The prompt is rebuilt from scratch every call -- this is what keeps the loop
    Markov in ``level`` alone.
    """
    level = int(np.clip(level, 0, NUM_LEVELS - 1))
    instruction = NEUTRALITY_LADDER[level]
    return base_prompt if not instruction else f"{base_prompt} {instruction}"


# --- the naive one-sided ladder ------------------------------------------------
# This is the design the PoC started from and the one that fails: the controller
# has a single move, "append one more copy of the instruction", so the ladder is
# unbounded above and has no rung on the negative side of the reference.  Kept in
# the package because the failure is the point -- see examples/llm_poc/README.md.
REPEAT_INSTRUCTION = "Be more neutral."


def render_repeat_prompt(level, base_prompt=BASE_PROMPT):
    """Base prompt followed by ``level`` verbatim copies of the instruction.

    Unbounded in ``level``: there is no clipping here, which is exactly what lets
    the closed-loop state escape to infinity.
    """
    level = max(0, int(level))
    if level == 0:
        return base_prompt
    return f"{base_prompt} " + " ".join([REPEAT_INSTRUCTION] * level)


#: Ladders addressable by name from the command line.
LADDERS = {"signed": render_prompt, "repeat": render_repeat_prompt}


def pick_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


class SentimentScorer:
    """Maps text to a scalar sentiment in [-1, 1] via a 3-class classifier.

    The score is ``P(positive) - P(negative)``, so 0 means "neutral" both when the
    classifier is confident about the neutral class and when positive and negative
    are balanced.  Using the probability difference rather than the argmax label
    keeps the observation continuous, which the density estimation needs.
    """

    def __init__(self, model_name="cardiffnlp/twitter-roberta-base-sentiment-latest",
                 device=None, batch_size=64):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = device or pick_device()
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device).eval()

        # Resolve label order from the config rather than assuming it.
        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.pos_idx = next(i for i, v in id2label.items() if v.startswith("pos"))
        self.neg_idx = next(i for i, v in id2label.items() if v.startswith("neg"))

    @torch.inference_mode()
    def score(self, texts, readout="prob"):
        """:param readout: ``"prob"`` gives ``P(pos) - P(neg)`` in [-1, 1]; ``"logit"``
            gives the unbounded log-odds difference, which does not saturate near
            the ends of the scale and so resolves differences between prompts that
            the probability difference compresses away.
        :return: np.ndarray of shape (len(texts),).
        """
        out = np.empty(len(texts), dtype=np.float64)
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start:start + self.batch_size]
            enc = self.tokenizer(chunk, return_tensors="pt", padding=True,
                                 truncation=True, max_length=128).to(self.device)
            logits = self.model(**enc).logits
            src = logits if readout == "logit" else torch.softmax(logits, dim=-1)
            out[start:start + len(chunk)] = (
                src[:, self.pos_idx] - src[:, self.neg_idx]
            ).float().cpu().numpy()
        return out


class LLMResponder:
    """A small instruction-tuned LLM sampling responses to the rendered prompt.

    Temperature must stay strictly positive: the LLM's sampling *is* the noise
    source that drives the loop, and at temperature 0 the closed loop is
    deterministic and its invariant measure degenerates to a point mass.
    """

    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct", device=None,
                 temperature=1.0, top_p=0.95, max_new_tokens=32, seed=None,
                 render=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        assert temperature > 0, "temperature must be > 0 or the loop is deterministic"
        self.device = device or pick_device()
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.render = render or render_prompt

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # required for batched decoder-only generation
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        self.model.to(self.device).eval()
        if seed is not None:
            torch.manual_seed(seed)

    @torch.inference_mode()
    def sample(self, level, n, base_prompt=BASE_PROMPT):
        """Draw ``n`` i.i.d. responses at instruction ``level``."""
        prompt = self.render(level, base_prompt)
        chat = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        enc = self.tokenizer([text] * n, return_tensors="pt", padding=True).to(self.device)
        out = self.model.generate(
            **enc,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        generated = out[:, enc["input_ids"].shape[1]:]
        return [t.strip() for t in self.tokenizer.batch_decode(generated, skip_special_tokens=True)]


class _BaseAgentLogic:
    """Shared ``Population`` logic plumbing."""

    def __init__(self):
        # ``k`` is the instruction level arriving from the controller.
        self.tensors = {"k": torch.tensor([0.0])}
        self.variables = ["k"]
        self.last_level = 0
        self.last_scores = None

    def _emit(self, scores):
        self.last_scores = np.asarray(scores, dtype=np.float64)
        # Mean sentiment over the ensemble.  Detached on purpose: this branch of
        # the toolkit certifies ergodicity by Markov-chain arguments, not by
        # autograd Jacobians, so no graph needs to be retained across iterations.
        return torch.tensor([float(self.last_scores.mean())])


class LLMAgentLogic(_BaseAgentLogic):
    """Live LLM population: generates and scores at every iteration."""

    def __init__(self, responder, scorer, base_prompt=BASE_PROMPT):
        super().__init__()
        self.responder = responder
        self.scorer = scorer
        self.base_prompt = base_prompt
        self.last_texts = None

    def forward(self, values, number_of_agents):
        self.tensors["k"] = values["k"]
        level = int(round(float(values["k"].item())))
        self.last_level = int(np.clip(level, 0, NUM_LEVELS - 1))
        self.last_texts = self.responder.sample(self.last_level, number_of_agents,
                                                self.base_prompt)
        return self._emit(self.scorer.score(self.last_texts))


class CachedLLMAgentLogic(_BaseAgentLogic):
    """Draws sentiment scores from precomputed per-level pools.

    ``pools[i]`` is a 1-D array of i.i.d. sentiment scores observed at ladder
    level ``levels[i]``.  Sampling with replacement from that pool reproduces the
    conditional law of the observation given the level, which is the only thing
    the loop dynamics depend on.

    ``levels`` defaults to ``0, 1, ..., len(pools)-1``, in which case the lookup
    is the obvious one.  It matters for the unbounded ladder, where the state can
    exceed every rung that was ever sampled: the pool of the *nearest probed*
    rung is then used as a surrogate, which is sound precisely when the
    conditional law has saturated -- the assumption
    ``guarantees.transience.saturation_report`` exists to test.
    """

    def __init__(self, pools, rng=None, levels=None):
        super().__init__()
        self.pools = [np.asarray(p, dtype=np.float64) for p in pools]
        self.levels = (np.arange(len(self.pools)) if levels is None
                       else np.asarray(levels, dtype=int))
        self.rng = rng if rng is not None else np.random.default_rng()

    @classmethod
    def from_npz(cls, path, rng=None):
        data = np.load(path)
        pools = [data[f"level_{i}"] for i in range(int(data["num_levels"]))]
        levels = data["levels"] if "levels" in data.files else None
        return cls(pools, rng=rng, levels=levels)

    def forward(self, values, number_of_agents):
        self.tensors["k"] = values["k"]
        level = int(round(float(values["k"].item())))
        i = int(np.argmin(np.abs(self.levels - level)))
        self.last_level = int(self.levels[i])
        pool = self.pools[i]
        return self._emit(self.rng.choice(pool, size=number_of_agents, replace=True))
