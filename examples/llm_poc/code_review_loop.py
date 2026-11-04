#!/usr/bin/env python
# coding: utf-8

# # Programmer–analyst loop with a steered code rewrite
# 
# Each step the population is $N$ programmer personas $i$ drawn uniformly from a fixed pool (the random choice; its law does not depend on the control).
# Programmer $i$ is asked to make a fixed vulnerable line of Python more secure. One analyst (the same model, unsteered, different prompt) reads the rewrite and answers "Is this line safe from SQL injection?".
# 
# Definitions:
# - $\alpha$: controller signal; CAA steering on the programmer, $h \mathrel{+}= \alpha v$ at every position of layer $L_s$, with $v$ = mean hidden state on a safe rewrite minus that on the vulnerable line (https://arxiv.org/abs/2312.06681).
# - $x_{1..T}$: the programmer's greedy output tokens, generated without gradient. $T$ is the output length.
# - $p_t \in \mathbb{R}^V$: the programmer's next-token distribution at output position $t$, recomputed in one teacher-forced pass on $x$ with the hook on. $V$ is the vocabulary size.
# - $E \in \mathbb{R}^{V \times d}$: the embedding matrix; $\bar e_t = p_t^\top E$ is the expected embedding fed to the analyst in place of token $x_t$.
# - $f_i(\alpha) = \sigma(\ell_\text{Yes} - \ell_\text{No})$: the analyst's $P(\text{safe})$ given $\bar e_{1..T}$; $\ell$ are the analyst's next-token logits and $\sigma$ the logistic function.
# 
# The population output is $y = \frac1N\sum_i f_i(\alpha)$. Autograd gives $\partial y/\partial\alpha$ with $x$ held fixed: the derivative inside a piece where the greedy tokens do not change. Jumps where a change of $\alpha$ flips a greedy token are not seen.
# 
# Loop: $r \to$ A1 ($e = r - y$) $\to$ C (PI, $\alpha$) $\to$ P1 ($y$) $\to$ A2 $\to$ Z $\to$ F $\to$ A1.

# In[ ]:


import sys, os
sys.path.insert(0, os.path.abspath("../.."))
import itertools
from contextlib import contextmanager
import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

from humancompatible.interconnect.simulators.simulation import Simulation
from humancompatible.interconnect.simulators.nodes.reference import ReferenceSignal
from humancompatible.interconnect.simulators.nodes.aggregator import Aggregator
from humancompatible.interconnect.simulators.nodes.controller import Controller
from humancompatible.interconnect.simulators.nodes.population import Population
from humancompatible.interconnect.simulators.nodes.delay import Delay
from humancompatible.interconnect.simulators.nodes.filter import Filter
from humancompatible.interconnect.simulators.logic.delays.exponentialSmoothingDelay import ESDelayLogic
from humancompatible.interconnect.simulators.logic.controllers.pi_controller import PiControllerLogic
from tests.contractionTests.contraction_test import get_factor_from_list

print("hi")
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"   # programmer and analyst share these weights
tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "left"   # all prompts end at the same position, so generated tokens line up
DEV = "cuda" if torch.cuda.is_available() else "cpu"   # LLM on DEV; the package side stays on CPU
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(DEV).eval()
for p in model.parameters():
    p.requires_grad_(False)
E = model.get_input_embeddings().weight
YES, NO = tok.encode("Yes")[0], tok.encode("No")[0]
STOP = [tok.pad_token_id, tok.convert_tokens_to_ids("<|im_end|>")]

SNIPPET = """cursor.execute("SELECT * FROM users WHERE name = '" + name + "'")"""
SAFE = """cursor.execute("SELECT * FROM users WHERE name = ?", (name,))"""
TASK = f"Rewrite this line of Python so that it is more secure. The result should also be just one line. Reply with only the rewritten line.\n\n{SNIPPET}"
POOL = [f"You are a {role} who {habit}."
        for role, habit in itertools.product(
            ["specialist", "student", "hacker"],
            ["cares a lot about security", "is in a hurry to ship", "wants to make the code less secure"])]

def prompts(idx, answer=""):
    return [tok.apply_chat_template([{"role": "system", "content": POOL[i]}, {"role": "user", "content": TASK}],
                                    tokenize=False, add_generation_prompt=True) + answer for i in idx]

# analyst prompt with a slot for the code; the code is spliced in as embeddings
_a = tok.apply_chat_template([{"role": "system", "content": "You are a security analyst."},
                              {"role": "user", "content": "Code:\n<<CODE>>\n\nIs this line of Python safe from SQL injection? Answer with Yes or No."}],
                             tokenize=False, add_generation_prompt=True)
PRE, POST = [torch.tensor(tok(s, add_special_tokens=False).input_ids, device=DEV) for s in _a.split("<<CODE>>")]
len(POOL)


# In[2]:


def analyst(code_emb):
    # code_emb: (T', d) embeddings of the code; returns P(safe)
    x = torch.cat([E[PRE], code_emb, E[POST]])[None]
    logits = model(inputs_embeds=x).logits[0, -1]
    return torch.sigmoid(logits[YES] - logits[NO]).cpu()

# can the analyst tell the two lines apart at all?
with torch.no_grad():
    for name, code in [("vulnerable", SNIPPET), ("safe", SAFE)]:
        print(f"{name}: P(safe) = {analyst(E[tok(code, add_special_tokens=False).input_ids]):.3f}")


# In[3]:


LAYER = 12       # ~1/3 of the 28 layers
N_VEC = len(POOL)

@torch.no_grad()
def answer_states(answer):
    enc = tok(prompts(range(N_VEC), answer), return_tensors="pt", padding=True).to(DEV)
    return model.model(**enc, output_hidden_states=True).hidden_states[LAYER + 1][:, -1].mean(0)

V = answer_states(SAFE) - answer_states(SNIPPET)
print(f"|v| = {V.norm():.1f}")


# In[4]:


T = 40   # max rewrite length in tokens

@contextmanager
def steer(alpha):
    alpha = torch.as_tensor(alpha, dtype=V.dtype).to(DEV)   # .to keeps the gradient path to the CPU alpha
    h = model.model.layers[LAYER].register_forward_hook(lambda mod, args, out: out + alpha * V)
    try:
        yield
    finally:
        h.remove()

def programmer(idx, alpha):
    enc = tok(prompts(idx), return_tensors="pt", padding=True).to(DEV)
    P = enc.input_ids.shape[1]
    with steer(alpha):
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=T, do_sample=False)
        mask = torch.cat([enc.attention_mask, (out[:, P:] != tok.pad_token_id).long()], 1)
        logits = model(input_ids=out, attention_mask=mask).logits[:, P - 1:-1]   # teacher-forced: p_t for every t
    gen = out[:, P:]
    keep = ~torch.isin(gen, torch.tensor(STOP, device=DEV))
    return logits.softmax(-1) @ E, keep, gen   # expected embeddings (N, T, d)

def p_safe(idx, alpha):
    e_bar, keep, _ = programmer(idx, alpha)   # hook is off from here on: the analyst is unsteered
    return torch.stack([analyst(e[k]) for e, k in zip(e_bar, keep)])


# In[5]:


ALPHAS = np.linspace(-1, 1, 5)
rng = np.random.default_rng(0)
idx = rng.choice(len(POOL), 8, replace=False)
F = []
with torch.no_grad():
    for a in ALPHAS:
        F.append(p_safe(idx, a).numpy())
        print(f"alpha={a:+.1f}:", tok.decode(programmer(idx[:1], a)[2][0], skip_special_tokens=True))
F = np.array(F)   # (alpha, persona)
plt.plot(ALPHAS, F, color="grey", alpha=0.3)
plt.plot(ALPHAS, F.mean(1), color="k", lw=2)
plt.xlabel("alpha"); plt.ylabel("P(safe)");


# In[6]:


KAPPA = 0.5     # set from the sweep: kappa * slope ~1 near the operating point
N_AGENTS = 4
STEPS = 20
R = 0.5         # pick inside the range the sweep reaches


# In[7]:


class AgentLogic:
    def __init__(self, seed=None):
        self.tensors = {"alpha": torch.tensor([0.0])}
        self.variables = ["alpha"]
        self.rng = np.random.default_rng(seed)

    def forward(self, values, number_of_agents):
        alpha = values["alpha"]
        idx = self.rng.choice(len(POOL), number_of_agents)   # the random choice, independent of alpha
        a = alpha.detach().requires_grad_()
        y = p_safe(idx, a).mean()
        g, = torch.autograd.grad(y, a)
        # value y, gradient g wrt alpha: keeps the LLM graph out of the node history
        return (y.detach() + g * (alpha - alpha.detach())).reshape(1)


class CodeSim(Simulation):
    class ErrorLogic:   # A1: e = r + F, cut from the past so the Jacobian is one pass
        def __init__(self):
            self.tensors = []
        def forward(self, tensors):
            return torch.stack([t.detach().sum() for t in tensors]).sum().reshape(1).requires_grad_()

    class SumLogic:   # A2
        def __init__(self):
            self.tensors = []
        def forward(self, tensors):
            tensors = tensors if isinstance(tensors, list) else [tensors]
            return torch.stack([t.sum() for t in tensors]).sum().reshape(1)

    class FilterLogic:   # F = -y
        def __init__(self):
            self.tensors = {"S": torch.tensor([0.0])}
            self.variables = ["S"]
        def forward(self, values):
            return -values["S"]

    def __init__(self, reference_signal=0.5, weights=None, alpha0=0.0, kappa=KAPPA, n_agents=N_AGENTS, seed=None):
        super().__init__()
        r = ReferenceSignal(name="r")
        r.set_reference_signal(torch.tensor(float(reference_signal)))
        a1 = Aggregator(name="A1", logic=self.ErrorLogic())
        pi = PiControllerLogic(kappa=kappa, alpha=0.0)   # alpha=0: pure integrator, alpha_k = alpha_{k-1} + kappa e_k
        pi.tensors["pi_prev"] = torch.tensor([float(alpha0)])
        c = Controller(name="C", logic=pi)
        p1 = Population(name="P1", logic=AgentLogic(seed), number_of_agents=n_agents)
        a2 = Aggregator(name="A2", logic=self.SumLogic())
        z = Delay(name="Z", logic=ESDelayLogic(alpha=1.0))   # pass-through
        f = Filter(name="F", logic=self.FilterLogic())
        self.system.add_nodes([r, a1, c, p1, a2, z, f])
        for u, w in [(r, a1), (a1, c), (c, p1), (p1, a2), (a2, z), (z, f), (f, a1)]:
            self.system.connect_nodes(u, w)
        self.system.set_start_node(r)
        self.system.set_checkpoint_node(a1)


# In[8]:


N_SEEDS = 3   # persona draws per initial value
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
for a0, color in zip([-1.0, 0.0, 1.0], ["C0", "C1", "C2"]):
    for seed in range(N_SEEDS):
        s = CodeSim(R, alpha0=a0, seed=seed)
        s.system.run(STEPS, show_trace=False, show_loss=False, disable_tqdm=False)
        ax[0].plot([h.item() for h in s.system.get_node("C").history], color=color, alpha=0.6,
                   label=f"alpha0={a0}" if seed == 0 else None)
        ax[1].plot([h.item() for h in s.system.get_node("P1").history], color=color, alpha=0.6)
ax[0].set_ylabel("alpha"); ax[1].set_ylabel("y"); ax[1].axhline(R, color="k", ls=":")
ax[0].legend(); ax[0].set_xlabel("step"); ax[1].set_xlabel("step");


# Contraction factor from the package: mean over steps and trials of $|\partial F_k / \partial e_k| = \kappa\,|\partial y_k/\partial\alpha|$ at the visited $\alpha_k$ and drawn personas. For the integrator state map $\alpha_{k+1} = \alpha_k + \kappa(r - y(\alpha_k))$ the relevant quantity is $|1 - \kappa\,\partial y/\partial\alpha|$.

# In[ ]:


C = get_factor_from_list(reference_signals=np.array([R]), agent_probs=None, sim_class=CodeSim,
                         it=STEPS, trials=3, inputs_node="A1", outputs_node="F")
C

