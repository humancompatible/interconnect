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
from tests.contractionTests.contraction_test import get_factor_from_list

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"   # programmer and analyst share these weights
tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "left"   # all prompts end at the same position, so generated tokens line up
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
for p in model.parameters():
    p.requires_grad_(False)
E = model.get_input_embeddings().weight
YES, NO = tok.encode("Yes")[0], tok.encode("No")[0]
STOP = [tok.pad_token_id, tok.convert_tokens_to_ids("<|im_end|>")]

SNIPPET = """cursor.execute("SELECT * FROM users WHERE name = '" + name + "'")"""
SAFE = """cursor.execute("SELECT * FROM users WHERE name = ?", (name,))"""
TASK = f"Rewrite this line of Python so that it is more secure. The result should also be just one line. Reply with only the rewritten line.\n\n{SNIPPET}"
POOL = [f"You are a {level} {role} who {habit}."
        for level, role, habit in itertools.product(
            ["junior", "experienced"],
            ["backend developer", "student", "hacker"],
            ["cares a lot about security", "is in a hurry to ship", "wants to make the code less secure"])]

def prompts(idx, answer=""):
    return [tok.apply_chat_template([{"role": "system", "content": POOL[i]}, {"role": "user", "content": TASK}],
                                    tokenize=False, add_generation_prompt=True) + answer for i in idx]

# analyst prompt with a slot for the code; the code is spliced in as embeddings
_a = tok.apply_chat_template([{"role": "system", "content": "You are a security analyst."},
                              {"role": "user", "content": "Code:\n<<CODE>>\n\nIs this line of Python safe from SQL injection? Answer with Yes or No."}],
                             tokenize=False, add_generation_prompt=True)
PRE, POST = [torch.tensor(tok(s, add_special_tokens=False).input_ids) for s in _a.split("<<CODE>>")]
# orchestrator prompt with a slot for the soft word
_o = tok.apply_chat_template([{"role": "system", "content": "You are the engineering manager of a software team."},
                              {"role": "user", "content": "The security review pass rate of the team's code is<<W>> the target. "
                                                          "Should you tell the programmers to focus more on security? Answer with Yes or No."}],
                             tokenize=False, add_generation_prompt=True)
OPRE, OPOST = [torch.tensor(tok(s, add_special_tokens=False).input_ids) for s in _o.split("<<W>>")]
(ABOVE,), (BELOW,) = tok.encode(" above"), tok.encode(" below")   # fails unless both are single tokens

CACHE = f"data/cache/orchestrator_loop/{MODEL.split('/')[-1]}"
os.makedirs(CACHE, exist_ok=True)

def cached(name, fn):
    # shortcut: the key is only `name`; delete CACHE after changing prompts, POOL or the model
    path = f"{CACHE}/{name}.pt"
    if os.path.exists(path):
        return torch.load(path, weights_only=False)
    out = fn()
    torch.save(out, path)
    return out
len(POOL)

def analyst(code_emb):
    # code_emb: (T', d) embeddings of the code; returns P(safe)
    x = torch.cat([E[PRE], code_emb, E[POST]])[None]
    logits = model(inputs_embeds=x).logits[0, -1]
    return torch.sigmoid(logits[YES] - logits[NO])

# can the analyst tell the two lines apart at all?
with torch.no_grad():
    for name, code in [("vulnerable", SNIPPET), ("safe", SAFE)]:
        print(f"{name}: P(safe) = {analyst(E[tok(code, add_special_tokens=False).input_ids]):.3f}")

LAYER = 12       # ~1/3 of the 28 layers
N_VEC = len(POOL)

@torch.no_grad()
def answer_states(answer):
    enc = tok(prompts(range(N_VEC), answer), return_tensors="pt", padding=True)
    return model.model(**enc, output_hidden_states=True).hidden_states[LAYER + 1][:, -1].mean(0)

V = cached(f"V_L{LAYER}", lambda: answer_states(SAFE) - answer_states(SNIPPET))
print(f"|v| = {V.norm():.1f}")

T = 40   # max rewrite length in tokens

@contextmanager
def steer(alpha):
    h = model.model.layers[LAYER].register_forward_hook(lambda mod, args, out: out + alpha * V)
    try:
        yield
    finally:
        h.remove()

def programmer(idx, alpha):
    enc = tok(prompts(idx), return_tensors="pt", padding=True)
    P = enc.input_ids.shape[1]
    with steer(alpha):
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=T, do_sample=False)
        mask = torch.cat([enc.attention_mask, (out[:, P:] != tok.pad_token_id).long()], 1)
        logits = model(input_ids=out, attention_mask=mask).logits[:, P - 1:-1]   # teacher-forced: p_t for every t
    gen = out[:, P:]
    keep = ~torch.isin(gen, torch.tensor(STOP))
    return logits.softmax(-1) @ E, keep, gen   # expected embeddings (N, T, d)

def p_safe(idx, alpha):
    e_bar, keep, _ = programmer(idx, alpha)   # hook is off from here on: the analyst is unsteered
    return torch.stack([analyst(e[k]) for e, k in zip(e_bar, keep)])

ALPHAS = np.linspace(-1, 1, 5)

@torch.no_grad()
def programmer_sweep():
    idx = np.random.default_rng(0).choice(len(POOL), 8, replace=False)
    F = np.array([p_safe(idx, a).numpy() for a in ALPHAS])   # (alpha, persona)
    texts = [tok.decode(programmer(idx[:1], a)[2][0], skip_special_tokens=True) for a in ALPHAS]
    return F, texts

F, texts = cached(f"sweep_programmer_L{LAYER}_T{T}", programmer_sweep)
for a, t in zip(ALPHAS, texts):
    print(f"alpha={a:+.1f}:", t)
plt.plot(ALPHAS, F, color="grey", alpha=0.3)
plt.plot(ALPHAS, F.mean(1), color="k", lw=2)
plt.xlabel("alpha"); plt.ylabel("P(safe)");

def d_orch(e):
    w = (1 + e) / 2   # e = r - y > 0: pass rate below target
    word = ((1 - w) * E[ABOVE] + w * E[BELOW]).reshape(1, -1)
    x = torch.cat([E[OPRE], word, E[OPOST]])[None]
    logits = model(inputs_embeds=x).logits[0, -1]
    return logits[YES] - logits[NO]

with torch.no_grad():
    D0 = d_orch(torch.tensor(0.0))
    W = np.linspace(0, 1, 11)
    Dw = cached("sweep_orchestrator", lambda: np.array([d_orch(torch.tensor(2 * w - 1)).item() for w in W]))
print(f"d(0) = {D0:.2f}")
plt.plot(W, Dw, "k.-"); plt.axhline(D0.item(), color="k", ls=":")
plt.xlabel("w  (0: above target, 1: below target)"); plt.ylabel("d = logit Yes - logit No");

C_GAIN = 0.1    # set from the sweeps: c * d' * g ~ 1 near the operating point, d' = dd/dw / 2
N_AGENTS = 4
STEPS = 20
R = 0.5         # pick inside the range the programmer sweep reaches

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


class OrchestratorLogic:
    # alpha_k = alpha_{k-1} + c (d(e_k) - d(0)): integrator with an LLM-chosen increment
    def __init__(self, c, alpha0):
        self.tensors = {"e": torch.tensor([0.0]), "pi_prev": torch.tensor([float(alpha0)])}
        self.variables = ["e"]
        self.c = c

    def forward(self, values):
        e = values["e"]
        a = e.detach().requires_grad_()
        d = d_orch(a)
        g, = torch.autograd.grad(d, a)
        # value and slope of d at e: keeps the orchestrator graph out of the node history
        result = self.tensors["pi_prev"] + self.c * ((d.detach() - D0) + g * (e - e.detach()))
        self.tensors["pi_prev"] = result
        return result


class OrchSim(Simulation):
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

    def __init__(self, reference_signal=0.5, weights=None, alpha0=0.0, c=C_GAIN, n_agents=N_AGENTS, seed=None):
        super().__init__()
        r = ReferenceSignal(name="r")
        r.set_reference_signal(torch.tensor(float(reference_signal)))
        a1 = Aggregator(name="A1", logic=self.ErrorLogic())
        ctrl = Controller(name="C", logic=OrchestratorLogic(c, alpha0))
        p1 = Population(name="P1", logic=AgentLogic(seed), number_of_agents=n_agents)
        a2 = Aggregator(name="A2", logic=self.SumLogic())
        z = Delay(name="Z", logic=ESDelayLogic(alpha=1.0))   # pass-through
        f = Filter(name="F", logic=self.FilterLogic())
        self.system.add_nodes([r, a1, ctrl, p1, a2, z, f])
        for u, w in [(r, a1), (a1, ctrl), (ctrl, p1), (p1, a2), (a2, z), (z, f), (f, a1)]:
            self.system.connect_nodes(u, w)
        self.system.set_start_node(r)
        self.system.set_checkpoint_node(a1)

N_SEEDS = 3   # persona draws per initial value

def run(a0, seed):
    s = OrchSim(R, alpha0=a0, seed=seed)
    s.system.run(STEPS, show_trace=False, show_loss=False, disable_tqdm=False)
    return {k: [h.item() for h in s.system.get_node(k).history] for k in ["C", "P1"]}

fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
for a0, color in zip([-1.0, 0.0, 1.0], ["C0", "C1", "C2"]):
    for seed in range(N_SEEDS):
        h = cached(f"run_R{R}_c{C_GAIN}_n{STEPS}_a{a0}_s{seed}", lambda: run(a0, seed))
        ax[0].plot(h["C"], color=color, alpha=0.6, label=f"alpha0={a0}" if seed == 0 else None)
        ax[1].plot(h["P1"], color=color, alpha=0.6)
ax[0].set_ylabel("alpha"); ax[1].set_ylabel("y"); ax[1].axhline(R, color="k", ls=":")
ax[0].legend(); ax[0].set_xlabel("step"); ax[1].set_xlabel("step");

C = cached(f"factor_R{R}_c{C_GAIN}_n{STEPS}",
           lambda: get_factor_from_list(reference_signals=np.array([R]), agent_probs=None, sim_class=OrchSim,
                                        it=STEPS, trials=3, inputs_node="A1", outputs_node="F"))
C
