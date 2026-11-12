# # Soft-token loop test: Qwen → Gumbel-softmax → Qwen sentiment head
# 
# Generator and judge share `Qwen2.5-1.5B-Instruct`. Each generated
# token is a Gumbel-softmax vector `y = softmax((logits/T + g)/λ)`; it is fed back to the
# generator and to the judge as the embedding `y @ E`. `λ = None` means hard Gumbel-max
# sampling (one-hot `y`). The judge never sees the instruction, only the response.
# 
# Sentiment readout: `s = logit[positive] − logit[negative]` of a linear head (trained on Amazon
# review texts, `finetune_sentiment_head.py`) applied to the mean hidden state over the response.

import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import MODEL, BASE, INSTRUCTIONS, FEAT_DIR

HEAD = FEAT_DIR / "score_head.pt"  # from finetune_sentiment_head.py
device = "cpu"  # the local GPU has 1 GB, too small for the fp32 model
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(device).eval()
for p in model.parameters():
    p.requires_grad_(False)
# judge = the same model's body + a linear head trained on its mean hidden states
score = torch.nn.Linear(model.config.hidden_size, 2, bias=False).to(device)
score.load_state_dict(torch.load(HEAD))
score.requires_grad_(False)
E = model.get_input_embeddings().weight          # (V, d), V = logit dimension
STOP = {tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}
print(device, E.shape)

def chat_ids(user_text):
    text = tok.apply_chat_template([{"role": "user", "content": user_text}],
                                   tokenize=False, add_generation_prompt=True)
    return torch.tensor(tok.encode(text), device=device)

def generate_agent_softtokens(prompt_emb, lam=None, seed=0, T=1.0, max_new=32, length=None):
    """Gumbel decoding from prompt embeddings (L0, d).
    Returns soft tokens Y (L, V) and the Gumbel-max ids (argmax of each y).
    `length` forces exactly that many tokens, ignoring stop tokens: the stop rule
    makes the map discontinuous, so perturbed runs must not change the length."""
    gen = torch.Generator(device=device).manual_seed(seed)
    out = model(inputs_embeds=prompt_emb.unsqueeze(0), use_cache=True)
    past, logits = out.past_key_values, out.logits[0, -1]
    Y, ids = [], []
    for _ in range(length or max_new):
        u = torch.rand(logits.shape, generator=gen, device=device).clamp_min(1e-20)
        z = logits / T - torch.log(-torch.log(u))            # perturbed logits
        i = int(z.argmax())
        if length is None and i in STOP:
            break
        if lam is None:
            y = F.one_hot(torch.tensor(i, device=device), z.numel()).float()
        else:
            y = torch.softmax(z / lam, dim=-1)
        Y.append(y)
        ids.append(i)
        out = model(inputs_embeds=(y @ E)[None, None], past_key_values=past, use_cache=True)
        past, logits = out.past_key_values, out.logits[0, -1]
    return torch.stack(Y), ids

def judge(Y):
    # the head was fit on states averaged over a text's tokens
    h = model.model(inputs_embeds=(Y @ E).unsqueeze(0)).last_hidden_state[0].mean(0)
    logits = score(h)
    return logits[1] - logits[0]   # log-odds of positive

def embedded_agent_prompt(instruction):
    return E[chat_ids(f"{BASE} {instruction}")]

# ## 1. Hard vs soft scores
# 
# For each prompt, seed and `λ`: the text read off the argmax tokens, the judge's score on
# the soft tokens `s_soft`, and on the one-hot version of the same argmax tokens `s_hard`.
# Same seed ⇒ same Gumbel noise across `λ`; the texts can still differ, because soft
# tokens fed back into the generator change its later logits.

LAMS = [None, 0.2]
rows = []
with torch.no_grad():
    for ins in INSTRUCTIONS:
        AGENT_PROMPT = embedded_agent_prompt(ins)
        for seed in range(5):
            for lam in LAMS:
                AGENT_SOFTTOKENS, ids = generate_agent_softtokens(AGENT_PROMPT, lam=lam, seed=seed)
                AGENT_HARDTOKENS = F.one_hot(torch.tensor(ids, device=device), AGENT_SOFTTOKENS.shape[1]).float()
                rows.append(dict(instruction=ins[:25], seed=seed, lam=lam,
                                 s_soft=judge(AGENT_SOFTTOKENS).item(), s_hard=judge(AGENT_HARDTOKENS).item(),
                                 peak=AGENT_SOFTTOKENS.max(1).values.mean().item(),
                                 text=tok.decode(ids)))
df = pd.DataFrame(rows)
pd.set_option("display.max_colwidth", 120)
print(df)

# Agreement between soft and hard scores, and mean peak mass of y, per lambda
print(df.assign(gap=(df.s_soft - df.s_hard).abs()).groupby("lam", dropna=False)[["gap", "peak"]].mean())

print(df.groupby(["instruction", "lam"], dropna=False)[["s_soft", "s_hard"]].mean())

del(df)

# ## 2. Gradient through the loop
# 
# Gradient of `s` with respect to the prompt embeddings. With hard tokens there is no path
# from the prompt to `s` (the judge sees only one-hot vectors), so the gradient is zero. With
# soft tokens it is not. The finite-difference check along a random direction `δ` (same seed
# on both sides) tests that autograd matches the relaxed system's actual sensitivity.

P0 = embedded_agent_prompt(INSTRUCTIONS[1])
torch.manual_seed(0)
delta = torch.randn_like(P0); delta /= delta.norm()
h = 1e-2
for lam in LAMS:
    AGENT_PROMPT = P0.clone().requires_grad_(True)
    AGENT_SOFTTOKENS, _ = generate_agent_softtokens(AGENT_PROMPT, lam=lam, seed=0)
    L = AGENT_SOFTTOKENS.shape[0]  # pin the perturbed runs to the base length
    judge_res = judge(AGENT_SOFTTOKENS)
    if not judge_res.requires_grad:
        print(f"lam={lam}: s={judge_res.item():+.3f}  no gradient path (hard tokens)")
        continue
    judge_res.backward()
    with torch.no_grad():
        fd = (judge(generate_agent_softtokens(P0 + h * delta, lam=lam, seed=0, length=L)[0])
              - judge(generate_agent_softtokens(P0 - h * delta, lam=lam, seed=0, length=L)[0])) / (2 * h)
    print(f"lam={lam}: judge result (s)={judge_res.item():+.3f}  |grad|={AGENT_PROMPT.grad.norm():.3f}  "
          f"grad·δ={(AGENT_PROMPT.grad * delta).sum():+.4f}  finite diff={fd.item():+.4f}")
