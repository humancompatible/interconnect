import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import MODEL, BASE, INSTRUCTIONS, FEAT_DIR, STOP_TOKENS

# ## 1. Train a linear head on cached mean hidden states

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

scorer_train_feats = torch.load(FEAT_DIR / "feat_train.pt")
scorer_val_feats = torch.load(FEAT_DIR / "feat_val.pt")
scorer_train_labels = torch.load(FEAT_DIR / "labels_train.pt")
scorer_val_labels = torch.load(FEAT_DIR / "labels_val.pt")

# label 0 = negative, 1 = positive; no bias, like the HF classification head
score = torch.nn.Linear(scorer_train_feats.shape[1], 2, bias=False)

ds = TensorDataset(scorer_train_feats, scorer_train_labels)
dl = DataLoader(ds, batch_size=1024, shuffle=True)
loss_fn = torch.nn.CrossEntropyLoss()
opt = torch.optim.AdamW(score.parameters(), lr=5e-4, weight_decay=0.1)
best_val_loss = float("inf")
patience = 100
epochs_val_plateau = 0

for epoch in range(1000):
    with torch.inference_mode():
        val_logits = score(scorer_val_feats)
        val_loss = loss_fn(val_logits, scorer_val_labels)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_score_state = {k: v.clone() for k, v in score.state_dict().items()}
            epochs_val_plateau = 0
        else:
            epochs_val_plateau +=1
            if epochs_val_plateau == patience:
                break

        if epoch % 10 == 0:
            print(
                compute_metrics((val_logits.numpy(), scorer_val_labels.numpy())), val_loss.item()
            )

    for b_features, b_labels in dl:
        out = score(b_features)
        loss = loss_fn(out, b_labels)
        loss.backward()
        opt.step()
        opt.zero_grad()

# keep the head with the lowest validation loss, not the last one
score.load_state_dict(best_score_state)
torch.save(best_score_state, FEAT_DIR / "score_head.pt")
print(f"best val loss {best_val_loss:.4f}")

# ## 2. Score hard-token generations as in the loop
# token embeddings E[ids] (= one-hot Y @ E) -> head reads the mean state over the response

device = "cpu"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(device).eval()
E = model.get_input_embeddings().weight
STOP = torch.tensor(tok.convert_tokens_to_ids(STOP_TOKENS))

with torch.inference_mode():
    for ins in INSTRUCTIONS:
        prompt_ids = tok.apply_chat_template(
            [{"role": "user", "content": f"{BASE} {ins}"}],
             add_generation_prompt=True,
             return_tensors="pt"
            )["input_ids"].to(device)
        s_all = []
        for seed in range(5):
            torch.manual_seed(seed)
            # plain T=1 sampling, as in the loop (override Qwen's generation_config defaults)
            ids = model.generate(prompt_ids, do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
                                 repetition_penalty=1.0, max_new_tokens=32,
                                 pad_token_id=tok.eos_token_id)[0, prompt_ids.shape[1]:]
            ids = ids[~torch.isin(ids, STOP)]
            # E[ids] == one_hot(ids) @ E
            h = model.model(inputs_embeds=E[ids][None]).last_hidden_state[0].mean(0)
            logits = score(h)
            s = logits[1] - logits[0]  # log-odds of positive
            s_all.append(s.item())
            print(f"{s.item():+.2f}  {tok.decode(ids)!r}")
        print(f"== {ins}  mean {np.mean(s_all):+.2f}  std {np.std(s_all):.2f}\n")
