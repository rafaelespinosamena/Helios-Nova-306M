"""
Helios Nova  ·  Training Script
================================
Professional training loop for a 306M dense LM on FineWeb-Edu.
Streams training data, pre-collects validation, logs to W&B, uploads
best checkpoints directly to HuggingFace Hub.

Key features
------------
• BPE tokenizer loaded from HuggingFace (`respinosamena/Helios-Nova`).
• Streaming data pipeline — no local copy of FineWeb-Edu required.
• Full validation pass every `eval.interval` optimiser steps.
• Best-checkpoint-only: saved to HF Hub, not stored locally (saves disk).
• WSD (Warmup-Stable-Decay) LR schedule — keeps LR high for most of
  training, then cosine-decays only in the final `decay_fraction` of steps.
  Outperforms cosine on long overtraining runs (MiniCPM, FineWeb ablations).
• torch.compile on CUDA for 10-30 % wall-time speedup.
• Mixed-precision bf16 on H100 via PyTorch.

Usage
-----
    python train.py                  # default config.yaml
    python train.py --config my.yaml
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import torch
import yaml

# ── Silence noisy loggers before any HF import ──────────────────────────────
for _n in ("datasets", "transformers", "urllib3", "filelock", "fsspec",
           "huggingface_hub", "huggingface_hub.utils"):
    logging.getLogger(_n).setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"

import wandb
from datasets import load_dataset
from huggingface_hub import HfApi
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm
from transformers import AutoTokenizer

from HeliosNova import HeliosNova, HeliosNovaConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_config(path: str) -> dict:
    """Load YAML configuration file."""
    with open(path) as f:
        return yaml.safe_load(f)


def select_device_dtype(cfg_dtype: str):
    """Pick the best available device and a sensible mixed-precision dtype."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        if cfg_dtype == "auto":
            dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dt = getattr(torch, cfg_dtype)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
        dt = torch.float32 if cfg_dtype == "auto" else getattr(torch, cfg_dtype)
    else:
        dev = torch.device("cpu")
        dt = torch.float32
    return dev, dt


def wsd_lr(
    step: int,
    warmup: int,
    peak: float,
    floor: float,
    total: int,
    decay_fraction: float = 0.10,
) -> float:
    """
    Warmup-Stable-Decay (WSD) schedule.

    Three phases:
      1. Warmup  : linear  0 → peak  over the first `warmup` steps.
      2. Stable  : hold    peak      from step `warmup` to `total - decay_steps`.
      3. Decay   : cosine  peak → floor over the last `decay_steps` steps,
                   where decay_steps = int(total * decay_fraction).

    Why WSD over cosine for long runs
    ----------------------------------
    A pure cosine schedule starts decaying immediately after warmup, so the
    model spends the majority of a 127k-step run at near-floor LR and learns
    very little.  WSD keeps LR at its peak for ~90% of training, then anneals
    sharply at the end.  This consistently outperforms cosine at long
    overtraining runs (MiniCPM, FineWeb ablations).
    """
    decay_steps = max(1, int(total * decay_fraction))
    stable_end  = total - decay_steps

    if step < warmup:                        # ── Phase 1: warmup
        return peak * (step + 1) / warmup
    if step < stable_end:                    # ── Phase 2: stable
        return peak
    if step >= total:                        # ── past end
        return floor
    # ── Phase 3: cosine decay
    t = (step - stable_end) / max(1, decay_steps)
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * t))


# ═══════════════════════════════════════════════════════════════════════════════
#  Datasets
# ═══════════════════════════════════════════════════════════════════════════════

class StreamingTokenDataset(IterableDataset):
    """
    Tokenise and pack documents into fixed-length chunks on the fly.

    Each document becomes:  [BOS] + token_ids + [EOS]
    Documents are concatenated into a continuous stream, then sliced into
    (seq_len + 1) chunks for next-token prediction.  No padding needed.
    """

    def __init__(self, hf_stream, tokenizer, seq_len: int):
        self.hf_stream = hf_stream
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.bos = tokenizer.bos_token_id
        self.eos = tokenizer.eos_token_id

    def __iter__(self):
        buf: list[int] = []
        chunk = self.seq_len + 1          # +1 so we can split into (input, target)
        for ex in self.hf_stream:
            ids = self.tokenizer.encode(ex["text"], add_special_tokens=False)
            buf.extend([self.bos] + ids + [self.eos])
            while len(buf) >= chunk:
                yield torch.tensor(buf[:chunk], dtype=torch.long)
                buf = buf[chunk:]


class TokenizedValDataset(Dataset):
    """Pre-tokenised, packed validation set held entirely in memory."""

    def __init__(self, docs: list[str], tokenizer, seq_len: int):
        self.chunks: list[torch.Tensor] = []
        buf: list[int] = []
        chunk = seq_len + 1
        bos, eos = tokenizer.bos_token_id, tokenizer.eos_token_id
        for text in docs:
            ids = tokenizer.encode(text, add_special_tokens=False)
            buf.extend([bos] + ids + [eos])
            while len(buf) >= chunk:
                self.chunks.append(torch.tensor(buf[:chunk], dtype=torch.long))
                buf = buf[chunk:]

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.chunks[idx]


# ═══════════════════════════════════════════════════════════════════════════════
#  Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, loader, device, dtype) -> tuple[float, float]:
    """Run the *full* validation set; returns (avg_loss, perplexity)."""
    model.eval()
    total_loss, total_tok = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        x, y = batch[:, :-1], batch[:, 1:]
        with torch.amp.autocast(device_type=device.type, dtype=dtype,
                                enabled=(dtype != torch.float32)):
            _, loss = model(x, targets=y)
        total_loss += loss.item() * y.numel()
        total_tok  += y.numel()
    model.train()
    avg = total_loss / max(total_tok, 1)
    return avg, math.exp(min(avg, 20.0))


# ═══════════════════════════════════════════════════════════════════════════════
#  HuggingFace Hub upload
# ═══════════════════════════════════════════════════════════════════════════════

def upload_checkpoint_to_hub(
    raw_model: HeliosNova,
    repo_id: str,
    step: int,
    val_loss: float,
    tokens_seen: int,
    hf_token: str,
) -> None:
    """
    Save the model in HF-compatible format (config.json + model.safetensors)
    to a temporary directory, upload to HuggingFace Hub, then delete locally.
    """
    api = HfApi(token=hf_token)

    # Ensure the repo exists (creates if needed, no-ops if it does)
    api.create_repo(repo_id=repo_id, exist_ok=True, private=False)

    with tempfile.TemporaryDirectory(prefix="helios_nova_ckpt_") as tmp:
        # Save model in HuggingFace-compatible format
        raw_model.save_pretrained(tmp)

        # Also write a small training-state metadata file
        import json
        meta = {
            "step": step,
            "val_loss": round(val_loss, 6),
            "tokens_seen": tokens_seen,
            "param_count": raw_model.param_count(),
        }
        with open(os.path.join(tmp, "training_state.json"), "w") as f:
            json.dump(meta, f, indent=2)

        # Upload the entire directory
        api.upload_folder(
            folder_path=tmp,
            repo_id=repo_id,
            commit_message=f"step {step}  |  val_loss={val_loss:.4f}",
        )
    # tmp is automatically deleted when the context manager exits


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    C = load_config(args.config)
    mc = C["model"]        # model config dict
    tc = C["training"]     # training config dict
    ec = C["eval"]         # eval config dict
    dc = C["data"]         # data config dict
    wc = C["wandb"]        # wandb config dict
    hc = C["hub"]          # huggingface hub config dict

    # ── Environment tokens ───────────────────────────────────────────────
    if hc.get("token"):
        os.environ["HF_TOKEN"] = hc["token"]
    hf_token = hc.get("token") or os.environ.get("HF_TOKEN", "")

    # ── Seed ─────────────────────────────────────────────────────────────
    torch.manual_seed(tc["seed"])

    # ── Device / dtype ───────────────────────────────────────────────────
    device, dtype = select_device_dtype(tc["dtype"])
    use_amp = dtype != torch.float32
    print(f"▸ device={device}  dtype={dtype}  amp={use_amp}")

    # ── Tokenizer (BPE, from HuggingFace Hub) ────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(hc["repo_id"], token=hf_token)
    mc["vocab_size"] = tokenizer.vocab_size
    print(f"▸ tokenizer: {hc['repo_id']}  vocab={tokenizer.vocab_size}")

    # ── Model ────────────────────────────────────────────────────────────
    cfg = HeliosNovaConfig.from_dict(mc)
    model = HeliosNova(cfg).to(device)
    n_params = model.param_count()
    print(f"▸ Helios Nova  {n_params:,} params ({n_params / 1e6:.1f}M)")
    print(f"  {cfg.n_layers} layers  |  d={cfg.d_model}  |  GQA {cfg.n_heads}q/{cfg.n_kv_heads}kv")
    print(f"  FFN={cfg.ffn_dim}  |  head_dim={cfg.head_dim}  |  QK-Norm={cfg.qk_norm}")

    # ── torch.compile (CUDA only; ~15-25 % speedup on A100) ─────────────
    raw_model = model                     # keep uncompiled ref for state_dict
    compiled = False
    if tc.get("compile", False) and device.type == "cuda":
        print("▸ torch.compile …")
        model = torch.compile(model)
        compiled = True
    elif tc.get("compile", False):
        print("▸ torch.compile skipped (CUDA only)")

    # ── W&B ──────────────────────────────────────────────────────────────
    if wc.get("token"):
        os.environ["WANDB_API_KEY"] = wc["token"]
    os.environ["WANDB_SILENT"] = "true"
    wandb.init(
        project=wc["project"],
        entity=wc.get("entity"),
        name=wc.get("run_name"),
        config={
            "model": mc, "training": tc, "eval": ec, "data": dc,
            "n_params": n_params, "compiled": compiled,
        },
    )

    # ── Load dataset (FineWeb-Edu, streamed) ─────────────────────────────
    ds_name  = dc["dataset"]
    ds_subset = dc.get("subset", None)
    ds_kwargs = {"path": ds_name, "streaming": True}
    if ds_subset:
        ds_kwargs["name"] = ds_subset

    has_val_split = dc.get("has_val_split", True)

    # ── Validation data (pre-collected into memory) ──────────────────────
    print("▸ Collecting validation set …", end=" ", flush=True)
    if has_val_split:
        val_raw = load_dataset(**ds_kwargs, split="validation")
    else:
        val_raw = load_dataset(**ds_kwargs, split="train")

    val_texts: list[str] = []
    for i, ex in enumerate(val_raw):
        if i >= dc["val_samples"]:
            break
        val_texts.append(ex["text"])
    val_ds = TokenizedValDataset(val_texts, tokenizer, mc["max_seq_len"])
    val_loader = DataLoader(val_ds, batch_size=tc["batch_size"],
                            shuffle=False, drop_last=True)
    print(f"{len(val_texts)} docs → {len(val_ds)} chunks")

    # ── Training data (streamed, skip val docs if carved from train) ─────
    print("▸ Streaming training data")
    train_raw = load_dataset(**ds_kwargs, split="train")
    if not has_val_split:
        train_raw = train_raw.skip(dc["val_samples"])
    train_raw = train_raw.shuffle(seed=tc["seed"], buffer_size=10_000)
    train_ds = StreamingTokenDataset(train_raw, tokenizer, mc["max_seq_len"])
    train_loader = DataLoader(
        train_ds,
        batch_size=tc["batch_size"],
        num_workers=dc.get("num_workers", 0),
        pin_memory=(device.type == "cuda"),
    )

    # ── Optimiser (AdamW with decoupled weight decay) ────────────────────
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Don't decay 1-D params (norms, biases) or embedding weights
        if p.dim() < 2 or "norm" in name or "emb" in name:
            no_decay.append(p)
        else:
            decay.append(p)

    optimizer = torch.optim.AdamW(
        [{"params": decay,    "weight_decay": tc["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=tc["learning_rate"],
        betas=(tc["beta1"], tc["beta2"]),
        fused=(device.type == "cuda"),
    )
    scaler = torch.amp.GradScaler(device.type,
                                  enabled=(use_amp and device.type == "cuda"))

    # ── Estimate total optimiser steps ───────────────────────────────────
    est_rows       = dc.get("est_num_rows", 9_660_000)
    est_avg_tokens = dc.get("est_avg_tokens", 1050)
    seq_len        = mc["max_seq_len"]
    est_total_tokens = est_rows * (est_avg_tokens + 2)    # +2 for BOS/EOS
    tok_per_step = tc["batch_size"] * tc["grad_accum_steps"] * seq_len
    est_steps = est_total_tokens // tok_per_step
    est_total_steps = est_steps * tc["num_epochs"]
    print(f"▸ ~{est_total_steps:,} optimiser steps for {tc['num_epochs']} epoch(s)")
    print(f"  effective batch = {tc['batch_size']}×{tc['grad_accum_steps']}×{seq_len}"
          f" = {tok_per_step:,} tokens/step")

    # ── Training state ───────────────────────────────────────────────────
    best_val_loss = float("inf")
    global_step   = 0
    tokens_seen   = 0
    running_loss  = 0.0
    running_count = 0

    model.train()
    optimizer.zero_grad(set_to_none=True)
    t0 = time.time()

    # ══════════════════════════════════════════════════════════════════════
    #  Training loop
    # ══════════════════════════════════════════════════════════════════════
    for epoch in range(tc["num_epochs"]):

        pbar = tqdm(
            total=est_steps,
            desc=f"Epoch {epoch + 1}/{tc['num_epochs']}",
            unit="step",
            dynamic_ncols=True,
            smoothing=0.05,
            bar_format=("{l_bar}{bar}| {n_fmt}/{total_fmt} steps "
                        "[{elapsed}<{remaining}, {rate_fmt}]{postfix}"),
        )

        micro = 0

        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            x, y = batch[:, :-1], batch[:, 1:]

            with torch.amp.autocast(device_type=device.type, dtype=dtype,
                                    enabled=use_amp):
                _, loss = model(x, targets=y)
                loss_scaled = loss / tc["grad_accum_steps"]

            scaler.scale(loss_scaled).backward()
            running_loss  += loss.item()
            running_count += 1
            tokens_seen   += y.numel()
            micro += 1

            # ── Accumulation boundary → optimiser step ───────────────
            if micro % tc["grad_accum_steps"] != 0:
                continue

            global_step += 1

            # Unscale → clip → get grad norm
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), tc["max_grad_norm"],
            ).item()

            # WSD LR schedule
            lr = wsd_lr(global_step, tc["warmup_steps"],
                        tc["learning_rate"], tc["min_lr"], est_total_steps,
                        tc.get("decay_fraction", 0.10))
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # ── tqdm update ──────────────────────────────────────────
            avg_loss = running_loss / max(running_count, 1)
            pbar.update(1)
            pbar.set_postfix_str(
                f"loss={avg_loss:.3f}  lr={lr:.1e}  gnorm={grad_norm:.2f}  "
                f"tok={tokens_seen / 1e9:.2f}B",
                refresh=False,
            )

            # ── Log to W&B ───────────────────────────────────────────
            if global_step % ec["log_interval"] == 0:
                ppl = math.exp(min(avg_loss, 20.0))
                wandb.log({
                    "train/loss":       avg_loss,
                    "train/perplexity": ppl,
                    "train/lr":         lr,
                    "train/grad_norm":  grad_norm,
                    "train/tokens":     tokens_seen,
                }, step=global_step)
                running_loss, running_count = 0.0, 0

            # ── Evaluation (full val set, no generation) ─────────────
            if global_step % ec["interval"] == 0:
                val_loss, val_ppl = evaluate(model, val_loader, device, dtype)

                wandb.log({
                    "eval/loss":       val_loss,
                    "eval/perplexity": val_ppl,
                }, step=global_step)

                # ── Upload to HF Hub if improved ─────────────────────
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    tqdm.write(
                        f"  ▸ step {global_step}  val_loss={val_loss:.4f}  "
                        f"val_ppl={val_ppl:.2f}  ✓ uploading to Hub …"
                    )
                    upload_checkpoint_to_hub(
                        raw_model=raw_model,
                        repo_id=hc["repo_id"],
                        step=global_step,
                        val_loss=val_loss,
                        tokens_seen=tokens_seen,
                        hf_token=hf_token,
                    )
                    tqdm.write(
                        f"    → uploaded to {hc['repo_id']}"
                    )
                else:
                    tqdm.write(
                        f"  ▸ step {global_step}  val_loss={val_loss:.4f}  "
                        f"val_ppl={val_ppl:.2f}  ✗ best={best_val_loss:.4f}"
                    )

        pbar.close()

    # ══════════════════════════════════════════════════════════════════════
    #  Final evaluation & upload
    # ══════════════════════════════════════════════════════════════════════
    val_loss, val_ppl = evaluate(model, val_loader, device, dtype)
    wandb.log({"eval/loss": val_loss, "eval/perplexity": val_ppl},
              step=global_step)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        print(f"\n▸ Final checkpoint improved — uploading to Hub …")
        upload_checkpoint_to_hub(
            raw_model=raw_model,
            repo_id=hc["repo_id"],
            step=global_step,
            val_loss=val_loss,
            tokens_seen=tokens_seen,
            hf_token=hf_token,
        )

    elapsed = time.time() - t0
    print(f"\n{'═' * 60}")
    print(f"  Done  |  best_val_loss={best_val_loss:.4f}  |  steps={global_step}")
    print(f"  Time: {elapsed / 3600:.1f}h  |  Tokens: {tokens_seen / 1e9:.2f}B")
    print(f"  Model: {hc['repo_id']}")
    print(f"{'═' * 60}")
    wandb.finish()


if __name__ == "__main__":
    main()
