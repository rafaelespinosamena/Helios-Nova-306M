#!/usr/bin/env python3
"""
Helios Nova  ·  Interactive Chat / Completion
==============================================
Downloads the model and tokenizer from HuggingFace Hub, then lets you
prompt the model interactively with streaming output.

Usage
-----
    python chat.py
    python chat.py --temperature 0.5 --top-k 20
    python chat.py --max-tokens 1024 --no-stream

Controls
--------
    Type any prompt and press Enter to see the completion.
    Type "!temp 0.5"   to change temperature on the fly.
    Type "!topk 30"    to change top-k on the fly.
    Type "!max 512"    to change max generation length.
    Type "!stream"     to toggle streaming (token-by-token) output.
    Type "quit" or "exit" or Ctrl+C to leave.
"""

from __future__ import annotations

import argparse
import os
import sys


import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from HeliosNova import HeliosNova, HeliosNovaConfig

# ── Default HuggingFace repo ────────────────────────────────────────────────
DEFAULT_REPO = "respinosamena/Helios-Nova"


def load_model(repo_id: str, device: torch.device) -> tuple[HeliosNova, AutoTokenizer]:
    """Download (or use cached) model + tokenizer from HuggingFace Hub."""
    print(f"Loading tokenizer from {repo_id} …")
    tokenizer = AutoTokenizer.from_pretrained(repo_id)

    print(f"Loading model from {repo_id} …")
    model = HeliosNova.from_pretrained(repo_id, device=str(device))
    model = model.to(device).eval()

    n = model.param_count()
    cfg = model.cfg
    print(f"  Model:    {n:,} params ({n / 1e6:.1f}M)")
    print(f"  Layers:   {cfg.n_layers}")
    print(f"  Context:  {cfg.max_seq_len} tokens")
    print(f"  Vocab:    {cfg.vocab_size}")
    print(f"  GQA:      {cfg.n_heads}q / {cfg.n_kv_heads}kv")
    print(f"  QK-Norm:  {cfg.qk_norm}")

    # Load training metadata if available
    try:
        from huggingface_hub import hf_hub_download
        import json
        meta_path = hf_hub_download(repo_id, "training_state.json")
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"  Step:     {meta.get('step', '?')}")
        print(f"  Val loss: {meta.get('val_loss', '?')}")
        tokens = meta.get("tokens_seen", "?")
        if isinstance(tokens, (int, float)):
            tokens = f"{tokens / 1e9:.2f}B"
        print(f"  Tokens:   {tokens}")
    except Exception:
        pass  # metadata file may not exist yet

    return model, tokenizer


@torch.no_grad()
def generate_streaming(
    model: HeliosNova,
    input_ids: torch.Tensor,
    tokenizer,
    max_new: int,
    temperature: float,
    top_k: int,
    repetition_penalty: float = 1.0,
    stream: bool = True,
) -> str:
    """Generate tokens one at a time, optionally printing as they arrive."""
    generated = []
    for _ in range(max_new):
        ctx = input_ids[:, -model.cfg.max_seq_len:]
        logits, _ = model(ctx)
        logits = logits[:, -1, :] / max(temperature, 1e-8)

        if repetition_penalty != 1.0:
            for token_id in set(input_ids[0].tolist()):
                if logits[0, token_id] < 0:
                    logits[0, token_id] *= repetition_penalty
                else:
                    logits[0, token_id] /= repetition_penalty

        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1)
        input_ids = torch.cat([input_ids, next_id], dim=1)

        token_id = next_id.item()

        # Stop on EOS
        if token_id == tokenizer.eos_token_id:
            break

        # Skip BOS / PAD in output
        if token_id in (tokenizer.bos_token_id, tokenizer.pad_token_id):
            continue

        # Decode this token
        tok = tokenizer.decode([token_id], skip_special_tokens=False)
        generated.append(tok)
        if stream:
            print(tok, end="", flush=True)

    if stream:
        print()  # newline after streaming

    return "".join(generated)


def pick_device() -> torch.device:
    """Best available device: CUDA → MPS (Apple Silicon) → CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description="Helios Nova interactive completion")
    parser.add_argument("--repo", type=str, default=DEFAULT_REPO,
                        help="HuggingFace repo ID (default: %(default)s)")
    parser.add_argument("--max-tokens", "-m", type=int, default=256,
                        help="Maximum tokens to generate per prompt")
    parser.add_argument("--temperature", "-t", type=float, default=0.5,
                        help="Sampling temperature (lower = more deterministic)")
    parser.add_argument("--top-k", "-k", type=int, default=50,
                        help="Top-k sampling (0 = disabled)")
    parser.add_argument("--no-stream", action="store_true",
                        help="Disable streaming output (print all at once)")
    parser.add_argument("--repetition-penalty", "-r", type=float, default=1.2,
                        help="Repetition penalty > 1.0 discourages repeated tokens (default: 1.2)")
    args = parser.parse_args()

    device = pick_device()
    print(f"Device: {device}\n")

    model, tokenizer = load_model(args.repo, device)

    # Mutable settings
    temperature = args.temperature
    top_k = args.top_k
    max_tokens = args.max_tokens
    repetition_penalty = args.repetition_penalty
    stream = not args.no_stream

    print(f"\n{'─' * 60}")
    print(f"  temperature={temperature}  top_k={top_k}  max_tokens={max_tokens}  rep_penalty={repetition_penalty}  stream={stream}")
    print(f"  Commands: !temp N  !topk N  !max N  !rep N  !stream  quit/exit")
    print(f"{'─' * 60}\n")

    while True:
        try:
            prompt = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not prompt:
            continue

        # ── Commands ─────────────────────────────────────────────────
        if prompt.lower() in ("quit", "exit"):
            print("Bye!")
            break
        if prompt.startswith("!temp "):
            try:
                temperature = float(prompt.split()[1])
                print(f"  → temperature={temperature}")
            except (IndexError, ValueError):
                print("  Usage: !temp 0.5")
            continue
        if prompt.startswith("!topk "):
            try:
                top_k = int(prompt.split()[1])
                print(f"  → top_k={top_k}")
            except (IndexError, ValueError):
                print("  Usage: !topk 30")
            continue
        if prompt.startswith("!max "):
            try:
                max_tokens = int(prompt.split()[1])
                print(f"  → max_tokens={max_tokens}")
            except (IndexError, ValueError):
                print("  Usage: !max 512")
            continue
        if prompt.startswith("!rep "):
            try:
                repetition_penalty = float(prompt.split()[1])
                print(f"  → repetition_penalty={repetition_penalty}")
            except (IndexError, ValueError):
                print("  Usage: !rep 1.2")
            continue
        if prompt == "!stream":
            stream = not stream
            print(f"  → stream={stream}")
            continue

        # ── Tokenise & generate ──────────────────────────────────────
        ids = [tokenizer.bos_token_id] + tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)

        remaining = model.cfg.max_seq_len - len(ids)
        gen_len = min(max_tokens, remaining)

        if gen_len <= 0:
            print("  (prompt fills entire context window, nothing to generate)")
            continue

        print(f"\nHelios Nova: ", end="", flush=True)
        if not stream:
            text = generate_streaming(
                model, input_ids, tokenizer, gen_len,
                temperature, top_k, repetition_penalty, stream=False,
            )
            print(text)
        else:
            generate_streaming(
                model, input_ids, tokenizer, gen_len,
                temperature, top_k, repetition_penalty, stream=True,
            )
        print()


if __name__ == "__main__":
    main()
