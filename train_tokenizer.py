"""
BPE Tokenizer Trainer — 16k vocab, FineWeb Edu 10BT, ~300k docs
Pushes to respinosamena/Helios-Nova-306M as a public HuggingFace repo.

Usage:
    huggingface-cli login        # once, to authenticate
    python train_tokenizer.py
"""

import os
import sys
from itertools import islice

from datasets import load_dataset
from tokenizers import (
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
)
from tokenizers.models import BPE
from transformers import PreTrainedTokenizerFast
from huggingface_hub import HfApi

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────
REPO_ID       = "respinosamena/Helios-Nova-306M"
VOCAB_SIZE    = 16_000
NUM_DOCS      = 1_000_000
BATCH_SIZE    = 1_000        # docs fed to trainer at a time
DATASET_NAME  = "HuggingFaceFW/fineweb-edu"
DATASET_CFG   = "sample-100BT"
DATASET_SPLIT = "train"
SAVE_DIR      = "./helios_tokenizer"

SPECIAL_TOKENS = [
    "<|endoftext|>",   # EOS / PAD (GPT-style)
    "<|pad|>",
    "<|unk|>",
    "<|bos|>",
    "<|eos|>",
    "<|sep|>",
    "<|mask|>",
    "<|user|>",
    "<|assistant|>",
    "<|system|>",
]


# ──────────────────────────────────────────────────────────────
# 1.  Stream text from FineWeb-Edu
# ──────────────────────────────────────────────────────────────
def stream_texts(num_docs: int, batch_size: int):
    """Yield batches of raw text strings from the dataset."""
    print(f"[data]  Streaming {num_docs:,} documents from {DATASET_NAME} ({DATASET_CFG}) …")
    ds = load_dataset(
        DATASET_NAME,
        name=DATASET_CFG,
        split=DATASET_SPLIT,
        streaming=True,
        trust_remote_code=True,
    )

    batch: list[str] = []
    seen = 0
    for example in islice(ds, num_docs):
        text = example.get("text", "")
        if not text:
            continue
        batch.append(text)
        seen += 1
        if len(batch) >= batch_size:
            yield batch
            batch = []
            print(f"  … {seen:,} / {num_docs:,} docs streamed", end="\r", flush=True)

    if batch:
        yield batch
    print(f"\n[data]  Done — {seen:,} documents collected.")


# ──────────────────────────────────────────────────────────────
# 2.  Build & train the tokenizer
# ──────────────────────────────────────────────────────────────
def build_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))

    # NFKC normalisation — handles Unicode compatibility forms
    tokenizer.normalizer = normalizers.NFKC()

    # ByteLevel pre-tokeniser (like GPT-2/RoBERTa): no OOV chars possible
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # ByteLevel decoder to reconstruct strings properly
    tokenizer.decoder = decoders.ByteLevel()

    return tokenizer


def train(tokenizer: Tokenizer) -> Tokenizer:
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        show_progress=True,
        # Aim for a reasonably expressive merge limit
        max_token_length=16,
    )

    print(f"\n[train] Starting BPE training  (vocab_size={VOCAB_SIZE:,}) …")
    tokenizer.train_from_iterator(
        stream_texts(NUM_DOCS, BATCH_SIZE),
        trainer=trainer,
        length=NUM_DOCS,
    )
    print("[train] Training complete.")
    return tokenizer


# ──────────────────────────────────────────────────────────────
# 3.  Wrap in PreTrainedTokenizerFast (full HF-compatible)
# ──────────────────────────────────────────────────────────────
def wrap_fast(tokenizer: Tokenizer) -> PreTrainedTokenizerFast:
    # Post-processor: append EOS after every sequence (optional but useful)
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="$A <|endoftext|>",
        pair="$A <|endoftext|> $B:1 <|endoftext|>:1",
        special_tokens=[("<|endoftext|>", eos_id)],
    )

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|bos|>",
        eos_token="<|endoftext|>",
        unk_token="<|unk|>",
        pad_token="<|pad|>",
        sep_token="<|sep|>",
        mask_token="<|mask|>",
        model_max_length=4096,
        # Ensure clean spaces are handled consistently
        add_prefix_space=False,
        trim_offsets=True,
    )
    return fast


# ──────────────────────────────────────────────────────────────
# 4.  Save locally
# ──────────────────────────────────────────────────────────────
def save_local(fast: PreTrainedTokenizerFast, directory: str):
    os.makedirs(directory, exist_ok=True)
    fast.save_pretrained(directory)
    print(f"[save]  Tokenizer saved to  {directory}/")

    # Quick sanity check
    sample = "Hello, world! This is a test of the Helios tokenizer. 42 + 7 = 49."
    ids = fast.encode(sample)
    decoded = fast.decode(ids)
    print(f"[check] Encoded {len(ids)} tokens  →  '{decoded[:80]}'")


# ──────────────────────────────────────────────────────────────
# 5.  Push to HuggingFace Hub
# ──────────────────────────────────────────────────────────────
def push_to_hub(fast: PreTrainedTokenizerFast, repo_id: str):
    print(f"\n[hub]   Pushing to  {repo_id}  …")

    # Create the repo (no-op if it already exists)
    api = HfApi()
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=False,
        exist_ok=True,
    )

    fast.push_to_hub(
        repo_id,
        commit_message="Add BPE tokenizer (16k vocab, trained on FineWeb-Edu 10BT)",
        private=False,
    )

    print(f"[hub]   ✓ Tokenizer live at  https://huggingface.co/{repo_id}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    # Check HF token early
    token = "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"
    if not token:
        print(
            "[auth]  No HF_TOKEN env var found.\n"
            "        Run  `huggingface-cli login`  or set  HF_TOKEN=<your_token>  "
            "before running this script.",
            file=sys.stderr,
        )
        # Don't exit — local training still works; push will just fail later.

    tok = build_tokenizer()
    tok = train(tok)
    fast = wrap_fast(tok)
    save_local(fast, SAVE_DIR)
    push_to_hub(fast, REPO_ID)

    print("\n[done]  All finished.")


if __name__ == "__main__":
    main()
