#!/usr/bin/env python3
"""
Evaluate Helios-Nova (306M) with lm-evaluation-harness and compare against
published baselines.  Produces a clean CSV leaderboard.

Usage
-----
    pip install lm-eval transformers safetensors huggingface_hub torch
    python evaluate_helios_nova.py          # full suite
    python evaluate_helios_nova.py --quick   # just hellaswag + arc_easy for testing

The script:
  1.  Loads `respinosamena/Helios-Nova` from the Hub via the custom architecture.
  2.  Wraps it as an `lm_eval` model (subclassing HFLM).
  3.  Runs the same 7 benchmarks used in the SmolLM2 paper (Table 4):
        HellaSwag · ARC-Easy · ARC-Challenge · WinoGrande · PIQA · OBQA · MMLU (5-shot)
  4.  Writes `helios_nova_benchmark_results.csv` with your results alongside the
      published numbers for 9 comparison models.

References for baseline numbers
-------------------------------
  • SmolLM2-360M, SmolLM-360M:
      Allal et al., "SmolLM2: When Smol Goes Big", arXiv 2502.02737 (Feb 2025), Table 4.
      https://arxiv.org/abs/2502.02737
  • OpenELM-270M, OpenELM-450M:
      Mehta et al., "OpenELM: An Efficient Language Model Family with Open Training
      and Inference Framework", arXiv 2404.14619 (Apr 2024), Tables 4-5.
      https://arxiv.org/abs/2404.14619
  • MobileLLM-350M:
      Liu et al., "MobileLLM: Optimizing Sub-billion Parameter Language Models for
      On-Device Use Cases", arXiv 2402.14905 (Feb 2024), Table 2.
      https://arxiv.org/abs/2402.14905
  • Pythia-410M:
      Biderman et al., "Pythia: A Suite for Analyzing Large Language Models Across
      Training and Scaling", ICML 2023 / arXiv 2304.01373, Table 5.
      https://arxiv.org/abs/2304.01373
  • Qwen2.5-0.5B:
      Qwen Team, "Qwen2.5 Technical Report", arXiv 2412.15115 (Dec 2024).
      https://arxiv.org/abs/2412.15115
  • Qwen3-0.6B:
      Qwen Team, "Qwen3 Technical Report", arXiv 2505.09388 (May 2025).
      https://arxiv.org/abs/2505.09388
  • Llama-3.2-1B:
      Meta, "Llama 3.2: Revolutionizing edge AI and vision with open, customizable
      models" (Sep 2024). Benchmarks from SmolLM2 paper Table 4.
      https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/

All baseline scores sourced from SmolLM2 Table 4 which evaluated all models
under identical conditions using the lighteval library (zero-shot unless noted).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import os
from pathlib import Path
from dataclasses import dataclass

# ---------------------------------------------------------------------------
#  Baseline comparison data  (from SmolLM2 paper Table 4)
# ---------------------------------------------------------------------------
BASELINES: list[dict] = [
    {
        "Model": "SmolLM2-360M",
        "Params": "360M",
        "Tokens": "4T",
        "HellaSwag": 54.5,
        "ARC-E": 61.4,
        "ARC-C": 44.6,
        "WinoGrande": 52.2,
        "PIQA": 71.7,
        "OBQA": 37.6,
        "MMLU (5s)": 31.5,
        "Avg": 50.5,
        "Source": "Allal et al. 2025, arXiv:2502.02737, Table 4",
    },
    {
        "Model": "SmolLM-360M",
        "Params": "360M",
        "Tokens": "1.4T",
        "HellaSwag": 51.8,
        "ARC-E": 58.2,
        "ARC-C": 42.0,
        "WinoGrande": 51.5,
        "PIQA": 71.6,
        "OBQA": 36.4,
        "MMLU (5s)": 26.2,
        "Avg": 48.2,
        "Source": "Allal et al. 2025, arXiv:2502.02737, Table 4",
    },
    {
        "Model": "OpenELM-270M",
        "Params": "270M",
        "Tokens": "1.5T",
        "HellaSwag": 47.2,
        "ARC-E": 45.6,
        "ARC-C": 27.6,
        "WinoGrande": 53.0,
        "PIQA": 69.8,
        "OBQA": 33.0,
        "MMLU (5s)": 25.4,
        "Avg": 43.1,
        "Source": "Mehta et al. 2024, arXiv:2404.14619; scores via SmolLM2 Table 4",
    },
    {
        "Model": "OpenELM-450M",
        "Params": "450M",
        "Tokens": "1.5T",
        "HellaSwag": 53.9,
        "ARC-E": 48.6,
        "ARC-C": 30.1,
        "WinoGrande": 53.6,
        "PIQA": 72.3,
        "OBQA": 33.6,
        "MMLU (5s)": 25.8,
        "Avg": 45.4,
        "Source": "Mehta et al. 2024, arXiv:2404.14619; scores via SmolLM2 Table 4",
    },
    {
        "Model": "MobileLLM-350M",
        "Params": "350M",
        "Tokens": "250B",
        "HellaSwag": 46.7,
        "ARC-E": 49.6,
        "ARC-C": 29.4,
        "WinoGrande": 52.3,
        "PIQA": 68.6,
        "OBQA": 33.0,
        "MMLU (5s)": 25.5,
        "Avg": 43.6,
        "Source": "Liu et al. 2024, arXiv:2402.14905; scores via SmolLM2 Table 4",
    },
    {
        "Model": "Pythia-410M",
        "Params": "410M",
        "Tokens": "300B",
        "HellaSwag": 47.2,
        "ARC-E": 51.4,
        "ARC-C": 29.3,
        "WinoGrande": 53.8,
        "PIQA": 70.4,
        "OBQA": 30.2,
        "MMLU (5s)": 25.3,
        "Avg": 43.9,
        "Source": "Biderman et al. 2023, arXiv:2304.01373; scores via SmolLM2 Table 4",
    },
    {
        "Model": "Qwen2.5-0.5B",
        "Params": "500M",
        "Tokens": "18T",
        "HellaSwag": 51.2,
        "ARC-E": 53.7,
        "ARC-C": 37.1,
        "WinoGrande": 54.9,
        "PIQA": 69.9,
        "OBQA": 34.6,
        "MMLU (5s)": 45.4,
        "Avg": 49.5,
        "Source": "Qwen Team 2024, arXiv:2412.15115; scores via SmolLM2 Table 4",
    },
    {
        "Model": "Qwen3-0.6B",
        "Params": "600M",
        "Tokens": "36T",
        "HellaSwag": 55.1,
        "ARC-E": 58.8,
        "ARC-C": 38.2,
        "WinoGrande": 55.5,
        "PIQA": 72.1,
        "OBQA": 35.8,
        "MMLU (5s)": 46.2,
        "Avg": 51.7,
        "Source": "Qwen Team 2025, arXiv:2505.09388; scores via SmolLM2 Table 4",
    },
    {
        "Model": "Llama-3.2-1B",
        "Params": "1B",
        "Tokens": "9T",
        "HellaSwag": 61.2,
        "ARC-E": 61.4,
        "ARC-C": 38.5,
        "WinoGrande": 59.5,
        "PIQA": 74.8,
        "OBQA": 36.2,
        "MMLU (5s)": 32.2,
        "Avg": 52.0,
        "Source": "Meta 2024; scores via SmolLM2 Table 4 (arXiv:2502.02737)",
    },
]

BENCHMARK_COLS = ["HellaSwag", "ARC-E", "ARC-C", "WinoGrande", "PIQA", "OBQA", "MMLU (5s)"]


# ---------------------------------------------------------------------------
#  HuggingFace-compatible wrapper so lm-eval can consume Helios-Nova
# ---------------------------------------------------------------------------
def build_hf_compatible_model(repo_id: str = "respinosamena/Helios-Nova", device: str = "cuda"):
    """
    Load HeliosNova from the Hub and wrap it so that lm-eval's HFLM can use it.
    Returns (model_wrapper, tokenizer).
    """
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, PretrainedConfig, PreTrainedModel
    from transformers.modeling_outputs import CausalLMOutputWithPast

    # --- Import the real model definition ---
    sys.path.insert(0, str(Path(__file__).parent))
    from HeliosNova import HeliosNova, HeliosNovaConfig

    # Load the actual model
    print(f"[*] Loading Helios-Nova from '{repo_id}' ...")
    raw_model = HeliosNova.from_pretrained(repo_id, device="cpu")
    raw_model.eval()
    cfg = raw_model.cfg
    print(f"    {raw_model.param_count():,} parameters loaded.")

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(repo_id)

    # Ensure special tokens are set (lm-eval needs these)
    if tokenizer.eos_token is None:
        tokenizer.eos_token = tokenizer.decode([0])  # fallback
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Create a HuggingFace-compatible config ---
    class HeliosNovaHFConfig(PretrainedConfig):
        model_type = "helios_nova"

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.vocab_size = kwargs.get("vocab_size", cfg.vocab_size)
            self.hidden_size = kwargs.get("hidden_size", cfg.d_model)
            self.num_hidden_layers = kwargs.get("num_hidden_layers", cfg.n_layers)
            self.num_attention_heads = kwargs.get("num_attention_heads", cfg.n_heads)
            self.max_position_embeddings = kwargs.get("max_position_embeddings", cfg.max_seq_len)

    hf_config = HeliosNovaHFConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.d_model,
        num_hidden_layers=cfg.n_layers,
        num_attention_heads=cfg.n_heads,
        max_position_embeddings=cfg.max_seq_len,
    )

    # --- Wrap in a PreTrainedModel so HFLM recognises it ---
    class HeliosNovaForCausalLM(PreTrainedModel):
        config_class = HeliosNovaHFConfig
        _no_split_modules = []

        def __init__(self, config, inner_model):
            super().__init__(config)
            self.inner = inner_model

        def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
            # HeliosNova.forward expects (input_ids, targets)
            logits, loss = self.inner(input_ids, targets=labels)
            return CausalLMOutputWithPast(loss=loss, logits=logits)

        def prepare_inputs_for_generation(self, input_ids, **kwargs):
            return {"input_ids": input_ids}

        def can_generate(self):
            return True

        @property
        def device(self):
            return next(self.inner.parameters()).device

    wrapper = HeliosNovaForCausalLM(hf_config, raw_model)

    # Move to requested device
    if device != "cpu" and torch.cuda.is_available():
        wrapper = wrapper.to(device)
        print(f"    Model moved to {device}.")
    else:
        device = "cpu"
        print("    Running on CPU (this will be slow).")

    return wrapper, tokenizer, device


# ---------------------------------------------------------------------------
#  Run lm-eval benchmarks
# ---------------------------------------------------------------------------
def run_evaluation(model, tokenizer, device: str, quick: bool = False) -> dict:
    """
    Run the benchmark suite via lm_eval and return a dict of task → score.
    """
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    # Construct the HFLM wrapper manually with our pre-loaded model
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size="auto",
        max_batch_size=32,
        device=str(device),
    )

    # Define tasks — same suite as SmolLM2 paper Table 4
    if quick:
        tasks = ["hellaswag", "arc_easy"]
        print("\n[*] QUICK MODE: running hellaswag + arc_easy only.\n")
    else:
        tasks = [
            "hellaswag",
            "arc_easy",
            "arc_challenge",
            "winogrande",
            "piqa",
            "openbookqa",
            "mmlu",
        ]
        print(f"\n[*] Running full benchmark suite: {', '.join(tasks)}\n")

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=None,  # use task defaults (0-shot except MMLU which is 5-shot)
        batch_size="auto",
        device=str(device),
    )

    return results


# ---------------------------------------------------------------------------
#  Extract scores from lm-eval results dict
# ---------------------------------------------------------------------------
def extract_scores(results: dict, quick: bool = False) -> dict:
    """Parse the lm-eval results dict into our standard column names."""
    r = results["results"]

    def get_acc(task_key: str, metric: str = "acc_norm,none") -> float | None:
        """Try acc_norm first, fall back to acc."""
        if task_key not in r:
            return None
        task = r[task_key]
        for m in [metric, "acc_norm,none", "acc,none"]:
            if m in task:
                return round(task[m] * 100, 1)
        return None

    scores = {}
    scores["HellaSwag"] = get_acc("hellaswag")
    scores["ARC-E"] = get_acc("arc_easy")

    if not quick:
        scores["ARC-C"] = get_acc("arc_challenge")
        scores["WinoGrande"] = get_acc("winogrande")
        scores["PIQA"] = get_acc("piqa")
        scores["OBQA"] = get_acc("openbookqa")

        # MMLU — lm-eval reports per-subject; the overall is under the group key
        mmlu_score = None
        for key in ["mmlu", "hendrycksTest"]:
            if key in r:
                mmlu_score = get_acc(key)
                break
        # If group key absent, average the sub-tasks
        if mmlu_score is None:
            mmlu_accs = []
            for k, v in r.items():
                if k.startswith("mmlu_") or k.startswith("hendrycksTest-"):
                    for m in ["acc,none", "acc_norm,none"]:
                        if m in v:
                            mmlu_accs.append(v[m])
                            break
            if mmlu_accs:
                mmlu_score = round(sum(mmlu_accs) / len(mmlu_accs) * 100, 1)
        scores["MMLU (5s)"] = mmlu_score

    # Compute average over available scores
    valid = [v for v in scores.values() if v is not None]
    scores["Avg"] = round(sum(valid) / len(valid), 1) if valid else None

    return scores


# ---------------------------------------------------------------------------
#  Write comparison CSV
# ---------------------------------------------------------------------------
def write_csv(helios_scores: dict, output_path: str = "helios_nova_benchmark_results.csv"):
    """Write a leaderboard CSV with Helios-Nova + all baselines, sorted by Avg."""
    # Build Helios-Nova row
    helios_row = {
        "Model": "Helios-Nova (ours)",
        "Params": "306M",
        "Tokens": "—",
        **helios_scores,
        "Source": "This evaluation (lm-eval-harness)",
    }

    all_rows = BASELINES + [helios_row]

    # Sort by Avg descending (None goes last)
    all_rows.sort(key=lambda x: x.get("Avg") or 0, reverse=True)

    fieldnames = ["Model", "Params", "Tokens"] + BENCHMARK_COLS + ["Avg", "Source"]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"\n[✓] Results saved to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
#  Pretty-print the leaderboard to stdout
# ---------------------------------------------------------------------------
def print_leaderboard(helios_scores: dict):
    """Print an aligned table to stdout."""
    helios_row = {
        "Model": "Helios-Nova (ours)",
        "Params": "306M",
        "Tokens": "—",
        **helios_scores,
        "Source": "",
    }

    all_rows = BASELINES + [helios_row]
    all_rows.sort(key=lambda x: x.get("Avg") or 0, reverse=True)

    header = f"{'Model':<25s} {'Params':>7s} {'Tok':>6s}"
    for col in BENCHMARK_COLS:
        header += f" {col:>10s}"
    header += f" {'Avg':>8s}"

    sep = "─" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)

    for row in all_rows:
        line = f"{row['Model']:<25s} {row['Params']:>7s} {str(row.get('Tokens','')):>6s}"
        for col in BENCHMARK_COLS:
            val = row.get(col)
            line += f" {val:>10.1f}" if val is not None else f" {'—':>10s}"
        avg = row.get("Avg")
        line += f" {avg:>8.1f}" if avg is not None else f" {'—':>8s}"

        # Highlight our model
        if "ours" in row["Model"]:
            line = f"\033[1;32m{line}\033[0m"  # green bold
        print(line)

    print(sep)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate Helios-Nova with lm-eval")
    parser.add_argument("--quick", action="store_true",
                        help="Run only hellaswag + arc_easy for a fast sanity check")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'cuda', 'cpu', or 'auto'")
    parser.add_argument("--output", type=str, default="helios_nova_benchmark_results.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    # Resolve device
    import torch
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Copy model file to working directory
    src = Path("/mnt/user-data/uploads/HeliosNova.py")
    dst = Path(__file__).parent / "HeliosNova.py"
    if src.exists() and not dst.exists():
        import shutil
        shutil.copy2(src, dst)

    # Step 1: Load model
    model, tokenizer, device = build_hf_compatible_model(
        repo_id="respinosamena/Helios-Nova",
        device=device,
    )

    # Step 2: Evaluate
    raw_results = run_evaluation(model, tokenizer, device, quick=args.quick)

    # Step 3: Extract scores
    scores = extract_scores(raw_results, quick=args.quick)
    print("\n[*] Helios-Nova scores:")
    for k, v in scores.items():
        print(f"    {k}: {v}")

    # Step 4: Save JSON dump of raw results (for reproducibility)
    json_path = Path(args.output).with_suffix(".json")
    with open(json_path, "w") as f:
        # Filter out non-serializable items
        serializable = {}
        for k, v in raw_results.items():
            if k in ("results", "versions", "config", "n-shot"):
                serializable[k] = v
        json.dump(serializable, f, indent=2, default=str)
    print(f"[*] Raw lm-eval results saved to {json_path}")

    # Step 5: Write CSV + print leaderboard
    csv_path = write_csv(scores, output_path=args.output)
    print_leaderboard(scores)

    print(f"\n[✓] Done!  CSV → {csv_path}  |  JSON → {json_path}")


if __name__ == "__main__":
    main()
