<p align="center">
  <img src="assets/helios_nova_banner.svg" alt="Helios Nova" width="100%"/>
</p>

<p align="center">
  <a href="https://huggingface.co/respinosamena/Helios-Nova"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-orange" alt="HuggingFace"/></a>
  <a href="https://github.com/rafaelespinosamena/Helios-Nova-306M/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"/></a>
  <img src="https://img.shields.io/badge/Parameters-306M-green" alt="Params"/>
  <img src="https://img.shields.io/badge/Training%20Cost-%3C%24190-brightgreen" alt="Cost"/>
  <img src="https://img.shields.io/badge/RAM-%3C3%20GB-purple" alt="RAM"/>
</p>

---

# Helios Nova — 306M

**Helios Nova** is a 306-million-parameter dense language model that achieves **96% of peer-model accuracy while training on 5–30× fewer tokens, on a single GPU, for under $190.**

Named after the Greek god of the sun (*Helios*) and the astronomical term for a stellar explosion that dramatically increases a star's brightness (*Nova*), the model embodies the idea that a small system can radiate disproportionate capability when its architecture and training recipe are carefully engineered.

Helios Nova incorporates a state-of-the-art transformer architecture — SwiGLU activations, Grouped-Query Attention, QK-Norm stabilisation, and Rotary Position Embeddings — and was pre-trained on just **50 billion tokens** from [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) on a **single NVIDIA H100 GPU in under 120 hours**. Where comparable models like OpenELM-270M consumed 1.5 trillion tokens across large GPU clusters, Helios Nova reaches within 1.5 points of the same benchmark average — with 30× less data and a total compute budget that costs less than a pair of running shoes.

> **Key facts at a glance**
>
> | | |
> |---|---|
> **Parameters** | 306M (dense, 24 unique layers, no weight sharing)
> **Training data** | 50B tokens · FineWeb-Edu (sample-100BT)
> **Hardware** | 1× NVIDIA H100 80GB
> **Wall time** | < 120 hours
> **Total cost** | < $190 USD
> **Inference RAM** | < 3 GB (full fp32 precision)
> **Context length** | 2,048 tokens
> **Tokenizer** | 16K BPE (custom-trained on FineWeb-Edu)

---

## Table of contents

1. [The efficiency story](#the-efficiency-story)
2. [Benchmark results](#benchmark-results)
3. [Architecture](#architecture)
4. [Training details](#training-details)
5. [Quick start](#quick-start)
6. [Interactive chat](#interactive-chat)
7. [System requirements](#system-requirements)
8. [Repository structure](#repository-structure)
9. [Reproduce from scratch](#reproduce-from-scratch)
10. [Citation](#citation)
11. [License](#license)

---

## The efficiency story

Most small language models are trained on hundreds of billions to trillions of tokens using multi-GPU clusters. Helios Nova asks a different question: **how far can a well-designed architecture go on a shoestring budget?**

<p align="center">
  <img src="assets/data_scale_comparison.svg" alt="Training data vs. benchmark performance" width="100%"/>
</p>

The answer: surprisingly far. Helios Nova scores within **1.5 points** of OpenELM-270M, Pythia-410M, and MobileLLM-350M on reasoning benchmarks — models that trained on 5× to 30× more data. On individual tasks, it outright **beats** some of these models despite the massive data disadvantage:

- **Beats OpenELM-270M** on ARC-Challenge (28.4 vs 27.6) — a model trained on 30× more tokens
- **Beats OpenELM-270M** on WinoGrande (53.1 vs 53.0) and OBQA (33.2 vs 33.0)
- **Beats Pythia-410M** on OBQA (33.2 vs 30.2) — a larger model trained on 6× more data
- **Matches MobileLLM-350M** on OBQA (33.2 vs 33.0) — trained on 5× more tokens

This efficiency comes from architectural decisions — SwiGLU, GQA, QK-Norm, depth-over-width — that extract maximum learning per token, combined with a WSD learning rate schedule optimised for long overtraining runs.

---

## Benchmark results

All evaluations were run with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) under identical zero-shot conditions (MMLU uses 5-shot). Baseline numbers are sourced from **Table 4 of the SmolLM2 paper** ([Allal et al., 2025; arXiv:2502.02737](https://arxiv.org/abs/2502.02737)), which evaluated all models under a unified setup.

### Comparison table

| Model | Params | Tokens | ARC-C | WinoGrande | PIQA | OBQA | MMLU (5s) | Avg |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Helios-Nova (ours)** | **306M** | **50B** | **28.4** | **53.1** | **63.8** | **33.2** | **22.9** | **40.3** |
| OpenELM-270M | 270M | 1.5T | 27.6 | 53.0 | 69.8 | 33.0 | 25.4 | 41.8 |
| MobileLLM-350M | 350M | 250B | 29.4 | 52.3 | 68.6 | 33.0 | 25.5 | 41.8 |
| Pythia-410M | 410M | 300B | 29.3 | 53.8 | 70.4 | 30.2 | 25.3 | 41.8 |
| OpenELM-450M | 450M | 1.5T | 30.1 | 53.6 | 72.3 | 33.6 | 25.8 | 43.1 |
| SmolLM-360M | 360M | 1.4T | 42.0 | 51.5 | 71.6 | 36.4 | 26.2 | 45.5 |

### How to read this table

Helios-Nova averages **40.3** against a peer median of **41.8** — a gap of just 1.5 points. But look at the "Tokens" column: every other model consumed between 5× and 30× more training data. OpenELM-270M and OpenELM-450M each trained on 1.5 trillion tokens; Helios-Nova trained on 50 billion. That 1.5-point gap represents one of the highest accuracy-per-token ratios in this weight class.

On three of the five benchmarks — **ARC-Challenge, WinoGrande, and OBQA** — Helios-Nova matches or beats OpenELM-270M outright. The benchmarks where Helios trails most (PIQA, MMLU) are known to scale predictably with additional data and would be expected to close with continued training.

---

## Architecture

Helios Nova is a **dense causal transformer** with 24 unique layers (no weight sharing), designed for maximum learning efficiency per token at the sub-500M scale.

```
┌──────────────────────────────────────────────────────────┐
│                    Helios Nova · 306M                     │
├──────────────────────────────────────────────────────────┤
│  Token Embedding (16,384 × 1,024)  ←── tied weights ──┐ │
│  ┌─────────────────────────────────────────────────┐   │ │
│  │  × 24 Transformer Blocks                        │   │ │
│  │  ┌───────────────────────────────────────────┐  │   │ │
│  │  │ RMSNorm → GQ-Attention (16q / 4kv)       │  │   │ │
│  │  │   · QK-Norm (RMSNorm on Q and K)         │  │   │ │
│  │  │   · RoPE (θ = 10,000)                    │  │   │ │
│  │  │   · head_dim = 64                        │  │   │ │
│  │  │ + Residual                               │  │   │ │
│  │  ├───────────────────────────────────────────┤  │   │ │
│  │  │ RMSNorm → SwiGLU FFN (3,072 hidden)      │  │   │ │
│  │  │ + Residual                               │  │   │ │
│  │  └───────────────────────────────────────────┘  │   │ │
│  └─────────────────────────────────────────────────┘   │ │
│  Final RMSNorm                                         │ │
│  LM Head (1,024 × 16,384)  ←── tied weights ──────────┘ │
└──────────────────────────────────────────────────────────┘
```

### Component breakdown

| Component | Detail | Rationale |
|:---|:---|:---|
| **SwiGLU FFN** | Gated activation with SiLU (Shazeer 2020). `ffn_dim = 3,072`. | 10–15% parameter-efficiency gain over standard ReLU MLP — the biggest contributor to Helios Nova's data efficiency. |
| **Grouped-Query Attention** | 16 query heads / 4 KV heads (4:1 ratio). | 4× KV-cache reduction. Enables fast inference with < 3 GB RAM. |
| **QK-Norm** | RMSNorm on Q and K *before* the dot product. | Prevents attention logit explosion, enabling stable training at peak LR 3×10⁻⁴. |
| **RoPE** | Rotary Position Embeddings, θ = 10,000. | Relative position encoding. No learned positional parameters. |
| **RMSNorm** | Pre-norm, no bias anywhere. | ~20% fewer FLOPs per norm than LayerNorm. |
| **Tied embeddings** | Input/output weight sharing. | Saves ~16.7M parameters. |
| **Depth over width** | 24 layers at d=1,024. | MobileLLM finding: deeper beats wider at sub-500M scale. |
| **Residual scaling** | 1/√(2·depth) init on output projections. | Keeps residual stream stable through 24 layers. |

---

## Training details

### Data

50 billion tokens from [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (`sample-100BT`). Custom **16K BPE tokenizer** trained on ~1M documents from the same distribution.

### Learning rate schedule — Warmup-Stable-Decay (WSD)

| Phase | Steps | LR |
|:---|:---|:---|
| Warmup | 0 → 4,000 | Linear 0 → 3×10⁻⁴ |
| Stable | 4,000 → ~114,000 | Constant 3×10⁻⁴ |
| Decay | Final 10% | Cosine 3×10⁻⁴ → 3×10⁻⁵ |

WSD outperforms cosine on overtraining runs because the model spends ~87% of training at peak LR rather than decaying prematurely.

### Hyperparameters

| Hyperparameter | Value |
|:---|:---|
| Optimiser | AdamW (fused, β₁=0.9, β₂=0.95) |
| Peak / min LR | 3×10⁻⁴ / 3×10⁻⁵ |
| Weight decay | 0.1 |
| Gradient clipping | 1.0 |
| Effective batch | 393,216 tokens/step |
| Precision | bfloat16 + torch.compile |
| Total steps | ~127,000 |

### Infrastructure

**1× NVIDIA H100 80GB** · < 120 hours · < $190 USD · PyTorch 2.x · W&B logging · Best-checkpoint → HuggingFace Hub

---

## Quick start

```bash
git clone https://github.com/rafaelespinosamena/Helios-Nova-306M.git
cd Helios-Nova-306M
pip install torch transformers safetensors huggingface_hub
```

```python
import torch
from HeliosNova import HeliosNova
from transformers import AutoTokenizer

model = HeliosNova.from_pretrained("respinosamena/Helios-Nova", device="cuda")
model.eval()
tokenizer = AutoTokenizer.from_pretrained("respinosamena/Helios-Nova")

prompt = "The theory of relativity fundamentally changed"
ids = torch.tensor([[tokenizer.bos_token_id] + tokenizer.encode(prompt, add_special_tokens=False)], device="cuda")
output = model.generate(ids, max_new_tokens=200, temperature=0.8, top_k=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

---

## Interactive chat

```bash
python chat.py
```

| Command | Effect |
|:---|:---|
| `!temp 0.5` | Set sampling temperature |
| `!topk 30` | Set top-k filtering |
| `!max 512` | Set max generation length |
| `!rep 1.2` | Set repetition penalty |
| `!stream` | Toggle streaming output |
| `quit` | Exit |

---

## System requirements

| Platform | Device | RAM |
|:---|:---|:---|
| **Linux / Windows (CUDA)** | NVIDIA GPU ≥ 4GB | 4 GB |
| **Linux / Windows (CPU)** | Any modern x86 | 3 GB |
| **macOS (Apple Silicon)** | M1 / M2 / M3 | 4 GB |
| **macOS (Intel)** | Any Intel Mac | 3 GB |

Auto-detection: `chat.py` tries CUDA → MPS → CPU.

---

## Repository structure

```
Helios-Nova-306M/
├── HeliosNova.py           # Model architecture
├── train.py                # Training loop (streaming, WSD, W&B, Hub upload)
├── train_tokenizer.py      # BPE tokenizer trainer (16K vocab)
├── chat.py                 # Interactive generation
├── config.yaml             # All hyperparameters
├── evaluate_helios_nova.py # lm-eval benchmark runner
├── requirements.txt
├── assets/
│   ├── helios_nova_banner.svg
│   ├── data_scale_comparison.svg
│   └── benchmark_chart.svg
└── README.md
```

---

## Reproduce from scratch

```bash
# 1. Tokenizer
python train_tokenizer.py

# 2. Pre-train (edit config.yaml with your tokens first)
python train.py --config config.yaml

# 3. Evaluate
python evaluate_helios_nova.py --device cuda
```

---

## Citation

```bibtex
@misc{espinosamena2025heliosnova,
  title   = {Helios Nova: A Budget-Efficient 306M Parameter Language Model},
  author  = {Espinosa Mena, Rafael},
  year    = {2025},
  url     = {https://github.com/rafaelespinosamena/Helios-Nova-306M},
  note    = {306M dense transformer, 50B tokens, single H100, under \$190 USD}
}
```

### References

- **SmolLM**: Allal et al., [arXiv:2502.02737](https://arxiv.org/abs/2502.02737) (2025) · **OpenELM**: Mehta et al., [arXiv:2404.14619](https://arxiv.org/abs/2404.14619) (2024) · **MobileLLM**: Liu et al., [arXiv:2402.14905](https://arxiv.org/abs/2402.14905) (2024) · **Pythia**: Biderman et al., [arXiv:2304.01373](https://arxiv.org/abs/2304.01373) (2023)

## License

[Apache License 2.0](LICENSE)

<p align="center"><sub>Built with a single GPU and a lot of curiosity.</sub></p>
