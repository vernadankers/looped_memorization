# Ouro Verbatim Memorization Extraction — Design

Date: 2026-10-27

## Purpose

Run prefix-based extraction experiments against ByteDance's Ouro looped
language models (base and Thinking variants, 1.4B and 2.6B) to measure
verbatim memorization: feed known-corpus prefixes, generate continuations,
and score how closely the model reproduces the true continuation.

Per-loop-iteration probing of *why*/*where* memorization happens inside the
recurrent architecture is an intentional non-goal of this spec — it is
deferred until the extraction results are in hand and a concrete probing
question has been chosen. The pipeline below should not block that future
work, but it also does not build it.

## Background

Ouro ("Looped Language Models", ByteDance, arXiv 2510.25741) replaces the
usual feedforward transformer stack with a single shared layer stack applied
R=4 times in latent space, with learned early-exit gates (easy tokens exit
early, hard tokens loop longer). Four checkpoints are released on Hugging
Face under Apache-2.0:

- `ByteDance/Ouro-1.4B` (24 layers × 4 loops, 7.7T pretraining tokens)
- `ByteDance/Ouro-2.6B` (48 layers × 4 loops, 7.7T pretraining tokens)
- `ByteDance/Ouro-1.4B-Thinking` (SFT, 32K context)
- `ByteDance/Ouro-2.6B-Thinking` (SFT, 32K context)

Native vLLM support exists only in vLLM v0.11.1–v0.26.0, and vLLM always
executes all 4 loops (no adaptive early exit) — so vLLM generation reflects
"full compute" behavior, not the model's natural early-exit dynamics.

Ouro's actual training mixture is not public. This pipeline uses
`monology/pile-uncopyrighted` (permissively licensed, streamable via
`datasets`, common in prior memorization literature) as a proxy corpus —
results characterize memorization of *plausible pretraining-like text*, not
verified members of Ouro's actual training set.

## Architecture

Three independent stages, each with a clear file-based input/output
contract, so any stage can be rerun without touching the others:

```
looped_memorization/
├── envs/
│   └── ouro-vllm.yml           # conda env: vllm==0.26.0, datasets
├── configs/
│   ├── models.yaml              # the 4 checkpoint HF names
│   └── extraction.yaml          # prefix/continuation lengths, sample size, seed, overlap threshold
├── data/
│   └── build_prefix_dataset.py  # streams Pile-uncopyrighted, samples docs, splits prefix/continuation
├── generate/
│   └── run_vllm_generation.py   # offline vLLM batch generation over prefixes.jsonl, per checkpoint
├── score/
│   └── score_extraction.py      # exact-match / longest-common-prefix / edit-distance vs true continuation
├── slurm/
│   ├── generate.sbatch          # GPU job array over the 4 checkpoints
│   └── score.sbatch             # CPU job, --dependency=afterok on the generate jobs
├── data/prefixes.jsonl          # generated, gitignored
├── results/                     # generations_<model>.jsonl, scores_<model>.jsonl — gitignored
└── docs/superpowers/specs/      # this file
```

## Components

### 1. Environment (`envs/ouro-vllm.yml`)

Single conda environment: `python=3.11`, `vllm==0.26.0`, `datasets`. Covers
both the data-build step (lightweight streaming, no GPU needed) and the
generation step (GPU, vLLM). Reproduced on the cluster via
`conda env create -f envs/ouro-vllm.yml`.

### 2. Data pipeline (`data/build_prefix_dataset.py`)

1. Stream `monology/pile-uncopyrighted` via `datasets` (`streaming=True`).
2. Filter to documents with at least `N + M` tokens (lengths from
   `configs/extraction.yaml`).
3. For each sampled document: tokenize, take first `N` tokens as `prefix`,
   next `M` tokens as `true_continuation`.
4. Sample a configured number of documents (default 10k), write
   `data/prefixes.jsonl` with fields `doc_id`, `prefix`, `true_continuation`,
   `source_meta` (shard/offset, so any hit traces back to its source
   document).
5. Documents shorter than `N + M` tokens are logged and skipped, not an
   error.

This file is produced once and reused across all four checkpoints.

### 3. Generation (`generate/run_vllm_generation.py`)

Runs in `ouro-vllm`, one GPU job per checkpoint:

1. Load the checkpoint via vLLM's offline `LLM` class, e.g.
   `LLM(model="ByteDance/Ouro-1.4B", trust_remote_code=True)`.
2. Read `data/prefixes.jsonl`; batch all prefixes through `llm.generate()`
   with greedy decoding (temperature=0, max_tokens=`M`) — deterministic, so
   overlap measures the model's actual argmax behavior rather than sampling
   variance.
3. Write `results/generations_<model_name>.jsonl`: `doc_id`, `prefix`,
   `true_continuation`, `generated_continuation`.
4. `--model` is a CLI arg (checkpoint name read from `configs/models.yaml`),
   so one script serves all four checkpoints.
5. Model load failures fail loudly — no fallback — since this is research
   code where silent degradation would corrupt results.

### 4. Scoring (`score/score_extraction.py`)

1. Read `results/generations_<model_name>.jsonl`.
2. Per row, compute:
   - **Exact match**: token-for-token equality of `generated_continuation`
     vs `true_continuation`.
   - **Longest common prefix length**: token count before first divergence.
   - **Edit distance / normalized overlap** (via `rapidfuzz`) as a softer
     signal.
3. Write `results/scores_<model_name>.jsonl` (per-doc metrics) plus a
   printed/saved aggregate summary (memorization rate at each threshold) per
   model, enabling direct comparison across the 4 checkpoints.
4. Rows above a configurable overlap threshold (`configs/extraction.yaml`)
   are flagged `high_overlap: true` — reserved for future probing work, not
   consumed by anything in this spec.

### 5. SLURM orchestration

- `slurm/generate.sbatch`: GPU job array over the 4 checkpoint names in
  `configs/models.yaml`, activates `ouro-vllm`, runs
  `generate/run_vllm_generation.py --model <name>`.
- `slurm/score.sbatch`: CPU-only job, `--dependency=afterok:<generate_job_id>`,
  runs `score/score_extraction.py` once all generations have landed.
- Partition/account/time-limit/GPU-type fields are left as placeholders for
  cluster-specific values.

## Testing

No unit-test suite — this is a one-off research pipeline, not a maintained
service. Each script accepts a `--limit` flag (e.g. 10 docs) so the full
pipeline can be sanity-checked on a login node before submitting the full
SLURM job and burning GPU hours.

## Explicitly out of scope

- Per-loop-iteration probing (hidden states, exit-gate/halting probabilities)
  via the HF `trust_remote_code` implementation. Deferred until extraction
  results motivate a specific probing question. The `high_overlap` flag in
  scoring output is the intended hook for this future work.
- Membership-inference-style methodology (loss/perplexity-based) — a
  possible future experiment, not this one.
- Containerization (Docker/Apptainer) — two-person research pipeline on a
  SLURM cluster, conda envs are sufficient.
