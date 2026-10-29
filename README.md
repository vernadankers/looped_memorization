# Ouro Verbatim Memorization Extraction

Measures verbatim memorization in ByteDance's [Ouro](https://ouro-llm.github.io/) looped language models: sample prefix/continuation pairs from a public corpus, generate continuations with each Ouro checkpoint via vLLM, and score how closely the model reproduces the true continuation.

Full design rationale and the task-by-task build plan live in `docs/superpowers/specs/2026-10-27-ouro-extraction-design.md` and `docs/superpowers/plans/2026-10-27-ouro-extraction-pipeline.md`.

## Pipeline

Three stages, each a thin CLI around pure, independently-tested functions. Every stage reads/writes `.jsonl` files, so any stage can be rerun on its own.

```
configs/extraction.yaml, configs/models.yaml   # experiment + checkpoint config
        │
        ▼
data/build_prefix_dataset.py   → data/prefixes.jsonl
        │
        ▼
generate/run_vllm_generation.py (× 4 checkpoints)  → results/generations_<model>.jsonl
        │
        ▼
score/score_extraction.py (× 4 checkpoints)  → results/scores_<model>.jsonl + summary
```

- **Data**: streams `monology/pile-uncopyrighted`, splits each document into a `prefix_tokens`-token prefix and a `continuation_tokens`-token true continuation (`configs/extraction.yaml`).
- **Generate**: loads one Ouro checkpoint via vLLM's offline `LLM` class, greedy-decodes a continuation for every prefix. Run once per checkpoint listed in `configs/models.yaml`.
- **Score**: compares generated vs. true continuation per row (exact match, longest common prefix, fuzzy overlap ratio) and reports an aggregate memorization rate per checkpoint. Rows above `high_overlap_threshold` are flagged `high_overlap: true`.

Per-loop-iteration probing of the recurrent architecture (why/where memorization happens inside the loop) is intentionally out of scope for this pipeline — see the spec's "Explicitly out of scope" section.

## Setup

```bash
conda env create -f envs/ouro-vllm.yml
conda activate ouro-vllm
```

## Running locally (sanity check)

Every stage CLI accepts `--limit` to run on a handful of rows before committing to a full SLURM job:

```bash
python -m data.build_prefix_dataset --limit 10
python -m generate.run_vllm_generation --model ByteDance/Ouro-1.4B --limit 10
python -m score.score_extraction --model Ouro-1.4B
```

Run all CLIs as modules (`python -m ...`) from the repo root — this is required for the `common`/`data`/`generate`/`score` package imports to resolve.

## Running on a cluster

Fill in `<FILL_IN_PARTITION>` and `<FILL_IN_ACCOUNT>` in `slurm/generate.sbatch` and `slurm/score.sbatch`, then:

```bash
python -m data.build_prefix_dataset   # build the full prefix set once, on a login node

GEN_JOB=$(sbatch --parsable slurm/generate.sbatch)
sbatch --dependency=afterok:$GEN_JOB slurm/score.sbatch
```

`generate.sbatch` runs the four checkpoints as a job array; `score.sbatch` runs after all four generation jobs succeed.

## Tests

```bash
pytest -v
```

Covers every pure function in the pipeline (document splitting, jsonl I/O, prompt/merge helpers, scoring metrics) with fakes — no GPU or network access required. The vLLM generation CLI and the HuggingFace `datasets`/`transformers` calls in the data builder are exercised manually via `--limit`, not by the test suite.
