# Ouro Verbatim Memorization Extraction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pipeline that samples prefix/continuation pairs from a public corpus, generates continuations from the four Ouro checkpoints via vLLM, and scores verbatim memorization (exact match, longest common prefix, overlap ratio), orchestrated as dependent SLURM jobs.

**Architecture:** Three sequential stages (data → generate → score), each a thin CLI wrapper around pure, independently-testable functions. Shared jsonl read/write lives in `common/`. No GPU or network access is required to run the test suite — every pure function is tested with fakes/fixtures; the real HF/vLLM calls are exercised manually via each CLI's `--limit` flag.

**Tech Stack:** Python 3.11, vLLM 0.26.0, HuggingFace `datasets` (streaming) and `transformers` (tokenizer), PyYAML, rapidfuzz, pytest, SLURM.

**Spec:** `docs/superpowers/specs/2026-10-27-ouro-extraction-design.md`

## Global Constraints

- vLLM version pinned to `0.26.0` (only version range with native Ouro support per spec; `0.11.1`–`0.26.0`, pin to the latest).
- `trust_remote_code=True` required on every model load (vLLM and transformers) — Ouro ships custom modeling code.
- Generation uses greedy decoding (`temperature=0.0`) — deterministic, so overlap reflects argmax behavior, not sampling variance.
- Four checkpoints, exact names: `ByteDance/Ouro-1.4B`, `ByteDance/Ouro-2.6B`, `ByteDance/Ouro-1.4B-Thinking`, `ByteDance/Ouro-2.6B-Thinking`.
- Proxy corpus: `monology/pile-uncopyrighted`, streamed (not fully downloaded).
- Per-loop probing is explicitly out of scope for this plan — do not add it.
- Model-load failures must fail loudly (no try/except swallowing errors around model loading).
- All CLI scripts are run as modules from the repo root (`python -m data.build_prefix_dataset`, etc.) so intra-project imports resolve.

## Review Focus

- Documents shorter than `prefix_tokens + continuation_tokens` must be skipped, not crash the run or get silently mis-split. → tested in Task 4.
- A mismatch between the number of input rows and the number of generated texts must raise a clear error, not silently zip rows to the wrong generations. → tested in Task 5.
- An empty generated continuation (model outputs nothing) must still score without crashing. → tested in Task 7.
- A row whose overlap ratio lands exactly on the configured `high_overlap_threshold` must be classified as high-overlap (threshold is inclusive), not excluded by an off-by-one boundary. → tested in Task 8.
- jsonl read/write must round-trip rows containing non-ASCII text (corpus documents may contain unicode) without corruption. → tested in Task 2.

---

## Task 1: Project scaffolding

**Files:**
- Create: `envs/ouro-vllm.yml`
- Create: `configs/models.yaml`
- Create: `configs/extraction.yaml`
- Create: `.gitignore`
- Create: `pyproject.toml`
- Create: `common/__init__.py`, `data/__init__.py`, `generate/__init__.py`, `score/__init__.py` (empty)

**Interfaces:**
- Produces: `configs/extraction.yaml` keys consumed by later tasks: `corpus`, `corpus_split`, `tokenizer_name`, `prefix_tokens`, `continuation_tokens`, `num_samples`, `seed`, `high_overlap_threshold`.
- Produces: `pyproject.toml` pytest config so `tests/` can import `common.*`, `data.*`, `generate.*`, `score.*` without a src-layout install step.

- [ ] **Step 1: Create the conda environment file**

`envs/ouro-vllm.yml`:
```yaml
name: ouro-vllm
channels:
  - conda-forge
dependencies:
  - python=3.11
  - pip
  - pip:
      - vllm==0.26.0
      - datasets
      - transformers
      - pyyaml
      - rapidfuzz
      - pytest
```

- [ ] **Step 2: Create the model list config**

`configs/models.yaml`:
```yaml
models:
  - ByteDance/Ouro-1.4B
  - ByteDance/Ouro-2.6B
  - ByteDance/Ouro-1.4B-Thinking
  - ByteDance/Ouro-2.6B-Thinking
```

- [ ] **Step 3: Create the extraction experiment config**

`configs/extraction.yaml`:
```yaml
corpus: monology/pile-uncopyrighted
corpus_split: train
tokenizer_name: ByteDance/Ouro-1.4B
prefix_tokens: 100
continuation_tokens: 100
num_samples: 10000
seed: 0
high_overlap_threshold: 0.8
```

`tokenizer_name` picks one checkpoint's tokenizer to use for splitting documents into prefix/continuation token spans; this is shared across all four checkpoints when building `prefixes.jsonl` so the dataset only needs to be built once.

- [ ] **Step 4: Create `.gitignore`**

```
data/prefixes.jsonl
results/
slurm_logs/
__pycache__/
*.pyc
.pytest_cache/
```

- [ ] **Step 5: Create `pyproject.toml`**

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

- [ ] **Step 6: Create empty package markers**

```bash
touch common/__init__.py data/__init__.py generate/__init__.py score/__init__.py
mkdir -p tests
touch tests/__init__.py
```

- [ ] **Step 7: Verify the YAML files parse**

Run:
```bash
python -c "import yaml; [yaml.safe_load(open(f)) for f in ['configs/models.yaml', 'configs/extraction.yaml', 'envs/ouro-vllm.yml']]; print('ok')"
```
Expected: prints `ok` with no traceback.

- [ ] **Step 8: Commit**

```bash
git add envs/ configs/ .gitignore pyproject.toml common/__init__.py data/__init__.py generate/__init__.py score/__init__.py tests/__init__.py
git commit -m "chore: scaffold project structure, configs, and env spec"
```

---

## Task 2: Shared jsonl I/O

**Files:**
- Create: `common/jsonl_io.py`
- Test: `tests/test_jsonl_io.py`

**Interfaces:**
- Produces: `read_jsonl(path: str) -> list[dict]`, `write_jsonl(path: str, rows: list[dict]) -> None` — used by every later task that reads or writes a `.jsonl` file.

- [ ] **Step 1: Write the failing tests**

`tests/test_jsonl_io.py`:
```python
from common.jsonl_io import read_jsonl, write_jsonl


def test_write_then_read_round_trip(tmp_path):
    path = tmp_path / "rows.jsonl"
    rows = [{"doc_id": 0, "text": "hello"}, {"doc_id": 1, "text": "world"}]
    write_jsonl(str(path), rows)
    result = read_jsonl(str(path))
    assert result == rows


def test_round_trip_preserves_unicode(tmp_path):
    path = tmp_path / "rows.jsonl"
    rows = [{"doc_id": 0, "text": "café élève 中文"}]
    write_jsonl(str(path), rows)
    result = read_jsonl(str(path))
    assert result == rows


def test_read_skips_blank_lines(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
    result = read_jsonl(str(path))
    assert result == [{"a": 1}, {"a": 2}]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_jsonl_io.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'common.jsonl_io'`

- [ ] **Step 3: Write the implementation**

`common/jsonl_io.py`:
```python
import json


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_jsonl_io.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add common/jsonl_io.py tests/test_jsonl_io.py
git commit -m "feat: add shared jsonl read/write helpers"
```

---

## Task 3: Prefix/continuation splitting

**Files:**
- Create: `data/splitting.py`
- Test: `tests/test_splitting.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `split_document(text: str, tokenizer, prefix_tokens: int, continuation_tokens: int) -> dict | None`, where the dict has keys `prefix` and `true_continuation`. `tokenizer` is any object with `.encode(text) -> list` and `.decode(list) -> str` (an HF tokenizer satisfies this). Returns `None` if `text` has fewer than `prefix_tokens + continuation_tokens` tokens. Used by Task 4.

- [ ] **Step 1: Write the failing tests**

`tests/test_splitting.py`:
```python
from data.splitting import split_document


class FakeTokenizer:
    def encode(self, text):
        return text.split()

    def decode(self, tokens):
        return " ".join(tokens)


def test_split_document_returns_prefix_and_continuation():
    tokenizer = FakeTokenizer()
    text = " ".join(f"tok{i}" for i in range(10))
    result = split_document(text, tokenizer, prefix_tokens=4, continuation_tokens=3)
    assert result == {
        "prefix": "tok0 tok1 tok2 tok3",
        "true_continuation": "tok4 tok5 tok6",
    }


def test_split_document_returns_none_for_short_document():
    tokenizer = FakeTokenizer()
    text = "tok0 tok1 tok2"
    result = split_document(text, tokenizer, prefix_tokens=4, continuation_tokens=3)
    assert result is None


def test_split_document_boundary_exact_length_is_not_short():
    tokenizer = FakeTokenizer()
    text = " ".join(f"tok{i}" for i in range(7))  # exactly prefix_tokens + continuation_tokens
    result = split_document(text, tokenizer, prefix_tokens=4, continuation_tokens=3)
    assert result is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_splitting.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data.splitting'`

- [ ] **Step 3: Write the implementation**

`data/splitting.py`:
```python
def split_document(text, tokenizer, prefix_tokens, continuation_tokens):
    token_ids = tokenizer.encode(text)
    needed = prefix_tokens + continuation_tokens
    if len(token_ids) < needed:
        return None
    prefix_ids = token_ids[:prefix_tokens]
    continuation_ids = token_ids[prefix_tokens:needed]
    return {
        "prefix": tokenizer.decode(prefix_ids),
        "true_continuation": tokenizer.decode(continuation_ids),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_splitting.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add data/splitting.py tests/test_splitting.py
git commit -m "feat: add prefix/continuation document splitting"
```

---

## Task 4: Prefix dataset builder

**Files:**
- Create: `data/build_prefix_dataset.py`
- Test: `tests/test_build_prefix_dataset.py`

**Interfaces:**
- Consumes: `split_document` from Task 3 (`data/splitting.py`); `write_jsonl` from Task 2 (`common/jsonl_io.py`).
- Produces: `build_dataset(documents: Iterable[str], tokenizer, prefix_tokens: int, continuation_tokens: int, num_samples: int) -> list[dict]`, where each returned row has keys `doc_id` (int, sequential), `prefix`, `true_continuation`, `source_meta` (dict). Also produces a `main()` CLI entry point writing to `data/prefixes.jsonl` by default. Rows from this file are consumed by Task 6.

- [ ] **Step 1: Write the failing tests**

`tests/test_build_prefix_dataset.py`:
```python
from data.build_prefix_dataset import build_dataset


class FakeTokenizer:
    def encode(self, text):
        return text.split()

    def decode(self, tokens):
        return " ".join(tokens)


def test_build_dataset_skips_short_documents_and_assigns_sequential_ids():
    tokenizer = FakeTokenizer()
    long_doc = " ".join(f"tok{i}" for i in range(10))
    short_doc = "only two tokens"
    documents = [short_doc, long_doc, long_doc]

    rows = build_dataset(documents, tokenizer, prefix_tokens=4, continuation_tokens=3, num_samples=10)

    assert len(rows) == 2
    assert [r["doc_id"] for r in rows] == [0, 1]
    assert rows[0]["prefix"] == "tok0 tok1 tok2 tok3"
    assert rows[0]["true_continuation"] == "tok4 tok5 tok6"


def test_build_dataset_stops_at_num_samples():
    tokenizer = FakeTokenizer()
    long_doc = " ".join(f"tok{i}" for i in range(10))
    documents = [long_doc] * 5

    rows = build_dataset(documents, tokenizer, prefix_tokens=4, continuation_tokens=3, num_samples=2)

    assert len(rows) == 2


def test_build_dataset_returns_fewer_rows_when_corpus_runs_out():
    tokenizer = FakeTokenizer()
    long_doc = " ".join(f"tok{i}" for i in range(10))
    documents = [long_doc]  # only one usable document available, far fewer than requested

    rows = build_dataset(documents, tokenizer, prefix_tokens=4, continuation_tokens=3, num_samples=10)

    assert len(rows) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_build_prefix_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data.build_prefix_dataset'`

- [ ] **Step 3: Write the implementation**

`data/build_prefix_dataset.py`:
```python
import argparse
import sys

import yaml
from datasets import load_dataset
from transformers import AutoTokenizer

from common.jsonl_io import write_jsonl
from data.splitting import split_document


def build_dataset(documents, tokenizer, prefix_tokens, continuation_tokens, num_samples):
    rows = []
    doc_id = 0
    for text in documents:
        if len(rows) >= num_samples:
            break
        split = split_document(text, tokenizer, prefix_tokens, continuation_tokens)
        if split is None:
            continue
        rows.append({
            "doc_id": doc_id,
            "prefix": split["prefix"],
            "true_continuation": split["true_continuation"],
            "source_meta": {"index": doc_id},
        })
        doc_id += 1
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/extraction.yaml")
    parser.add_argument("--out", default="data/prefixes.jsonl")
    parser.add_argument("--limit", type=int, default=None,
                         help="Override num_samples for a quick sanity-check run")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    num_samples = args.limit if args.limit is not None else config["num_samples"]

    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])
    dataset = load_dataset(config["corpus"], split=config["corpus_split"], streaming=True)
    dataset = dataset.shuffle(seed=config["seed"], buffer_size=10_000)
    documents = (row["text"] for row in dataset)

    rows = build_dataset(documents, tokenizer, config["prefix_tokens"],
                          config["continuation_tokens"], num_samples)

    if len(rows) < num_samples:
        print(f"warning: only found {len(rows)} usable documents out of requested {num_samples}",
              file=sys.stderr)

    write_jsonl(args.out, rows)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_build_prefix_dataset.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add data/build_prefix_dataset.py tests/test_build_prefix_dataset.py
git commit -m "feat: add prefix dataset builder with streaming corpus support"
```

---

## Task 5: Generation prompt/merge utilities

**Files:**
- Create: `generate/prompt_utils.py`
- Test: `tests/test_prompt_utils.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (operates on plain dicts matching the row shape produced by Task 4).
- Produces: `build_prompts(rows: list[dict]) -> list[str]`, `merge_generations(rows: list[dict], generated_texts: list[str]) -> list[dict]` (raises `ValueError` on length mismatch). Used by Task 6.

- [ ] **Step 1: Write the failing tests**

`tests/test_prompt_utils.py`:
```python
import pytest

from generate.prompt_utils import build_prompts, merge_generations


def test_build_prompts_extracts_prefix_field():
    rows = [{"doc_id": 0, "prefix": "hello"}, {"doc_id": 1, "prefix": "world"}]
    assert build_prompts(rows) == ["hello", "world"]


def test_merge_generations_adds_generated_continuation_field():
    rows = [{"doc_id": 0, "prefix": "hello"}]
    result = merge_generations(rows, ["generated text"])
    assert result == [{"doc_id": 0, "prefix": "hello", "generated_continuation": "generated text"}]


def test_merge_generations_raises_on_length_mismatch():
    rows = [{"doc_id": 0, "prefix": "hello"}, {"doc_id": 1, "prefix": "world"}]
    with pytest.raises(ValueError):
        merge_generations(rows, ["only one"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_prompt_utils.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'generate.prompt_utils'`

- [ ] **Step 3: Write the implementation**

`generate/prompt_utils.py`:
```python
def build_prompts(rows):
    return [row["prefix"] for row in rows]


def merge_generations(rows, generated_texts):
    if len(rows) != len(generated_texts):
        raise ValueError(
            f"row count ({len(rows)}) does not match generated text count ({len(generated_texts)})"
        )
    merged = []
    for row, generated_text in zip(rows, generated_texts):
        merged_row = dict(row)
        merged_row["generated_continuation"] = generated_text
        merged.append(merged_row)
    return merged
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_prompt_utils.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add generate/prompt_utils.py tests/test_prompt_utils.py
git commit -m "feat: add vLLM prompt-building and generation-merging helpers"
```

---

## Task 6: vLLM generation CLI

**Files:**
- Create: `generate/run_vllm_generation.py`

**Interfaces:**
- Consumes: `read_jsonl`/`write_jsonl` (Task 2), `build_prompts`/`merge_generations` (Task 5).
- Produces: `results/generations_<short_model_name>.jsonl`, rows with keys `doc_id`, `prefix`, `true_continuation`, `source_meta`, `generated_continuation`. Consumed by Task 8.

No unit tests — this script's only logic beyond Task 5's tested functions is the vLLM model call, which needs a GPU and cannot run in this test suite. Correctness of the surrounding logic is already covered by Task 5's tests; this script is verified by syntax-compiling it and, later, by a manual `--limit` run on the cluster (see spec's Testing section).

- [ ] **Step 1: Write the implementation**

`generate/run_vllm_generation.py`:
```python
import argparse

import yaml
from vllm import LLM, SamplingParams

from common.jsonl_io import read_jsonl, write_jsonl
from generate.prompt_utils import build_prompts, merge_generations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF checkpoint name, e.g. ByteDance/Ouro-1.4B")
    parser.add_argument("--config", default="configs/extraction.yaml")
    parser.add_argument("--prefixes", default="data/prefixes.jsonl")
    parser.add_argument("--out", default=None,
                         help="defaults to results/generations_<model short name>.jsonl")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process the first N rows, for a quick sanity-check run")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    rows = read_jsonl(args.prefixes)
    if args.limit is not None:
        rows = rows[:args.limit]

    prompts = build_prompts(rows)

    llm = LLM(model=args.model, trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=config["continuation_tokens"])
    outputs = llm.generate(prompts, sampling_params)
    generated_texts = [output.outputs[0].text for output in outputs]

    merged_rows = merge_generations(rows, generated_texts)

    out_path = args.out
    if out_path is None:
        short_name = args.model.split("/")[-1]
        out_path = f"results/generations_{short_name}.jsonl"

    write_jsonl(out_path, merged_rows)
    print(f"wrote {len(merged_rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script compiles**

Run: `python -m py_compile generate/run_vllm_generation.py`
Expected: no output, exit code 0

- [ ] **Step 3: Commit**

```bash
git add generate/run_vllm_generation.py
git commit -m "feat: add vLLM offline batch generation CLI"
```

---

## Task 7: Memorization scoring metrics

**Files:**
- Create: `score/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `exact_match(generated_tokens: list[str], true_tokens: list[str]) -> bool`, `longest_common_prefix_len(generated_tokens: list[str], true_tokens: list[str]) -> int`, `overlap_ratio(generated_text: str, true_text: str) -> float` (0.0–1.0). Used by Task 8.

- [ ] **Step 1: Write the failing tests**

`tests/test_metrics.py`:
```python
from score.metrics import exact_match, longest_common_prefix_len, overlap_ratio


def test_exact_match_true_for_identical_token_lists():
    assert exact_match(["a", "b", "c"], ["a", "b", "c"]) is True


def test_exact_match_false_for_different_token_lists():
    assert exact_match(["a", "b", "c"], ["a", "b", "d"]) is False


def test_longest_common_prefix_len_stops_at_first_divergence():
    assert longest_common_prefix_len(["a", "b", "x", "d"], ["a", "b", "c", "d"]) == 2


def test_longest_common_prefix_len_handles_empty_generated_tokens():
    assert longest_common_prefix_len([], ["a", "b", "c"]) == 0


def test_overlap_ratio_is_one_for_identical_text():
    assert overlap_ratio("hello world", "hello world") == 1.0


def test_overlap_ratio_handles_empty_generated_text_without_crashing():
    ratio = overlap_ratio("", "hello world")
    assert 0.0 <= ratio < 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'score.metrics'`

- [ ] **Step 3: Write the implementation**

`score/metrics.py`:
```python
from rapidfuzz import fuzz


def exact_match(generated_tokens, true_tokens):
    return generated_tokens == true_tokens


def longest_common_prefix_len(generated_tokens, true_tokens):
    count = 0
    for g, t in zip(generated_tokens, true_tokens):
        if g != t:
            break
        count += 1
    return count


def overlap_ratio(generated_text, true_text):
    return fuzz.ratio(generated_text, true_text) / 100.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_metrics.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add score/metrics.py tests/test_metrics.py
git commit -m "feat: add exact-match, longest-common-prefix, and overlap-ratio metrics"
```

---

## Task 8: Scoring CLI

**Files:**
- Create: `score/score_extraction.py`
- Test: `tests/test_score_extraction.py`

**Interfaces:**
- Consumes: `exact_match`, `longest_common_prefix_len`, `overlap_ratio` (Task 7); `read_jsonl`/`write_jsonl` (Task 2).
- Produces: `score_rows(rows: list[dict], overlap_threshold: float) -> tuple[list[dict], dict]` — `scored_rows` have keys `doc_id`, `exact_match`, `longest_common_prefix_len`, `overlap_ratio`, `high_overlap`; `summary` has keys `num_rows`, `num_exact_match`, `exact_match_rate`, `num_high_overlap`, `high_overlap_rate`. Also produces a `main()` CLI writing `results/scores_<model>.jsonl` and printing the summary as JSON.

- [ ] **Step 1: Write the failing tests**

`tests/test_score_extraction.py`:
```python
from score.score_extraction import score_rows


def make_row(doc_id, generated, true):
    return {"doc_id": doc_id, "generated_continuation": generated, "true_continuation": true}


def test_score_rows_counts_exact_matches():
    rows = [
        make_row(0, "hello world", "hello world"),
        make_row(1, "goodbye", "hello world"),
    ]
    scored, summary = score_rows(rows, overlap_threshold=0.9)
    assert summary["num_rows"] == 2
    assert summary["num_exact_match"] == 1
    assert scored[0]["exact_match"] is True
    assert scored[1]["exact_match"] is False


def test_score_rows_flags_overlap_exactly_at_threshold_as_high_overlap():
    rows = [make_row(0, "hello world", "hello world")]  # overlap_ratio == 1.0
    scored, _ = score_rows(rows, overlap_threshold=1.0)
    assert scored[0]["high_overlap"] is True


def test_score_rows_handles_empty_row_list():
    scored, summary = score_rows([], overlap_threshold=0.8)
    assert scored == []
    assert summary["num_rows"] == 0
    assert summary["exact_match_rate"] == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_score_extraction.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'score.score_extraction'`

- [ ] **Step 3: Write the implementation**

`score/score_extraction.py`:
```python
import argparse
import json

import yaml

from common.jsonl_io import read_jsonl, write_jsonl
from score.metrics import exact_match, longest_common_prefix_len, overlap_ratio


def score_rows(rows, overlap_threshold):
    scored = []
    num_exact_match = 0
    num_high_overlap = 0

    for row in rows:
        generated_tokens = row["generated_continuation"].split()
        true_tokens = row["true_continuation"].split()

        is_exact_match = exact_match(generated_tokens, true_tokens)
        prefix_len = longest_common_prefix_len(generated_tokens, true_tokens)
        overlap = overlap_ratio(row["generated_continuation"], row["true_continuation"])
        is_high_overlap = overlap >= overlap_threshold

        if is_exact_match:
            num_exact_match += 1
        if is_high_overlap:
            num_high_overlap += 1

        scored.append({
            "doc_id": row["doc_id"],
            "exact_match": is_exact_match,
            "longest_common_prefix_len": prefix_len,
            "overlap_ratio": overlap,
            "high_overlap": is_high_overlap,
        })

    summary = {
        "num_rows": len(rows),
        "num_exact_match": num_exact_match,
        "exact_match_rate": num_exact_match / len(rows) if rows else 0.0,
        "num_high_overlap": num_high_overlap,
        "high_overlap_rate": num_high_overlap / len(rows) if rows else 0.0,
    }
    return scored, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="short model name used in the generations filename")
    parser.add_argument("--config", default="configs/extraction.yaml")
    parser.add_argument("--generations", default=None,
                         help="defaults to results/generations_<model>.jsonl")
    parser.add_argument("--out", default=None,
                         help="defaults to results/scores_<model>.jsonl")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    generations_path = args.generations or f"results/generations_{args.model}.jsonl"
    out_path = args.out or f"results/scores_{args.model}.jsonl"

    rows = read_jsonl(generations_path)
    scored_rows, summary = score_rows(rows, config["high_overlap_threshold"])

    write_jsonl(out_path, scored_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_score_extraction.py -v`
Expected: 3 passed

- [ ] **Step 5: Run the full test suite**

Run: `pytest -v`
Expected: all tests across every task pass (18 tests total)

- [ ] **Step 6: Commit**

```bash
git add score/score_extraction.py tests/test_score_extraction.py
git commit -m "feat: add memorization scoring CLI with aggregate summary"
```

---

## Task 9: SLURM orchestration

**Files:**
- Create: `slurm/generate.sbatch`
- Create: `slurm/score.sbatch`

**Interfaces:**
- Consumes: `generate/run_vllm_generation.py` (Task 6), `score/score_extraction.py` (Task 8).
- Produces: submittable SLURM scripts; no code interface (terminal artifact of the plan).

- [ ] **Step 1: Write the generation job array script**

`slurm/generate.sbatch`:
```bash
#!/bin/bash
#SBATCH --job-name=ouro-generate
#SBATCH --partition=<FILL_IN_PARTITION>
#SBATCH --account=<FILL_IN_ACCOUNT>
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-3
#SBATCH --output=slurm_logs/generate_%A_%a.out

set -euo pipefail

MODELS=(
    "ByteDance/Ouro-1.4B"
    "ByteDance/Ouro-2.6B"
    "ByteDance/Ouro-1.4B-Thinking"
    "ByteDance/Ouro-2.6B-Thinking"
)
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ouro-vllm

mkdir -p results slurm_logs

python -m generate.run_vllm_generation --model "$MODEL"
```

- [ ] **Step 2: Write the dependent scoring job script**

`slurm/score.sbatch`:
```bash
#!/bin/bash
#SBATCH --job-name=ouro-score
#SBATCH --partition=<FILL_IN_PARTITION>
#SBATCH --account=<FILL_IN_ACCOUNT>
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=slurm_logs/score_%j.out

set -euo pipefail

MODELS=(
    "Ouro-1.4B"
    "Ouro-2.6B"
    "Ouro-1.4B-Thinking"
    "Ouro-2.6B-Thinking"
)

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ouro-vllm

for MODEL in "${MODELS[@]}"; do
    python -m score.score_extraction --model "$MODEL"
done
```

`<FILL_IN_PARTITION>` and `<FILL_IN_ACCOUNT>` are cluster-specific values the spec explicitly leaves as placeholders for you to fill in before submitting — this is the one intentional placeholder in this plan.

- [ ] **Step 3: Verify shell syntax**

Run:
```bash
bash -n slurm/generate.sbatch
bash -n slurm/score.sbatch
```
Expected: no output, exit code 0 for both

- [ ] **Step 4: Commit**

```bash
mkdir -p slurm_logs
touch slurm_logs/.gitkeep
git add slurm/generate.sbatch slurm/score.sbatch slurm_logs/.gitkeep
git commit -m "feat: add SLURM job scripts for generation and dependent scoring"
```

**To run on the cluster (after filling in partition/account, and after Task 1's config, Task 4's `data/build_prefix_dataset.py` has been run once to produce `data/prefixes.jsonl`):**
```bash
GEN_JOB=$(sbatch --parsable slurm/generate.sbatch)
sbatch --dependency=afterok:$GEN_JOB slurm/score.sbatch
```
