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
    for text, meta in documents:
        if len(rows) >= num_samples:
            break
        split = split_document(text, tokenizer, prefix_tokens, continuation_tokens)
        if split is None:
            continue
        rows.append({
            "doc_id": doc_id,
            "prefix": split["prefix"],
            "true_continuation": split["true_continuation"],
            "source_meta": meta,
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

    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"], trust_remote_code=True)
    dataset = load_dataset(config["corpus"], split=config["corpus_split"], streaming=True)
    dataset = dataset.shuffle(seed=config["seed"], buffer_size=10_000)
    documents = (
        (row["text"], {"stream_index": i, "pile_set_name": row.get("meta", {}).get("pile_set_name")})
        for i, row in enumerate(dataset)
    )

    rows = build_dataset(documents, tokenizer, config["prefix_tokens"],
                          config["continuation_tokens"], num_samples)

    if len(rows) < num_samples:
        print(f"warning: only found {len(rows)} usable documents out of requested {num_samples}",
              file=sys.stderr)

    write_jsonl(args.out, rows)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
