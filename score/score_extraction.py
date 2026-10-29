import argparse
import json

import yaml
from transformers import AutoTokenizer

from common.jsonl_io import read_jsonl, write_jsonl
from score.metrics import exact_match, longest_common_prefix_len, overlap_ratio


def score_rows(rows, overlap_threshold, tokenizer=None):
    scored = []
    num_exact_match = 0
    num_high_overlap = 0

    for row in rows:
        if tokenizer is not None:
            generated_tokens = tokenizer.encode(row["generated_continuation"])
            true_tokens = tokenizer.encode(row["true_continuation"])
        else:
            generated_tokens = row["generated_continuation"].split()
            true_tokens = row["true_continuation"].split()

        is_exact_match = exact_match(generated_tokens, true_tokens) and len(true_tokens) > 0
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

    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])

    rows = read_jsonl(generations_path)
    scored_rows, summary = score_rows(rows, config["high_overlap_threshold"], tokenizer=tokenizer)

    write_jsonl(out_path, scored_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
