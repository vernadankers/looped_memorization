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
