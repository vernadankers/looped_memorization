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
