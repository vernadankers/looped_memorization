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
