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
