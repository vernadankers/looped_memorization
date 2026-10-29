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
