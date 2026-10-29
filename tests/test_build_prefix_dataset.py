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
    documents = [
        (short_doc, {"stream_index": 0}),
        (long_doc, {"stream_index": 1}),
        (long_doc, {"stream_index": 2}),
    ]

    rows = build_dataset(documents, tokenizer, prefix_tokens=4, continuation_tokens=3, num_samples=10)

    assert len(rows) == 2
    assert [r["doc_id"] for r in rows] == [0, 1]
    assert rows[0]["prefix"] == "tok0 tok1 tok2 tok3"
    assert rows[0]["true_continuation"] == "tok4 tok5 tok6"
    assert rows[0]["source_meta"] == {"stream_index": 1}


def test_build_dataset_stops_at_num_samples():
    tokenizer = FakeTokenizer()
    long_doc = " ".join(f"tok{i}" for i in range(10))
    documents = [(long_doc, {"stream_index": i}) for i in range(5)]

    rows = build_dataset(documents, tokenizer, prefix_tokens=4, continuation_tokens=3, num_samples=2)

    assert len(rows) == 2


def test_build_dataset_returns_fewer_rows_when_corpus_runs_out():
    tokenizer = FakeTokenizer()
    long_doc = " ".join(f"tok{i}" for i in range(10))
    documents = [(long_doc, {"stream_index": 0})]  # only one usable document available, far fewer than requested

    rows = build_dataset(documents, tokenizer, prefix_tokens=4, continuation_tokens=3, num_samples=10)

    assert len(rows) == 1
