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


def test_score_rows_empty_generated_continuation_does_not_crash_or_match():
    rows = [make_row(0, "", "hello world")]
    scored, _ = score_rows(rows, overlap_threshold=0.8)
    assert scored[0]["exact_match"] is False


def test_score_rows_both_empty_is_not_counted_as_exact_match():
    rows = [make_row(0, "", "")]
    scored, _ = score_rows(rows, overlap_threshold=0.8)
    assert scored[0]["exact_match"] is False


class LowercasingFakeTokenizer:
    def encode(self, text):
        return text.lower().split()


def test_score_rows_uses_provided_tokenizer_instead_of_plain_split():
    # "Hello" vs "hello" differ under .split() but are identical once the
    # tokenizer's encode() lowercases them first, proving the tokenizer path
    # is actually exercised rather than silently falling back to .split().
    rows = [make_row(0, "Hello World", "hello world")]
    scored, _ = score_rows(rows, overlap_threshold=0.8, tokenizer=LowercasingFakeTokenizer())
    assert scored[0]["exact_match"] is True
    assert scored[0]["longest_common_prefix_len"] == 2
