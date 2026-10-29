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
