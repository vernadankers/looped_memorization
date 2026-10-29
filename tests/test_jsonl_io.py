from common.jsonl_io import read_jsonl, write_jsonl


def test_write_then_read_round_trip(tmp_path):
    path = tmp_path / "rows.jsonl"
    rows = [{"doc_id": 0, "text": "hello"}, {"doc_id": 1, "text": "world"}]
    write_jsonl(str(path), rows)
    result = read_jsonl(str(path))
    assert result == rows


def test_round_trip_preserves_unicode(tmp_path):
    path = tmp_path / "rows.jsonl"
    rows = [{"doc_id": 0, "text": "café élève 中文"}]
    write_jsonl(str(path), rows)
    result = read_jsonl(str(path))
    assert result == rows


def test_read_skips_blank_lines(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
    result = read_jsonl(str(path))
    assert result == [{"a": 1}, {"a": 2}]


def test_write_jsonl_creates_missing_parent_directories(tmp_path):
    path = tmp_path / "nested" / "dir" / "rows.jsonl"
    rows = [{"doc_id": 0, "text": "hello"}]
    write_jsonl(str(path), rows)
    result = read_jsonl(str(path))
    assert result == rows
