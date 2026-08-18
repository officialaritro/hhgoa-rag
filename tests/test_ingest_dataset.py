import json
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.ingest_dataset import extract_english_records, ingest


def test_extract_english_records_pairs_passages_with_is_selected_by_index():
    records = extract_english_records(
        passages_batch=[
            {
                "English_passages": ["passage A", "passage B"],
                "is_selected": [1, 0],
            }
        ],
        eng_query_batch=["what is X"],
        eng_answer_batch=["X is Y"],
        query_id_batch=[123],
    )

    assert len(records) == 1
    record = records[0]
    assert record["query_id"] == 123
    assert record["query"] == "what is X"
    assert record["answer"] == "X is Y"
    assert record["passages"] == [
        {"text": "passage A", "is_selected": True},
        {"text": "passage B", "is_selected": False},
    ]


def test_extract_english_records_handles_multiple_rows():
    records = extract_english_records(
        passages_batch=[
            {"English_passages": ["p1"], "is_selected": [1]},
            {"English_passages": ["p2"], "is_selected": [0]},
        ],
        eng_query_batch=["q1", "q2"],
        eng_answer_batch=["a1", "a2"],
        query_id_batch=[1, 2],
    )

    assert [r["query_id"] for r in records] == [1, 2]


def _write_fixture_parquet(path, num_rows):
    table = pa.table(
        {
            "query_id": list(range(num_rows)),
            "Eng_Query": [f"query {i}" for i in range(num_rows)],
            "Eng_Answer": [f"answer {i}" for i in range(num_rows)],
            "passages": [
                {"English_passages": [f"passage {i}"], "is_selected": [1]}
                for i in range(num_rows)
            ],
        }
    )
    pq.write_table(table, path)


@patch("scripts.ingest_dataset.download_parquet")
def test_ingest_respects_row_limit(mock_download, tmp_path):
    fixture_path = tmp_path / "fixture.parquet"
    _write_fixture_parquet(fixture_path, num_rows=10)
    mock_download.return_value = str(fixture_path)
    output_path = tmp_path / "corpus.jsonl"

    written = ingest(row_limit=3, output_path=str(output_path))

    assert written == 3
    lines = output_path.read_text().strip().split("\n")
    assert len(lines) == 3
    first = json.loads(lines[0])
    assert first["query_id"] == 0
