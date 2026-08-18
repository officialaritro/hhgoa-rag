"""Downloads train/hintrain.parquet and extracts the first N rows' English
fields as the ingestion corpus (plan Task 2). Row cap and file choice per
plan Global Constraints -- corrected mid-implementation after discovering the
dataset's real per-language-file structure (see plan Context for Implementer).
"""

import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

DEFAULT_ROW_LIMIT = 10_000
REPO_ID = "ai4bharat/MSMARCO-XI"
FILENAME = "train/hintrain.parquet"
DEFAULT_OUTPUT_PATH = "data/corpus.jsonl"


def extract_english_records(
    passages_batch: list[dict[str, Any]],
    eng_query_batch: list[str],
    eng_answer_batch: list[str],
    query_id_batch: list[int],
) -> list[dict[str, Any]]:
    """Pure function: pairs each row's English_passages with its is_selected
    flag by list index (confirmed real schema -- the two lists are parallel,
    not nested per-passage)."""
    records = []
    for query_id, eng_query, eng_answer, passages in zip(
        query_id_batch, eng_query_batch, eng_answer_batch, passages_batch
    ):
        texts = passages["English_passages"]
        selected_flags = passages["is_selected"]
        records.append(
            {
                "query_id": query_id,
                "query": eng_query,
                "answer": eng_answer,
                "passages": [
                    {"text": text, "is_selected": bool(flag)}
                    for text, flag in zip(texts, selected_flags)
                ],
            }
        )
    return records


def download_parquet() -> str:
    return hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=FILENAME)


def ingest(
    row_limit: int = DEFAULT_ROW_LIMIT, output_path: str = DEFAULT_OUTPUT_PATH
) -> int:
    path = download_parquet()
    parquet_file = pq.ParquetFile(path)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with open(output_path, "w") as f:
        for batch in parquet_file.iter_batches(
            batch_size=1000, columns=["query_id", "Eng_Query", "Eng_Answer", "passages"]
        ):
            if rows_written >= row_limit:
                break
            remaining = row_limit - rows_written
            columns = batch.to_pydict()
            records = extract_english_records(
                columns["passages"][:remaining],
                columns["Eng_Query"][:remaining],
                columns["Eng_Answer"][:remaining],
                columns["query_id"][:remaining],
            )
            f.writelines(json.dumps(record) + "\n" for record in records)
            rows_written += len(records)
    return rows_written


if __name__ == "__main__":
    written = ingest()
    print(f"wrote {written} rows to {DEFAULT_OUTPUT_PATH}")
