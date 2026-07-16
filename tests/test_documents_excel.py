from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from rag_app.documents import (
    DocumentProcessor,
    normalize_repeated_table_cells,
)


@dataclass
class _Cell:
    text: str
    start_row_offset_idx: int
    start_col_offset_idx: int


class _Meta:
    def export_json_dict(self) -> dict:
        return {"headings": ["Тестовый лист"]}


@dataclass
class _Chunk:
    text: str
    meta: _Meta


class _Converter:
    def __init__(self, document) -> None:
        self.document = document

    def convert(self, source: Path):
        return SimpleNamespace(document=self.document)


class _Chunker:
    def __init__(self, chunks: list[_Chunk]) -> None:
        self.chunks = chunks

    def chunk(self, dl_doc):
        return iter(self.chunks)

    def contextualize(self, chunk: _Chunk) -> str:
        return chunk.text


def test_normalize_repeated_table_cells_collapses_only_long_repeats() -> None:
    cells = [
        _Cell("Повтор", 0, 0),
        _Cell("Повтор", 0, 1),
        _Cell("Повтор", 0, 2),
        _Cell("Повтор", 0, 3),
        _Cell("Два", 1, 0),
        _Cell("Два", 1, 1),
        _Cell("Поле", 2, 0),
        _Cell("Значение", 2, 1),
    ]
    document = SimpleNamespace(
        tables=[SimpleNamespace(data=SimpleNamespace(table_cells=cells))]
    )

    normalized = normalize_repeated_table_cells(document)

    assert normalized == 3
    assert [cell.text for cell in cells[:4]] == ["Повтор", "", "", ""]
    assert [cell.text for cell in cells[4:6]] == ["Два", "Два"]
    assert [cell.text for cell in cells[6:]] == ["Поле", "Значение"]


def test_text_file_gets_chunk_metadata_and_respects_zero_overlap(
    tmp_path: Path,
) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("Первый абзац. Второй абзац. Третий абзац.", encoding="utf-8")
    processor = DocumentProcessor(chunk_size=20, chunk_overlap=0)

    documents = processor.load(path)

    assert documents
    assert processor._splitter._chunk_overlap == 0
    assert len({document.metadata["doc_id"] for document in documents}) == len(documents)
    assert all(document.metadata["source"] == path.name for document in documents)
    assert all(document.metadata["source_id"] == str(path.resolve()) for document in documents)
    assert [document.metadata["chunk_index"] for document in documents] == list(
        range(1, len(documents) + 1)
    )


def test_docling_chunks_are_deduplicated_and_receive_stable_ids(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sales.xlsx"
    path.write_bytes(b"fake workbook")
    converted_document = SimpleNamespace(tables=[])
    chunks = [
        _Chunk("Общая выручка: 100", _Meta()),
        _Chunk("Общая   выручка: 100", _Meta()),
        _Chunk("Максимальная выручка: 60", _Meta()),
    ]
    processor = DocumentProcessor(chunk_size=100, chunk_overlap=0)
    processor._docling_converter = _Converter(converted_document)
    processor._docling_chunker = _Chunker(chunks)

    documents = processor.load_file(path)

    assert [document.page_content for document in documents] == [
        "Общая выручка: 100",
        "Максимальная выручка: 60",
    ]
    assert [document.metadata["chunk_index"] for document in documents] == [1, 2]
    assert len({document.metadata["doc_id"] for document in documents}) == 2
    assert all(document.metadata["document_type"] == "xlsx" for document in documents)
    assert all("_prechunked" not in document.metadata for document in documents)
