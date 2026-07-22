from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from docling.datamodel.base_models import InputFormat
from rag_app.config import Settings
from rag_app.documents import DocumentProcessor


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


def test_text_file_gets_chunk_metadata_and_respects_zero_overlap(
    tmp_path: Path,
) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("Первый абзац. Второй абзац. Третий абзац.", encoding="utf-8")
    processor = DocumentProcessor(Settings(chunk_size=20, chunk_overlap=0))

    documents = processor.load(path)

    assert documents
    assert processor._splitter._chunk_overlap == 0
    assert len({document.metadata["doc_id"] for document in documents}) == len(documents)
    assert all(document.metadata["source"] == path.name for document in documents)
    assert all(document.metadata["source_id"] == str(path.resolve()) for document in documents)
    assert all(document.metadata["document_type"] == "txt" for document in documents)
    assert [document.metadata["chunk_index"] for document in documents] == list(
        range(1, len(documents) + 1)
    )


def test_docling_chunks_preserve_repeated_rows_and_receive_stable_ids(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sales.xlsx"
    path.write_bytes(b"fake workbook")
    converted_document = SimpleNamespace()
    chunks = [
        _Chunk("Общая выручка: 100", _Meta()),
        _Chunk("Общая   выручка: 100", _Meta()),
        _Chunk("Максимальная выручка: 60", _Meta()),
    ]
    processor = DocumentProcessor(Settings(chunk_size=100, chunk_overlap=0))
    processor._docling_converter = _Converter(converted_document)
    processor._docling_chunker = _Chunker(chunks)

    documents = processor.load_file(path)

    assert [document.page_content for document in documents] == [
        "Общая выручка: 100",
        "Общая   выручка: 100",
        "Максимальная выручка: 60",
    ]
    assert [document.metadata["chunk_index"] for document in documents] == [1, 2, 3]
    assert len({document.metadata["doc_id"] for document in documents}) == 3
    assert all(document.metadata["document_type"] == "xlsx" for document in documents)
    assert all("_prechunked" not in document.metadata for document in documents)


def test_pdf_is_processed_by_docling(tmp_path: Path) -> None:
    path = tmp_path / "report.pdf"
    path.write_bytes(b"fake pdf")
    processor = DocumentProcessor(Settings(chunk_size=100, chunk_overlap=0))
    processor._docling_converter = _Converter(SimpleNamespace())
    processor._docling_chunker = _Chunker(
        [_Chunk("Выручка по региону Урал: 638 000", _Meta())]
    )

    documents = processor.load_file(path)

    assert [document.page_content for document in documents] == [
        "Выручка по региону Урал: 638 000"
    ]
    assert documents[0].metadata["document_type"] == "pdf"


def test_docling_pdf_pipeline_disables_ocr(monkeypatch) -> None:
    captured: dict = {}

    class _ConfiguredConverter:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("rag_app.documents.DocumentConverter", _ConfiguredConverter)
    processor = DocumentProcessor(Settings(chunk_size=100, chunk_overlap=0))
    processor._docling_chunker = object()

    processor._get_docling_components()

    pdf_option = captured["format_options"][InputFormat.PDF]
    assert pdf_option.pipeline_options.do_ocr is False
    assert pdf_option.pipeline_options.do_table_structure is True
