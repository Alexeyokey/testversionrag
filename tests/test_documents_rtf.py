from __future__ import annotations

from pathlib import Path

import pytest

from rag_app.config import Settings
from rag_app.documents import DocumentProcessor, _rtf_to_text


def _sample_rtf() -> bytes:
    return (
        rb"{\rtf1\ansi\ansicpg1251\deff0"
        rb"{\fonttbl{\f0\fcharset204 Arial;}}"
        rb"{\*\generator Hidden generator;}"
        rb"\uc1\pard "
        + "Федеральный ".encode("cp1251")
        + rb"\b "
        + "закон ".encode("cp1251")
        + rb"\b0 \u8470? 273-\u1060?\u1047?\par "
        + "Уровни:".encode("cp1251")
        + rb"\tab "
        + "дошкольное".encode("cp1251")
        + rb"\line "
        + "среднее.".encode("cp1251")
        + rb"\par \'cf\'f0\'e8\'e2\'e5\'f2\par "
        + rb"{\field{\*\fldinst HYPERLINK \"https://example.test\"}{\fldrslt "
        + "текст ссылки".encode("cp1251")
        + rb"}}}"
    )


def test_rtf_to_text_decodes_cyrillic_unicode_and_visible_field_result() -> None:
    text = _rtf_to_text(_sample_rtf())

    assert "Федеральный закон № 273-ФЗ" in text
    assert "Уровни: дошкольное\nсреднее." in text
    assert "Привет" in text
    assert "текст ссылки" in text
    assert "Hidden generator" not in text
    assert "https://example.test" not in text


def test_rtf_without_ansicpg_uses_cyrillic_font_charset() -> None:
    raw = (
        rb"{\rtf1\ansi{\fonttbl{\f0\fcharset204 Arial;}}\f0 " + "Кириллица".encode("cp1251") + b"}"
    )

    assert _rtf_to_text(raw) == "Кириллица"


def test_rtf_file_uses_text_splitter_and_receives_metadata(tmp_path: Path) -> None:
    path = tmp_path / "law.rtf"
    path.write_bytes(_sample_rtf())
    processor = DocumentProcessor(Settings(chunk_size=80, chunk_overlap=10))

    documents = processor.load_directory(tmp_path)

    assert documents
    assert "Федеральный закон № 273-ФЗ" in "\n".join(
        document.page_content for document in documents
    )
    assert all(document.metadata["source"] == path.name for document in documents)
    assert all(document.metadata["document_type"] == "rtf" for document in documents)
    assert all(document.metadata["source_id"] == str(path.resolve()) for document in documents)
    assert len({document.metadata["doc_id"] for document in documents}) == len(documents)


def test_invalid_rtf_header_is_rejected() -> None:
    with pytest.raises(ValueError, match="заголовок RTF"):
        _rtf_to_text("обычный текст".encode("utf-8"))
