from __future__ import annotations

import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import urlparse
from uuid import NAMESPACE_URL, uuid5

import pandas as pd
import requests
from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

LOGGER = logging.getLogger(__name__)


class DocumentProcessor:
    supported_extensions = {".docx", ".txt", ".md", ".rst", ".pdf", ".xls", ".xlsx"}

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
            length_function=len,
        )

    def load(self, source: str) -> list[Document]:
        source = source.strip().strip('"')
        if source.startswith(("http://", "https://")):
            documents = self._load_url(source)
        else:
            path = Path(source).expanduser().resolve()
            if path.is_file():
                documents = self._load_file(path)
            elif path.is_dir():
                documents = self._load_directory(path)
            else:
                raise FileNotFoundError(f"Файл или папка не найдены: {source}")

        chunks = self.splitter.split_documents(documents)
        for index, chunk in enumerate(chunks):
            identity = "|".join(
                [
                    str(chunk.metadata.get("source", "unknown")),
                    str(chunk.metadata.get("page", chunk.metadata.get("sheet", ""))),
                    str(index),
                    chunk.page_content,
                ]
            )
            chunk.metadata["doc_id"] = str(uuid5(NAMESPACE_URL, identity))
            chunk.metadata["chunk_index"] = index
        return chunks

    def _load_directory(self, directory: Path) -> list[Document]:
        documents: list[Document] = []
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in self.supported_extensions:
                continue
            LOGGER.info("Обработка %s", path)
            try:
                documents.extend(self._load_file(path))
            except Exception as error:
                LOGGER.warning("Не удалось обработать %s: %s", path, error)
        return documents

    def _load_file(self, path: Path, source: str | None = None) -> list[Document]:
        suffix = path.suffix.lower()
        display_source = source or str(path)
        if suffix in {".txt", ".md", ".rst"}:
            return [Document(page_content=self._read_text(path), metadata={"source": display_source})]
        if suffix == ".pdf":
            return self._load_pdf(path, display_source)
        if suffix == ".docx":
            return self._load_docx(path, display_source)
        if suffix in {".xls", ".xlsx"}:
            return self._load_excel(path, display_source)
        raise ValueError(f"Формат {suffix or '<без расширения>'} не поддерживается")

    @staticmethod
    def _read_text(path: Path) -> str:
        for encoding in ("utf-8-sig", "utf-8", "cp1251"):
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        return path.read_text(encoding="utf-8", errors="replace")

    @staticmethod
    def _load_pdf(path: Path, source: str) -> list[Document]:
        reader = PdfReader(str(path))
        documents = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": source, "page": page_number},
                    )
                )
        return documents

    @staticmethod
    def _load_docx(path: Path, source: str) -> list[Document]:
        document = DocxDocument(str(path))
        text = "\n".join(paragraph.text for paragraph in document.paragraphs if paragraph.text)
        return [Document(page_content=text, metadata={"source": source})]

    def _load_excel(self, path: Path, source: str) -> list[Document]:
        workbook = pd.ExcelFile(path)
        documents: list[Document] = []
        for sheet_name in workbook.sheet_names:
            raw_frame = pd.read_excel(workbook, sheet_name=sheet_name, header=None)
            raw_frame = raw_frame.dropna(axis="columns", how="all")
            if raw_frame.empty:
                continue

            header_index = self._detect_excel_header(raw_frame)
            headers = self._unique_headers(raw_frame.iloc[header_index].tolist())
            data_frame = raw_frame.iloc[header_index + 1 :].copy()
            data_frame.columns = headers
            data_frame = data_frame.dropna(axis="rows", how="all")

            header_line = "Колонки: " + " | ".join(headers)
            current_lines = [header_line]
            first_row_number: int | None = None
            last_row_number: int | None = None

            for row_index, row in data_frame.iterrows():
                fields = [
                    f"{column}: {self._format_cell(value)}"
                    for column, value in row.items()
                    if not pd.isna(value) and str(value).strip()
                ]
                if not fields:
                    continue
                row_number = int(row_index) + 1
                row_line = "; ".join(fields)
                candidate = "\n".join([*current_lines, row_line])
                if len(candidate) > self.chunk_size and len(current_lines) > 1:
                    documents.append(
                        self._excel_document(
                            source,
                            sheet_name,
                            current_lines,
                            first_row_number,
                            last_row_number,
                        )
                    )
                    current_lines = [header_line]
                    first_row_number = None

                current_lines.append(row_line)
                first_row_number = first_row_number or row_number
                last_row_number = row_number

            if len(current_lines) > 1:
                documents.append(
                    self._excel_document(
                        source,
                        sheet_name,
                        current_lines,
                        first_row_number,
                        last_row_number,
                    )
                )
        return documents

    @staticmethod
    def _detect_excel_header(frame: pd.DataFrame) -> int:
        sample = frame.head(25)
        if sample.empty:
            return 0
        scored_rows: list[tuple[int, int, int]] = []
        for index, row in sample.iterrows():
            values = [str(value).strip() for value in row.tolist() if not pd.isna(value)]
            unique_count = len(set(values))
            scored_rows.append((unique_count, len(values), int(index)))
        best_unique = max(item[0] for item in scored_rows)
        best_non_empty = max(item[1] for item in scored_rows if item[0] == best_unique)
        return next(
            index
            for unique_count, non_empty_count, index in scored_rows
            if unique_count == best_unique and non_empty_count == best_non_empty
        )

    @staticmethod
    def _unique_headers(values: list[object]) -> list[str]:
        headers: list[str] = []
        counts: dict[str, int] = {}
        for index, value in enumerate(values, start=1):
            base = str(value).strip() if not pd.isna(value) else f"column_{index}"
            count = counts.get(base, 0) + 1
            counts[base] = count
            headers.append(base if count == 1 else f"{base}_{count}")
        return headers

    @staticmethod
    def _format_cell(value: object) -> str:
        if isinstance(value, pd.Timestamp):
            return value.isoformat(sep=" ")
        return str(value)

    @staticmethod
    def _excel_document(
        source: str,
        sheet_name: str,
        lines: list[str],
        first_row: int | None,
        last_row: int | None,
    ) -> Document:
        return Document(
            page_content="\n".join(lines),
            metadata={
                "source": source,
                "sheet": sheet_name,
                "first_row": first_row,
                "last_row": last_row,
            },
        )

    def _load_url(self, url: str) -> list[Document]:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        suffix = Path(urlparse(url).path).suffix.lower()
        if suffix in self.supported_extensions:
            with NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
                temp_file.write(response.content)
                temp_path = Path(temp_file.name)
            try:
                return self._load_file(temp_path, source=url)
            finally:
                temp_path.unlink(missing_ok=True)

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())
        return [Document(page_content=text, metadata={"source": url})]
