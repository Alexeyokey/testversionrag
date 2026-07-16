from __future__ import annotations

from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from docling.chunking import HybridChunker
from docling.datamodel.backend_options import MsExcelBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, ExcelFormatOption
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.chunker.tokenizer.huggingface import (
    HuggingFaceTokenizer,
)
from docling_core.transforms.serializer.base import BaseDocSerializer
from docling_core.transforms.serializer.markdown import MarkdownTableSerializer
from docling_core.types import DoclingDocument as DoclingCoreDocument
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from .config import Settings


_MULTILINGUAL_SEPARATORS = [
    "\n\n",
    "\n",
    ". ",
    "! ",
    "? ",
    ".\n",
    "!\n",
    "?\n",
    "; ",
    ", ",
    " ",
    "",
]


class MarkdownTableSerializerProvider(ChunkingSerializerProvider):
    """Serialize tables as Markdown so they can be split by complete rows."""

    def get_serializer(self, doc: DoclingCoreDocument) -> BaseDocSerializer:
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),
        )


def normalize_repeated_table_cells(
    document: DoclingCoreDocument,
    minimum_repetitions: int = 3,
) -> int:
    """Blank copied merged-like cells without changing the source workbook."""
    normalized_cells = 0

    for table in document.tables:
        cells_by_row: dict[int, list] = {}
        for cell in table.data.table_cells:
            if cell.text.strip():
                cells_by_row.setdefault(cell.start_row_offset_idx, []).append(cell)

        for row_cells in cells_by_row.values():
            if len(row_cells) < minimum_repetitions:
                continue

            normalized_texts = {
                " ".join(cell.text.split()).casefold()
                for cell in row_cells
            }
            if len(normalized_texts) != 1:
                continue

            ordered_cells = sorted(
                row_cells,
                key=lambda cell: cell.start_col_offset_idx,
            )
            for cell in ordered_cells[1:]:
                cell.text = ""
                normalized_cells += 1

    return normalized_cells


class DocumentProcessor:
    """
    Загружает документы и преобразует их в LangChain Document.

    TXT/Markdown/PDF:
        LangChain loader -> RecursiveCharacterTextSplitter

    XLSX:
        Docling -> HybridChunker -> LangChain Document
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        docling_chunk_tokens: int = 512,
        embedding_model: str | None = None,
        trust_remote_code: bool | None = None,
    ) -> None:
        runtime_settings = Settings.from_env()
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size or runtime_settings.chunk_size,
            chunk_overlap=chunk_overlap or runtime_settings.chunk_overlap,
            separators=_MULTILINGUAL_SEPARATORS,
            length_function=len,
            is_separator_regex=False,
        )

        self.docling_chunk_tokens = docling_chunk_tokens
        self.embedding_model = embedding_model or runtime_settings.embedding_model
        self.trust_remote_code = (
            runtime_settings.trust_remote_code
            if trust_remote_code is None
            else trust_remote_code
        )

        # Создаются лениво: при обработке только TXT/PDF
        # Docling и HF-токенизатор загружаться не будут.
        self._docling_converter: DocumentConverter | None = None
        self._docling_chunker: HybridChunker | None = None

    def load(self, source: str | Path) -> list[Document]:
        """Load a file or all supported files from a directory."""
        path = Path(source).expanduser()
        if path.is_file():
            return self.load_file(path)
        if path.is_dir():
            return self.load_directory(path)
        raise FileNotFoundError(f"Файл или папка не найдены: {source}")

    def load_file(self, path: str | Path) -> list[Document]:
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix in {".xlsx", ".odt", ".docx"}:
            return self._load_docling(path)

        if suffix in {".txt", ".md", ".rst"}:
            loader = TextLoader(
                str(path),
                encoding="utf-8",
                autodetect_encoding=True,
            )
        elif suffix == ".pdf":
            loader = PyPDFLoader(str(path))
        else:
            raise ValueError(
                f"Неподдерживаемый тип файла: {suffix!r}. "
                "Поддерживаемые форматы: "
                ".txt .md .rst .pdf .docx .xlsx .odt"
            )

        raw_documents = loader.load()

        for document in raw_documents:
            document.metadata["source"] = path.name

        documents = self._splitter.split_documents(raw_documents)
        return self._add_document_ids(documents, path)

    def _load_docling(self, path: Path) -> list[Document]:
        converter, chunker = self._get_docling_components()

        conversion = converter.convert(source=path)
        docling_document = conversion.document
        if path.suffix.lower() == ".xlsx":
            normalize_repeated_table_cells(docling_document)

        documents: list[Document] = []
        seen_content: set[str] = set()

        for chunk in chunker.chunk(dl_doc=docling_document):
            content = chunker.contextualize(chunk=chunk).strip()

            if not content:
                continue

            content_key = " ".join(content.split()).casefold()
            if content_key in seen_content:
                continue
            seen_content.add(content_key)

            chunk_index = len(documents) + 1

            metadata = chunk.meta.export_json_dict()
            metadata.update(
                {
                    "source": path.name,
                    "chunk_index": chunk_index,
                    "document_type": path.suffix.lower().lstrip("."),
                }
            )

            documents.append(
                Document(
                    page_content=content,
                    metadata=metadata,
                )
            )

        return self._add_document_ids(documents, path)

    @staticmethod
    def _add_document_ids(
        documents: list[Document],
        path: Path,
    ) -> list[Document]:
        source_identity = str(path.resolve())
        for chunk_index, document in enumerate(documents, start=1):
            document.metadata.setdefault("source", path.name)
            document.metadata.setdefault("chunk_index", chunk_index)
            identity = "|".join(
                (
                    source_identity,
                    str(document.metadata["chunk_index"]),
                    document.page_content,
                )
            )
            document.metadata["doc_id"] = str(uuid5(NAMESPACE_URL, identity))
        return documents

    def _get_docling_components(
        self,
    ) -> tuple[DocumentConverter, HybridChunker]:
        if self._docling_converter is None:
            self._docling_converter = DocumentConverter(
                allowed_formats=[
                    InputFormat.XLSX,
                    InputFormat.ODT,
                    InputFormat.DOCX,
                ],
                format_options={
                    InputFormat.XLSX: ExcelFormatOption(
                        backend_options=MsExcelBackendOptions(
                            parse_charts=False,
                        )
                    )
                },
            )

        if self._docling_chunker is None:
            hf_tokenizer = AutoTokenizer.from_pretrained(
                self.embedding_model,
                trust_remote_code=self.trust_remote_code,
            )

            docling_tokenizer = HuggingFaceTokenizer(
                tokenizer=hf_tokenizer,
                max_tokens=self.docling_chunk_tokens,
            )

            self._docling_chunker = HybridChunker(
                tokenizer=docling_tokenizer,
                serializer_provider=MarkdownTableSerializerProvider(),
                merge_peers=True,
                repeat_table_header=True,
            )

        return self._docling_converter, self._docling_chunker

    def load_directory(self, directory: str | Path) -> list[Document]:
        directory = Path(directory)
        all_chunks: list[Document] = []

        supported = {
            ".txt",
            ".md",
            ".rst",
            ".docx",
            ".pdf",
            ".xlsx",
            ".odt",
        }

        for file_path in sorted(directory.rglob("*")):
            if not file_path.is_file():
                continue

            if file_path.suffix.lower() not in supported:
                continue

            print(f"[Processor] Загрузка {file_path.name} ...")

            try:
                all_chunks.extend(self.load_file(file_path))
            except Exception as error:
                print(
                    f"[Processor] Пропущено "
                    f"{file_path.name}: {error}"
                )

        return all_chunks
