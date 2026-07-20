from __future__ import annotations

from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from docling.chunking import HybridChunker
from docling.datamodel.backend_options import MsExcelBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import (
    DocumentConverter,
    ExcelFormatOption,
    PdfFormatOption,
)
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
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .config import Settings


# RecursiveCharacterTextSplitter пробует разделители по порядку; пустая строка
# в конце гарантирует жёсткое деление по символам для очень длинного фрагмента.
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

# Форматы разделены по двум конвейерам, чтобы список каталога и load_file()
# не расходились при добавлении нового типа документа.
_DOCLING_EXTENSIONS = frozenset({".pdf", ".xlsx", ".odt", ".docx"})
_TEXT_EXTENSIONS = frozenset({".txt", ".md", ".rst"})
_SUPPORTED_EXTENSIONS = _DOCLING_EXTENSIONS | _TEXT_EXTENSIONS


class MarkdownTableSerializerProvider(ChunkingSerializerProvider):
    """Сериализовать таблицы в Markdown для сохранения строк и заголовков."""

    def get_serializer(self, doc: DoclingCoreDocument) -> BaseDocSerializer:
        """Создать Markdown-сериализатор таблиц для документа Docling."""
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),
        )


class DocumentProcessor:
    """
    Загружает документы и преобразует их в LangChain Document.

    TXT/Markdown/RST:
        LangChain loader -> RecursiveCharacterTextSplitter

    PDF/XLSX/ODT/DOCX:
        Docling -> HybridChunker -> LangChain Document
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        docling_chunk_tokens: int = 512,
        embedding_model: str | None = None,
        trust_remote_code: bool | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> None:
        """Настроить обычный и токен-ориентированный разделители текста."""
        runtime_settings = Settings.from_env()
        resolved_chunk_size = (
            runtime_settings.chunk_size if chunk_size is None else chunk_size
        )
        resolved_chunk_overlap = (
            runtime_settings.chunk_overlap
            if chunk_overlap is None
            else chunk_overlap
        )
        if resolved_chunk_size <= 0:
            raise ValueError("chunk_size должен быть больше нуля")
        if resolved_chunk_overlap < 0:
            raise ValueError("chunk_overlap не может быть отрицательным")
        if resolved_chunk_overlap >= resolved_chunk_size:
            raise ValueError("chunk_overlap должен быть меньше chunk_size")
        if docling_chunk_tokens <= 0:
            raise ValueError("docling_chunk_tokens должен быть больше нуля")

        # Для простого текста размер и overlap измеряются в символах. Docling-файлы
        # ниже делятся отдельно по токенам embedding-модели через HybridChunker.
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=resolved_chunk_size,
            chunk_overlap=resolved_chunk_overlap,
            separators=_MULTILINGUAL_SEPARATORS,
            length_function=len,
            is_separator_regex=False,
        )

        self.docling_chunk_tokens = docling_chunk_tokens
        self.embedding_model = embedding_model or runtime_settings.embedding_model
        # RagService передаёт сюда уже загруженный токенизатор embedding-модели,
        # чтобы чанкование и последующее построение векторов считали токены одинаково.
        self._hf_tokenizer = tokenizer
        self.trust_remote_code = (
            runtime_settings.trust_remote_code
            if trust_remote_code is None
            else trust_remote_code
        )

        # Тяжёлые компоненты создаются только при первом Docling-файле. При работе
        # исключительно с TXT/Markdown/RST модельный токенизатор здесь не загружается.
        self._docling_converter: DocumentConverter | None = None
        self._docling_chunker: HybridChunker | None = None

    def load(self, source: str | Path) -> list[Document]:
        """Загрузить один файл или все поддерживаемые файлы каталога."""
        path = Path(source).expanduser()
        if path.is_file():
            return self.load_file(path)
        if path.is_dir():
            return self.load_directory(path)
        raise FileNotFoundError(f"Файл или папка не найдены: {source}")

    def load_file(self, path: str | Path) -> list[Document]:
        """Преобразовать один поддерживаемый файл в индексируемые чанки."""
        path = Path(path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Файл не найден: {path}")
        suffix = path.suffix.lower()

        # Структурированные документы сначала переводятся в единый DoclingDocument.
        if suffix in _DOCLING_EXTENSIONS:
            return self._load_docling(path)

        if suffix in _TEXT_EXTENSIONS:
            loader = TextLoader(
                str(path),
                encoding="utf-8",
                autodetect_encoding=True,
            )
        else:
            raise ValueError(
                f"Неподдерживаемый тип файла: {suffix!r}. "
                "Поддерживаемые форматы: "
                ".txt .md .rst .pdf .docx .xlsx .odt"
            )
        raw_documents = loader.load()
        for document in raw_documents:
            # Метаданные копируются в каждый чанк функцией split_documents().
            document.metadata["source"] = path.name
        documents = self._splitter.split_documents(raw_documents)
        return self._add_document_ids(documents, path)

    def _load_docling(self, path: Path) -> list[Document]:
        """Преобразовать структурированный документ через Docling и HybridChunker."""
        converter, chunker = self._get_docling_components()
        conversion = converter.convert(source=path)
        docling_document = conversion.document
        documents: list[Document] = []

        for chunk in chunker.chunk(dl_doc=docling_document):
            # contextualize() добавляет к содержимому структурный контекст Docling:
            # заголовки разделов, подписи и повторённый заголовок таблицы.
            content = chunker.contextualize(chunk=chunk).strip()
            if not content:
                continue

            # Одинаковые тексты намеренно не удаляются: повторяющиеся строки таблицы
            # могут описывать разные реальные операции и должны участвовать в расчётах.
            chunk_index = len(documents) + 1

            # Сохраняем исходные метаданные Docling (страницы, заголовки, provenance)
            # и дополняем их единообразными полями, используемыми остальным RAG.
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
        """Добавить стабильные идентификаторы источника и каждого чанка."""
        # Абсолютный путь отличает одноимённые файлы из разных каталогов и служит
        # ключом для удаления старых чанков этого источника перед переиндексацией.
        source_identity = str(path.resolve())
        for chunk_index, document in enumerate(documents, start=1):
            document.metadata.setdefault("source", path.name)
            document.metadata["source_id"] = source_identity
            document.metadata.setdefault(
                "document_type",
                path.suffix.lower().lstrip("."),
            )
            document.metadata.setdefault("chunk_index", chunk_index)
            identity = f"{source_identity}|{document.metadata['chunk_index']}"
            document.metadata["doc_id"] = str(uuid5(NAMESPACE_URL, identity))
        return documents

    def _get_docling_components(
        self,
    ) -> tuple[DocumentConverter, HybridChunker]:
        """Лениво создать конвертер Docling и совместимый с эмбеддингами чункер."""
        if self._docling_converter is None:
            # Явный allow-list не даёт случайно отправить в Docling неподдерживаемый
            # текущим маршрутизатором формат.
            self._docling_converter = DocumentConverter(
                allowed_formats=[
                    InputFormat.PDF,
                    InputFormat.XLSX,
                    InputFormat.ODT,
                    InputFormat.DOCX,
                ],
                format_options={
                    # Если PDF содержат готовый текстовый слой то OCR должен быть отключён, чтобы
                    # не загружать модели RapidOCR и не распознавать текст повторно.
                    # Layout-анализ и восстановление структуры таблиц остаются включены.
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=PdfPipelineOptions(
                            do_ocr=False,
                            do_table_structure=True,
                        )
                    ),
                    # Диаграммы часто повторяют значения исходных ячеек; их разбор
                    # отключён, чтобы не индексировать одни финансовые данные дважды.
                    InputFormat.XLSX: ExcelFormatOption(
                        backend_options=MsExcelBackendOptions(
                            parse_charts=False,
                        )
                    )
                },
            )

        if self._docling_chunker is None:
            hf_tokenizer = self._hf_tokenizer
            if hf_tokenizer is None:
                # Fallback нужен при автономном использовании DocumentProcessor;
                # в RagService сюда обычно передаётся готовый токенизатор.
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
                # Таблицы сериализуются в Markdown, небольшие соседние элементы
                # объединяются, а заголовок повторяется в каждой части большой таблицы.
                serializer_provider=MarkdownTableSerializerProvider(),
                merge_peers=True,
                repeat_table_header=True,
            )

        return self._docling_converter, self._docling_chunker

    def load_directory(self, directory: str | Path) -> list[Document]:
        """Рекурсивно загрузить поддерживаемые документы из каталога."""
        directory = Path(directory).expanduser()
        if not directory.is_dir():
            raise NotADirectoryError(f"Каталог не найден: {directory}")
        all_chunks: list[Document] = []

        # Сортировка делает порядок чанков и диагностические сообщения воспроизводимыми.
        for file_path in sorted(directory.rglob("*")):
            if not file_path.is_file():
                continue

            if file_path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
                continue

            print(f"[Processor] Загрузка {file_path.name} ...")

            try:
                all_chunks.extend(self.load_file(file_path))
            except Exception as error:
                # Best-effort режим: повреждённый файл не блокирует остальные.
                # Сообщение важно проверять — возвращаемый индекс может быть частичным.
                print(
                    f"[Processor] Пропущено "
                    f"{file_path.name}: {error}"
                )

        return all_chunks
