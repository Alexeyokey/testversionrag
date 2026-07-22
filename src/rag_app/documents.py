from __future__ import annotations

import codecs
import re
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
_RTF_EXTENSIONS = frozenset({".rtf"})
_SUPPORTED_EXTENSIONS = _DOCLING_EXTENSIONS | _TEXT_EXTENSIONS | _RTF_EXTENSIONS

# Служебные группы RTF не являются содержимым документа. fldrslt намеренно не
# включён: отображаемый результат поля (например, текст гиперссылки) полезен RAG.
_RTF_IGNORABLE_DESTINATIONS = frozenset(
    """
    annotation atnauthor atndate atnid author background bkmkend bkmkstart
    blipuid buptim category colorschememapping colortbl comment company creatim
    datafield datastore defchp defpap do doccomm docvar dptxbxtext factoidname
    falt fchars ffdeftext ffentrymcr ffexitmcr ffformat ffhelptext ffl ffname
    ffstattext file filetbl fldinst fldtype fname fontemb fontfile fonttbl footer
    footerf footerl footerr footnote formfield generator gridtbl header headerf
    headerl headerr hlinkbase info keycode keywords latentstyles lchars
    levelnumbers leveltext lfolevel linkval list listlevel listname listoverride
    listoverridetable listpicture liststylename listtable listtext manager
    mmath mmathPr object objalias objclass objdata objname objsect objtime
    oldcprops oldpprops oldsprops oldtprops oleclsid operator panose pgp pgptbl
    picprop pict pn pnseclvl pntext pntxta pntxtb printim private propname
    revtbl revtim rsidtbl shp shpgrp shpinst shppict shprslt shptxt stylesheet
    subject template themedata title txe ud upr userprops xmlattrname
    xmlattrvalue xmlclose xmlname xmlnstbl xmlopen
    """.split()
)

_RTF_SPECIAL_CHARACTERS = {
    "par": "\n",
    "line": "\n",
    "tab": "\t",
    "emdash": "—",
    "endash": "–",
    "emspace": "\u2003",
    "enspace": "\u2002",
    "qmspace": "\u2005",
    "bullet": "•",
    "lquote": "‘",
    "rquote": "’",
    "ldblquote": "“",
    "rdblquote": "”",
}

_RTF_TOKEN_RE = re.compile(
    r"\\([a-zA-Z]{1,32})(-?\d{1,10})?[ ]?|"
    r"\\'([0-9a-fA-F]{2})|\\([^a-zA-Z])|([{}])|[\r\n]+|(.)",
    re.DOTALL,
)
_RTF_CODEPAGE_RE = re.compile(br"\\ansicpg(\d+)")


def _rtf_to_text(raw: bytes) -> str:
    """Извлечь видимый текст RTF без внешнего конвертера или офисного пакета."""
    if not raw.lstrip().startswith(b"{\\rtf"):
        raise ValueError("Файл не содержит заголовок RTF")

    encoding = _detect_rtf_encoding(raw)
    source = raw.decode(encoding, errors="replace")
    stack: list[tuple[int, bool]] = []
    ignorable = False
    unicode_fallback_length = 1
    fallback_characters_to_skip = 0
    output: list[str] = []

    for match in _RTF_TOKEN_RE.finditer(source):
        word, argument, hex_code, escaped, brace, character = match.groups()

        if brace:
            if brace == "{":
                stack.append((unicode_fallback_length, ignorable))
            elif stack:
                unicode_fallback_length, ignorable = stack.pop()
            continue

        if escaped:
            if escaped == "*":
                ignorable = True
            elif escaped in "{}\\" and not ignorable:
                output.append(escaped)
            elif escaped == "~" and not ignorable:
                output.append("\u00a0")
            continue

        if word:
            normalized_word = word.casefold()
            if normalized_word in _RTF_IGNORABLE_DESTINATIONS:
                ignorable = True
            elif normalized_word == "uc" and argument is not None:
                unicode_fallback_length = max(0, int(argument))
            elif normalized_word == "u" and argument is not None:
                codepoint = int(argument)
                if codepoint < 0:
                    codepoint += 0x10000
                if not ignorable:
                    output.append(chr(codepoint))
                # После \uN идёт однобайтовый fallback для старых RTF readers.
                fallback_characters_to_skip = unicode_fallback_length
            elif normalized_word in _RTF_SPECIAL_CHARACTERS and not ignorable:
                output.append(_RTF_SPECIAL_CHARACTERS[normalized_word])
            continue

        if hex_code:
            if fallback_characters_to_skip:
                fallback_characters_to_skip -= 1
            elif not ignorable:
                output.append(
                    bytes.fromhex(hex_code).decode(encoding, errors="replace")
                )
            continue

        if character:
            if fallback_characters_to_skip:
                fallback_characters_to_skip -= 1
            elif not ignorable:
                output.append(character)

    return _normalize_rtf_text("".join(output))


def _detect_rtf_encoding(raw: bytes) -> str:
    """Определить ANSI code page, указанную в заголовке RTF."""
    match = _RTF_CODEPAGE_RE.search(raw[:16384])
    if match:
        codepage = int(match.group(1))
        aliases = {65001: "utf-8", 10000: "mac_roman"}
        encoding = aliases.get(codepage, f"cp{codepage}")
        try:
            codecs.lookup(encoding)
        except LookupError:
            pass
        else:
            return encoding

    # fcharset204 обозначает кириллический Windows charset.
    if br"\fcharset204" in raw[:16384]:
        return "cp1251"
    return "cp1252"


def _normalize_rtf_text(text: str) -> str:
    """Убрать артефакты форматирования, сохранив границы абзацев."""
    text = text.replace("\r", "").replace("\x00", " ")
    lines = [
        re.sub(r"[ \t\u00a0]+", " ", line).strip()
        for line in text.split("\n")
    ]
    compact: list[str] = []
    for line in lines:
        if line:
            compact.append(line)
        elif compact and compact[-1] != "":
            compact.append("")
    while compact and not compact[-1]:
        compact.pop()
    return "\n".join(compact)


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

    TXT/Markdown/RST/RTF:
        LangChain loader -> RecursiveCharacterTextSplitter

    PDF/XLSX/ODT/DOCX:
        Docling -> HybridChunker -> LangChain Document
    """

    def __init__(
        self,
        settings: Settings,
        *,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> None:
        """Настроить обычный и токен-ориентированный разделители текста."""
        settings.validate()

        # Для простого текста размер и overlap измеряются в символах. Docling-файлы
        # ниже делятся отдельно по токенам embedding-модели через HybridChunker.
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=_MULTILINGUAL_SEPARATORS,
            length_function=len,
            is_separator_regex=False,
        )

        self.docling_chunk_tokens = settings.docling_chunk_tokens
        self.embedding_model = settings.embedding_model
        # RagService передаёт сюда уже загруженный токенизатор embedding-модели,
        # чтобы чанкование и последующее построение векторов считали токены одинаково.
        self._hf_tokenizer = tokenizer
        self.trust_remote_code = settings.trust_remote_code

        # Тяжёлые компоненты создаются только при первом Docling-файле. При работе
        # исключительно с TXT/Markdown/RST/RTF токенизатор здесь не загружается.
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

        if suffix in _RTF_EXTENSIONS:
            # Управляющие слова и служебные группы удаляются до обычного
            # символьного чанкования; Docling для RTF не загружается.
            text = _rtf_to_text(path.read_bytes())
            if not text:
                raise ValueError(f"В RTF-файле не найдено текста: {path}")
            raw_documents = [
                Document(page_content=text, metadata={"source": path.name})
            ]
        elif suffix in _TEXT_EXTENSIONS:
            loader = TextLoader(
                str(path),
                encoding="utf-8",
                autodetect_encoding=True,
            )
            raw_documents = loader.load()
        else:
            raise ValueError(
                f"Неподдерживаемый тип файла: {suffix!r}. "
                "Поддерживаемые форматы: "
                ".txt .md .rst .rtf .pdf .docx .xlsx .odt"
            )
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
                    ),
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
