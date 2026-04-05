from pathlib import Path
from typing import Any, Dict, List, Union

from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    JSONLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_core.documents import Document

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".csv", ".xlsx", ".docx", ".json"}


def _normalize_metadata(doc: Document, file_path: Path) -> Document:
    metadata = dict(doc.metadata or {})
    metadata["source"] = str(file_path)
    metadata["file_name"] = file_path.name
    metadata["file_type"] = file_path.suffix.lower().lstrip(".")
    metadata.setdefault("page", metadata.get("page", None))
    return Document(page_content=doc.page_content, metadata=metadata)


LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".csv": CSVLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".docx": Docx2txtLoader,
    ".json": JSONLoader,
}


def load_file(file_path: Union[str, Path]) -> List[Document]:
    path = Path(file_path).resolve()
    suffix = path.suffix.lower()

    if suffix not in LOADER_MAP:
        raise ValueError(f"Unsupported file type: {suffix}")

    loader_cls = LOADER_MAP[suffix]

    if suffix == ".json":
        loader = loader_cls(str(path), jq_schema=".", text_content=False)
    else:
        loader = loader_cls(str(path))

    loaded_docs = loader.load()
    return [_normalize_metadata(doc, path) for doc in loaded_docs]


def load_all_documents(data_dir: Union[str, Path]) -> List[Document]:
    data_path = Path(data_dir).resolve()
    documents: List[Document] = []

    if not data_path.exists():
        data_path.mkdir(parents=True, exist_ok=True)
        return documents

    for file_path in sorted(data_path.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        try:
            documents.extend(load_file(file_path))
        except Exception as exc:
            print(f"[ERROR] Failed to load {file_path}: {exc}")

    return documents


def list_supported_files(data_dir: Union[str, Path]) -> List[Dict[str, Any]]:
    data_path = Path(data_dir).resolve()

    if not data_path.exists():
        return []

    files: List[Dict[str, Any]] = []
    for file_path in sorted(data_path.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            stat = file_path.stat()
            files.append(
                {
                    "name": file_path.name,
                    "path": str(file_path),
                    "size_bytes": stat.st_size,
                    "suffix": file_path.suffix.lower(),
                }
            )

    return files


if __name__ == "__main__":
    docs = load_all_documents("data")
    print(f"Loaded {len(docs)} documents.")
    print("Example document:", docs[0] if docs else None)