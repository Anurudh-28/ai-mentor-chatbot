from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LoadedDoc:
    doc_id: str
    source: str
    text: str


def load_local_documents(docs_dir: str) -> list[LoadedDoc]:
    base = Path(docs_dir)
    if not base.exists():
        return []

    docs: list[LoadedDoc] = []
    for p in sorted(base.rglob("*")):
        if not p.is_file():
            continue
        suffix = p.suffix.lower()
        if suffix in {".txt", ".md"}:
            text = p.read_text(encoding="utf-8", errors="ignore")
        elif suffix == ".pdf":
            text = _read_pdf(p)
        else:
            continue
        doc_id = str(p.relative_to(base)).replace("\\", "/")
        docs.append(LoadedDoc(doc_id=doc_id, source=str(p), text=text))
    return docs


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PDF support requires pypdf. Install: pip install pypdf") from e

    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts)

