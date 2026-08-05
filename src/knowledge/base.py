"""Dependency-free Markdown knowledge index with JSON metadata."""

from __future__ import annotations

import json
import re
from pathlib import Path
from pydantic import BaseModel


class KnowledgeChunk(BaseModel):
    document_id: str
    title: str
    model: str
    source: str
    section: str
    content: str


class KnowledgeHit(BaseModel):
    chunk: KnowledgeChunk
    score: float


class KnowledgeBase:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.chunks = self._load()

    def _load(self) -> list[KnowledgeChunk]:
        catalog_path = self.root / "catalog.json"
        if not catalog_path.exists():
            return []
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        chunks = []
        for item in catalog.get("documents", []):
            path = self.root / item["file"]
            if not path.exists():
                continue
            for section, content in _split_markdown(path.read_text(encoding="utf-8")):
                chunks.append(KnowledgeChunk(
                    document_id=item["id"], title=item["title"], model=item["model"],
                    source=item["file"], section=section, content=content,
                ))
        return chunks

    def search(self, query: str, *, model: str | None = None, top_k: int = 3) -> list[KnowledgeHit]:
        query_terms = _terms(query)
        query_codes = set(re.findall(r"\b(?:[a-z]\d+|\d{3,})\b", query.lower()))
        hits = []
        for chunk in self.chunks:
            if model and chunk.model != model:
                continue
            document_text = f"{chunk.title} {chunk.section} {chunk.content}".lower()
            if query_codes and not query_codes.issubset(set(re.findall(r"\b(?:[a-z]\d+|\d{3,})\b", document_text))):
                continue
            document_terms = _terms(document_text)
            overlap = query_terms & document_terms
            if not overlap:
                continue
            score = len(overlap) / max(1, len(query_terms))
            hits.append(KnowledgeHit(chunk=chunk, score=score))
        return sorted(hits, key=lambda hit: (-hit.score, hit.chunk.document_id, hit.chunk.section))[:top_k]


def _split_markdown(text: str) -> list[tuple[str, str]]:
    sections = []
    current_title = "正文"
    current_lines = []
    for line in text.splitlines():
        if line.startswith("## "):
            if current_lines:
                sections.append((current_title, "\n".join(current_lines).strip()))
            current_title = line[3:].strip()
            current_lines = []
        elif not line.startswith("# "):
            current_lines.append(line)
    if current_lines:
        sections.append((current_title, "\n".join(current_lines).strip()))
    return [(title, content) for title, content in sections if content]


def _terms(text: str) -> set[str]:
    lowered = text.lower()
    latin = set(re.findall(r"[a-z0-9_-]+", lowered))
    chinese = "".join(re.findall(r"[\u4e00-\u9fff]", lowered))
    grams = {chinese[index:index + 2] for index in range(max(0, len(chinese) - 1))}
    return latin | grams
