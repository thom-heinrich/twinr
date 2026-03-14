from __future__ import annotations

from dataclasses import dataclass
import sqlite3

from twinr.text_utils import fts_match_query


@dataclass(frozen=True, slots=True)
class FullTextDocument:
    doc_id: str
    category: str
    content: str


class FullTextSelector:
    def __init__(self, documents: tuple[FullTextDocument, ...]) -> None:
        self._documents = documents

    def search(
        self,
        query_text: str | None,
        *,
        limit: int,
        category: str | None = None,
        allow_fallback: bool = False,
    ) -> tuple[str, ...]:
        normalized_query = fts_match_query(query_text)
        if not normalized_query:
            return self._fallback(limit=limit, category=category)
        with sqlite3.connect(":memory:") as connection:
            connection.execute(
                "CREATE VIRTUAL TABLE memory_search USING fts5(doc_id UNINDEXED, category UNINDEXED, content, tokenize='unicode61 remove_diacritics 2')"
            )
            connection.executemany(
                "INSERT INTO memory_search(doc_id, category, content) VALUES (?, ?, ?)",
                (
                    (item.doc_id, item.category, item.content)
                    for item in self._documents
                    if item.content.strip()
                ),
            )
            if category is None:
                rows = connection.execute(
                    "SELECT doc_id FROM memory_search WHERE memory_search MATCH ? ORDER BY bm25(memory_search), rowid LIMIT ?",
                    (normalized_query, max(1, limit)),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT doc_id FROM memory_search WHERE category = ? AND memory_search MATCH ? ORDER BY bm25(memory_search), rowid LIMIT ?",
                    (category, normalized_query, max(1, limit)),
                ).fetchall()
        result = tuple(str(row[0]) for row in rows if row and str(row[0]).strip())
        if result:
            return result
        if allow_fallback:
            return self._fallback(limit=limit, category=category)
        return ()

    def _fallback(self, *, limit: int, category: str | None) -> tuple[str, ...]:
        selected: list[str] = []
        for item in self._documents:
            if category is not None and item.category != category:
                continue
            if item.doc_id in selected:
                continue
            selected.append(item.doc_id)
            if len(selected) >= max(1, limit):
                break
        return tuple(selected)


__all__ = ["FullTextDocument", "FullTextSelector"]
