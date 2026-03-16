from __future__ import annotations

from dataclasses import dataclass
from contextlib import closing  # AUDIT-FIX(#2): sqlite3 connection context managers do not close connections.
import sqlite3

from twinr.text_utils import fts_match_query


@dataclass(frozen=True, slots=True)
class FullTextDocument:
    doc_id: str
    category: str
    content: str


class FullTextSelector:
    _CREATE_TABLE_SQL = (
        "CREATE VIRTUAL TABLE memory_search USING fts5("
        "doc_id UNINDEXED, category UNINDEXED, content, "
        "tokenize='unicode61 remove_diacritics 2')"
    )
    _INSERT_SQL = "INSERT INTO memory_search(doc_id, category, content) VALUES (?, ?, ?)"
    _SEARCH_SQL = (
        "SELECT doc_id FROM memory_search "
        "WHERE memory_search MATCH ? "
        "ORDER BY bm25(memory_search), rowid"
    )
    _SEARCH_BY_CATEGORY_SQL = (
        "SELECT doc_id FROM memory_search "
        "WHERE category = ? AND memory_search MATCH ? "
        "ORDER BY bm25(memory_search), rowid"
    )

    def __init__(self, documents: tuple[FullTextDocument, ...]) -> None:
        # AUDIT-FIX(#3): materialize once so live searches do not depend on a possibly mutable/non-tuple iterable.
        self._documents = tuple(documents)

    def search(
        self,
        query_text: str | None,
        *,
        limit: int,
        category: str | None = None,
        allow_fallback: bool = False,
    ) -> tuple[str, ...]:
        # AUDIT-FIX(#6): non-positive or invalid limits now mean "no results", instead of silently forcing one hit.
        resolved_limit = self._normalize_limit(limit)
        if resolved_limit == 0:
            return ()

        normalized_category = self._normalize_category(category)
        normalized_query = self._normalize_query(query_text)

        if not normalized_query:
            raw_query_text = "" if query_text is None else str(query_text).strip()
            # AUDIT-FIX(#4): only a genuinely blank query gets the unconditional default fallback.
            if not raw_query_text:
                return self._fallback(limit=resolved_limit, category=normalized_category)
            if allow_fallback:
                return self._fallback(limit=resolved_limit, category=normalized_category)
            return ()

        try:
            result = self._search_fts(
                normalized_query=normalized_query,
                limit=resolved_limit,
                category=normalized_category,
            )
        except sqlite3.Error:
            # AUDIT-FIX(#1): degrade gracefully when FTS5 is unavailable or SQLite rejects the MATCH query.
            if allow_fallback:
                return self._fallback(limit=resolved_limit, category=normalized_category)
            return ()

        if result:
            return result
        if allow_fallback:
            return self._fallback(limit=resolved_limit, category=normalized_category)
        return ()

    def _search_fts(
        self,
        *,
        normalized_query: str,
        limit: int,
        category: str | None,
    ) -> tuple[str, ...]:
        # AUDIT-FIX(#2): close the transient SQLite connection explicitly on every search.
        with closing(sqlite3.connect(":memory:")) as connection:
            connection.execute(self._CREATE_TABLE_SQL)
            connection.executemany(self._INSERT_SQL, self._iter_index_rows())

            if category is None:
                cursor = connection.execute(self._SEARCH_SQL, (normalized_query,))
            else:
                cursor = connection.execute(
                    self._SEARCH_BY_CATEGORY_SQL,
                    (category, normalized_query),
                )

            try:
                # AUDIT-FIX(#5): de-duplicate doc_ids while preserving SQLite ranking order.
                selected: list[str] = []
                seen: set[str] = set()
                for row in cursor:
                    if not row:
                        continue
                    doc_id = self._normalize_doc_id(row[0])
                    if not doc_id or doc_id in seen:
                        continue
                    seen.add(doc_id)
                    selected.append(doc_id)
                    if len(selected) >= limit:
                        break
                return tuple(selected)
            finally:
                cursor.close()

    def _fallback(self, *, limit: int, category: str | None) -> tuple[str, ...]:
        selected: list[str] = []
        seen: set[str] = set()
        for item in self._documents:
            # AUDIT-FIX(#3): sanitize persisted document fields so malformed rows do not crash fallback or leak blank ids.
            doc_id = self._normalize_doc_id(getattr(item, "doc_id", None))
            item_category = self._normalize_category(getattr(item, "category", None))
            if not doc_id:
                continue
            if category is not None and item_category != category:
                continue
            if doc_id in seen:
                continue
            seen.add(doc_id)
            selected.append(doc_id)
            if len(selected) >= limit:
                break
        return tuple(selected)

    def _iter_index_rows(self):
        for item in self._documents:
            # AUDIT-FIX(#3): sanitize persisted document fields before indexing so bad data degrades instead of exploding.
            doc_id = self._normalize_doc_id(getattr(item, "doc_id", None))
            content = self._normalize_content(getattr(item, "content", None))
            if not doc_id or not content.strip():
                continue
            yield (
                doc_id,
                self._normalize_category(getattr(item, "category", None)) or "",
                content,
            )

    # AUDIT-FIX(#6): centralize limit normalization so every path obeys the same "0 means no results" rule.
    @staticmethod
    def _normalize_limit(limit: int) -> int:
        if not isinstance(limit, int):
            return 0
        return limit if limit > 0 else 0

    @staticmethod
    def _normalize_query(query_text: str | None) -> str | None:
        try:
            # AUDIT-FIX(#1): protect the selector from normalization helper failures.
            normalized_query = fts_match_query(query_text)
        except Exception:
            return None
        if normalized_query is None:
            return None
        normalized_query_text = str(normalized_query).strip()
        return normalized_query_text or None

    # AUDIT-FIX(#3): centralize document field sanitization so bad persisted data cannot crash the selector.
    @staticmethod
    def _normalize_doc_id(value: object) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @staticmethod
    def _normalize_category(value: object) -> str | None:
        if value is None:
            return None
        return str(value).strip()

    @staticmethod
    def _normalize_content(value: object) -> str:
        if not isinstance(value, str):
            return ""
        return value


__all__ = ["FullTextDocument", "FullTextSelector"]