from __future__ import annotations

from contextlib import nullcontext

from chonkydb.index.base_index.base_index import (
    mixins_metadata as mod,
)


def test_log_skip_metadata_persist_read_only_uses_info(monkeypatch) -> None:
    calls = []

    monkeypatch.setattr(
        mod.chonk_errors,
        "info",
        lambda message: calls.append(("info", str(message))),
        raising=True,
    )

    mod.BaseIndexMetadataMixin._log_skip_metadata_persist_read_only(None, index_name="ccodex_memory_fulltext")

    assert calls == [
        (
            "info",
            "[BaseIndex._persist_metadata_now] CHONK_READ_ONLY=1 -> skip metadata persist for index=ccodex_memory_fulltext",
        )
    ]


def test_read_only_env_enabled_accepts_chonky_alias(monkeypatch) -> None:
    monkeypatch.delenv("CHONK_READ_ONLY", raising=False)
    monkeypatch.setenv("CHONKY_READ_ONLY", "1")

    assert mod._read_only_env_enabled() is True


def test_persist_metadata_now_skips_when_opened_read_only_without_env(monkeypatch) -> None:
    monkeypatch.delenv("CHONK_READ_ONLY", raising=False)
    monkeypatch.delenv("CHONKY_READ_ONLY", raising=False)

    calls = []

    class _FakeIndex:
        router = object()
        _is_open = True
        _opened_read_only = True
        index_name = "ccodex_memory_vector_768"

        def _log_skip_metadata_persist_read_only(self, *, index_name: str) -> None:
            calls.append(index_name)

    mod.BaseIndexMetadataMixin._persist_metadata_now(_FakeIndex(), tx_id=None)

    assert calls == ["ccodex_memory_vector_768"]


def test_persist_metadata_now_forces_non_transactional_router_write_when_tx_absent(
    monkeypatch,
) -> None:
    monkeypatch.delenv("CHONK_READ_ONLY", raising=False)
    monkeypatch.delenv("CHONKY_READ_ONLY", raising=False)

    seen_tx_ids = []

    class _FakeRouter:
        def store_index_data(self, *, file_type, key, data, tx_id):  # noqa: ANN001
            _ = (file_type, key, data)
            seen_tx_ids.append(tx_id)
            if tx_id is None:
                raise AssertionError("router received ambient transaction path")
            return (0, 1)

    class _FakeIndex:
        router = _FakeRouter()
        _is_open = True
        _opened_read_only = False
        index_name = "ccodex_memory_hash"
        _file_type = "hash"
        _index_structure_data = {"persist": {"mode": "delta_v1"}}

        def acquire_lock(self, *_args, **_kwargs):
            return nullcontext()

        def _validate_structure_data(self) -> None:
            return None

        def _calc_baseindex_docid(self) -> int:
            return 7

        def _prepare_data_for_serialization(self, data):  # noqa: ANN001
            return data

    mod.BaseIndexMetadataMixin._persist_metadata_now(
        _FakeIndex(),
        tx_id=None,
        skip_fsync=True,
    )

    assert seen_tx_ids == [""]


def test_prepare_data_for_serialization_snapshots_mutating_dict_views() -> None:
    class _MutatingDict(dict):
        def items(self):  # noqa: ANN001
            iterator = super().items()
            injected = False
            for item in iterator:
                if not injected:
                    injected = True
                    self["late"] = {"nested": "mutation"}
                yield item

    class _Serializer(mod.BaseIndexMetadataMixin):
        pass

    serializer = _Serializer()
    payload = _MutatingDict({"stable": {"nested": "value"}})

    normalized = serializer._prepare_data_for_serialization(payload)

    assert normalized == {"stable": {"nested": "value"}}
