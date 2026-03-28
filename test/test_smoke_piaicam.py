from __future__ import annotations

from hardware.piaicam.smoke_piaicam import _metadata_entries, _metadata_keys, _summarize_ai_metadata


def test_metadata_helpers_accept_dict_payload() -> None:
    payload = {
        "ExposureTime": 1234,
        "CnnKpiInfo": [13098, 12362],
    }

    assert _metadata_entries(payload) == (payload,)
    assert _metadata_keys(payload) == ["CnnKpiInfo", "ExposureTime"]

    summary = _summarize_ai_metadata(payload)

    assert summary["metadata_entry_count"] == 1
    assert summary["selected_metadata_index"] == 0
    assert summary["has_any_cnn_metadata"] is True
    assert summary["kpi_ms"] == {
        "dnn_runtime_ms": 13.098,
        "dsp_runtime_ms": 12.362,
    }


def test_metadata_helpers_accept_rpicam_frame_list_payload() -> None:
    payload = [
        {
            "ExposureTime": 1111,
            "SensorTimestamp": 1,
        },
        {
            "ExposureTime": 2222,
            "CnnOutputTensorInfo": [1, 2, 3],
            "CnnKpiInfo": [13098, 12362],
        },
        {
            "ExposureTime": 3333,
            "SensorTimestamp": 3,
        },
    ]

    assert len(_metadata_entries(payload)) == 3
    assert _metadata_keys(payload) == [
        "CnnKpiInfo",
        "CnnOutputTensorInfo",
        "ExposureTime",
        "SensorTimestamp",
    ]

    summary = _summarize_ai_metadata(payload)

    assert summary["metadata_entry_count"] == 3
    assert summary["selected_metadata_index"] == 1
    assert summary["has_any_cnn_metadata"] is True
    assert summary["metadata_keys"] == [
        "CnnKpiInfo",
        "CnnOutputTensorInfo",
        "ExposureTime",
        "SensorTimestamp",
    ]
    assert summary["CnnOutputTensorInfo"] == {
        "present": True,
        "len": 3,
        "value": [1, 2, 3],
    }
