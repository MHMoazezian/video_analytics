from datetime import datetime, timezone
import json

import pytest
from pydantic import ValidationError
import numpy as np

from app.analytics.cli import _management_spatial_layers
from app.management.api import _authorized, _query
from app.management.models import AnalyticsQuery, CameraMinute, IngestBatch
from app.management.publisher import MinutePublisher, clamp_observation
from app.management.repository import AnalyticsRepository
from app.management.service import (
    ManagementAnalyticsService,
    _difference_points,
    _shift_month,
)


def test_camera_minute_accepts_live_job_sample_counts() -> None:
    value = CameraMinute.model_validate({
        "cameraId": "camera-1", "bucketStart": "2026-08-11T12:00:00Z",
        "sampleCount": 179, "expectedSamples": 30, "confidenceSum": 179.0,
    })
    assert value.sample_count == 179
    value = CameraMinute.model_validate({
        "cameraId": "camera-1", "bucketStart": "2026-08-11T12:00:00Z",
        "sampleCount": 30, "expectedSamples": 30,
    })
    assert value.camera_id == "camera-1"
    assert value.bucket_start.tzinfo is not None

    with pytest.raises(ValidationError, match="timezone"):
        CameraMinute.model_validate({
            "cameraId": "camera-1", "bucketStart": "2026-08-11T12:00:00",
            "sampleCount": 30, "expectedSamples": 30,
        })


def test_ingest_key_comparison_rejects_wrong_service_key(monkeypatch) -> None:
    monkeypatch.setenv("ANALYTICS_INGEST_KEY", "correct-key")
    assert not _authorized("wrong-key", "ANALYTICS_INGEST_KEY")
    assert _authorized("correct-key", "ANALYTICS_INGEST_KEY")


def test_management_query_parses_business_dates_in_configured_timezone(monkeypatch) -> None:
    monkeypatch.setenv("ANALYTICS_TIMEZONE", "Asia/Tehran")
    query = _query("organization", None, "all", "2026-08-11", "2026-08-11", "none", "hour", None, None)
    assert query.from_date.tzinfo is timezone.utc
    assert (query.to_date-query.from_date).total_seconds() == 86_400


def test_difference_grid_preserves_signed_change() -> None:
    difference = _difference_points(
        [{"x": 10.0, "y": 20.0, "value": 7.0, "intensity": 1.0}],
        [{"x": 10.0, "y": 20.0, "value": 10.0, "intensity": 1.0}],
    )
    assert difference == [{"x": 10, "y": 20, "value": -3.0, "intensity": -1.0}]


def test_previous_month_clamps_end_of_month() -> None:
    assert _shift_month(datetime(2024, 3, 31, tzinfo=timezone.utc)) == datetime(2024, 2, 29, tzinfo=timezone.utc)


def test_management_spatial_grid_is_bounded_and_has_all_layers() -> None:
    layers = _management_spatial_layers(np.ones((36, 64)), np.full((36, 64), 120.0))
    assert set(layers) == {"occupancy", "dwell", "traffic", "congestion"}
    assert all(len(points) == 12 for points in layers.values())
    assert layers["dwell"][0]["value"] == 2.0


def test_publisher_is_zero_cost_when_ingestion_is_disabled(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("VIDEO_ANALYTICS_INGEST_URL", raising=False)
    monkeypatch.setenv("ANALYTICS_OUTBOX_DIR", str(tmp_path))
    publisher = MinutePublisher("camera-1")
    publisher.observe({"current_people": 5})
    publisher.close()
    assert list(tmp_path.iterdir()) == []


def test_publisher_snapshots_the_current_minute_without_waiting_for_clock_roll(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("VIDEO_ANALYTICS_INGEST_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("ANALYTICS_OUTBOX_DIR", str(tmp_path))
    monkeypatch.setenv("ANALYTICS_PUBLISH_INTERVAL_SECONDS", "0")
    publisher = MinutePublisher("camera-1", "Cam")
    try:
        publisher.observe({"current_people": 1})
        publisher.observe({"current_people": 4})
        files = list((tmp_path / "camera-1").glob("*.json"))
        assert len(files) == 1
        payload = json.loads(files[0].read_text(encoding="utf-8"))
        observation = payload["observations"][0]
        assert observation["sampleCount"] == 2
        assert observation["occupancyLast"] == 4
        assert observation["occupancyMax"] == 4
    finally:
        publisher.close()


class _BrokenPool:
    """Stands in for a pool whose schema has not been migrated yet."""

    def __init__(self, error: Exception) -> None:
        self.error = error

    def connection(self):
        raise self.error


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError('relation "analytics_camera_source" does not exist'),
        KeyError("unexpected driver failure"),
    ],
)
def test_health_reports_degraded_instead_of_raising(monkeypatch, error) -> None:
    monkeypatch.delenv("ANALYTICS_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    repository = AnalyticsRepository()
    assert repository.health() is False  # no database configured

    repository.pool = _BrokenPool(error)
    assert repository.health() is False


def test_peak_period_labels_use_the_business_timezone(monkeypatch) -> None:
    monkeypatch.setenv("ANALYTICS_TIMEZONE", "Asia/Tehran")
    service = ManagementAnalyticsService(repository=object())  # type: ignore[arg-type]
    rows = [
        {"timestamp": datetime(2026, 1, 15, 9, 0, tzinfo=timezone.utc), "occupancy": 12.0},
        {"timestamp": datetime(2026, 1, 15, 5, 0, tzinfo=timezone.utc), "occupancy": 30.0},
        {"timestamp": datetime(2026, 1, 15, 7, 0, tzinfo=timezone.utc), "occupancy": None},
    ]
    monkeypatch.setattr(service, "_flow_trend", lambda *_args: rows)
    query = AnalyticsQuery(
        from_date=datetime(2026, 1, 15, tzinfo=timezone.utc),
        to_date=datetime(2026, 1, 16, tzinfo=timezone.utc),
    )

    peaks = service._peak_periods(query, query.from_date, query.to_date, "field")

    # Tehran is UTC+03:30 all year, so 05:00Z is 08:30 local.
    assert [item["label"] for item in peaks] == ["اوج 08:30", "اوج 12:30"]
    assert peaks[0]["from"] == "2026-01-15T05:00:00+00:00"

    monkeypatch.setenv("ANALYTICS_TIMEZONE", "UTC")
    assert service._peak_periods(query, query.from_date, query.to_date, "field")[0]["label"] == "اوج 05:00"


def _oversized_observation() -> dict[str, object]:
    point = {"x": 50.0, "y": 50.0, "value": 1.0, "intensity": 0.5}
    queue = {
        "queueId": "q", "queueName": "q", "sampleCount": 7200, "lengthSum": 14400.0,
        "lengthMax": 4, "lengthLast": 2, "waitSumSeconds": 40000.0, "waitSampleCount": 4000,
        "completedWaitSeconds": [float(index) for index in range(900)], "throughput": 900,
        "slaMet": 300, "warningSamples": 0, "criticalSamples": 0,
        "movementSpeedSumMpm": 0.0, "movementSpeedSamples": 0, "physicallyCalibrated": False,
    }
    event = {
        "eventType": "line_crossed", "severity": "info", "title": "عبور از خط شمارش",
        "occurredAt": "2026-08-11T12:00:30+00:00", "metricKey": "entries", "payload": {},
    }
    return {
        "cameraId": "camera-1", "cameraName": "Cam", "bucketStart": "2026-08-11T12:00:00+00:00",
        "sampleCount": 7200, "expectedSamples": 30, "confidenceSum": 7200.0,
        "occupancySum": 21600.0, "occupancyMax": 5, "occupancyLast": 3, "entries": 4, "exits": 2,
        "uniqueVisitors": None, "physicallyCalibrated": False,
        "queues": [{**queue, "queueId": f"q-{index}"} for index in range(130)],
        "spatial": [
            {"zoneId": f"zone-{index}", "zoneName": "کل مکان", "layer": "occupancy",
             "bucketSeconds": 300, "coveragePercent": None, "points": [dict(point) for _ in range(300)]}
            for index in range(40)
        ],
        "events": [{**event, "eventId": f"event-{index}"} for index in range(650)],
    }


def test_oversized_observation_is_rejected_until_it_is_clamped() -> None:
    with pytest.raises(ValidationError):
        IngestBatch.model_validate({"observations": [_oversized_observation()]})

    observation = clamp_observation(_oversized_observation())
    minute = IngestBatch.model_validate({"observations": [observation]}).observations[0]

    assert minute.sample_count == 3600
    assert minute.confidence_sum == 3600
    # The occupancy sum is scaled with the sample count, so the mean stays 3 people.
    assert minute.occupancy_sum / minute.sample_count == pytest.approx(3.0)
    assert len(minute.queues) == 100
    assert len(minute.spatial) == 32
    assert all(len(layer.points) == 256 for layer in minute.spatial)
    assert len(minute.events) == 500
    assert minute.events[-1].event_id == "event-649"
    queue = minute.queues[0]
    assert queue.sample_count == 3600
    assert queue.length_sum / queue.sample_count == pytest.approx(2.0)
    assert queue.wait_sample_count == 1000
    assert queue.wait_sum_seconds / queue.wait_sample_count == pytest.approx(10.0)
    # The oldest completed waits keep their list position (idempotent event ids).
    assert queue.completed_wait_seconds[:3] == [0.0, 1.0, 2.0]
    assert len(queue.completed_wait_seconds) == 500


def test_clamp_never_lets_confidence_exceed_the_sample_count() -> None:
    observation = clamp_observation({"sampleCount": 12, "confidenceSum": 40.0, "expectedSamples": 0})
    assert observation["confidenceSum"] == 12.0
    assert observation["expectedSamples"] == 1


def test_fast_recorded_job_outbox_document_stays_within_ingest_limits(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("VIDEO_ANALYTICS_INGEST_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("ANALYTICS_OUTBOX_DIR", str(tmp_path))
    monkeypatch.setenv("ANALYTICS_PUBLISH_INTERVAL_SECONDS", "3600")
    publisher = MinutePublisher("camera-fast", "Fast")
    try:
        # Pin the bucket: a wall-clock minute roll mid-loop would split the samples.
        bucket = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)
        publisher._bucket = bucket
        publisher._occupancy = [2] * 4000
        publisher._queues = {"q": {
            "lengths": [1] * 4000, "waits": [5.0] * 4000, "speeds": [], "completed_first": 0,
            "completed_last": 0, "completed_waits": [], "calibrated": False,
        }}
        publisher._write_outbox()
        files = list((tmp_path / "camera-fast").glob("*.json"))
        assert len(files) == 1
        batch = IngestBatch.model_validate(json.loads(files[0].read_text(encoding="utf-8")))
    finally:
        publisher._reset()
        publisher.close()
    minute = batch.observations[0]
    assert minute.sample_count == 3600
    assert minute.queues[0].wait_sample_count == 1000
    assert minute.queues[0].wait_sum_seconds == pytest.approx(5000.0)
