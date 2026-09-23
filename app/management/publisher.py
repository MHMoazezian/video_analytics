"""Producer-side minute compaction with a durable HTTP outbox.

Frame-rate data never crosses the service boundary. Each worker keeps constant-size
minute accumulators and overwrites one small outbox document as the current minute
grows, then closes that minute when the clock rolls.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import threading
from time import monotonic
from typing import Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

from app.core.models import Event


class MinutePublisher:
    def __init__(self, camera_id: str, camera_name: str | None = None) -> None:
        self.camera_id = camera_id
        self.camera_name = camera_name or camera_id
        self.url = os.environ.get("VIDEO_ANALYTICS_INGEST_URL")
        self.key = os.environ.get("ANALYTICS_INGEST_KEY")
        self.expected = max(1, int(os.environ.get("ANALYTICS_EXPECTED_SAMPLES_PER_MINUTE", "30")))
        self.warning_seconds = float(os.environ.get("ANALYTICS_QUEUE_WARNING_SECONDS", "300"))
        self.critical_seconds = float(os.environ.get("ANALYTICS_QUEUE_CRITICAL_SECONDS", "600"))
        self.outbox = Path(os.environ.get("ANALYTICS_OUTBOX_DIR", "/tmp/video-analytics-outbox")) / camera_id
        self.max_outbox_files = max(60, int(os.environ.get("ANALYTICS_OUTBOX_MAX_FILES_PER_CAMERA", "1440")))
        try:
            self.publish_interval = max(0.0, float(os.environ.get("ANALYTICS_PUBLISH_INTERVAL_SECONDS", "10")))
        except (TypeError, ValueError):
            self.publish_interval = 10.0
        self.enabled = bool(self.url)
        self._bucket: datetime | None = None
        self._last_snapshot_at: float | None = None
        self._occupancy: list[int] = []
        self._first_entries: int | None = None
        self._last_entries = 0
        self._first_exits: int | None = None
        self._last_exits = 0
        self._queues: dict[str, dict[str, object]] = {}
        self._events: list[dict[str, object]] = []
        self._spatial: dict[str, dict[str, object]] = {}
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        if self.enabled:
            self.outbox.mkdir(parents=True, exist_ok=True)
            self._worker = threading.Thread(target=self._send_loop, name=f"analytics-outbox-{camera_id}", daemon=True)
            self._worker.start()

    def observe(self, metrics: Mapping[str, object], events: Sequence[Event] = ()) -> None:
        if not self.enabled:
            return
        now = datetime.now(timezone.utc)
        bucket = now.replace(second=0, microsecond=0)
        if self._bucket is not None and bucket != self._bucket:
            self._write_outbox()
            self._reset()
        self._bucket = bucket
        occupancy = metrics.get("current_people")
        if isinstance(occupancy, int) and not isinstance(occupancy, bool):
            self._occupancy.append(max(0, occupancy))
        self._observe_counter(metrics, "entry_count", entries=True)
        self._observe_counter(metrics, "exit_count", entries=False)
        details = metrics.get("queue_details")
        if isinstance(details, Mapping):
            for queue_id, raw in details.items():
                if isinstance(raw, Mapping):
                    self._observe_queue(str(queue_id), raw)
        crowded = metrics.get("top_crowded_regions")
        if isinstance(crowded, list):
            self._observe_spatial(crowded)
        layers = metrics.get("management_spatial_layers")
        if isinstance(layers, Mapping):
            for layer, points in layers.items():
                if layer in {"occupancy", "dwell", "traffic", "congestion"} and isinstance(points, list):
                    self._spatial[str(layer)] = {"zoneId": "location", "zoneName": "کل مکان", "layer": layer,
                                                  "bucketSeconds": 300, "coveragePercent": None, "points": points[:256]}
        for event in events:
            self._events.append({
                "eventId": event.event_id, "eventType": event.event_type,
                "severity": _severity(event.event_type), "title": _event_title(event.event_type),
                "occurredAt": now.isoformat(), "metricKey": _metric_key(event.event_type),
                "payload": dict(event.payload or {}),
            })
        if len(self._events) > 500:
            self._events = self._events[-500:]
        if self._should_snapshot():
            self._write_outbox()
            self._last_snapshot_at = monotonic()

    def close(self) -> None:
        if not self.enabled:
            return
        self._write_outbox()
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=3)

    def _observe_counter(self, metrics: Mapping[str, object], key: str, *, entries: bool) -> None:
        value = metrics.get(key)
        if not isinstance(value, int) or isinstance(value, bool):
            return
        if entries:
            self._first_entries = value if self._first_entries is None else self._first_entries
            self._last_entries = value
        else:
            self._first_exits = value if self._first_exits is None else self._first_exits
            self._last_exits = value

    def _observe_queue(self, queue_id: str, raw: Mapping[str, object]) -> None:
        item = self._queues.setdefault(queue_id, {
            "lengths": [], "waits": [], "speeds": [], "completed_first": None,
            "completed_last": 0, "completed_waits": [], "calibrated": False,
        })
        people = raw.get("people")
        if isinstance(people, int) and not isinstance(people, bool):
            item["lengths"].append(max(0, people))  # type: ignore[union-attr]
        wait = raw.get("current_wait_seconds")
        if isinstance(wait, (int, float)) and not isinstance(wait, bool):
            item["waits"].append(max(0.0, float(wait)))  # type: ignore[union-attr]
        speed = raw.get("average_speed_metres_per_second")
        if isinstance(speed, (int, float)) and not isinstance(speed, bool):
            item["speeds"].append(abs(float(speed)) * 60)  # type: ignore[union-attr]
            item["calibrated"] = True
        completed = raw.get("completed_wait_count")
        if isinstance(completed, int) and not isinstance(completed, bool):
            previous_completed = int(item["completed_last"])
            item["completed_first"] = completed if item["completed_first"] is None else item["completed_first"]
            item["completed_last"] = completed
            last_wait = raw.get("last_completed_wait_seconds")
            if completed > previous_completed and isinstance(last_wait, (int, float)):
                item["completed_waits"].append(float(last_wait))  # type: ignore[union-attr]

    def _observe_spatial(self, crowded: list[object]) -> None:
        points = []
        maximum = max((float(item.get("average_occupancy", 0)) for item in crowded if isinstance(item, Mapping)), default=1) or 1
        for item in crowded:
            if not isinstance(item, Mapping):
                continue
            bounds = item.get("normalized_bounds")
            if not isinstance(bounds, list) or len(bounds) != 4:
                continue
            value = float(item.get("average_occupancy", 0))
            points.append({"x": (float(bounds[0])+float(bounds[2]))*50, "y": (float(bounds[1])+float(bounds[3]))*50,
                           "value": value, "intensity": min(1, value/maximum)})
        if points:
            self._spatial["congestion"] = {"zoneId": "location", "zoneName": "کل مکان", "layer": "congestion",
                                             "bucketSeconds": 300, "coveragePercent": None, "points": points}

    def _should_snapshot(self) -> bool:
        if not self._occupancy:
            return False
        if self._last_snapshot_at is None or self.publish_interval <= 0:
            return True
        return (monotonic() - self._last_snapshot_at) >= self.publish_interval

    def _write_outbox(self) -> None:
        if self._bucket is None or not self._occupancy:
            return
        queues = []
        for queue_id, raw in self._queues.items():
            lengths = raw["lengths"]
            waits = raw["waits"]
            speeds = raw["speeds"]
            completed = max(0, int(raw["completed_last"]) - int(raw["completed_first"] or 0))
            completed_waits = list(raw["completed_waits"])
            queues.append({
                "queueId": queue_id, "queueName": queue_id, "sampleCount": len(lengths),
                "lengthSum": sum(lengths), "lengthMax": max(lengths) if lengths else None,
                "lengthLast": lengths[-1] if lengths else None, "waitSumSeconds": sum(waits),
                "waitSampleCount": len(waits), "completedWaitSeconds": completed_waits,
                "throughput": completed, "slaMet": sum(wait <= float(os.environ.get("ANALYTICS_QUEUE_SLA_SECONDS", "300")) for wait in completed_waits),
                "warningSamples": sum(wait >= self.warning_seconds for wait in waits),
                "criticalSamples": sum(wait >= self.critical_seconds for wait in waits),
                "movementSpeedSumMpm": sum(speeds), "movementSpeedSamples": len(speeds),
                "physicallyCalibrated": bool(raw["calibrated"]),
            })
        entry_start = self._first_entries or 0
        exit_start = self._first_exits or 0
        observation = {
            "cameraId": self.camera_id, "cameraName": self.camera_name,
            "bucketStart": self._bucket.isoformat(), "sampleCount": len(self._occupancy),
            "expectedSamples": self.expected, "confidenceSum": float(len(self._occupancy)),
            "occupancySum": float(sum(self._occupancy)), "occupancyMax": max(self._occupancy),
            "occupancyLast": self._occupancy[-1], "entries": max(0, self._last_entries-entry_start),
            "exits": max(0, self._last_exits-exit_start), "uniqueVisitors": None,
            "physicallyCalibrated": any(bool(item["physicallyCalibrated"]) for item in queues),
            "queues": queues, "spatial": list(self._spatial.values()), "events": self._events,
        }
        destination = self.outbox / f"{self._bucket.strftime('%Y%m%dT%H%M')}-{self.camera_id}.json"
        pending = sorted(self.outbox.glob("*.json"))
        if len(pending) >= self.max_outbox_files:
            for expired in pending[:len(pending)-self.max_outbox_files+1]:
                expired.unlink(missing_ok=True)
        temporary = destination.with_suffix(".tmp")
        clamp_observation(observation)
        temporary.write_text(json.dumps({"observations": [observation]}, separators=(",", ":")), encoding="utf-8")
        temporary.replace(destination)

    def _reset(self) -> None:
        self._bucket = None
        self._last_snapshot_at = None
        self._occupancy = []
        self._first_entries = None
        self._last_entries = 0
        self._first_exits = None
        self._last_exits = 0
        self._queues = {}
        self._events = []
        self._spatial = {}

    def _send_loop(self) -> None:
        while not self._stop.wait(1):
            for path in sorted(self.outbox.glob("*.json"))[:100]:
                if self._send(path):
                    path.unlink(missing_ok=True)

    def _send(self, path: Path) -> bool:
        try:
            payload = path.read_bytes()
            headers = {"Content-Type": "application/json"}
            if self.key:
                headers["X-Analytics-Key"] = self.key
            request = Request(str(self.url), data=payload, headers=headers, method="POST")
            with urlopen(request, timeout=5) as response:
                return 200 <= response.status < 300
        except HTTPError as exc:
            detail = exc.read()[:400]
            if exc.code in {400, 401, 404, 422}:
                logger.warning("dropping invalid analytics outbox %s: %s %s", path.name, exc.code, detail)
                return True
            logger.warning("analytics ingest retry %s: %s %s", path.name, exc.code, detail)
            return False
        except (OSError, URLError, TimeoutError) as exc:
            logger.warning("analytics ingest unavailable for %s: %s", path.name, exc)
            return False


# Upper bounds of the ingest contract in ``app.management.models``. A recorded job
# can process far more than one frame per wall-clock second, and a payload above
# these bounds is rejected with 422 and dropped, losing the whole minute.
MAX_SAMPLES_PER_MINUTE = 3600
MAX_WAIT_SAMPLES = 1000
MAX_COMPLETED_WAITS = 500
MAX_QUEUES = 100
MAX_EVENTS = 500
MAX_SPATIAL_LAYERS = 32
MAX_SPATIAL_POINTS = 256


def _clamp_count(item: dict[str, object], count_key: str, limit: int, sum_key: str) -> None:
    """Cap a sample counter and scale its paired sum so the average is preserved."""

    count = item.get(count_key)
    if isinstance(count, bool) or not isinstance(count, int) or count <= limit:
        return
    total = item.get(sum_key)
    if isinstance(total, (int, float)) and not isinstance(total, bool):
        item[sum_key] = float(total) * limit / count
    item[count_key] = limit


def clamp_observation(observation: dict[str, object]) -> dict[str, object]:
    """Fit one ``CameraMinute`` document into the ingest limits, in place."""

    _clamp_count(observation, "sampleCount", MAX_SAMPLES_PER_MINUTE, "occupancySum")
    expected = observation.get("expectedSamples")
    if isinstance(expected, int) and not isinstance(expected, bool):
        observation["expectedSamples"] = min(max(expected, 1), MAX_SAMPLES_PER_MINUTE)
    samples = observation.get("sampleCount")
    confidence = observation.get("confidenceSum")
    if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
        ceiling = float(min(MAX_SAMPLES_PER_MINUTE, samples)) if isinstance(samples, int) else float(MAX_SAMPLES_PER_MINUTE)
        observation["confidenceSum"] = max(0.0, min(float(confidence), ceiling))
    queues = observation.get("queues")
    if isinstance(queues, list):
        del queues[MAX_QUEUES:]
        for queue in queues:
            if not isinstance(queue, dict):
                continue
            _clamp_count(queue, "sampleCount", MAX_SAMPLES_PER_MINUTE, "lengthSum")
            _clamp_count(queue, "waitSampleCount", MAX_WAIT_SAMPLES, "waitSumSeconds")
            waits = queue.get("completedWaitSeconds")
            if isinstance(waits, list):
                # Keep the oldest entries: ingestion derives idempotent event ids
                # from the list position, which must stay stable between snapshots.
                del waits[MAX_COMPLETED_WAITS:]
    events = observation.get("events")
    if isinstance(events, list) and len(events) > MAX_EVENTS:
        observation["events"] = events[-MAX_EVENTS:]
    spatial = observation.get("spatial")
    if isinstance(spatial, list):
        del spatial[MAX_SPATIAL_LAYERS:]
        for layer in spatial:
            if isinstance(layer, dict) and isinstance(layer.get("points"), list):
                layer["points"] = layer["points"][:MAX_SPATIAL_POINTS]
    return observation


def _severity(event_type: str) -> str:
    return "critical" if "restricted_area_confirmed" in event_type else "warning" if "overflow" in event_type else "info"


def _metric_key(event_type: str) -> str | None:
    return "criticalMinutes" if "overflow" in event_type else "entries" if "line_crossed" in event_type else None


def _event_title(event_type: str) -> str:
    return {"queue_overflow_started": "عبور صف از آستانه بحرانی", "queue_overflow_ended": "پایان ازدحام صف",
            "restricted_area_confirmed": "ورود تأییدشده به محدوده", "line_crossed": "عبور از خط شمارش"}.get(event_type, event_type)
