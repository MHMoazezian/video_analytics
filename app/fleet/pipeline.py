"""Per-camera 0.5 FPS analyzer that publishes location rollup facts."""

from __future__ import annotations

from threading import Lock
from time import monotonic, time
from typing import Protocol

import numpy as np

from app.analytics.counting import CameraCountingConfig, PeopleCounter
from app.analytics.heatmap import MovementHeatmaps
from app.analytics.queue import CameraQueueConfig, QueueAnalyzer
from app.analytics.restricted_area import CameraRestrictedAreaConfig, RestrictedAreaDetector
from app.analytics.speed import CameraSpeedConfig, SpeedEstimator
from app.api.live import processing_frame_size, resize_processing_frame
from app.detection.base import DetectionResult, DetectionTimings
from app.fleet.catalog import FleetCamera
from app.fleet.geometry import build_camera_config
from app.fleet.metrics import collect_events, live_metrics
from app.fleet.settings import FleetSettings
from app.geometry.config import CameraConfig
from app.management.publisher import MinutePublisher
from app.tracking.base import BaseTracker
from app.tracking.factory import create_tracker


class PersonDetector(Protocol):
    def detect(self, frame: np.ndarray) -> DetectionResult: ...


class CameraPipeline:
    """Detect, count, queue, and heat-map one camera without writing video."""

    def __init__(
        self,
        camera: FleetCamera,
        settings: FleetSettings,
        detector: PersonDetector,
        detector_lock: Lock,
        *,
        publisher: MinutePublisher | None = None,
    ) -> None:
        self.camera = camera
        self.settings = settings
        self._detector = detector
        self._detector_lock = detector_lock
        self._config = build_camera_config(camera, settings)
        self._publisher = publisher or MinutePublisher(camera.camera_id, camera.name)
        self._tracker: BaseTracker | None = None
        self._counter: PeopleCounter | None = None
        self._restricted: RestrictedAreaDetector | None = None
        self._queues: QueueAnalyzer | None = None
        self._speed: SpeedEstimator | None = None
        self._heatmaps: MovementHeatmaps | None = None
        self._frame_size: tuple[int, int] | None = None
        self._started = monotonic()
        self.processed_frames = 0
        self.restricted_violations = 0
        self._last_spatial_publish: float | None = None
        self._has_counting_lines = False
        self._expired_track_ids: set[int] = set()
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._publisher.close()

    def process(self, frame: np.ndarray, *, timestamp: float | None = None) -> dict[str, object]:
        height, width = frame.shape[:2]
        target = processing_frame_size(width, height, self.settings.processing_width)
        frame = resize_processing_frame(frame, target)
        self._ensure_modules(target)
        assert self._tracker is not None and self._counter is not None and self._heatmaps is not None
        started = monotonic()
        sample_time = time() if timestamp is None else timestamp
        with self._detector_lock:
            detected = self._detector.detect(frame)
        tracked = self._tracker.update(
            detected.detections,
            camera_id=self.camera.camera_id,
            timestamp=sample_time,
            frame_index=self.processed_frames,
            frame=frame,
        )
        observations = tracked.observations
        if self._speed is not None:
            observations = self._speed.update(
                self.camera.camera_id, observations, timestamp=sample_time
            ).observations
        counted = self._counter.update(self.camera.camera_id, observations, timestamp=sample_time)
        intrusion = (
            self._restricted.update(self.camera.camera_id, observations, timestamp=sample_time)
            if self._restricted is not None
            else None
        )
        queue_result = (
            self._queues.update(self.camera.camera_id, observations, timestamp=sample_time)
            if self._queues is not None
            else None
        )
        heatmap_snapshot = self._heatmaps.update(observations, timestamp=sample_time)
        self.processed_frames += 1
        elapsed = max(monotonic() - self._started, 0.001)
        if intrusion is not None:
            self.restricted_violations += sum(
                event.event_type == "restricted_area_confirmed" for event in intrusion.events
            )
        include_spatial = (
            self._last_spatial_publish is None
            or sample_time - self._last_spatial_publish >= self.settings.spatial_publish_seconds
        )
        if include_spatial:
            self._last_spatial_publish = sample_time
        snapshot = counted.snapshot
        self._expired_track_ids.update(tracked.expired_track_ids)
        if self._has_counting_lines:
            entries, exits = snapshot.cumulative_entries, snapshot.cumulative_exits
        else:
            # Fleet cameras are mapped without counting lines, so directed line
            # crossings never happen. Traffic is approximated from identities:
            # every confirmed person entered the view, every expired track left it.
            entries = snapshot.total_unique_people
            exits = min(entries, len(self._expired_track_ids))
        metrics = live_metrics(
            current_people=snapshot.current_people,
            unique_people=snapshot.total_unique_people,
            active_tracks=len(observations),
            entries=entries,
            exits=exits,
            occupancy={item.zone_id: item.current for item in snapshot.occupancy},
            restricted=intrusion.snapshot if intrusion is not None else None,
            restricted_violations=self.restricted_violations,
            queue_statuses=queue_result.snapshot.queues if queue_result is not None else (),
            crowded=heatmap_snapshot.top_crowded_regions,
            ground=heatmap_snapshot.ground or heatmap_snapshot.image,
            include_spatial_layers=include_spatial,
            processing_fps=self.processed_frames / elapsed,
            frame_count=self.processed_frames,
            active_tracker=self._tracker.name,
        )
        metrics["last_frame_ms"] = (monotonic() - started) * 1000.0
        self._publisher.observe(
            metrics,
            collect_events(
                counted.events,
                intrusion.events if intrusion is not None else None,
                queue_result.events if queue_result is not None else None,
            ),
        )
        return metrics

    def _ensure_modules(self, frame_size: tuple[int, int]) -> None:
        if self._frame_size == frame_size and self._tracker is not None:
            return
        config: CameraConfig = self._config
        counting = CameraCountingConfig.from_camera_config(config, frame_size)
        restricted_config = CameraRestrictedAreaConfig.from_camera_config(config, frame_size)
        queue_config = CameraQueueConfig.from_camera_config(config, frame_size)
        lost_track_buffer = max(3, round(8.0 * self.settings.fps))
        self._tracker = create_tracker(
            self.settings.tracker_type,
            lost_track_buffer=lost_track_buffer,
            history_size=max(16, lost_track_buffer * 4),
            frame_rate=self.settings.fps,
            confirmation_frames=1,
            frame_size=frame_size,
        )
        self._counter = PeopleCounter((counting,))
        self._has_counting_lines = bool(counting.counting_lines)
        # A rebuilt tracker restarts its identities, so the exit ledger restarts too.
        self._expired_track_ids = set()
        self._restricted = (
            RestrictedAreaDetector((restricted_config,)) if restricted_config.zones else None
        )
        self._queues = QueueAnalyzer((queue_config,)) if queue_config.queues else None
        speed_config = CameraSpeedConfig.from_camera_config(config, frame_size)
        self._speed = SpeedEstimator((speed_config,)) if self._queues is not None else None
        self._heatmaps = MovementHeatmaps.from_camera_config(config, frame_size)
        self._frame_size = frame_size


class EmptyDetector:
    """Test double that reports no people."""

    def detect(self, frame: np.ndarray) -> DetectionResult:
        return DetectionResult((), DetectionTimings(0.0, 0.0, 0.0))
