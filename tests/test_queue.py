"""Phase 8 synthetic configured-queue analytics tests."""

from __future__ import annotations

import numpy as np

from app.analytics import (
    CameraQueueConfig,
    QueueAnalyzer,
    QueueResult,
    QueueTrackState,
    annotate_queues,
)
from app.analytics.cli import build_parser
from app.core.models import TrackObservation, TrajectoryPoint
from app.geometry.config import NormalizedPoint, QueueRegion, ServicePoint


def _queue(
    queue_id: str = "checkout",
    *,
    left: float = 0.2,
    right: float = 0.8,
    service_x: float | None = None,
    dwell: float = 1.0,
    maximum_speed: float = 20.0,
    gap: float = 1.0,
    overflow: int = 3,
    smoothing: float = 0.5,
) -> QueueRegion:
    return QueueRegion(
        queue_id=queue_id,
        polygon=(
            NormalizedPoint(left, 0.2),
            NormalizedPoint(right, 0.2),
            NormalizedPoint(right, 0.8),
            NormalizedPoint(left, 0.8),
        ),
        service_point=ServicePoint(
            NormalizedPoint(right if service_x is None else service_x, 0.5),
            "desk",
        ),
        overflow_threshold=overflow,
        minimum_dwell_seconds=dwell,
        maximum_speed_pixels_per_second=maximum_speed,
        gap_tolerance_seconds=gap,
        service_completion_radius=0.08,
        count_smoothing_alpha=smoothing,
    )


def _observation(
    track_id: int,
    point: tuple[float, float],
    timestamp: float,
    *,
    camera_id: str = "cam-a",
    confirmed: bool = True,
    velocity: tuple[float, float] | None = (0.0, 0.0),
) -> TrackObservation:
    x, y = point
    return TrackObservation(
        camera_id=camera_id,
        track_id=track_id,
        timestamp=timestamp,
        frame_index=int(timestamp * 10),
        xyxy=(x - 3, y - 12, x + 3, y),
        foot_point=point,
        detection_confidence=0.9,
        confirmed=confirmed,
        trajectory=(),
        smoothed_velocity=velocity,
    )


def _analyzer(*queues: QueueRegion, camera_id: str = "cam-a") -> QueueAnalyzer:
    return QueueAnalyzer((CameraQueueConfig(camera_id, (101, 101), queues),))


def _types(result: QueueResult) -> tuple[str, ...]:
    return tuple(event.event_type for event in result.events)


def test_passer_by_crossing_polygon_never_becomes_queue_member() -> None:
    analyzer = _analyzer(_queue(dwell=2))

    entered = analyzer.update("cam-a", [_observation(1, (50, 50), 0)], timestamp=0)
    left = analyzer.update("cam-a", [_observation(1, (10, 50), 0.5)], timestamp=0.5)

    assert entered.events == left.events == ()
    assert left.snapshot.queue_for("checkout").raw_count == 0
    assert left.snapshot.state_for("checkout", 1) is None


def test_person_dwelling_inside_joins_and_fast_person_does_not() -> None:
    analyzer = _analyzer(_queue(dwell=1, maximum_speed=10))
    analyzer.update(
        "cam-a",
        [
            _observation(1, (50, 50), 0),
            _observation(2, (60, 50), 0, velocity=(20, 0)),
        ],
        timestamp=0,
    )
    joined = analyzer.update(
        "cam-a",
        [
            _observation(1, (50, 50), 1),
            _observation(2, (60, 50), 1, velocity=(20, 0)),
        ],
        timestamp=1,
    )

    assert _types(joined) == ("queue_joined",)
    assert joined.events[0].track_id == 1
    assert joined.events[0].payload["membership_is_heuristic"] is True
    assert joined.snapshot.state_for("checkout", 1) is QueueTrackState.MEMBER
    assert joined.snapshot.state_for("checkout", 2) is None


def test_speed_is_derived_from_shared_smoothed_trajectory_when_needed() -> None:
    analyzer = _analyzer(_queue(dwell=0, maximum_speed=10))
    trajectory = (
        TrajectoryPoint(0, 0, (20, 50), (20, 50)),
        TrajectoryPoint(1, 10, (50, 50), (50, 50)),
    )
    observation = _observation(1, (50, 50), 1, velocity=None)
    observation = TrackObservation(
        camera_id=observation.camera_id,
        track_id=observation.track_id,
        timestamp=observation.timestamp,
        frame_index=observation.frame_index,
        xyxy=observation.xyxy,
        foot_point=observation.foot_point,
        detection_confidence=observation.detection_confidence,
        confirmed=observation.confirmed,
        trajectory=trajectory,
    )

    result = analyzer.update("cam-a", [observation], timestamp=1)

    assert result.events == ()
    assert result.snapshot.state_for("checkout", 1) is None


def test_join_and_leave_near_service_emit_completed_wait() -> None:
    analyzer = _analyzer(_queue(dwell=1))
    analyzer.update("cam-a", [_observation(7, (50, 50), 0)], timestamp=0)
    joined = analyzer.update("cam-a", [_observation(7, (60, 50), 1)], timestamp=1)
    left = analyzer.update("cam-a", [_observation(7, (85, 50), 3)], timestamp=3)

    assert _types(joined) == ("queue_joined",)
    assert _types(left) == ("queue_left",)
    assert left.events[0].zone_id == "checkout"
    assert left.events[0].payload == {
        "waiting_seconds": 3,
        "completed_near_service": True,
        "leave_reason": "outside_polygon",
    }
    status = left.snapshot.queue_for("checkout")
    assert status.raw_count == 0
    assert status.completed_wait_count == 1
    assert status.last_completed_waiting_seconds == 3
    assert status.average_completed_waiting_seconds == 3


def test_short_tracking_gaps_preserve_candidate_and_member_state() -> None:
    analyzer = _analyzer(_queue(dwell=1, gap=1))
    analyzer.update("cam-a", [_observation(1, (50, 50), 0)], timestamp=0)
    missing_candidate = analyzer.update("cam-a", [], timestamp=0.5)
    resumed = analyzer.update("cam-a", [_observation(1, (50, 50), 1)], timestamp=1)
    missing_member = analyzer.update("cam-a", [], timestamp=1.8)
    expired = analyzer.update("cam-a", [], timestamp=2.1)

    assert (
        missing_candidate.snapshot.state_for("checkout", 1)
        is QueueTrackState.CANDIDATE
    )
    assert _types(resumed) == ("queue_joined",)
    assert missing_member.snapshot.state_for("checkout", 1) is QueueTrackState.MEMBER
    assert _types(expired) == ("queue_left",)
    assert expired.events[0].payload["leave_reason"] == "tracking_gap"


def test_overflow_events_are_edge_triggered() -> None:
    analyzer = _analyzer(_queue(dwell=0, overflow=2))
    started = analyzer.update(
        "cam-a",
        [_observation(1, (40, 50), 0), _observation(2, (60, 50), 0)],
        timestamp=0,
    )
    steady = analyzer.update(
        "cam-a",
        [_observation(1, (40, 50), 1), _observation(2, (60, 50), 1)],
        timestamp=1,
    )
    ended = analyzer.update(
        "cam-a",
        [_observation(1, (40, 50), 2), _observation(2, (10, 50), 2)],
        timestamp=2,
    )

    assert _types(started) == (
        "queue_joined",
        "queue_joined",
        "queue_overflow_started",
    )
    assert steady.events == ()
    assert _types(ended) == ("queue_left", "queue_overflow_ended")
    assert not ended.snapshot.queue_for("checkout").overflow


def test_multiple_configured_queues_keep_independent_state() -> None:
    analyzer = _analyzer(
        _queue("left", left=0.05, right=0.45, dwell=0),
        _queue("right", left=0.55, right=0.95, dwell=0),
    )
    result = analyzer.update(
        "cam-a",
        [_observation(1, (25, 50), 0), _observation(2, (75, 50), 0)],
        timestamp=0,
    )

    assert result.snapshot.queue_for("left").raw_count == 1
    assert result.snapshot.queue_for("right").raw_count == 1
    assert result.snapshot.state_for("left", 1) is QueueTrackState.MEMBER
    assert result.snapshot.state_for("right", 2) is QueueTrackState.MEMBER
    assert result.snapshot.state_for("left", 2) is None


def test_reset_clears_tracks_counts_smoothing_overflow_and_wait_totals() -> None:
    analyzer = _analyzer(_queue(dwell=0, overflow=1))
    analyzer.update("cam-a", [_observation(1, (50, 50), 0)], timestamp=0)
    analyzer.update("cam-a", [_observation(1, (85, 50), 1)], timestamp=1)

    analyzer.reset("cam-a")

    status = analyzer.snapshot("cam-a").queue_for("checkout")
    assert analyzer.snapshot("cam-a").tracks == ()
    assert status.raw_count == 0
    assert status.smoothed_count == 0
    assert not status.overflow
    assert status.completed_wait_count == 0


def test_current_wait_is_mean_elapsed_wait_for_members() -> None:
    analyzer = _analyzer(_queue(dwell=0))
    analyzer.update("cam-a", [_observation(1, (40, 50), 0)], timestamp=0)
    analyzer.update(
        "cam-a",
        [_observation(1, (40, 50), 2), _observation(2, (60, 50), 2)],
        timestamp=2,
    )
    result = analyzer.update(
        "cam-a",
        [_observation(1, (40, 50), 4), _observation(2, (60, 50), 4)],
        timestamp=4,
    )

    assert (
        result.snapshot.queue_for("checkout").approximate_current_waiting_seconds
        == 3
    )


def test_smoothed_count_uses_configured_exponential_average() -> None:
    analyzer = _analyzer(_queue(dwell=0, smoothing=0.5))
    joined = analyzer.update("cam-a", [_observation(1, (50, 50), 0)], timestamp=0)
    left = analyzer.update("cam-a", [_observation(1, (10, 50), 1)], timestamp=1)

    assert joined.snapshot.queue_for("checkout").raw_count == 1
    assert joined.snapshot.queue_for("checkout").smoothed_count == 0.5
    assert left.snapshot.queue_for("checkout").raw_count == 0
    assert left.snapshot.queue_for("checkout").smoothed_count == 0.25


def test_unconfirmed_tracks_are_ignored_and_overlay_is_composable() -> None:
    queue = _queue(dwell=0)
    config = CameraQueueConfig("cam-a", (101, 101), (queue,))
    analyzer = QueueAnalyzer((config,))
    ignored = _observation(1, (50, 50), 0, confirmed=False)
    result = analyzer.update("cam-a", [ignored], timestamp=0)
    frame = np.zeros((101, 101, 3), dtype=np.uint8)

    annotated = annotate_queues(frame, config, result.snapshot, [ignored])

    assert result.events == ()
    assert result.snapshot.queue_for("checkout").raw_count == 0
    assert np.count_nonzero(frame) == 0
    assert np.count_nonzero(annotated) > 0


def test_queue_cli_is_disabled_until_explicitly_enabled() -> None:
    parser = build_parser()

    disabled = parser.parse_args(["input.mp4"])
    enabled = parser.parse_args(["input.mp4", "--enable-queue"])

    assert not disabled.enable_queue
    assert not disabled.enable_restricted_area
    assert enabled.enable_queue
    assert not enabled.enable_restricted_area
    assert enabled.queue_mode == "vertical"
    assert parser.parse_args(
        ["input.mp4", "--enable-restricted-area"]
    ).enable_restricted_area


def test_foot_point_inside_but_bbox_mostly_outside_queue_does_not_qualify() -> None:
    # Queue spans x in [20, 80]; this bbox straddles the right edge so only
    # two-thirds of its area overlaps the queue, below the 0.75 requirement.
    analyzer = _analyzer(_queue(dwell=0.0))
    observation = TrackObservation(
        camera_id="cam-a",
        track_id=1,
        timestamp=0.0,
        frame_index=0,
        xyxy=(76.0, 38.0, 82.0, 50.0),
        foot_point=(79.0, 50.0),
        detection_confidence=0.9,
        confirmed=True,
        trajectory=(),
        smoothed_velocity=(0.0, 0.0),
    )

    result = analyzer.update("cam-a", [observation], timestamp=0.0)

    assert result.events == ()
    assert result.snapshot.queue_for("checkout").raw_count == 0
    assert result.snapshot.state_for("checkout", 1) is None


def test_overlay_uses_one_stable_color_per_queue_and_one_summary_line() -> None:
    left = _queue("left", left=0.05, right=0.45, dwell=0)
    right = _queue("right", left=0.55, right=0.95, dwell=0)
    config = CameraQueueConfig("cam-a", (101, 101), (left, right))
    analyzer = QueueAnalyzer((config,))
    observations = (
        _observation(1, (25, 40), 0),
        _observation(2, (35, 60), 0),
        _observation(3, (75, 50), 0),
    )
    result = analyzer.update("cam-a", observations, timestamp=0)

    annotated = annotate_queues(
        np.zeros((101, 101, 3), dtype=np.uint8),
        config,
        result.snapshot,
        observations,
    )

    first_left_color = tuple(annotated[28, 22])
    second_left_color = tuple(annotated[48, 32])
    right_color = tuple(annotated[38, 72])
    assert first_left_color == second_left_color == (255, 160, 0)
    assert right_color == (0, 200, 255)
    assert right_color != first_left_color
    # The summary bar covers the bottom 26 rows (75-100). Its text is wider than
    # this tiny frame and runs over the middle rows, so sample the bar's top
    # edge, which no glyph reaches.
    assert tuple(annotated[76, 100]) == (20, 20, 20)
