from pathlib import Path
import json
import threading
import time

import app.api.jobs as jobs_module
import numpy as np
from app.api.jobs import JobManager, _last_json_object
from app.api.live import LiveReporter, processing_frame_size
from app.api.presets import APPLICATIONS, get_application


def test_application_catalog_has_unique_ids() -> None:
    identifiers = [item.application_id for item in APPLICATIONS]
    assert len(identifiers) == len(set(identifiers))
    assert {"people_counting", "heatmap", "vertical_queue", "configured_queue"} <= set(identifiers)
    assert all(item.metrics for item in APPLICATIONS)
    assert all(
        definition.display in {"card", "chart", "status", "counter", "table"}
        for item in APPLICATIONS
        for definition in item.metrics
    )


def test_people_counting_exposes_unique_people_metric() -> None:
    keys = {item.key for item in get_application("people_counting").metrics}
    definition = next(item for item in get_application("people_counting").metrics if item.key == "total_unique_people")

    assert "current_people" in keys
    assert "total_unique_people" in keys
    assert definition.aggregation == "total"
    assert definition.display == "counter"


def test_configured_applications_declare_camera_config_requirement() -> None:
    assert get_application("restricted_area").requires_camera_config
    assert get_application("configured_queue").requires_camera_config
    assert not get_application("people_counting").requires_camera_config


def test_configured_queue_exposes_occupancy_and_speed_metrics() -> None:
    keys = {item.key for item in get_application("configured_queue").metrics}

    assert {"queue_length", "queue_speed", "queue_wait_seconds", "queue_details"} <= keys
    assert "entry_count" not in keys
    assert "exit_count" not in keys
    assert "total_unique_people" not in keys


def test_restricted_area_exposes_lifecycle_counters() -> None:
    keys = {item.key for item in get_application("restricted_area").metrics}

    assert {
        "restricted_occupancy",
        "restricted_entries",
        "restricted_exits",
        "restricted_violations",
    } <= keys
    assert "entry_count" not in keys
    assert "exit_count" not in keys
    assert "total_unique_people" not in keys


def test_heatmap_applications_expose_top_crowded_regions_metric() -> None:
    metrics = get_application("heatmap").metrics
    definition = next(item for item in metrics if item.key == "top_crowded_regions")
    assert definition.value_type == "table"
    assert definition.display == "table"


def test_job_command_uses_existing_analytics_cli(tmp_path: Path) -> None:
    manager = JobManager(tmp_path, python_executable="python-test")
    job_dir = tmp_path / "job-1"
    job_dir.mkdir()
    source = job_dir / "input.mp4"
    source.touch()
    record = manager.register(
        job_id="job-1",
        application_id="vertical_queue",
        original_filename="shop.mp4",
        camera_id="test-camera",
        input_video=source,
        camera_config=None,
        max_frames=25,
    )

    command, expected = manager._build_command(record, get_application("vertical_queue"))

    assert command[:3] == ["python-test", "-m", "app.analytics.cli"]
    assert "--enable-queue" in command
    assert command[command.index("--queue-mode") + 1] == "vertical"
    assert command[command.index("--max-frames") + 1] == "25"
    assert command[command.index("--processing-width") + 1] == "1280"
    assert command[command.index("--frame-stride") + 1] == "5"
    assert expected["counts_csv"] == job_dir / "counts.csv"


def test_configured_queue_command_uses_camera_geometry(tmp_path: Path) -> None:
    manager = JobManager(tmp_path, python_executable="python-test")
    job_dir = tmp_path / "job-queue"
    job_dir.mkdir()
    source = job_dir / "input.mp4"
    source.touch()
    camera_config = job_dir / "camera.yaml"
    camera_config.write_text("camera: {id: camera, name: Camera, source: uploaded}\nanalytics: {enabled: [queue]}\n")
    record = manager.register(
        job_id="job-queue",
        application_id="configured_queue",
        original_filename="shop.mp4",
        camera_id="test-camera",
        input_video=source,
        camera_config=camera_config,
        max_frames=40,
    )

    command, expected = manager._build_command(record, get_application("configured_queue"))

    assert command[:3] == ["python-test", "-m", "app.analytics.cli"]
    assert command.count("--enable-queue") == 1
    assert command[command.index("--queue-mode") + 1] == "configured"
    assert command[command.index("--camera-config") + 1] == str(camera_config)
    assert expected["counts_csv"] == job_dir / "counts.csv"


def test_stream_job_passes_rtsp_url_to_existing_pipeline(tmp_path: Path) -> None:
    manager = JobManager(tmp_path, python_executable="python-test")
    job_dir = tmp_path / "live-job"
    job_dir.mkdir()
    source = "rtsp://mediamtx:8554/mobile-1"
    record = manager.register(
        job_id="live-job",
        application_id="people_counting",
        original_filename="live:phone",
        camera_id="phone",
        input_video=source,
        camera_config=None,
        max_frames=100,
        source_type="rtsp",
    )

    command, _ = manager._build_command(record, get_application("people_counting"))
    configuration = json.loads((job_dir / "configuration.json").read_text())

    assert command[3] == source
    assert configuration["source_type"] == "rtsp"
    assert configuration["input_video"] == "<live-rtsp-stream>"


def test_combined_live_job_enables_selected_modules_once(tmp_path: Path) -> None:
    manager = JobManager(tmp_path, python_executable="python-test")
    job_dir = tmp_path / "combined-live"
    job_dir.mkdir()
    record = manager.register(
        job_id="combined-live",
        application_id="people_counting",
        original_filename="live:camera",
        camera_id="camera",
        input_video="rtsp://mediamtx:8554/camera",
        camera_config=None,
        max_frames=100,
        source_type="rtsp",
        enabled_tasks=["people_counting", "heatmap", "vertical_queue"],
    )

    command, expected = manager._build_command(record, get_application("people_counting"))
    configuration = json.loads((job_dir / "configuration.json").read_text())

    assert command.count("--enable-heatmap") == 1
    assert command.count("--enable-queue") == 1
    assert command[command.index("--queue-mode") + 1] == "vertical"
    assert expected["counts_csv"] == job_dir / "counts.csv"
    assert configuration["enabled_tasks"] == [
        "people_counting",
        "heatmap",
        "vertical_queue",
    ]


def test_combined_live_job_enables_restricted_area_with_camera_geometry(tmp_path: Path) -> None:
    manager = JobManager(tmp_path, python_executable="python-test")
    job_dir = tmp_path / "restricted-live"
    job_dir.mkdir()
    camera_config = job_dir / "camera.yaml"
    camera_config.write_text("camera: {id: camera, name: Camera, source: live}\nanalytics: {enabled: []}\n")
    record = manager.register(
        job_id="restricted-live",
        application_id="people_counting",
        original_filename="live:camera",
        camera_id="camera",
        input_video="rtsp://mediamtx:8554/camera",
        camera_config=camera_config,
        max_frames=100,
        source_type="rtsp",
        enabled_tasks=["people_counting", "restricted_area"],
    )

    command, _ = manager._build_command(record, get_application("people_counting"))

    assert command.count("--enable-restricted-area") == 1
    assert command[command.index("--camera-config") + 1] == str(camera_config)


def test_combined_live_job_enables_configured_queue_with_camera_geometry(tmp_path: Path) -> None:
    manager = JobManager(tmp_path, python_executable="python-test")
    job_dir = tmp_path / "queue-live"
    job_dir.mkdir()
    camera_config = job_dir / "camera.yaml"
    camera_config.write_text("camera: {id: camera, name: Camera, source: live}\nanalytics: {enabled: []}\n")
    record = manager.register(
        job_id="queue-live",
        application_id="people_counting",
        original_filename="live:camera",
        camera_id="camera",
        input_video="rtsp://mediamtx:8554/camera",
        camera_config=camera_config,
        max_frames=100,
        source_type="rtsp",
        enabled_tasks=["people_counting", "configured_queue"],
    )

    command, _ = manager._build_command(record, get_application("people_counting"))

    assert command.count("--enable-queue") == 1
    assert command[command.index("--queue-mode") + 1] == "configured"
    assert command[command.index("--camera-config") + 1] == str(camera_config)


def test_tracking_command_does_not_receive_analytics_camera_config(tmp_path: Path) -> None:
    manager = JobManager(tmp_path)
    job_dir = tmp_path / "job-3"
    job_dir.mkdir()
    source = job_dir / "input.mp4"
    source.touch()
    camera_config = job_dir / "camera.yaml"
    camera_config.touch()
    record = manager.register(
        job_id="job-3",
        application_id="tracking",
        original_filename="shop.mp4",
        camera_id="test-camera",
        input_video=source,
        camera_config=camera_config,
        max_frames=10,
    )

    command, _ = manager._build_command(record, get_application("tracking"))

    assert "--camera-config" not in command
    assert command[command.index("--camera-id") + 1] == "test-camera"
    assert command[command.index("--max-frames") + 1] == "10"
    assert command[command.index("--tracker") + 1] == "bytetrack"


def test_selected_tracker_is_persisted_and_added_to_command(tmp_path: Path) -> None:
    manager = JobManager(tmp_path, python_executable="python-test")
    job_dir = tmp_path / "job-stable"
    job_dir.mkdir()
    source = job_dir / "input.mp4"
    source.touch()
    record = manager.register(
        job_id="job-stable",
        application_id="tracking",
        original_filename="shop.mp4",
        camera_id="test-camera",
        input_video=source,
        camera_config=None,
        max_frames=8,
        tracker_type="stabletrack",
    )

    command, _ = manager._build_command(record, get_application("tracking"))
    configuration = json.loads((job_dir / "configuration.json").read_text())

    assert command[command.index("--tracker") + 1] == "stabletrack"
    assert record.tracker_type == "stabletrack"
    assert manager.public_dict(record)["tracker_type"] == "stabletrack"
    assert configuration["tracker_type"] == "stabletrack"


def test_admin_selected_reid_is_persisted_and_added_to_tracking_command(
    tmp_path: Path,
) -> None:
    manager = JobManager(tmp_path)
    job_dir = tmp_path / "job-reid"
    job_dir.mkdir()
    source = job_dir / "input.mp4"
    source.touch()
    record = manager.register(
        job_id="job-reid",
        application_id="people_counting",
        original_filename="shop.mp4",
        camera_id="test-camera",
        input_video=source,
        camera_config=None,
        max_frames=None,
        enable_reid=True,
    )

    command, _ = manager._build_command(record, get_application("people_counting"))
    configuration = json.loads((job_dir / "configuration.json").read_text())

    assert "--enable-reid" in command
    assert record.enable_reid is True
    assert manager.public_dict(record)["enable_reid"] is True
    assert configuration["enable_reid"] is True


def test_heatmap_occupancy_video_gets_stable_browser_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    manager = JobManager(tmp_path)
    job_dir = tmp_path / "job-4"
    heatmap_dir = job_dir / "heatmaps"
    heatmap_dir.mkdir(parents=True)
    source = job_dir / "input.mp4"
    source.touch()
    raw_video = job_dir / "annotated_raw.mp4"
    raw_video.touch()
    occupancy_video = heatmap_dir / "camera_image_occupancy.mp4"
    occupancy_video.touch()
    log_path = job_dir / "process.log"
    log_path.touch()
    record = manager.register(
        job_id="job-4",
        application_id="heatmap",
        original_filename="shop.mp4",
        camera_id="test-camera",
        input_video=source,
        camera_config=None,
        max_frames=None,
    )

    def fake_browser_video(_source: Path, destination: Path) -> bool:
        destination.touch()
        return True

    monkeypatch.setattr(jobs_module, "_make_browser_video", fake_browser_video)
    artifacts = manager._collect_artifacts(
        record, {"annotated_video_raw": raw_video}, log_path
    )

    assert Path(artifacts["heatmap_video"]).name == "heatmap_occupancy.mp4"
    assert Path(artifacts["heatmap_video"]).parent == job_dir


def test_last_json_object_ignores_leading_output() -> None:
    assert _last_json_object('progress\n{"frames": 12, "ok": true}\n') == {
        "frames": 12,
        "ok": True,
    }


def test_live_reporter_persists_sampled_preview_metrics_and_events(tmp_path: Path) -> None:
    reporter = LiveReporter(
        tmp_path,
        "job-live",
        total_frames=10,
        metric_interval=10,
        preview_interval=10,
    )
    reporter.publish(
        1,
        {"current_people": 2, "processing_fps": 12.5},
        frame=np.zeros((720, 1280, 3), dtype=np.uint8),
        force=True,
    )

    assert (tmp_path / "preview.jpg").is_file()
    assert (tmp_path / "metrics.jsonl").is_file()
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert {event["type"] for event in events} == {
        "preview_updated",
        "metrics_updated",
        "progress_updated",
    }
    assert events[-1]["job_id"] == "job-live"
    assert events[-1]["progress"] == 20.0
    assert events[-1]["preview_reference"] == "/api/v1/jobs/job-live/preview"


def test_cancel_request_is_persisted_for_running_process(tmp_path: Path) -> None:
    manager = JobManager(tmp_path)
    job_dir = tmp_path / "job-cancel"
    job_dir.mkdir()
    source = job_dir / "input.mp4"
    source.touch()
    record = manager.register(
        job_id="job-cancel",
        application_id="tracking",
        original_filename="shop.mp4",
        camera_id="camera",
        input_video=source,
        camera_config=None,
        max_frames=None,
    )

    cancelled = manager.cancel(record.id)

    assert cancelled.status == "cancelling"
    assert (job_dir / "cancel.requested").is_file()


class _FakeProcess:
    """Popen stand-in whose reaction to terminate() is scripted by the test."""

    def __init__(self, *, exits_on_terminate: bool, already_exited: bool = False) -> None:
        self.exits_on_terminate = exits_on_terminate
        self.returncode: int | None = 0 if already_exited else None
        self.terminated = threading.Event()
        self.killed = threading.Event()

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated.set()
        if self.exits_on_terminate:
            self.returncode = -15

    def kill(self) -> None:
        self.killed.set()
        self.returncode = -9


def _running_job(tmp_path: Path, process: _FakeProcess, **options: float) -> tuple[JobManager, str]:
    manager = JobManager(tmp_path, **options)
    job_dir = tmp_path / "job-stuck"
    job_dir.mkdir()
    source = job_dir / "input.mp4"
    source.touch()
    record = manager.register(
        job_id="job-stuck",
        application_id="tracking",
        original_filename="shop.mp4",
        camera_id="camera",
        input_video=source,
        camera_config=None,
        max_frames=None,
    )
    manager._update(record, status="running")
    manager._processes[record.id] = process  # type: ignore[assignment]
    return manager, record.id


def test_cancel_terminates_then_kills_a_process_that_ignores_the_marker(tmp_path: Path) -> None:
    process = _FakeProcess(exits_on_terminate=False)
    manager, job_id = _running_job(
        tmp_path, process, cancel_grace_seconds=0.5, cancel_kill_delay_seconds=0.05
    )

    started = time.monotonic()
    record = manager.cancel(job_id)
    returned_after = time.monotonic() - started

    # The HTTP request is answered immediately; the hard stop runs on a daemon timer.
    assert record.status == "cancelling"
    assert not process.terminated.is_set()
    assert returned_after < 0.5
    assert manager._cancel_timers[job_id].daemon is True
    assert (tmp_path / "job-stuck" / "cancel.requested").is_file()

    assert process.terminated.wait(timeout=5)
    assert process.killed.wait(timeout=5)


def test_cancel_does_not_kill_a_process_that_exits_on_terminate(tmp_path: Path) -> None:
    process = _FakeProcess(exits_on_terminate=True)
    manager, job_id = _running_job(
        tmp_path, process, cancel_grace_seconds=0.05, cancel_kill_delay_seconds=0.2
    )

    manager.cancel(job_id)
    manager.cancel(job_id)  # a repeated request must not stack timers

    assert process.terminated.wait(timeout=5)
    deadline = time.monotonic() + 5
    while job_id in manager._cancel_timers and time.monotonic() < deadline:
        time.sleep(0.01)
    assert job_id not in manager._cancel_timers
    assert not process.killed.is_set()


def test_cancel_leaves_a_cooperative_process_alone(tmp_path: Path) -> None:
    process = _FakeProcess(exits_on_terminate=True)
    manager, job_id = _running_job(
        tmp_path, process, cancel_grace_seconds=0.1, cancel_kill_delay_seconds=0.05
    )

    manager.cancel(job_id)
    process.returncode = 0  # the CLI noticed cancel.requested within the grace period
    timer = manager._cancel_timers[job_id]
    timer.join(timeout=5)

    assert not process.terminated.is_set()
    assert not process.killed.is_set()


def test_cancel_grace_period_comes_from_the_environment(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("VIDEO_ANALYTICS_CANCEL_GRACE_SECONDS", raising=False)
    assert JobManager(tmp_path / "default").cancel_grace_seconds == 10.0
    monkeypatch.setenv("VIDEO_ANALYTICS_CANCEL_GRACE_SECONDS", "2.5")
    assert JobManager(tmp_path / "configured").cancel_grace_seconds == 2.5
    monkeypatch.setenv("VIDEO_ANALYTICS_CANCEL_GRACE_SECONDS", "soon")
    assert JobManager(tmp_path / "invalid").cancel_grace_seconds == 10.0


def test_dashboard_processing_size_preserves_aspect_ratio_and_avoids_upscale() -> None:
    assert processing_frame_size(3840, 2160, 1280) == (1280, 720)
    assert processing_frame_size(640, 480, 1280) == (640, 480)
