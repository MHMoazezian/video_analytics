import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import zipfile

import pytest
from fastapi import HTTPException

from app.api import main
from app.api.jobs import JobManager, JobRecord
from app.api.reports import (
    build_job_report,
    exportable_artifacts,
    metrics_csv,
    write_job_export_zip,
)


def _jsonl(path: Path, rows: list[object]) -> None:
    path.write_text(
        "".join((row if isinstance(row, str) else json.dumps(row)) + "\n" for row in rows),
        encoding="utf-8",
    )


def _metrics_row(frame: int, elapsed: float, **metrics: object) -> dict[str, object]:
    return {
        "type": "metrics_updated", "job_id": "job-1", "status": "running",
        "timestamp": f"2026-09-19T10:00:0{frame}+00:00", "frame_index": frame,
        "elapsed_seconds": elapsed, "metrics": metrics,
    }


@pytest.fixture()
def job(tmp_path: Path) -> tuple[JobRecord, Path]:
    job_dir = tmp_path / "jobs" / "job-1"
    (job_dir / "heatmaps").mkdir(parents=True)
    (job_dir / "input.mp4").write_bytes(b"the customer's uploaded video")
    (job_dir / "annotated.mp4").write_bytes(b"annotated")
    (job_dir / "counts.csv").write_text("frame_index,confirmed_humans\n0,2\n", encoding="utf-8")
    (job_dir / "heatmaps" / "shop_occupancy.png").write_bytes(b"png")
    (job_dir / "configuration.json").write_text(
        json.dumps({"job_id": "job-1", "frame_stride": 10, "tracker_type": "bytetrack"}), encoding="utf-8"
    )
    (job_dir / "final_metrics.json").write_text(
        json.dumps({"total_unique_people": 7, "frames": 3}), encoding="utf-8"
    )
    _jsonl(job_dir / "metrics.jsonl", [
        _metrics_row(0, 0.5, current_people=2, processing_fps=10.0, active_tracker="bytetrack",
                     zone_occupancy={"frame": 2}, queue_length=None),
        "{this line was cut off by a crash",
        _metrics_row(1, 1.0, current_people=4, processing_fps=12.5, active_tracker="bytetrack",
                     calibrated=True),
        _metrics_row(2, 1.5, current_people=3, queue_length=5),
    ])
    _jsonl(job_dir / "events.jsonl", [
        {"type": "job_started", "elapsed_seconds": 0.0},
        {"type": "metrics_updated", "elapsed_seconds": 0.5},
        {"type": "metrics_updated", "elapsed_seconds": 1.5},
        {"type": "job_completed", "elapsed_seconds": 1.75},
    ])
    _jsonl(job_dir / "analytics_events.jsonl", [
        {"event_type": "restricted_area_entered", "track_id": 1},
        {"event_type": "restricted_area_entered", "track_id": 2},
        {"event_type": "restricted_area_exited", "track_id": 1},
        {"no_type": True},
    ])
    outside = tmp_path / "secrets.txt"
    outside.write_text("not part of the job", encoding="utf-8")
    record = JobRecord(
        id="job-1",
        application_id="people_counting",
        original_filename="shop.mp4",
        camera_id="camera",
        status="completed",
        created_at="2026-09-19T10:00:00+00:00",
        updated_at="2026-09-19T10:00:09+00:00",
        job_directory=str(job_dir),
        input_video=str(job_dir / "input.mp4"),
        artifacts={
            "annotated_video": str(job_dir / "annotated.mp4"),
            "annotated_video_raw": str(job_dir / "annotated.mp4"),  # no ffmpeg: same file twice
            "counts_csv": str(job_dir / "counts.csv"),
            "heatmap_1": str(job_dir / "heatmaps" / "shop_occupancy.png"),
            "configuration": str(job_dir / "configuration.json"),
            "uploaded": str(job_dir / "input.mp4"),
            "escape": str(outside),
            "traversal": str(job_dir / ".." / ".." / "secrets.txt"),
            "missing": str(job_dir / "tracking.jsonl"),
        },
    )
    return record, job_dir


def test_report_summarises_the_job_files(job) -> None:
    record, job_dir = job
    now = datetime(2026, 9, 19, 11, 0, tzinfo=timezone.utc)

    report = build_job_report(record, job_dir, now=now)

    assert set(report) == {
        "job", "configuration", "final_metrics", "metric_statistics",
        "event_counts", "analytics_event_counts", "duration_seconds", "generated_at",
    }
    assert report["job"]["id"] == "job-1"
    # Server paths never leave the service.
    assert not {"job_directory", "input_video", "camera_config"} & set(report["job"])
    assert report["configuration"]["frame_stride"] == 10
    assert report["final_metrics"] == {"total_unique_people": 7, "frames": 3}
    assert report["metric_statistics"] == {
        "current_people": {"min": 2, "max": 4, "mean": 3.0, "last": 3, "samples": 3},
        "processing_fps": {"min": 10.0, "max": 12.5, "mean": 11.25, "last": 12.5, "samples": 2},
        "queue_length": {"min": 5, "max": 5, "mean": 5.0, "last": 5, "samples": 1},
    }
    assert report["event_counts"] == {"job_completed": 1, "job_started": 1, "metrics_updated": 2}
    assert report["analytics_event_counts"] == {
        "restricted_area_entered": 2, "restricted_area_exited": 1,
    }
    assert report["duration_seconds"] == 1.75
    assert report["generated_at"] == "2026-09-19T11:00:00+00:00"
    json.dumps(report)  # the report is plain JSON


def test_report_embeds_the_public_job_view_when_given(job) -> None:
    record, job_dir = job
    public = {"id": "job-1", "status": "completed", "artifacts": {}, "input_video": "/leak"}

    report = build_job_report(record, job_dir, public_job=public)

    assert report["job"] == {"id": "job-1", "status": "completed", "artifacts": {}}


def test_report_of_a_job_without_files_is_empty_but_complete(tmp_path: Path, job) -> None:
    record, _job_dir = job
    empty = tmp_path / "empty"
    empty.mkdir()

    report = build_job_report(record, empty)

    assert report["configuration"] == {} and report["final_metrics"] == {}
    assert report["metric_statistics"] == {} and report["event_counts"] == {}
    assert report["analytics_event_counts"] == {}
    # Without pipeline events the record's own timestamps give the duration.
    assert report["duration_seconds"] == 9.0


def test_metrics_csv_has_one_row_per_line_and_the_union_of_numeric_keys(job) -> None:
    _record, job_dir = job

    rows = list(csv.reader(io.StringIO(metrics_csv(job_dir))))

    assert rows[0] == [
        "timestamp", "frame_index", "elapsed_seconds",
        "current_people", "processing_fps", "queue_length",
    ]
    assert rows[1:] == [
        ["2026-09-19T10:00:00+00:00", "0", "0.5", "2", "10.0", ""],
        ["2026-09-19T10:00:01+00:00", "1", "1.0", "4", "12.5", ""],
        ["2026-09-19T10:00:02+00:00", "2", "1.5", "3", "", "5"],
    ]


def test_metrics_csv_of_a_job_without_metrics_is_only_the_header(tmp_path: Path) -> None:
    assert metrics_csv(tmp_path) == "timestamp,frame_index,elapsed_seconds\n"


def test_exportable_artifacts_stay_inside_the_job_and_skip_the_upload(job) -> None:
    record, job_dir = job

    selected = exportable_artifacts(record.artifacts, job_dir, input_video=record.input_video)

    assert sorted(name for name, _path in selected) == [
        "artifacts/annotated.mp4",
        "artifacts/configuration.json",
        "artifacts/counts.csv",
        "artifacts/heatmaps/shop_occupancy.png",
    ]


def test_export_zip_contains_report_csv_and_artifacts_only(tmp_path: Path, job) -> None:
    record, job_dir = job
    report = build_job_report(record, job_dir)
    destination = tmp_path / "export.zip"

    names = write_job_export_zip(
        destination, report, job_dir, record.artifacts, input_video=record.input_video
    )

    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
        assert sorted(archive.namelist()) == sorted(names) == [
            "artifacts/annotated.mp4",
            "artifacts/configuration.json",
            "artifacts/counts.csv",
            "artifacts/heatmaps/shop_occupancy.png",
            "metrics.csv",
            "report.json",
        ]
        assert json.loads(archive.read("report.json"))["final_metrics"]["frames"] == 3
        assert archive.read("metrics.csv").decode("utf-8").startswith("timestamp,frame_index,elapsed_seconds,")
        assert archive.read("artifacts/annotated.mp4") == b"annotated"
        assert all(b"uploaded video" not in archive.read(name) for name in archive.namelist())


def _served_job(tmp_path: Path, monkeypatch, job) -> JobManager:
    record, _job_dir = job
    manager = JobManager(tmp_path / "jobs")
    manager._jobs[record.id] = record
    monkeypatch.setattr(main, "manager", manager)
    monkeypatch.setattr(main, "JOBS_ROOT", manager.root)
    return manager


def test_report_route_wraps_the_report_and_404s_unknown_jobs(tmp_path: Path, monkeypatch, job) -> None:
    _served_job(tmp_path, monkeypatch, job)

    data = main.job_report("job-1")["data"]

    assert data["job"]["application"]["id"] == "people_counting"
    assert set(data["job"]["artifacts"]["annotated_video"]) == {"filename", "media_type", "url"}
    assert data["metric_statistics"]["current_people"]["samples"] == 3
    with pytest.raises(HTTPException) as raised:
        main.job_report("missing")
    assert raised.value.status_code == 404
    with pytest.raises(HTTPException):
        main.job_export("missing")


def test_export_route_serves_a_zip_and_removes_its_scratch_file(tmp_path: Path, monkeypatch, job) -> None:
    manager = _served_job(tmp_path, monkeypatch, job)

    response = main.job_export("job-1")

    archive_path = Path(response.path)
    assert archive_path.parent == manager.root and archive_path.name.startswith(".export-")
    assert response.media_type == "application/zip"
    assert 'filename="job-job-1.zip"' in response.headers["content-disposition"]
    with zipfile.ZipFile(archive_path) as archive:
        assert "report.json" in archive.namelist()
        assert not any("input" in name for name in archive.namelist())
    # Starlette runs the background task after the body is sent.
    import asyncio

    asyncio.run(response.background())
    assert not archive_path.exists()
    # The scratch file is not mistaken for a job on the next start.
    assert [item.id for item in JobManager(manager.root).list()] == []
