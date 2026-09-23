from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from app.api.jobs import JobManager, JobRecord, select_expired_jobs


NOW = datetime(2026, 9, 19, 12, 0, tzinfo=timezone.utc)


def _record(job_id: str, status: str, age_days: float, root: Path = Path("/jobs")) -> JobRecord:
    stamp = (NOW - timedelta(days=age_days)).isoformat()
    return JobRecord(
        id=job_id,
        application_id="tracking",
        original_filename="shop.mp4",
        camera_id="camera",
        status=status,
        created_at=stamp,
        updated_at=stamp,
        job_directory=str(root / job_id),
        input_video=str(root / job_id / "input.mp4"),
    )


def test_retention_is_disabled_by_default() -> None:
    records = [_record("old", "completed", 400)]

    assert select_expired_jobs(records, retention_days=0, now=NOW) == []
    assert select_expired_jobs(records, retention_days=-3, now=NOW) == []


def test_only_finished_jobs_past_the_window_are_selected() -> None:
    records = [
        _record("old-completed", "completed", 31),
        _record("old-failed", "failed", 45),
        _record("old-cancelled", "cancelled", 30.5),
        _record("fresh-completed", "completed", 29),
        _record("old-running", "running", 90),
        _record("old-queued", "queued", 90),
        _record("old-cancelling", "cancelling", 90),
    ]
    damaged = _record("damaged-stamp", "completed", 90)
    damaged.updated_at = "yesterday"

    expired = select_expired_jobs([*records, damaged], retention_days=30, now=NOW)

    assert [item.id for item in expired] == ["old-completed", "old-failed", "old-cancelled"]


def _persist(manager: JobManager, record: JobRecord) -> Path:
    job_dir = Path(record.job_directory)
    job_dir.mkdir(parents=True)
    (job_dir / "input.mp4").write_bytes(b"video")
    (job_dir / "job.json").write_text(json.dumps({"id": record.id}), encoding="utf-8")
    manager._jobs[record.id] = record
    return job_dir


def test_purge_deletes_expired_directories_and_forgets_the_jobs(tmp_path: Path) -> None:
    manager = JobManager(tmp_path)
    root = manager.root
    settings = root / "_settings"
    settings.mkdir()
    (settings / "camera.yaml").write_text("camera: {}", encoding="utf-8")
    expired_dir = _persist(manager, _record("expired", "completed", 40, root))
    fresh_dir = _persist(manager, _record("fresh", "completed", 2, root))
    running_dir = _persist(manager, _record("running", "running", 40, root))
    # A finished record that (wrongly) points at the settings directory or outside
    # the jobs root must never cause a deletion.
    rogue = _record("rogue", "failed", 40, root)
    rogue.job_directory = str(settings)
    manager._jobs[rogue.id] = rogue
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    escaped = _record("escaped", "failed", 40, root)
    escaped.job_directory = str(outside)
    manager._jobs[escaped.id] = escaped

    assert manager.purge_expired(0, now=NOW) == []
    assert expired_dir.is_dir()

    removed = manager.purge_expired(30, now=NOW)

    assert removed == ["expired"]
    assert not expired_dir.exists()
    assert fresh_dir.is_dir() and running_dir.is_dir()
    assert (settings / "camera.yaml").is_file()
    assert outside.is_dir()
    assert {item.id for item in manager.list()} == {"fresh", "running", "rogue", "escaped"}
