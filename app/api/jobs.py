"""Persistent local job runner for recorded-video analytics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timedelta, timezone
from concurrent.futures import Future, ThreadPoolExecutor
import json
import mimetypes
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Iterable

from app.api.presets import ApplicationPreset, get_application
from app.core.config import PROJECT_ROOT


TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
SETTINGS_DIRECTORY_NAME = "_settings"
DEFAULT_CANCEL_GRACE_SECONDS = 10.0
CANCEL_KILL_DELAY_SECONDS = 5.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class JobRecord:
    id: str
    application_id: str
    original_filename: str
    camera_id: str
    status: str
    created_at: str
    updated_at: str
    job_directory: str
    input_video: str
    camera_config: str | None = None
    max_frames: int | None = None
    enable_reid: bool = False
    tracker_type: str = "bytetrack"
    error: str | None = None
    summary: dict[str, Any] | None = None
    artifacts: dict[str, str] = field(default_factory=dict)
    source_type: str = "file"
    enabled_tasks: list[str] = field(default_factory=list)


class JobManager:
    """Run one process per job while persisting metadata beside its artifacts."""

    def __init__(
        self,
        root: Path,
        *,
        python_executable: str | None = None,
        max_workers: int = 1,
        processing_width: int = 1280,
        frame_stride: int = 5,
        cancel_grace_seconds: float | None = None,
        cancel_kill_delay_seconds: float = CANCEL_KILL_DELAY_SECONDS,
    ) -> None:
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        if processing_width < 2:
            raise ValueError("processing_width must be at least 2")
        if frame_stride <= 0:
            raise ValueError("frame_stride must be positive")
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.python_executable = python_executable or sys.executable
        self.processing_width = processing_width
        self.frame_stride = frame_stride
        self.cancel_grace_seconds = (
            _cancel_grace_from_environ()
            if cancel_grace_seconds is None
            else max(0.0, cancel_grace_seconds)
        )
        self.cancel_kill_delay_seconds = max(0.0, cancel_kill_delay_seconds)
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="video-analytics-job"
        )
        self._futures: dict[str, Future[None]] = {}
        self._processes: dict[str, subprocess.Popen[str]] = {}
        self._cancel_timers: dict[str, threading.Timer] = {}
        self._load_existing()

    def register(
        self,
        *,
        job_id: str,
        application_id: str,
        original_filename: str,
        camera_id: str,
        input_video: Path | str,
        camera_config: Path | None,
        max_frames: int | None,
        enable_reid: bool = False,
        tracker_type: str = "bytetrack",
        source_type: str = "file",
        enabled_tasks: list[str] | None = None,
    ) -> JobRecord:
        get_application(application_id)
        from app.tracking.factory import available_tracker_types

        selected_tracker = tracker_type.strip().lower() or "bytetrack"
        if selected_tracker not in available_tracker_types():
            known = ", ".join(available_tracker_types())
            raise ValueError(f"unknown tracker type {selected_tracker!r}; expected one of: {known}")
        now = _now()
        record = JobRecord(
            id=job_id,
            application_id=application_id,
            original_filename=original_filename,
            camera_id=camera_id,
            status="queued",
            created_at=now,
            updated_at=now,
            job_directory=(
                str(input_video.parent.resolve())
                if isinstance(input_video, Path)
                else str((self.root / job_id).resolve())
            ),
            input_video=str(input_video.resolve()) if isinstance(input_video, Path) else input_video,
            camera_config=str(camera_config.resolve()) if camera_config else None,
            max_frames=max_frames,
            enable_reid=enable_reid,
            tracker_type=selected_tracker,
            source_type=source_type,
            enabled_tasks=list(enabled_tasks or ()),
        )
        with self._lock:
            self._jobs[job_id] = record
            self._save(record)
            self._save_configuration(record)
        return record

    def list(self) -> list[JobRecord]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda item: item.created_at, reverse=True)

    def get(self, job_id: str) -> JobRecord:
        with self._lock:
            try:
                return self._jobs[job_id]
            except KeyError as exc:
                raise KeyError(f"job not found: {job_id}") from exc

    def run(self, job_id: str) -> None:
        record = self.get(job_id)
        preset = get_application(record.application_id)
        if (Path(record.job_directory) / "cancel.requested").exists():
            self._update(record, status="cancelled", error=None)
            self._append_event(record, "job_cancelled", status="cancelled")
            return
        self._update(record, status="running", error=None)
        self._append_event(record, "job_started", status="running")
        command, expected = self._build_command(record, preset)
        job_dir = Path(record.job_directory)
        log_path = job_dir / "process.log"
        stdout_path = job_dir / "stdout.log"
        stderr_path = job_dir / "stderr.log"
        try:
            with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
                "w", encoding="utf-8"
            ) as stderr:
                process = subprocess.Popen(
                    command,
                    cwd=PROJECT_ROOT,
                    env=os.environ.copy(),
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                )
                with self._lock:
                    self._processes[job_id] = process
                while process.poll() is None:
                    time.sleep(0.2)
                returncode = process.returncode
            with self._lock:
                self._processes.pop(job_id, None)
            process_stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
            process_stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
            log_path.write_text(
                f"$ {' '.join(command)}\n\nSTDOUT\n{process_stdout}\n\nSTDERR\n{process_stderr}",
                encoding="utf-8",
            )
            cancelled = (job_dir / "cancel.requested").exists()
            if cancelled:
                self._update(record, status="cancelled", error=None)
                self._append_event(record, "job_cancelled", status="cancelled")
                return
            if returncode != 0:
                detail = process_stderr.strip() or process_stdout.strip()
                raise RuntimeError(detail[-6000:] or f"process exited with {returncode}")

            artifacts = self._collect_artifacts(record, expected, log_path)
            summary = _last_json_object(process_stdout)
            live_state = _read_json(job_dir / "live_state.json") or {}
            final_metrics = {
                **(live_state.get("metrics") if isinstance(live_state.get("metrics"), dict) else {}),
                **(summary or {}),
            }
            if final_metrics:
                (job_dir / "final_metrics.json").write_text(
                    json.dumps(final_metrics, indent=2), encoding="utf-8"
                )
                artifacts["final_metrics"] = str((job_dir / "final_metrics.json").resolve())
            self._update(
                record,
                status="completed",
                summary=summary,
                artifacts=artifacts,
                error=None,
            )
            self._append_event(
                record, "job_completed", status="completed", metrics=final_metrics
            )
        except Exception as exc:
            with self._lock:
                self._processes.pop(job_id, None)
            artifacts = {"process_log": str(log_path)} if log_path.is_file() else {}
            self._update(record, status="failed", error=str(exc), artifacts=artifacts)
            self._append_event(
                record, "job_failed", status="failed", message=str(exc)
            )

    def enqueue(self, job_id: str) -> Future[None]:
        """Submit a job without blocking the HTTP request thread."""

        self.get(job_id)
        future = self._executor.submit(self.run, job_id)
        with self._lock:
            self._futures[job_id] = future
        return future

    def cancel(self, job_id: str) -> JobRecord:
        record = self.get(job_id)
        if record.status in TERMINAL_STATUSES:
            return record
        job_dir = Path(record.job_directory)
        (job_dir / "cancel.requested").touch()
        self._update(record, status="cancelling")
        self._append_event(record, "warning", status="cancelling", message="Cancellation requested")
        self._schedule_forced_stop(job_id)
        return record

    def _schedule_forced_stop(self, job_id: str) -> None:
        """Back the cooperative marker with a hard stop, off the request thread.

        The CLI only notices ``cancel.requested`` when it publishes a frame, so a
        process blocked on an RTSP open, a model load, or ffmpeg would otherwise
        hold its worker slot forever.
        """

        with self._lock:
            if job_id in self._cancel_timers:
                return
            timer = threading.Timer(
                self.cancel_grace_seconds, self._force_stop, args=(job_id,)
            )
            timer.daemon = True
            timer.name = f"video-analytics-cancel-{job_id}"
            self._cancel_timers[job_id] = timer
        timer.start()

    def _force_stop(self, job_id: str) -> None:
        """terminate() a process that outlived the grace period, then kill() it."""

        try:
            with self._lock:
                process = self._processes.get(job_id)
            if process is None or process.poll() is not None:
                return
            process.terminate()
            deadline = time.monotonic() + self.cancel_kill_delay_seconds
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    return
                time.sleep(min(0.1, self.cancel_kill_delay_seconds))
            if process.poll() is None:
                process.kill()
        except OSError:
            # The process exited between poll() and the signal.
            pass
        finally:
            with self._lock:
                self._cancel_timers.pop(job_id, None)

    def purge_expired(
        self, retention_days: float, *, now: datetime | None = None
    ) -> list[str]:
        """Delete finished jobs older than the retention window; return their ids."""

        with self._lock:
            expired = select_expired_jobs(
                self._jobs.values(), retention_days=retention_days, now=now
            )
        removed: list[str] = []
        for record in expired:
            job_dir = Path(record.job_directory).resolve()
            if not _is_purgeable_directory(job_dir, self.root):
                continue
            shutil.rmtree(job_dir, ignore_errors=True)
            if job_dir.exists():
                continue
            with self._lock:
                self._jobs.pop(record.id, None)
                self._futures.pop(record.id, None)
            removed.append(record.id)
        return removed

    def public_dict(self, record: JobRecord) -> dict[str, Any]:
        data = asdict(record)
        data.pop("job_directory")
        data.pop("input_video")
        data.pop("camera_config")
        try:
            data["application"] = get_application(record.application_id).public_dict()
        except ValueError:
            data["application"] = {
                "id": record.application_id,
                "name": record.application_id,
                "description": "This application preset is no longer available.",
                "requires_camera_config": False,
                "metric_schema": [],
            }
        state_path = Path(record.job_directory) / "live_state.json"
        data["live"] = _read_json(state_path)
        data["preview_url"] = (
            f"/api/v1/jobs/{record.id}/preview"
            if (Path(record.job_directory) / "preview.jpg").is_file()
            else None
        )
        data["artifacts"] = {
            key: {
                "filename": Path(path).name,
                "media_type": mimetypes.guess_type(path)[0] or "application/octet-stream",
                "url": f"/api/v1/jobs/{record.id}/artifacts/{key}",
            }
            for key, path in record.artifacts.items()
            if Path(path).is_file()
        }
        return data

    def job_file(self, job_id: str, filename: str) -> Path:
        record = self.get(job_id)
        path = Path(record.job_directory) / filename
        if not path.is_file():
            raise KeyError(f"job file not found: {filename}")
        return path

    def artifact_path(self, job_id: str, key: str) -> Path:
        record = self.get(job_id)
        try:
            path = Path(record.artifacts[key]).resolve()
        except KeyError as exc:
            raise KeyError(f"artifact not found: {key}") from exc
        job_dir = Path(record.job_directory).resolve()
        if not path.is_relative_to(job_dir) or not path.is_file():
            raise KeyError(f"artifact not found: {key}")
        return path

    def _build_command(
        self, record: JobRecord, preset: ApplicationPreset
    ) -> tuple[list[str], dict[str, Path]]:
        job_dir = Path(record.job_directory)
        raw_video = job_dir / "annotated_raw.mp4"
        counts_csv = job_dir / "counts.csv"
        events_jsonl = job_dir / "analytics_events.jsonl"
        heatmap_dir = job_dir / "heatmaps"
        command = [
            self.python_executable,
            "-m",
            preset.module,
            record.input_video,
            "--output",
            str(raw_video),
            "--live-dir",
            str(job_dir),
            "--job-id",
            record.id,
            "--processing-width",
            str(self.processing_width),
            "--frame-stride",
            str(self.frame_stride),
        ]
        expected = {"annotated_video_raw": raw_video}
        if preset.module == "app.analytics.cli":
            command.extend(("--counts-csv", str(counts_csv), "--events-jsonl", str(events_jsonl)))
            expected["counts_csv"] = counts_csv
            expected["analytics_events"] = events_jsonl
            if "--enable-heatmap" in preset.arguments or "heatmap" in record.enabled_tasks:
                command.extend(("--heatmap-dir", str(heatmap_dir)))
        # A dashboard camera YAML describes analytics geometry.  The standalone
        # tracking and detection CLIs accept application settings under
        # ``--config`` instead and must not receive ``--camera-config``.
        if record.camera_config and preset.module == "app.analytics.cli":
            command.extend(("--camera-config", record.camera_config))
        if preset.module != "app.detection.cli":
            command.extend(("--camera-id", record.camera_id))
            if record.enable_reid:
                command.append("--enable-reid")
            if record.tracker_type:
                command.extend(("--tracker", record.tracker_type))
        if record.max_frames is not None:
            command.extend(("--max-frames", str(record.max_frames)))
        command.extend(preset.arguments)
        if record.enabled_tasks:
            if "heatmap" in record.enabled_tasks and "--enable-heatmap" not in command:
                command.append("--enable-heatmap")
            if "vertical_queue" in record.enabled_tasks and "--enable-queue" not in command:
                command.extend(("--enable-queue", "--queue-mode", "vertical"))
            if "configured_queue" in record.enabled_tasks and "--enable-queue" not in command:
                command.extend(("--enable-queue", "--queue-mode", "configured"))
            if "restricted_area" in record.enabled_tasks and "--enable-restricted-area" not in command:
                command.append("--enable-restricted-area")
        return command, expected

    def _collect_artifacts(
        self, record: JobRecord, expected: dict[str, Path], log_path: Path
    ) -> dict[str, str]:
        artifacts = {
            key: str(path.resolve()) for key, path in expected.items() if path.is_file()
        }
        raw_video = expected["annotated_video_raw"]
        browser_video = Path(record.job_directory) / "annotated.mp4"
        if raw_video.is_file() and _make_browser_video(raw_video, browser_video):
            artifacts["annotated_video"] = str(browser_video.resolve())
        elif raw_video.is_file():
            artifacts["annotated_video"] = str(raw_video.resolve())
        heatmap_dir = Path(record.job_directory) / "heatmaps"
        if heatmap_dir.is_dir():
            for index, path in enumerate(sorted(p for p in heatmap_dir.rglob("*") if p.is_file()), 1):
                artifacts[f"heatmap_{index}"] = str(path.resolve())
                if path.name.endswith("_occupancy.mp4"):
                    browser_heatmap = Path(record.job_directory) / "heatmap_occupancy.mp4"
                    if _make_browser_video(path, browser_heatmap):
                        artifacts["heatmap_video"] = str(browser_heatmap.resolve())
                    else:
                        artifacts["heatmap_video"] = str(path.resolve())
        artifacts["process_log"] = str(log_path.resolve())
        for key, filename in (
            ("events", "events.jsonl"),
            ("metric_time_series", "metrics.jsonl"),
            ("tracking_results", "tracking.jsonl"),
            ("configuration", "configuration.json"),
        ):
            path = Path(record.job_directory) / filename
            if path.is_file():
                artifacts[key] = str(path.resolve())
        return artifacts

    def _update(self, record: JobRecord, **changes: Any) -> None:
        with self._lock:
            for name, value in changes.items():
                setattr(record, name, value)
            record.updated_at = _now()
            self._save(record)

    def _save(self, record: JobRecord) -> None:
        path = Path(record.job_directory) / "job.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")
        temporary.replace(path)

    def _save_configuration(self, record: JobRecord) -> None:
        payload = {
            "job_id": record.id,
            "application_id": record.application_id,
            "camera_id": record.camera_id,
            "original_filename": record.original_filename,
            "input_video": (
                Path(record.input_video).name
                if record.source_type == "file"
                else "<live-rtsp-stream>"
            ),
            "source_type": record.source_type,
            "camera_config": Path(record.camera_config).name if record.camera_config else None,
            "max_frames": record.max_frames,
            "enable_reid": record.enable_reid,
            "tracker_type": record.tracker_type,
            "enabled_tasks": record.enabled_tasks,
            "processing_width": self.processing_width,
            "frame_stride": self.frame_stride,
            "created_at": record.created_at,
        }
        (Path(record.job_directory) / "configuration.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    def _append_event(
        self,
        record: JobRecord,
        event_type: str,
        *,
        status: str,
        metrics: dict[str, Any] | None = None,
        message: str | None = None,
    ) -> None:
        state = _read_json(Path(record.job_directory) / "live_state.json") or {}
        payload = {
            "type": event_type,
            "job_id": record.id,
            "timestamp": _now(),
            "status": status,
            "frame_index": state.get("frame_index"),
            "progress": 100.0 if status == "completed" else state.get("progress"),
            "elapsed_seconds": state.get("elapsed_seconds", 0.0),
            "metrics": metrics if metrics is not None else state.get("metrics", {}),
            "preview_reference": state.get("preview_reference"),
            "message": message,
        }
        with (Path(record.job_directory) / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, separators=(",", ":")) + "\n")
        temporary = Path(record.job_directory) / "live_state.json.tmp"
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(Path(record.job_directory) / "live_state.json")

    def _load_existing(self) -> None:
        for metadata in self.root.glob("*/job.json"):
            try:
                values = json.loads(metadata.read_text(encoding="utf-8"))
                allowed = {item.name for item in fields(JobRecord)}
                values = {key: value for key, value in values.items() if key in allowed}
                values.setdefault("tracker_type", "bytetrack")
                record = JobRecord(**values)
                if record.status in {"queued", "running", "cancelling"}:
                    record.status = "failed"
                    record.error = "The analytics service stopped before this job finished."
                    record.updated_at = _now()
                    self._save(record)
                self._jobs[record.id] = record
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                continue


def _cancel_grace_from_environ() -> float:
    try:
        value = float(
            os.environ.get("VIDEO_ANALYTICS_CANCEL_GRACE_SECONDS", DEFAULT_CANCEL_GRACE_SECONDS)
        )
    except (TypeError, ValueError):
        return DEFAULT_CANCEL_GRACE_SECONDS
    return max(0.0, value)


def select_expired_jobs(
    records: Iterable[JobRecord],
    *,
    retention_days: float,
    now: datetime | None = None,
) -> list[JobRecord]:
    """Return finished jobs whose last update is older than ``retention_days``.

    Retention is opt-in: zero or a negative value selects nothing. Jobs that are
    still queued, running, or cancelling are never selected, whatever their age.
    """

    if retention_days <= 0:
        return []
    cutoff = (now or datetime.now(timezone.utc)) - timedelta(days=retention_days)
    expired: list[JobRecord] = []
    for record in records:
        if record.status not in TERMINAL_STATUSES:
            continue
        try:
            updated = datetime.fromisoformat(record.updated_at)
        except (TypeError, ValueError):
            continue
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=timezone.utc)
        if updated < cutoff:
            expired.append(record)
    return expired


def _is_purgeable_directory(job_dir: Path, root: Path) -> bool:
    """Only direct children of the jobs root, and never the settings directory."""

    return (
        job_dir.parent == root
        and job_dir.name != SETTINGS_DIRECTORY_NAME
        and not job_dir.name.startswith(".")
    )


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _make_browser_video(source: Path, destination: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False
    result = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(destination),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and destination.is_file()


def _last_json_object(output: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, character in enumerate(output):
        if character != "{":
            continue
        try:
            value, end = decoder.raw_decode(output[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and not output[index + end :].strip():
            return value
    return None
