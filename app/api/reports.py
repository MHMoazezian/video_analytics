"""Exportable job reports built from the files a job leaves in its directory.

Everything here is pure file processing: no FastAPI objects and no job manager,
so the builders can be tested on a temporary directory.
"""

from __future__ import annotations

from collections import Counter
import csv
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
from typing import Any, BinaryIO, Iterator, Mapping
import zipfile


METRICS_BASE_COLUMNS = ("timestamp", "frame_index", "elapsed_seconds")
PRIVATE_JOB_FIELDS = ("job_directory", "input_video", "camera_config")
# Already-compressed media gains nothing from deflate and costs a lot of CPU.
STORED_SUFFIXES = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".jpg", ".jpeg", ".png", ".zip"})


def _is_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield the JSON objects of a JSONL file, skipping damaged lines."""

    try:
        stream = path.open("r", encoding="utf-8", errors="replace")
    except OSError:
        return
    with stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if isinstance(value, dict):
                yield value


def _metric_values(row: Mapping[str, Any]) -> Mapping[str, Any]:
    """Metrics live under ``metrics`` in reporter payloads; accept flat rows too."""

    metrics = row.get("metrics")
    if isinstance(metrics, dict):
        return metrics
    return {key: value for key, value in row.items() if key not in METRICS_BASE_COLUMNS}


def _record_mapping(record: object) -> dict[str, Any]:
    if is_dataclass(record) and not isinstance(record, type):
        return asdict(record)
    if isinstance(record, Mapping):
        return dict(record)
    raise TypeError("record must be a JobRecord or a mapping")


def _parse_time(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def metric_statistics(job_dir: Path) -> dict[str, dict[str, float | int]]:
    """min/max/mean/last/samples for every numeric top-level metric key."""

    totals: dict[str, dict[str, float | int]] = {}
    for row in _iter_jsonl(job_dir / "metrics.jsonl"):
        for key, value in _metric_values(row).items():
            if not _is_number(value):
                continue
            item = totals.get(key)
            if item is None:
                totals[key] = {"min": value, "max": value, "sum": float(value), "last": value, "samples": 1}
                continue
            item["min"] = min(item["min"], value)
            item["max"] = max(item["max"], value)
            item["sum"] += float(value)
            item["last"] = value
            item["samples"] += 1
    return {
        key: {
            "min": item["min"],
            "max": item["max"],
            "mean": round(float(item["sum"]) / int(item["samples"]), 4),
            "last": item["last"],
            "samples": item["samples"],
        }
        for key, item in totals.items()
    }


def _count_field(path: Path, field: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in _iter_jsonl(path):
        value = row.get(field)
        if isinstance(value, str) and value:
            counts[value] += 1
    return dict(sorted(counts.items()))


def _duration_seconds(record: Mapping[str, Any], job_dir: Path) -> float | None:
    """Processing time reported by the pipeline, else the record's wall clock."""

    elapsed: float | None = None
    for row in _iter_jsonl(job_dir / "events.jsonl"):
        value = row.get("elapsed_seconds")
        if _is_number(value) and value > 0:
            elapsed = float(value)
    if elapsed is not None:
        return round(elapsed, 3)
    created = _parse_time(record.get("created_at"))
    updated = _parse_time(record.get("updated_at"))
    if created is None or updated is None:
        return None
    return round(max(0.0, (updated - created).total_seconds()), 3)


def build_job_report(
    record: object,
    job_dir: Path,
    *,
    public_job: Mapping[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Summarise one job. ``public_job`` is the API's ``JobPublic`` view of it."""

    values = _record_mapping(record)
    job = dict(public_job) if public_job is not None else {
        key: value for key, value in values.items() if key not in PRIVATE_JOB_FIELDS
    }
    for private in PRIVATE_JOB_FIELDS:
        job.pop(private, None)
    return {
        "job": job,
        "configuration": _read_json_object(job_dir / "configuration.json"),
        "final_metrics": _read_json_object(job_dir / "final_metrics.json"),
        "metric_statistics": metric_statistics(job_dir),
        "event_counts": _count_field(job_dir / "events.jsonl", "type"),
        "analytics_event_counts": _count_field(job_dir / "analytics_events.jsonl", "event_type"),
        "duration_seconds": _duration_seconds(values, job_dir),
        "generated_at": (now or datetime.now(timezone.utc)).isoformat(),
    }


def metrics_csv(job_dir: Path) -> str:
    """Flatten ``metrics.jsonl`` into one CSV row per line.

    Columns are ``timestamp, frame_index, elapsed_seconds`` followed by the union
    of numeric top-level metric keys in first-seen order; a cell is blank when the
    row has no numeric value for that key.
    """

    path = job_dir / "metrics.jsonl"
    columns: list[str] = []
    seen: set[str] = set()
    for row in _iter_jsonl(path):
        for key, value in _metric_values(row).items():
            if key not in seen and _is_number(value):
                seen.add(key)
                columns.append(key)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow([*METRICS_BASE_COLUMNS, *columns])
    for row in _iter_jsonl(path):
        metrics = _metric_values(row)
        base = [row.get(name) if row.get(name) is not None else "" for name in METRICS_BASE_COLUMNS]
        writer.writerow(
            [*base, *(metrics.get(key) if _is_number(metrics.get(key)) else "" for key in columns)]
        )
    return buffer.getvalue()


def exportable_artifacts(
    artifacts: Mapping[str, str], job_dir: Path, *, input_video: str | None = None
) -> list[tuple[str, Path]]:
    """Resolve artifact files to ``(archive name, path)``, confined to the job.

    Paths outside the job directory, missing files, duplicates and the uploaded
    input video are skipped.
    """

    root = job_dir.resolve()
    excluded = {path.resolve() for path in root.glob("input.*")}
    if input_video:
        try:
            excluded.add(Path(input_video).resolve())
        except (OSError, RuntimeError, ValueError):
            pass
    selected: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for _key, raw_path in sorted(artifacts.items()):
        try:
            path = Path(raw_path).resolve()
        except (OSError, RuntimeError, TypeError, ValueError):
            continue
        if path in seen or path in excluded:
            continue
        if not path.is_relative_to(root) or not path.is_file():
            continue
        seen.add(path)
        selected.append((f"artifacts/{path.relative_to(root).as_posix()}", path))
    return selected


def write_job_export_zip(
    destination: Path | BinaryIO,
    report: Mapping[str, Any],
    job_dir: Path,
    artifacts: Mapping[str, str],
    *,
    input_video: str | None = None,
) -> list[str]:
    """Write ``report.json``, ``metrics.csv`` and the job artifacts; return names."""

    names: list[str] = []
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        archive.writestr("report.json", json.dumps(report, ensure_ascii=False, indent=2, default=str))
        names.append("report.json")
        archive.writestr("metrics.csv", metrics_csv(job_dir))
        names.append("metrics.csv")
        for name, path in exportable_artifacts(artifacts, job_dir, input_video=input_video):
            compression = (
                zipfile.ZIP_STORED if path.suffix.lower() in STORED_SUFFIXES else zipfile.ZIP_DEFLATED
            )
            try:
                archive.write(path, arcname=name, compress_type=compression)
            except OSError:
                continue
            names.append(name)
    return names
