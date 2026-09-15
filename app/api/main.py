"""FastAPI service used by the Tarebar dashboard."""

from __future__ import annotations

import os
import json
from pathlib import Path
import re
import shutil
import asyncio
import tempfile
import time
from typing import Annotated, Iterator
from uuid import uuid4

import cv2
from fastapi import FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.api.frames import (
    FrameCaptureError,
    encode_jpeg_data_url,
    grab_stream_frame,
    iter_mjpeg_preview,
    require_rtsp_url,
)
from app.api.jobs import JobManager
from app.api.presets import APPLICATIONS, get_application
from app.core.config import DEFAULT_CAMERA_CONFIG_PATH, PROJECT_ROOT
from app.geometry.config import load_camera_config
from app.fleet.supervisor import fleet_supervisor
from app.management.api import rollup_worker, router as management_router
from app.management.repository import analytics_repository
from app.insights import VideoInsightError, VideoInsightService
from app.tracking.factory import available_tracker_types, public_tracker_catalog


JOBS_ROOT = Path(
    os.environ.get("VIDEO_ANALYTICS_JOBS_DIR", PROJECT_ROOT / "output" / "dashboard")
)
DEFAULT_CAMERA_CONFIG_PATH = PROJECT_ROOT / "configs" / "cameras" / "example_lobby.yaml"
SAVED_CAMERA_CONFIG_PATH = Path(
    os.environ.get(
        "VIDEO_ANALYTICS_CAMERA_CONFIG_PATH",
        JOBS_ROOT / "_settings" / "camera.yaml",
    )
)
MAX_UPLOAD_BYTES = int(os.environ.get("VIDEO_ANALYTICS_MAX_UPLOAD_BYTES", 1_073_741_824))
ALLOWED_VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"})
manager = JobManager(
    JOBS_ROOT,
    max_workers=int(os.environ.get("VIDEO_ANALYTICS_JOB_WORKERS", "1")),
    processing_width=int(os.environ.get("VIDEO_ANALYTICS_PROCESSING_WIDTH", "1280")),
    frame_stride=int(os.environ.get("VIDEO_ANALYTICS_FRAME_STRIDE", "10")),
)
video_insight_service = VideoInsightService()

app = FastAPI(title="Video Analytics MVP API", version="0.1.0")
origins = [
    item.strip()
    for item in os.environ.get(
        "VIDEO_ANALYTICS_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",")
    if item.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.include_router(management_router)


@app.on_event("startup")
async def start_analytics_store() -> None:
    analytics_repository.open()
    app.state.analytics_rollup_task = asyncio.create_task(rollup_worker())
    app.state.fleet_task = asyncio.create_task(asyncio.to_thread(fleet_supervisor.start))
    if os.environ.get("VIDEO_INSIGHT_PRELOAD", "false").strip().lower() in {
        "1", "true", "yes", "on"
    }:
        await asyncio.to_thread(video_insight_service.preload)


@app.on_event("shutdown")
async def stop_analytics_store() -> None:
    fleet_supervisor.stop()
    task = getattr(app.state, "analytics_rollup_task", None)
    if task is not None:
        task.cancel()
    analytics_repository.close()


class StreamJobRequest(BaseModel):
    """Start the existing pipeline with a normalized RTSP camera source."""

    model_config = ConfigDict(extra="forbid")

    stream_url: str = Field(min_length=1, max_length=2048)
    application_id: str | None = None
    application_ids: list[str] | None = Field(default=None, min_length=1, max_length=4)
    camera_config_yaml: str | None = Field(default=None, max_length=2_000_000)
    camera_id: str = "live-camera"
    max_frames: int | None = Field(default=None, gt=0)
    enable_reid: bool = False
    tracker_type: str = "bytetrack"

    @field_validator("stream_url")
    @classmethod
    def validate_stream_url(cls, value: str) -> str:
        return require_rtsp_url(value)

    @field_validator("camera_id")
    @classmethod
    def validate_camera_id(cls, value: str) -> str:
        candidate = value.strip()
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", candidate):
            raise ValueError(
                "camera_id must use 1-100 letters, digits, dots, underscores, or hyphens"
            )
        return candidate


    @field_validator("tracker_type")
    @classmethod
    def validate_tracker_type(cls, value: str) -> str:
        selected = value.strip().lower()
        allowed = available_tracker_types()
        if selected not in allowed:
            raise ValueError(f"tracker_type must be one of: {', '.join(allowed)}")
        return selected


class StreamFrameRequest(BaseModel):
    """Capture a single still from the same RTSP source used by live jobs."""

    model_config = ConfigDict(extra="forbid")

    stream_url: str = Field(min_length=1, max_length=2048)

    @field_validator("stream_url")
    @classmethod
    def validate_stream_url(cls, value: str) -> str:
        return require_rtsp_url(value)


class StreamVideoInsightRequest(BaseModel):
    """Interpret a short sequence sampled from a live RTSP source."""

    model_config = ConfigDict(extra="forbid")

    stream_url: str = Field(min_length=1, max_length=2048)
    query: str = Field(min_length=1, max_length=4000)
    num_frames: int = Field(default=8, ge=2, le=16)
    max_new_tokens: int = Field(default=80, ge=16, le=512)
    detailed: bool = False

    @field_validator("stream_url")
    @classmethod
    def validate_stream_url(cls, value: str) -> str:
        return require_rtsp_url(value)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        candidate = value.strip()
        if not candidate:
            raise ValueError("query must not be blank")
        return candidate


@app.get("/health")
def health() -> dict[str, object]:
    analytical_store = analytics_repository.health()
    fleet = fleet_supervisor.status()
    return {
        "status": "ok" if analytical_store else "degraded",
        "analyticsStore": analytical_store,
        "fleet": {
            "enabled": fleet["enabled"],
            "fps": fleet["fps"],
            "cameras": fleet["cameras"],
            "running": fleet["running"],
        },
    }


@app.get("/api/v1/fleet/status")
def fleet_status() -> dict[str, object]:
    return {"data": fleet_supervisor.status()}


@app.get("/api/v1/applications")
def applications() -> dict[str, list[dict[str, object]]]:
    return {"data": [item.public_dict() for item in APPLICATIONS]}


@app.get("/api/v1/trackers")
def trackers() -> dict[str, list[dict[str, object]]]:
    return {"data": public_tracker_catalog()}


@app.get("/api/v1/jobs")
def list_jobs() -> dict[str, list[dict[str, object]]]:
    return {"data": [manager.public_dict(item) for item in manager.list()]}


@app.get("/api/v1/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, object]:
    try:
        return {"data": manager.public_dict(manager.get(job_id))}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/v1/jobs/{job_id}/cancel", status_code=202)
def cancel_job(job_id: str) -> dict[str, object]:
    try:
        return {"data": manager.public_dict(manager.cancel(job_id))}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/v1/jobs/{job_id}/events")
def stream_job_events(
    job_id: str,
    last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
    after: Annotated[int | None, Query(ge=0)] = None,
) -> StreamingResponse:
    try:
        manager.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    cursor = after or 0
    if last_event_id and last_event_id.isdigit():
        cursor = max(cursor, int(last_event_id))
    return StreamingResponse(
        _event_stream(job_id, cursor),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/v1/jobs/{job_id}/preview")
def job_preview(job_id: str) -> FileResponse:
    try:
        path = manager.job_file(job_id, "preview.jpg")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/v1/jobs/{job_id}/preview-stream")
def job_preview_stream(job_id: str) -> StreamingResponse:
    try:
        manager.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return StreamingResponse(
        _preview_stream(job_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@app.post("/api/v1/frames/first")
async def extract_first_frame(video: Annotated[UploadFile, File(...)]) -> dict[str, object]:
    """Decode the first frame of an uploaded video with OpenCV.

    Used as a fallback when the browser's <video> element cannot preview a
    file it will nonetheless accept for processing (e.g. some codecs inside
    an otherwise-supported container).
    """

    video_suffix = Path(video.filename or "").suffix.lower()
    if video_suffix not in ALLOWED_VIDEO_SUFFIXES:
        raise HTTPException(status_code=422, detail="unsupported video file type")

    JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(dir=JOBS_ROOT) as scratch_dir:
            scratch_path = Path(scratch_dir) / f"source{video_suffix}"
            try:
                await _save_upload(video, scratch_path, limit=MAX_UPLOAD_BYTES)
            finally:
                await video.close()

            capture = cv2.VideoCapture(str(scratch_path))
            try:
                if not capture.isOpened():
                    raise HTTPException(status_code=422, detail="could not open video file")
                ok, frame = capture.read()
            finally:
                capture.release()
    except HTTPException:
        raise
    except OSError as exc:
        raise HTTPException(status_code=422, detail=f"could not read video file: {exc}") from exc

    if not ok or frame is None:
        raise HTTPException(status_code=422, detail="could not read the first frame of this video")

    try:
        return {"data": encode_jpeg_data_url(frame)}
    except FrameCaptureError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@app.post("/api/v1/frames/from-stream")
def extract_stream_frame(request: StreamFrameRequest) -> dict[str, object]:
    """Grab one still from a live RTSP camera for region drawing.

    Live analytics already opens this same URL with OpenCV. The zone editor
    only needs a single reference frame, not a browser WebRTC/HLS session.
    """

    try:
        frame = grab_stream_frame(request.stream_url)
        return {"data": encode_jpeg_data_url(frame)}
    except FrameCaptureError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@app.post("/api/v1/preview-stream")
def live_camera_preview(request: StreamFrameRequest) -> StreamingResponse:
    """Stream an unannotated MJPEG view of the camera used by live analytics."""

    try:
        frames = iter_mjpeg_preview(request.stream_url)
        first = next(frames)
    except StopIteration as exc:
        raise HTTPException(
            status_code=503,
            detail="could not read a frame from the camera",
        ) from exc
    except FrameCaptureError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    def body() -> Iterator[bytes]:
        yield first
        yield from frames

    return StreamingResponse(
        body(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@app.post("/api/v1/video-insights")
async def interpret_recorded_video(
    video: Annotated[UploadFile, File(...)],
    query: Annotated[str, Form(min_length=1, max_length=4000)],
    num_frames: Annotated[int, Form(ge=2, le=16)] = 8,
    max_new_tokens: Annotated[int, Form(ge=16, le=512)] = 80,
    detailed: Annotated[bool, Form()] = False,
) -> dict[str, object]:
    """Answer a query about uniformly sampled frames from an uploaded video."""

    video_suffix = Path(video.filename or "").suffix.lower()
    if video_suffix not in ALLOWED_VIDEO_SUFFIXES:
        raise HTTPException(status_code=422, detail="unsupported recorded-video file type")
    query = query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="query must not be blank")
    JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(dir=JOBS_ROOT) as scratch_dir:
            video_path = Path(scratch_dir) / f"input{video_suffix}"
            await _save_upload(video, video_path, limit=MAX_UPLOAD_BYTES)
            from app.insights.service import extract_video_frames

            frames = await asyncio.to_thread(
                extract_video_frames, video_path, num_frames=num_frames
            )
            result = await asyncio.to_thread(
                video_insight_service.interpret,
                frames,
                query,
                max_new_tokens=max_new_tokens,
                detailed=detailed,
            )
    except VideoInsightError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        await video.close()
    return {"data": result}


@app.post("/api/v1/video-insights/from-stream")
async def interpret_live_video(request: StreamVideoInsightRequest) -> dict[str, object]:
    """Answer a query about consecutive frames from a live camera."""

    from app.insights.service import extract_stream_frames

    try:
        frames = await asyncio.to_thread(
            extract_stream_frames,
            request.stream_url,
            num_frames=request.num_frames,
            sample_interval_seconds=float(
                os.environ.get("VIDEO_INSIGHT_LIVE_SAMPLE_INTERVAL_SECONDS", "0.75")
            ),
        )
        result = await asyncio.to_thread(
            video_insight_service.interpret,
            frames,
            request.query,
            max_new_tokens=request.max_new_tokens,
            detailed=request.detailed,
        )
    except VideoInsightError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"data": result}


@app.post("/api/v1/jobs", status_code=202)
async def create_job(
    video: Annotated[UploadFile, File(...)],
    application_id: Annotated[str, Form(...)],
    camera_id: Annotated[str, Form()] = "uploaded-video",
    max_frames: Annotated[int | None, Form()] = None,
    enable_reid: Annotated[bool, Form()] = False,
    tracker_type: Annotated[str, Form()] = "bytetrack",
    camera_config: Annotated[UploadFile | None, File()] = None,
    persist_camera_config: Annotated[bool, Form()] = True,
) -> dict[str, object]:
    camera_config_upload = (
        camera_config if _has_camera_config_upload(camera_config) else None
    )
    try:
        preset = get_application(application_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if max_frames is not None and max_frames <= 0:
        raise HTTPException(status_code=422, detail="max_frames must be positive")
    if enable_reid and preset.module == "app.detection.cli":
        raise HTTPException(status_code=422, detail="ReID is only available for tracking jobs")
    selected_tracker = tracker_type.strip().lower()
    if selected_tracker not in available_tracker_types():
        raise HTTPException(
            status_code=422,
            detail=f"tracker_type must be one of: {', '.join(available_tracker_types())}",
        )
    camera_id = camera_id.strip()
    if not camera_id or not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", camera_id):
        raise HTTPException(
            status_code=422,
            detail="camera_id must use 1-100 letters, digits, dots, underscores, or hyphens",
        )
    video_suffix = Path(video.filename or "").suffix.lower()
    if video_suffix not in ALLOWED_VIDEO_SUFFIXES:
        raise HTTPException(status_code=422, detail="unsupported recorded-video file type")
    # Browsers submit an empty file part for an untouched <input type="file">,
    # so an unset optional upload still arrives here with camera_config.filename == "".
    if camera_config is not None and not (camera_config.filename or "").strip():
        await camera_config.close()
        camera_config = None
    if (
        preset.requires_camera_config
        and camera_config is None
        and not DEFAULT_CAMERA_CONFIG_PATH.exists()
    ):
        raise HTTPException(
            status_code=422,
            detail=f"{preset.name} requires a camera YAML configuration",
        )

    job_id = uuid4().hex
    job_directory = manager.root / job_id
    job_directory.mkdir(parents=True)
    video_path = job_directory / f"input{video_suffix}"
    config_path: Path | None = None
    try:
        await _save_upload(video, video_path, limit=MAX_UPLOAD_BYTES)
        if camera_config_upload is not None:
            config_suffix = Path(camera_config_upload.filename or "").suffix.lower()
            if config_suffix not in {".yaml", ".yml"}:
                raise HTTPException(status_code=422, detail="camera configuration must be YAML")
            config_path = job_directory / "camera.yaml"
            await _save_upload(camera_config, config_path, limit=2_000_000)
        elif DEFAULT_CAMERA_CONFIG_PATH.exists():
            config_path = job_directory / "camera.yaml"
            shutil.copyfile(DEFAULT_CAMERA_CONFIG_PATH, config_path)
        if config_path is not None:
            try:
                load_camera_config(config_path)
            except (OSError, ValueError) as exc:
                raise HTTPException(status_code=422, detail=f"invalid camera YAML: {exc}") from exc
            if persist_camera_config:
                _save_default_camera_config(config_path)
        elif preset.module == "app.analytics.cli":
            config_path = job_directory / "camera.yaml"
            _copy_default_camera_config(config_path)
        record = manager.register(
            job_id=job_id,
            application_id=application_id,
            original_filename=video.filename or video_path.name,
            camera_id=camera_id,
            input_video=video_path,
            camera_config=config_path,
            max_frames=max_frames,
            enable_reid=enable_reid,
            tracker_type=selected_tracker,
        )
    except Exception:
        shutil.rmtree(job_directory, ignore_errors=True)
        raise
    finally:
        await video.close()
        if camera_config is not None:
            await camera_config.close()

    manager.enqueue(job_id)
    return {"data": manager.public_dict(record)}


def _active_camera_config_path() -> Path:
    """Return the saved camera YAML, falling back to the bundled lobby example."""

    return (
        SAVED_CAMERA_CONFIG_PATH
        if SAVED_CAMERA_CONFIG_PATH.is_file()
        else DEFAULT_CAMERA_CONFIG_PATH
    )


def _has_camera_config_upload(upload: UploadFile | None) -> bool:
    """Ignore the empty multipart file part emitted by an unselected picker."""

    return bool(
        upload is not None
        and (upload.filename or "").strip()
        and upload.size != 0
    )


def _copy_default_camera_config(destination: Path) -> None:
    source = _active_camera_config_path()
    try:
        load_camera_config(source)
        shutil.copyfile(source, destination)
    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"default camera YAML is unavailable or invalid: {exc}",
        ) from exc


def _save_default_camera_config(source: Path) -> None:
    """Atomically persist a validated upload for subsequent recorded jobs."""

    SAVED_CAMERA_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = SAVED_CAMERA_CONFIG_PATH.with_name(f".{uuid4().hex}.tmp")
    try:
        shutil.copyfile(source, temporary)
        temporary.replace(SAVED_CAMERA_CONFIG_PATH)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail=f"could not save camera YAML as the default: {exc}",
        ) from exc


@app.post("/api/v1/stream-jobs", status_code=202)
def create_stream_job(request: StreamJobRequest) -> dict[str, object]:
    """Run a detector/tracker/analytics preset directly against RTSP.

    The transport contract is deliberately source-agnostic: Larix, an IP
    camera, or another publisher all appear here as an RTSP URL. MediaMTX owns
    protocol conversion and the CV pipeline remains unchanged.
    """

    if request.application_id is not None and request.application_ids is not None:
        raise HTTPException(
            status_code=422,
            detail="provide application_id or application_ids, not both",
        )

    enabled_tasks = request.application_ids
    if enabled_tasks is not None:
        allowed_live_tasks = {
            "people_counting",
            "heatmap",
            "vertical_queue",
            "configured_queue",
            "restricted_area",
        }
        if len(enabled_tasks) != len(set(enabled_tasks)):
            raise HTTPException(status_code=422, detail="application_ids must be unique")
        unsupported = sorted(set(enabled_tasks) - allowed_live_tasks)
        if unsupported:
            raise HTTPException(
                status_code=422,
                detail=f"unsupported combined live analytics: {', '.join(unsupported)}",
            )
        if "vertical_queue" in enabled_tasks and "configured_queue" in enabled_tasks:
            raise HTTPException(
                status_code=422,
                detail="vertical_queue and configured_queue cannot run together",
            )
        # These modules share detection and tracking, so a single analytics CLI
        # process can calculate all selected results without decoding the stream
        # or running the person detector more than once.
        application_id = "people_counting"
        preset = get_application(application_id)
    else:
        application_id = request.application_id or "people_counting"
        try:
            preset = get_application(application_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    needs_camera_config = preset.requires_camera_config or bool(
        enabled_tasks
        and ({"restricted_area", "configured_queue"} & set(enabled_tasks))
    )
    if needs_camera_config and not request.camera_config_yaml:
        raise HTTPException(
            status_code=422,
            detail=f"{preset.name} requires camera geometry",
        )
    if request.enable_reid and preset.module == "app.detection.cli":
        raise HTTPException(status_code=422, detail="ReID is only available for tracking jobs")

    job_id = uuid4().hex
    job_directory = manager.root / job_id
    job_directory.mkdir(parents=True)
    config_path: Path | None = None
    try:
        if request.camera_config_yaml:
            config_path = job_directory / "camera.yaml"
            config_path.write_text(request.camera_config_yaml, encoding="utf-8")
            try:
                load_camera_config(config_path)
            except (OSError, ValueError) as exc:
                raise HTTPException(status_code=422, detail=f"invalid camera YAML: {exc}") from exc
        record = manager.register(
            job_id=job_id,
            application_id=application_id,
            original_filename=f"live:{request.camera_id}",
            camera_id=request.camera_id,
            input_video=request.stream_url,
            camera_config=config_path,
            max_frames=request.max_frames,
            enable_reid=request.enable_reid,
            tracker_type=request.tracker_type,
            source_type="rtsp",
            enabled_tasks=enabled_tasks,
        )
    except Exception:
        shutil.rmtree(job_directory, ignore_errors=True)
        raise

    manager.enqueue(job_id)
    return {"data": manager.public_dict(record)}


@app.get("/api/v1/restricted-area-events")
def restricted_area_events(
    camera_id: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> dict[str, list[dict[str, object]]]:
    """Return recent entry/exit lifecycle events across live and recorded jobs."""

    rows: list[dict[str, object]] = []
    allowed = {"restricted_area_entered", "restricted_area_exited"}
    for record in manager.list():
        if camera_id and record.camera_id != camera_id:
            continue
        path = Path(record.job_directory) / "analytics_events.jsonl"
        if not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event_type") not in allowed:
                continue
            event["job_id"] = record.id
            rows.append(event)
    rows.sort(key=lambda item: float(item.get("timestamp", 0)), reverse=True)
    return {"data": rows[:limit]}


@app.get("/api/v1/jobs/{job_id}/artifacts/{artifact_key}")
def download_artifact(job_id: str, artifact_key: str) -> FileResponse:
    try:
        path = manager.artifact_path(job_id, artifact_key)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    media_type = None
    if path.suffix.lower() == ".mp4":
        media_type = "video/mp4"
    elif path.suffix.lower() == ".png":
        media_type = "image/png"
    return FileResponse(
        path,
        media_type=media_type,
        filename=path.name,
        content_disposition_type="inline" if media_type else "attachment",
    )


async def _save_upload(upload: UploadFile, destination: Path, *, limit: int) -> None:
    written = 0
    with destination.open("wb") as stream:
        while chunk := await upload.read(1024 * 1024):
            written += len(chunk)
            if written > limit:
                raise HTTPException(status_code=413, detail="uploaded file is too large")
            stream.write(chunk)
    if written == 0:
        raise HTTPException(status_code=422, detail="uploaded file is empty")


def _event_stream(job_id: str, cursor: int) -> Iterator[str]:
    """Tail persisted JSONL events; line numbers are stable resumable SSE IDs."""

    idle_started = time.monotonic()
    stream = None
    line_number = 0
    terminal_seen_at: float | None = None
    try:
        while True:
            try:
                record = manager.get(job_id)
            except KeyError:
                return
            path = Path(record.job_directory) / "events.jsonl"
            if stream is None and path.is_file():
                try:
                    stream = path.open("r", encoding="utf-8")
                    while line_number < cursor and stream.readline():
                        line_number += 1
                except OSError:
                    stream = None
            line = stream.readline() if stream is not None else ""
            if line:
                line_number += 1
                try:
                    event_type = str(json.loads(line).get("type", "message"))
                except (json.JSONDecodeError, AttributeError):
                    event_type = "message"
                yield f"id: {line_number}\nevent: {event_type}\ndata: {line.rstrip()}\n\n"
                idle_started = time.monotonic()
                continue
            if record.status in {"completed", "failed", "cancelled"}:
                terminal_seen_at = terminal_seen_at or time.monotonic()
                if time.monotonic() - terminal_seen_at >= 0.5:
                    return
            else:
                terminal_seen_at = None
            if time.monotonic() - idle_started >= 15.0:
                yield ": keep-alive\n\n"
                idle_started = time.monotonic()
            time.sleep(0.25)
    finally:
        if stream is not None:
            stream.close()


def _preview_stream(job_id: str) -> Iterator[bytes]:
    """Stream completed JPEGs over one connection, retaining the last browser frame."""

    last_modified = -1
    terminal_seen_at: float | None = None
    while True:
        try:
            record = manager.get(job_id)
        except KeyError:
            return
        preview = Path(record.job_directory) / "preview.jpg"
        try:
            modified = preview.stat().st_mtime_ns
            if modified != last_modified:
                image = preview.read_bytes()
                last_modified = modified
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(image)).encode("ascii")
                    + b"\r\n\r\n"
                    + image
                    + b"\r\n"
                )
        except OSError:
            pass
        if record.status in {"completed", "failed", "cancelled"}:
            terminal_seen_at = terminal_seen_at or time.monotonic()
            if time.monotonic() - terminal_seen_at >= 1.0:
                return
        else:
            terminal_seen_at = None
        time.sleep(0.05)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.api.main:app", host="0.0.0.0", port=8000, reload=False)
