"""FastAPI service used by the Tarebar dashboard."""

from __future__ import annotations

import os
import json
import logging
from contextlib import contextmanager
from pathlib import Path
import re
import shutil
import asyncio
import tempfile
import time
from typing import Annotated, Iterator, Literal
from uuid import uuid4

import cv2
from fastapi import FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.background import BackgroundTask

from app.api.auth import ServiceTokenMiddleware
from app.api.frames import (
    FrameCaptureError,
    encode_jpeg_data_url,
    grab_stream_frame,
    iter_mjpeg_preview,
    require_rtsp_url,
)
from app.api.jobs import JobManager
from app.api.presets import APPLICATIONS, get_application
from app.api.reports import build_job_report, write_job_export_zip
from app.core.config import DEFAULT_CAMERA_CONFIG_PATH, PROJECT_ROOT
from app.geometry.config import load_camera_config
from app.fleet.supervisor import fleet_supervisor
from app.management.api import rollup_worker, router as management_router
from app.management.repository import analytics_repository
from app.insights import VideoInsightError, VideoInsightService
from app.tracking.factory import available_tracker_types, public_tracker_catalog


logger = logging.getLogger(__name__)

JOBS_ROOT = Path(
    os.environ.get("VIDEO_ANALYTICS_JOBS_DIR", PROJECT_ROOT / "output" / "dashboard")
)
SAVED_CAMERA_CONFIG_PATH = Path(
    os.environ.get(
        "VIDEO_ANALYTICS_CAMERA_CONFIG_PATH",
        JOBS_ROOT / "_settings" / "camera.yaml",
    )
)
MAX_UPLOAD_BYTES = int(os.environ.get("VIDEO_ANALYTICS_MAX_UPLOAD_BYTES", 1_073_741_824))
ALLOWED_VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"})
ALLOWED_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})
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
# Starlette inserts every added middleware at the front of its list and wraps the
# router from the back, so the middleware added LAST is the OUTERMOST one. The
# token check is therefore registered first and CORS last: CORS answers preflight
# requests itself and decorates the 401 responses produced by the token check.
app.add_middleware(ServiceTokenMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.include_router(management_router)

RETENTION_SWEEP_SECONDS = 6 * 60 * 60
EXPORT_SCRATCH_PREFIX = ".export-"


def _retention_days() -> float:
    """Opt-in job retention; anything but a positive number keeps every job."""

    try:
        return max(0.0, float(os.environ.get("VIDEO_ANALYTICS_RETENTION_DAYS", "0") or 0))
    except ValueError:
        return 0.0


async def retention_worker(days: float) -> None:
    """Purge expired finished jobs at startup and then every six hours."""

    while True:
        try:
            removed = await asyncio.to_thread(manager.purge_expired, days)
            if removed:
                logger.info("retention removed %d job(s) older than %s day(s)", len(removed), days)
        except Exception:  # noqa: BLE001 - housekeeping must never stop the service
            logger.exception("job retention sweep failed")
        await asyncio.sleep(RETENTION_SWEEP_SECONDS)


@app.on_event("startup")
async def start_analytics_store() -> None:
    analytics_repository.open()
    app.state.analytics_rollup_task = asyncio.create_task(rollup_worker())
    app.state.fleet_task = asyncio.create_task(asyncio.to_thread(fleet_supervisor.start))
    for leftover in JOBS_ROOT.glob(f"{EXPORT_SCRATCH_PREFIX}*.zip"):
        leftover.unlink(missing_ok=True)
    retention_days = _retention_days()
    if retention_days > 0:
        app.state.retention_task = asyncio.create_task(retention_worker(retention_days))
    if os.environ.get("VIDEO_INSIGHT_PRELOAD", "false").strip().lower() in {
        "1", "true", "yes", "on"
    }:
        await asyncio.to_thread(video_insight_service.preload)


@app.on_event("shutdown")
async def stop_analytics_store() -> None:
    fleet_supervisor.stop()
    for name in ("analytics_rollup_task", "retention_task"):
        task = getattr(app.state, name, None)
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


class StreamSamplingRequest(BaseModel):
    """One live sampling window shared by the Qwen endpoints."""

    model_config = ConfigDict(extra="forbid")

    stream_url: str = Field(min_length=1, max_length=2048)
    num_frames: int = Field(default=8, ge=2, le=16)
    interval_seconds: float = Field(default=10.0, ge=10.0, le=60.0)

    @field_validator("stream_url")
    @classmethod
    def validate_stream_url(cls, value: str) -> str:
        return require_rtsp_url(value)


class StreamVideoInsightRequest(StreamSamplingRequest):
    """Evaluate the two fixed questions over one live sampling window."""

    include_thumbnail: bool = False


class StreamFruitQualityRequest(StreamSamplingRequest):
    """Evaluate fruit freshness over one live sampling window."""

    detail_level: Literal["summary", "detailed"] = "summary"
    per_frame: bool = False
    include_thumbnails: bool = False


INSIGHT_UNAVAILABLE_DETAIL = "the analysis could not be completed; please try again"


@contextmanager
def _insight_errors() -> Iterator[None]:
    """Map every insight failure to 503; only known messages reach the client."""

    try:
        yield
    except HTTPException:
        raise
    except VideoInsightError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - e.g. CUDA or decoder errors must not be a 500
        logger.exception("video insight request failed")
        raise HTTPException(status_code=503, detail=INSIGHT_UNAVAILABLE_DETAIL) from exc


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
        # Which model answers the insight requests and at what precision and
        # image resolution; cheap, it never loads the model.
        "insights": video_insight_service.describe(),
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
    num_frames: Annotated[int, Form(ge=2, le=16)] = 8,
    window_start_seconds: Annotated[float, Form(ge=0)] = 0,
    window_end_seconds: Annotated[float | None, Form(gt=0)] = None,
    include_thumbnail: Annotated[bool, Form()] = False,
) -> dict[str, object]:
    """Evaluate the two fixed questions over a recorded-video time window."""

    video_suffix = Path(video.filename or "").suffix.lower()
    if video_suffix not in ALLOWED_VIDEO_SUFFIXES:
        raise HTTPException(status_code=422, detail="unsupported recorded-video file type")
    if window_end_seconds is not None and window_end_seconds <= window_start_seconds:
        raise HTTPException(status_code=422, detail="window end must be after window start")
    JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        with _insight_errors(), tempfile.TemporaryDirectory(dir=JOBS_ROOT) as scratch_dir:
            video_path = Path(scratch_dir) / f"input{video_suffix}"
            await _save_upload(video, video_path, limit=MAX_UPLOAD_BYTES)
            from app.insights.service import extract_video_frames

            frames = await asyncio.to_thread(
                extract_video_frames,
                video_path,
                num_frames=num_frames,
                window_start_seconds=window_start_seconds,
                window_end_seconds=window_end_seconds,
            )
            result = await asyncio.to_thread(
                video_insight_service.interpret_video,
                frames,
                include_thumbnail=include_thumbnail,
            )
    finally:
        await video.close()
    return {"data": result}


@app.post("/api/v1/video-insights/from-stream")
async def interpret_live_video(request: StreamVideoInsightRequest) -> dict[str, object]:
    """Evaluate the two fixed questions over one live-camera time window."""

    from app.insights.service import extract_stream_frames

    with _insight_errors():
        frames = await asyncio.to_thread(
            extract_stream_frames,
            request.stream_url,
            num_frames=request.num_frames,
            sample_interval_seconds=request.interval_seconds / (request.num_frames - 1),
        )
        result = await asyncio.to_thread(
            video_insight_service.interpret_video,
            frames,
            include_thumbnail=request.include_thumbnail,
        )
    return {"data": result}


@app.post("/api/v1/fruit-quality")
async def interpret_fruit_quality(
    media: Annotated[UploadFile, File(...)],
    num_frames: Annotated[int, Form(ge=1, le=16)] = 8,
    detail_level: Annotated[Literal["summary", "detailed"], Form()] = "summary",
    per_frame: Annotated[bool, Form()] = False,
    include_thumbnails: Annotated[bool, Form()] = False,
) -> dict[str, object]:
    """Score visible fruit freshness in one image or a sampled video."""

    started = time.perf_counter()
    suffix = Path(media.filename or "").suffix.lower()
    if suffix not in ALLOWED_IMAGE_SUFFIXES | ALLOWED_VIDEO_SUFFIXES:
        raise HTTPException(status_code=422, detail="unsupported fruit image or video type")
    JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        with _insight_errors(), tempfile.TemporaryDirectory(dir=JOBS_ROOT) as scratch_dir:
            media_path = Path(scratch_dir) / f"input{suffix}"
            await _save_upload(media, media_path, limit=MAX_UPLOAD_BYTES)
            if suffix in ALLOWED_IMAGE_SUFFIXES:
                from app.insights.service import sample_image_frame

                samples = await asyncio.to_thread(sample_image_frame, media_path)
            else:
                from app.insights.service import sample_video_frames

                samples = await asyncio.to_thread(
                    sample_video_frames, media_path, num_frames=num_frames
                )
            result = await asyncio.to_thread(
                video_insight_service.interpret_fruit_quality,
                [item.image for item in samples],
                detail_level=detail_level,
                per_frame=per_frame,
                include_thumbnails=include_thumbnails,
                timestamps=[item.timestamp_seconds for item in samples],
            )
    finally:
        await media.close()
    # Report the whole request, including the upload and frame sampling.
    result["total_seconds"] = round(time.perf_counter() - started, 3)
    return {"data": result}


@app.post("/api/v1/fruit-quality/from-stream")
async def interpret_live_fruit_quality(
    request: StreamFruitQualityRequest,
) -> dict[str, object]:
    """Score visible fruit freshness over one live-camera window."""

    from app.insights.service import sample_stream_frames

    started = time.perf_counter()
    with _insight_errors():
        samples = await asyncio.to_thread(
            sample_stream_frames,
            request.stream_url,
            num_frames=request.num_frames,
            sample_interval_seconds=request.interval_seconds / (request.num_frames - 1),
        )
        result = await asyncio.to_thread(
            video_insight_service.interpret_fruit_quality,
            [item.image for item in samples],
            detail_level=request.detail_level,
            per_frame=request.per_frame,
            include_thumbnails=request.include_thumbnails,
            timestamps=[item.timestamp_seconds for item in samples],
        )
    # Report the whole request, including the live sampling window.
    result["total_seconds"] = round(time.perf_counter() - started, 3)
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
        elif _active_camera_config_path().exists():
            # No upload: reuse the saved default, else the bundled example.
            config_path = job_directory / "camera.yaml"
            shutil.copyfile(_active_camera_config_path(), config_path)
        if config_path is not None:
            try:
                load_camera_config(config_path)
            except (OSError, ValueError) as exc:
                raise HTTPException(status_code=422, detail=f"invalid camera YAML: {exc}") from exc
            # Only a real upload may replace the saved default; persisting the
            # fallback copy would overwrite it with the bundled example.
            if persist_camera_config and camera_config_upload is not None:
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


def _job_report(job_id: str) -> tuple[dict[str, object], Path, dict[str, str], str | None]:
    try:
        record = manager.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    job_dir = Path(record.job_directory)
    report = build_job_report(record, job_dir, public_job=manager.public_dict(record))
    input_video = record.input_video if record.source_type == "file" else None
    return report, job_dir, dict(record.artifacts), input_video


@app.get("/api/v1/jobs/{job_id}/report")
def job_report(job_id: str) -> dict[str, object]:
    """Configuration, final metrics, metric statistics and event counts of one job."""

    report, _job_dir, _artifacts, _input_video = _job_report(job_id)
    return {"data": report}


@app.get("/api/v1/jobs/{job_id}/export.zip")
def job_export(job_id: str) -> FileResponse:
    """ZIP with report.json, metrics.csv and the job artifacts (never the upload)."""

    report, job_dir, artifacts, input_video = _job_report(job_id)
    JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    archive = JOBS_ROOT / f"{EXPORT_SCRATCH_PREFIX}{uuid4().hex}.zip"
    try:
        write_job_export_zip(archive, report, job_dir, artifacts, input_video=input_video)
    except OSError as exc:
        archive.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"could not build the export: {exc}") from exc
    return FileResponse(
        archive,
        media_type="application/zip",
        filename=f"job-{job_id}.zip",
        background=BackgroundTask(archive.unlink, missing_ok=True),
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
