# Live dashboard integration

Dashboard jobs are stored under `output/dashboard/<job_id>/`. The FastAPI job
runner passes `--live-dir` and `--job-id` to the existing CLIs; direct CLI and
batch use remains unchanged when those options are omitted.

## HTTP API

- `GET /api/v1/applications` returns application metadata and its
  `metric_schema`.
- `POST /api/v1/jobs` creates and queues a recorded-video job.
- `GET /api/v1/jobs` and `GET /api/v1/jobs/{job_id}` return history and the
  latest persisted live snapshot. These endpoints are the polling fallback.
- `GET /api/v1/jobs/{job_id}/events` is an SSE stream. Its `id` is the stable
  one-based line number in `events.jsonl`; browser `Last-Event-ID` reconnection
  and the optional `?after=<id>` cursor are supported.
- `GET /api/v1/jobs/{job_id}/preview` returns the latest sampled JPEG with
  `Cache-Control: no-store`.
- `GET /api/v1/jobs/{job_id}/preview-stream` keeps one MJPEG connection open;
  the browser retains the last decoded frame until the next complete JPEG is
  available.
- `POST /api/v1/jobs/{job_id}/cancel` requests cancellation. Processing stops
  at the next frame boundary and records `job_cancelled`. A process that has
  not stopped after `VIDEO_ANALYTICS_CANCEL_GRACE_SECONDS` (default 10) is
  terminated, and killed five seconds later.
- `GET /api/v1/jobs/{job_id}/report` and `GET /api/v1/jobs/{job_id}/export.zip`
  return the job report and a ZIP of the report, `metrics.csv` and artifacts.

The complete route list, the optional service token and the insight contracts
are documented in the README ("HTTP API reference").

## Event contract

Every SSE `data` value is a JSON object with:

```json
{
  "type": "metrics_updated",
  "job_id": "<id>",
  "timestamp": "2026-08-03T08:00:00+00:00",
  "status": "running",
  "frame_index": 149,
  "progress": 50.0,
  "elapsed_seconds": 12.4,
  "metrics": {"current_people": 7, "processing_fps": 12.1},
  "preview_reference": "/api/v1/jobs/<id>/preview",
  "message": null
}
```

Event types are `job_started`, `preview_updated`, `metrics_updated`,
`progress_updated`, `warning`, `job_completed`, `job_failed`, and
`job_cancelled`. Failures retain the original tracker/model exception and add
frame index, detection count, bbox format, and frame size when tracking fails.

## Metric schema

Each application publishes a schema containing `key`, `label`, `value_type`,
`unit`, `aggregation`, `display`, and `availability`. The dashboard renders
cards, counters, statuses, tables, and sampled in-memory chart windows from
this schema. Metric values themselves always come from the real processing
loop.

## Job files

Depending on the selected application, a job directory contains:

- `configuration.json`, `job.json`, uploaded input, and optional `camera.yaml`
- `events.jsonl`, `metrics.jsonl`, `live_state.json`, and `preview.jpg`
- `tracking.jsonl`, `counts.csv`, `analytics_events.jsonl`, and heatmap exports
  as applicable
- `final_metrics.json`, `process.log`, and final processed video when produced

Only one compressed preview is retained and replaced atomically. Metrics and
events are streamed to JSONL; source frames are never accumulated in memory.
The default update intervals are 0.5 seconds for metrics and 0.2 seconds for a
maximum-width 960px JPEG preview.

The dashboard job runner also caps processing frames at 1280 pixels wide by
default, preserving aspect ratio and never upscaling smaller inputs. Override
this with `VIDEO_ANALYTICS_PROCESSING_WIDTH`. This cap applies only to dashboard
jobs; direct batch CLI processing retains source resolution unless
`--processing-width` is explicitly supplied. Dashboard jobs process every fifth
source frame by default (`VIDEO_ANALYTICS_FRAME_STRIDE=5`). Source timestamps are
preserved and the output FPS is divided by the stride, so the saved video keeps
approximately the original duration. Direct batch CLIs default to stride 1 and
support an explicit `--frame-stride` override.

## Verification

```bash
cd CV_applications/video_analytics_mvp
.venv/bin/python -m pytest -q

cd ../Tarebar-Smart-Monitoring-Platform
npx eslint 'src/app/(dashboard)/analytics/_components/video-analytics-client.tsx'
npx tsc --noEmit
```

Start the backend with `uvicorn app.api.main:app --host 0.0.0.0 --port 8000`
from `video_analytics_mvp`, then start the Next.js dashboard with `npm run dev`.
