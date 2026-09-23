# Video Analytics MVP

This is the isolated application scaffold for a medium-complexity, video-based
people analytics MVP. Existing computer-vision experiments and model weights
remain outside this directory and are treated as read-only inputs.

Phase 9 adds timestamp-window movement speed in explicitly labelled pixels per
second and, only with valid metre-based ground calibration, metres per second.
It also adds queue progress toward each configured service point. A FastAPI
integration layer now exposes recorded-video upload jobs and their generated
artifacts to the Tarebar dashboard.

The event-driven video-insight API uses a 4-bit Qwen3-VL-2B flow derived from
`Action_recognition/test_vlm_qwen_optimized_video.py`: eight uniformly sampled
RGB frames, bounded processor resolution, and deterministic generation. Precision
and resolution follow the GPU (`VIDEO_INSIGHT_PRECISION=auto`): unquantized
bf16/fp16 weights and full-resolution frames when the card can hold them, 4-bit
NF4 weights and ~384x384 frames only on a small one; see "Model quality" below. `POST /api/v1/video-insights` accepts a
recorded upload and video-time window; `POST /api/v1/video-insights/from-stream`
samples a configurable live RTSP window. Both endpoints evaluate only whether
people are fighting and whether the floor is clean. Generated model text stays
inside the service and the public response contains normalized Yes/No answers.
The model is loaded lazily from
`VIDEO_INSIGHT_MODEL_PATH` and is deliberately not copied into the image.
Production enables `VIDEO_INSIGHT_PRELOAD`, avoiding model-loading latency on
the first dashboard query.

## Planned scope

The MVP will process recorded video, webcams, and basic RTSP sources through one
shared pipeline:

```text
video source
  -> ONNX person detector
  -> ByteTrack adapter
  -> shared timestamped tracks
  -> camera geometry
  -> analytics
  -> events, metrics, SQLite, annotated video, API, and dashboard
```

Planned analytics include occupancy and directional counts, restricted-area
intrusion, movement and dwell heatmaps, configured queue heuristics, and speed
in pixels/second or metres/second when valid calibration is available.

## Architecture

```text
app/
  core/       settings and shared data contracts
  detection/  ONNX detector adapters
  tracking/   multi-object tracker adapters
  geometry/   zones, lines, and calibration
  analytics/  counting, intrusion, heatmap, queue, and speed modules
  storage/    JSONL, CSV, SQLite, and latest-status output
  api/        FastAPI application
  dashboard/  lightweight dashboard
configs/      YAML application and camera configuration
scripts/      command-line helpers
tests/        unit and integration tests
outputs/      ignored generated artifacts
```

The detector is separated into:

- `app.core.models`: shared immutable `Detection` representation
- `app.detection.preprocessing`: letterbox and BGR-to-RGB tensor conversion
- `app.detection.onnx_detector`: model validation and ONNX Runtime inference
- `app.detection.postprocessing`: output parsing, coordinate restoration, and NMS
- `app.detection.visualization`: optional OpenCV annotation
- `app.detection.cli`: headless image and recorded-video runner

Preprocessing, inference, post-processing, and visualization timing/behavior do
not overlap.

Tracking is separated into:

- `app.tracking.base`: `BaseTracker` ABC, normalized `track_id`/`bbox`/`confidence`/`class_id` outputs
- `app.tracking.factory`: configuration-based construction (`tracker.type` in YAML)
- `app.tracking.bytetrack`: ByteTrack baseline adapter (unchanged production behavior)
- `app.tracking.stabletrack_adapter`: StableTrack adapter for 0.5 FPS / 2 s gaps
- `app.tracking.deepocsort_adapter`: Deep OC-SORT adapter (OCM + Adaptive Weighting)
- `app.tracking.botsort_adapter`: BoT-SORT adapter (GMC + width-height Kalman + optional ReID)
- `app.tracking.ucmctrack_adapter`: UCMCTrack adapter (mapped Mahalanobis association, optional camera geometry)
- `app.tracking.calibration`: per-camera UCMCTrack geometry catalog (uncalibrated by default)
- `app.tracking.third_party.stabletrack`: isolated paper implementation (no official repo)
- `app.tracking.third_party.deepocsort`: isolated Deep OC-SORT backend (MIT, official algorithm)
- `app.tracking.third_party.botsort`: isolated BoT-SORT backend (MIT, official algorithm)
- `app.tracking.third_party.ucmctrack`: isolated UCMCTrack backend (MIT, official algorithm)
- `app.tracking.benchmark`: cached-detection HOTA/IDF1/MOTA runner
- `app.tracking.visualization`: track IDs, state, foot points, and trajectories
- `app.tracking.cli`: recorded-video detector/tracker runner using source times

Select a tracker with `tracker.type: bytetrack|stabletrack|deepocsort|botsort|ucmctrack`, `--tracker`, or the
temporary dashboard selector. `GET /api/v1/trackers` lists registered types so
future adapters do not require a dashboard rebuild.

`TrackObservation` and `TrajectoryPoint` live in `app.core.models`, so future
analytics do not depend on ByteTrack or Supervision objects. Raw trajectory
positions are bounding-box bottom centers; each sample also retains an
EMA-smoothed position for later speed and queue analytics.

Camera geometry is separated into:

- `app.geometry.config`: validated camera/analytics YAML models, normalized
  zones, directed counting lines, queue service points, and calibration pairs
- `app.geometry.primitives`: inclusive-boundary polygon membership, polygon
  validation, directed line sides, and finite-segment crossing results
- `app.geometry.calibration`: fixed-resolution homography construction and an
  explicit unavailable result when calibration is not configured
- `scripts/configure_camera.py`: optional Tk reference-frame point selector

People counting is separated into:

- `app.analytics.counting`: confirmed-track polygon occupancy, hysteresis-based
  finite-line crossings, per-camera state, cumulative totals, and explicit reset
- `app.analytics.visualization`: composable occupancy and entry/exit overlays
- `app.core.models.Event`: shared event envelope used for `line_crossed` events

Each counting line has a normalized `hysteresis` value. It is interpreted as a
fraction of the frame diagonal, creating a resolution-independent dead band on
both sides of the line. A track must move from one stable side to the other and
cross the configured finite segment before it is counted. Tracks on the line or
moving only within the dead band do not increment totals.

Restricted-area detection is separated into:

- `app.analytics.restricted_area`: foot-point membership, independent
  camera/track/zone state, entry dwell, exit grace, cooldown, and reset
- `app.analytics.restricted_visualization`: named zone status plus pending and
  confirmed intrusion overlays
- `app.storage.EventSink`: the persistence boundary used by analytics
- `app.storage.JsonlEventSink`: append-only persistence for shared `Event`
  envelopes; SQLite remains a later phase

Each restricted zone supports `entry_dwell_seconds`, `exit_grace_seconds`, and
`alert_cooldown_seconds`. Entry and exit lifecycle events remain visible for a
transient crossing, while only a dwell-qualified
`restricted_area_confirmed` event is the cooldown-controlled alert. During a
short missing-observation or outside period, prior state is kept until the exit
grace expires.

Movement heatmaps are separated into:

- `app.analytics.heatmap`: image-grid mapping, optional calibrated ground-grid
  mapping, sample-count occupancy, elapsed-seconds dwell, bounded track state,
  reset/tumbling-window aggregation, and CSV/PNG export
- `HeatmapSnapshot.occupancy`: number of confirmed position samples per cell
- `HeatmapSnapshot.dwell_seconds`: timestamp-derived time assigned to the
  previous confirmed position cell

These are people-movement analytics heatmaps, not detector feature or neural
network activation heatmaps. Long gaps are not treated as dwell: intervals over
`max_sample_gap_seconds` are discarded, and per-track state is evicted after
`track_idle_seconds`. `aggregation_window_seconds` creates constant-memory
tumbling windows; setting it to `null` retains run totals until explicit reset.
Calibration creates a parallel ground-plane grid, using configured
`ground_bounds` or the calibration correspondence extents. Missing calibration
leaves image heatmaps available and reports a clear ground-unavailable reason.
Image heatmaps always evaluate the complete frame: zero-value cells use
the low (blue) end of the selected color map and increasingly occupied cells
progress through green/yellow/orange to red. `smoothing_sigma_cells` spreads
each foot-point cell into a readable density region without changing the exact
numeric CSV values. Each image snapshot is also divided into 12 row-major
regions (3 rows by 4 columns). The three regions with the highest average
occupancy are returned as `top_crowded_regions`, including their row, column,
normalized frame bounds, and average occupancy. Heatmap videos and live previews
draw all 12 region boundaries and highlight the top three with matching rank and
region labels for direct comparison with the dashboard report.

Queue analytics are separated into:

- `app.analytics.queue`: independent camera/queue/track state, join/leave and
  edge-triggered overflow events, raw and exponentially smoothed counts, and
  current/completed waiting-time estimates
- `app.analytics.queue_visualization`: polygons, service points, queue metrics,
  overflow state, and candidate/member labels on tracked people
- `app.analytics.vertical_queue`: automatic grouping by horizontal proximity
  of confirmed bbox centers, stable row IDs, and per-frame reassignment
- `app.analytics.vertical_queue_visualization`: same-color member boxes,
  vertical row lines, and a single queue-count summary line

Queue membership is heuristic. A confirmed track becomes a queue candidate
only while its foot point is inside the manually configured polygon and its
smoothed image speed does not exceed
`maximum_speed_pixels_per_second`. It becomes a member after
`minimum_dwell_seconds`. State survives missing tracker observations for
`gap_tolerance_seconds`; an explicitly observed polygon exit is handled
immediately. `service_completion_radius` is a normalized image-coordinate
distance from the manual service point and determines whether a leaving track's
wait is considered completed. The approximate current wait is the mean elapsed
time since qualifying presence among current members. These values are useful
operational estimates, not proof that a person is queueing or was served.

`raw_count` contains dwell-qualified active members. `smoothed_count` is an
exponential moving average controlled by `count_smoothing_alpha`. Queue presence
uses `raw_count > 0`, and overflow uses `raw_count >= overflow_threshold`.
Overflow start/end events are emitted only when that Boolean state changes.
Configured mode performs no automatic discovery, ordering, or group inference.
Vertical mode is a deliberately small geometric heuristic: confirmed people
whose bbox-center X positions are within `--queue-column-distance` (a fraction
of frame width) are grouped into a nearly vertical row. Groups smaller than
`--queue-min-people` are omitted. Row centers are matched between adjacent
frames so their IDs and colors remain stable; membership is recalculated every
frame, so a track moving into another row adopts that row's color. This does not
prove that the detected column is a real-world queue.

Speed analytics are separated into:

- `app.analytics.speed`: bounded timestamp-window estimation over smoothed
  trajectory points, image and calibrated-ground jump rejection, camera/track
  metrics, and explicit physical-speed unavailability
- `TrackObservation.speed_pixels_per_second`: image speed, always labelled
  `px/s`; it is never presented as a physical measurement
- `TrackObservation.speed_metres_per_second`: physical speed available only
  when a valid homography declares metre-based ground units
- queue track and aggregate metrics: signed progress velocity toward the
  configured service point in `px/s` and, when calibrated, `m/s`

The `speed` camera section configures `window_seconds`,
`minimum_displacement_pixels`, `maximum_speed_pixels_per_second`, and
`maximum_speed_metres_per_second`. Estimation uses timestamps rather than video
FPS, so irregular observations and skipped frames are supported. At least two
accepted samples within the window are required. Motion below the minimum
displacement is reported as stationary (`0 px/s`); excessive jumps are rejected
instead of being allowed to contaminate the track speed.

Queue-progress speed is the signed component of the smoothed velocity pointing
toward the service point. Positive values mean progress, negative values mean
movement away, and exactly sideways motion is zero. Queue snapshots and CSV
metrics include average member movement and progress speeds.

Validate a configured homography against an independently measured distance:

```bash
python scripts/validate_calibration_distance.py configs/cameras/example_lobby.yaml \
  --frame-size 1920 1080 --first 383.8 377.65 --second 1535.2 377.65 \
  --known-metres 8.0
```

The script reports projected and known distances plus absolute and relative
error. The two image points should mark surveyed ground locations; this is a
validation aid, not automatic calibration.

An optional `active_schedule` accepts `start`/`end` in `HH:MM`, weekday names
or integers (`0` is Monday), and an IANA timezone. Schedule evaluation requires
Unix timestamps. Recorded-video source-relative timestamps should leave the
schedule unset unless the caller supplies an epoch time basis.

Normalized `(0, 0)` and `(1, 1)` map to inclusive pixel corners `(0, 0)` and
`(width - 1, height - 1)`. Ground projection is only available through a
validated calibration containing at least four non-degenerate image/ground
correspondences.

## Dependencies

The base dependency set is CPU-capable: NumPy, ONNX Runtime, OpenCV Headless,
PyYAML, and the maintained `trackers` package. API, dashboard, and development
tools are optional dependency groups. ByteTrack does not use BoT-SORT, OSNet,
or PyTorch in this phase. The current `trackers` package itself declares the
regular OpenCV distribution, although this application uses no GUI APIs and
remains headless at runtime.

Python 3.10 or newer is required.

## Current commands

From this directory:

```bash
python -m pip install -e ".[dev]"
python -m pytest
python scripts/check_imports.py
```

### Run the dashboard integration API

Install the API dependencies and start the service from this directory:

```bash
python -m pip install -e ".[api,dev]"
python -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```

OpenAPI documentation is available at `http://localhost:8000/docs`. Uploaded
videos and generated artifacts from the dashboard are stored under
`output/dashboard/<job-id>/`.
The API exposes detection, tracking, counting, restricted-area, heatmap,
vertical-queue, configured-queue, and combined-analysis presets. Recorded-video
analytics use `configs/cameras/example_lobby.yaml` when no camera YAML is
uploaded. A validated YAML uploaded with a recorded job is persisted under the
analytics jobs directory and becomes the default for later recorded jobs, so
it only needs to be uploaded once; a job submitted without a YAML reuses that
saved default and never overwrites it. Set `VIDEO_ANALYTICS_CAMERA_CONFIG_PATH`
to choose a different persistent path.

Live RTSP sources use the same processing commands through
`POST /api/v1/stream-jobs`. The JSON body accepts `stream_url`,
`application_id`, `camera_id`, optional `max_frames`, and optional
`enable_reid`. The Tarebar camera backend calls this endpoint when live
analysis is started from a monitoring card. Presets that require camera YAML
remain recorded-job-only for now.

Useful environment settings:

```text
VIDEO_ANALYTICS_JOBS_DIR=/path/to/job-storage
VIDEO_ANALYTICS_JOB_WORKERS=1
VIDEO_ANALYTICS_MAX_UPLOAD_BYTES=1073741824
VIDEO_ANALYTICS_CORS_ORIGINS=http://localhost:3000
VIDEO_ANALYTICS_DETECTOR_MODEL=/absolute/path/to/model.onnx
SERVICE_AUTH_SECRET=                      # unset/empty = no token check (default)
VIDEO_ANALYTICS_RETENTION_DAYS=0          # 0 = keep finished jobs forever (default)
VIDEO_ANALYTICS_CANCEL_GRACE_SECONDS=10   # then terminate(), kill() 5 s later
VIDEO_INSIGHT_PRECISION=auto              # auto | bf16 | fp16 | 8bit | 4bit
VIDEO_INSIGHT_MIN_PIXELS=                 # unset = follow the GPU size
VIDEO_INSIGHT_MAX_PIXELS=
```

#### Model quality

The insight model is never weakened when the hardware can afford it.

- `VIDEO_INSIGHT_PRECISION=auto` (default) loads the weights unquantized (bf16,
  or fp16 on GPUs without bf16) when they fit in the free VRAM with 4 GiB of
  headroom (`weights x 1.2 + 4 GiB`), and falls back to 4-bit NF4 otherwise.
  The 3B model runs at full precision from a 16 GB card, the 7B model from
  32 GB. `bf16`/`fp16`/`8bit`/`4bit` force one mode.
- `VIDEO_INSIGHT_MIN_PIXELS` / `VIDEO_INSIGHT_MAX_PIXELS` bound what the model
  sees per frame. Unset, they follow the GPU: from 12 GiB of VRAM on
  `200704`-`1003520` (256-1280 patches of 28x28, the range of the model card; a
  1280x720 camera frame is not downscaled), below that `65536`-`147456`. Small
  defects such as bruises and mould spots depend on this more than on anything
  else.
- Another checkpoint only needs `VIDEO_INSIGHT_MODEL_PATH`: any `Qwen3-VL-*-Instruct`
  or `Qwen2.5-VL-*-Instruct` directory (the loader dispatches on the config). The
  default is `Qwen3-VL-2B-Instruct`, which needs transformers >= 4.57 (in the image).
- `GET /health` reports what is in effect: `"insights": {"model", "loaded",
  "requested_precision", "precision", "min_pixels", "max_pixels"}`; the same
  line is logged when the model loads.

### HTTP API reference

Every successful JSON response is wrapped as `{"data": ...}`.

| Method and path | Purpose |
|---|---|
| `GET /health` | Always HTTP 200: `{"status":"ok"\|"degraded","analyticsStore":bool,"fleet":{...}}`. Any database problem (unreachable, not migrated yet) reports `degraded`. |
| `GET /api/v1/fleet/status` | Always-on 0.5 FPS camera fleet: settings, last refresh/error and one entry per worker. |
| `GET /api/v1/applications` | Application presets with their `metric_schema`. |
| `GET /api/v1/trackers` | Registered tracker types. |
| `GET /api/v1/jobs`, `GET /api/v1/jobs/{id}` | Job history and one job (`JobPublic`). |
| `POST /api/v1/jobs` | Multipart recorded-video job: `video`, `application_id`, `camera_id`, `max_frames`, `enable_reid`, `tracker_type`, `camera_config`, `persist_camera_config`. |
| `POST /api/v1/stream-jobs` | JSON live RTSP job: `stream_url`, `application_id` or `application_ids`, `camera_config_yaml`, `camera_id`, `max_frames`, `enable_reid`, `tracker_type`. |
| `POST /api/v1/jobs/{id}/cancel` | Cooperative cancel marker, backed by a hard stop (see below). |
| `GET /api/v1/jobs/{id}/events` | SSE tail of `events.jsonl` (`Last-Event-ID` / `?after=`). |
| `GET /api/v1/jobs/{id}/preview`, `GET /api/v1/jobs/{id}/preview-stream` | Latest preview JPEG / MJPEG stream. |
| `GET /api/v1/jobs/{id}/artifacts/{key}` | One artifact listed in `JobPublic.artifacts`. |
| `GET /api/v1/jobs/{id}/report` | Job report (see "Job report and export"). |
| `GET /api/v1/jobs/{id}/export.zip` | ZIP with `report.json`, `metrics.csv` and the job artifacts. |
| `POST /api/v1/frames/first` | Multipart `video` → first frame as a JPEG data URL (zone editor fallback). |
| `POST /api/v1/frames/from-stream` | JSON `{stream_url}` → one live still as a JPEG data URL. |
| `POST /api/v1/preview-stream` | JSON `{stream_url}` → unannotated MJPEG of the camera. |
| `POST /api/v1/video-insights`, `POST /api/v1/video-insights/from-stream` | Qwen2.5-VL answers to the two fixed questions. |
| `POST /api/v1/fruit-quality`, `POST /api/v1/fruit-quality/from-stream` | Qwen2.5-VL fruit freshness (see "Fruit quality contract"). |
| `GET /api/v1/restricted-area-events?camera_id=&limit=` | Recent restricted-area entry/exit events across jobs. |
| `POST /api/v1/ingest/minutes` | Minute facts from producers; header `X-Analytics-Key` = `ANALYTICS_INGEST_KEY`. |
| `GET /api/v1/management/overview\|people-flow\|queues\|spatial` | Management read models; header `X-Analytics-Key` = `ANALYTICS_READ_KEY`; query `from`, `to`, `locationType`, `locationId`, `placeType`, `comparison`, `bucket`, `timeFrom`, `timeTo`. Peak-period labels are rendered in `ANALYTICS_TIMEZONE`. |

#### Service token (optional)

Set `SERVICE_AUTH_SECRET` to the same value as the dashboard to require a token
on every `/api/v1/*` route. `/api/v1/management/*` and `/api/v1/ingest/*` keep
their `X-Analytics-Key` check instead, and `/health`, `/docs`, `/openapi.json`,
`/redoc` and `OPTIONS` requests are never checked. With the variable unset or
empty nothing is enforced.

The token is a compact JWS signed with HS256 (unpadded base64url), header
`{"alg":"HS256","typ":"JWT"}`, payload
`{"iss":"tarebar","sub":"<userId>","role":"<Role>","iat":<unix>,"exp":<unix>}`.
It must not be expired (30 s leeway) and `iss` must be `tarebar`. Send it as
`Authorization: Bearer <token>` or, where headers cannot be set (`EventSource`,
`<img>`, `<video>`, download links), as the `access_token=<token>` query
parameter. A missing or invalid token is answered with HTTP 401
`{"detail":"invalid or missing service token"}`; the response still carries the
CORS headers, so the browser can read it.

#### Video insight contract

`POST /api/v1/video-insights` takes the form fields `video`, `num_frames`
(2–16), `window_start_seconds`, `window_end_seconds` and `include_thumbnail`
(default `false`); `/from-stream` takes the JSON fields `stream_url`,
`num_frames`, `interval_seconds` (10–60) and `include_thumbnail`. The response
is `{"answers":{"fighting":"Yes"|"No","floor_clean":"Yes"|"No"},"frame_count",
"inference_seconds","thumbnail","model"}`; `thumbnail` is the middle sampled
frame as a JPEG data URL (or `null`).

#### Fruit quality contract

`POST /api/v1/fruit-quality` takes the form fields `media` (image or video),
`num_frames` (1–16), and the optional `detail_level` (`summary` default, or
`detailed`), `per_frame` (default `false`) and `include_thumbnails` (default
`false`). `/from-stream` takes `stream_url`, `num_frames` (2–16),
`interval_seconds` and the same three optional JSON fields. The defaults return
exactly the original verdict plus the new, empty fields.

```json
{
  "has_fruit": true, "label": "تازه", "freshness_score": 92,
  "distribution": {"fresh": 90, "middle": 8, "rotten": 2},
  "fruit_count_estimate": 14, "confidence": 85, "summary_fa": "…", "verdict_fa": "…",
  "frame_count": 8, "inference_seconds": 5.2,
  "analysis_version": 2, "detail_level": "detailed", "model": "Qwen2.5-VL-3B-Instruct",
  "grade": "A",
  "fruit_types": [{"name_fa": "سیب", "share_percent": 70}],
  "defects": [{"type": "bruising", "label_fa": "کوفتگی", "severity": "low",
               "affected_percent": 10, "note_fa": "…"}],
  "shelf_life_days_estimate": 5, "recommendation_fa": "…", "model_recommendation_fa": "…",
  "storage_advice_fa": "…",
  "frames": [{"index": 0, "timestamp_seconds": 1.2, "has_fruit": true, "freshness_score": 90,
              "label": "تازه", "note_fa": "…", "thumbnail": "data:image/jpeg;base64,…", "error": null}],
  "frame_statistics": {"analyzed": 8, "score_min": 85, "score_max": 95,
                       "score_mean": 90.1, "score_stddev": 3.2},
  "total_seconds": 12.3
}
```

- The core fields are validated strictly (a violation is HTTP 503). `grade` is
  derived by the service from `freshness_score` (≥85 A, ≥70 B, ≥50 C, else D;
  `null` without fruit) and never taken from the model.
- **Demo-safe output (analysis version 3).** The scene is a market hall with
  several pallets and passing people, so the prompts describe that scene and
  ask the model to ignore people, floor and packaging. Nothing a bystander can
  dispute is asked for or shown: no fruit count (`fruit_count_estimate` is
  always `null`), no shares per fruit kind, no shelf-life days, no storage
  advice, and Persian text sentences that mention colours, numbers or people
  are dropped. Instead the service composes `quality_profile`, a list of
  observable aspects derived from the validated numbers and parsed defects —
  `overall` (grade), `uniformity` (dominant share), `spoilage` (rotten share +
  mold/decay), `mechanical` and `surface` (detailed only, from the defect
  groups), `coverage` (confidence) — each `{key, label_fa, value_fa, status:
  good|watch|poor, note_fa}`. Defects carry `extent` (`few|some|most`) and
  `extent_label_fa` instead of a percentage; `verdict_fa` uses bands, not
  percentages. `fruit_types` (names only), `summary_fa` and `defects[].note_fa`
  remain in the response for the record but the dashboard does not display them.
- The analysis starts with a one-word presence question (is any fruit or
  vegetable visible?) on at most three evenly spaced frames. Only a clear "No"
  ends it there with the no-fruit result (`has_fruit: false`, label `نامشخص`,
  score 0, `grade: null`, HTTP 200). The 3B model answers that question
  reliably, whereas it fills `has_fruit` of the long answer with `true` for any
  picture.
- Text a manager acts on is composed by the service from the validated numbers,
  never by the model: `verdict_fa` puts label, score, grade, shares and count
  into one Persian sentence, and `recommendation_fa` is the action for the grade
  (`FRUIT_ACTIONS_FA`), present at every detail level and `null` without fruit.
  The model's own advice is kept as `model_recommendation_fa` for the record
  only; on real photos it contradicted its own score ("sell it quickly" for a
  rotten apple).
- `summary_fa` is the model's observation after cleaning: echoed instruction
  prefixes, first-person chatter and repetition loops are removed and the text
  is cut to 500 characters. When nothing usable is left it equals `verdict_fa`.
- An answer cut off by the token limit is recovered instead of failing: the
  object is closed after its last complete member and a long cut-off sentence
  ends at its last full stop. Only an answer without the core numbers is a 503.
- `fruit_types`, `defects`, `shelf_life_days_estimate`, `model_recommendation_fa`
  and `storage_advice_fa` are filled only for `detail_level=detailed` (otherwise
  `[]`/`null`) and parsed leniently: an invalid entry (for example a sentence in
  place of a fruit name) is dropped, never an error. Defect `type` is one of `bruising, mold, discoloration, soft_spot,
  wrinkling, dryness, decay, cut_damage, pest_damage, other` (an unlisted type
  becomes `other`), `severity` one of `low, medium, high`; `label_fa` is the
  service's Persian label for the type. If the detailed answer is unusable the
  service asks again with the summary prompt and returns empty detail fields.
- `frames` is `[]` unless `per_frame` or `include_thumbnails` is set.
  `per_frame` runs one compact single-image prompt per sampled frame after the
  aggregate verdict; a failing frame gets an `error` string and `null` scores
  and never fails the request. With only `include_thumbnails` the scores stay
  `null`. `timestamp_seconds` is the video time, the seconds since the first
  live sample, `0` for an image, or `null` when the video has no frame rate.
- `frame_statistics.analyzed` counts the successfully analysed frames; the score
  figures (population standard deviation) cover the analysed frames that show
  fruit and are `null` when there are none.
- Thumbnails are JPEG data URLs, at most 320 px wide, quality 70.
- On a GPU out-of-memory error the service frees the CUDA cache and retries once
  with half of the frames (`frame_count` reports the frames actually used). If
  that fails too, and for any other unexpected failure, the answer is HTTP 503
  with a readable `detail`, never a 500.

#### Job report and export

`GET /api/v1/jobs/{id}/report` returns
`{"job": JobPublic, "configuration": {...}, "final_metrics": {...},
"metric_statistics": {"<key>": {"min","max","mean","last","samples"}},
"event_counts": {"<type>": n}, "analytics_event_counts": {"<event_type>": n},
"duration_seconds": n, "generated_at": iso}`. Statistics cover every numeric
top-level metric in `metrics.jsonl`; `duration_seconds` is the processing time
reported by the pipeline, falling back to `updated_at - created_at`.

`GET /api/v1/jobs/{id}/export.zip` downloads `report.json`, `metrics.csv` (one
row per `metrics.jsonl` line: `timestamp, frame_index, elapsed_seconds`, then
every numeric metric key, blank when missing) and every artifact of the job
under `artifacts/`. The uploaded input video is never included.

#### Cancellation and retention

Cancelling a job writes the cooperative `cancel.requested` marker. If the
subprocess is still alive after `VIDEO_ANALYTICS_CANCEL_GRACE_SECONDS`
(default 10) it is sent `terminate()`, and `kill()` five seconds later, so a
process stuck on an RTSP connect or a model load no longer blocks the worker.

Set `VIDEO_ANALYTICS_RETENTION_DAYS` to a positive number to delete the
directories of completed, failed and cancelled jobs whose last update is older
than that many days. The sweep runs at startup and every six hours; the default
`0` disables it. Running jobs and the `_settings` directory are never touched.

#### Management store notes

Before a minute document is written to the outbox it is clamped to the ingest
limits (`sampleCount` ≤ 3600, `waitSampleCount` ≤ 1000 with the sums scaled to
keep the averages, ≤ 500 completed waits and events, ≤ 100 queues, ≤ 32 spatial
layers of ≤ 256 points), so a fast recorded job can no longer lose a minute to
an HTTP 422. Fleet cameras have no counting lines; their `entries` are the
cumulative unique confirmed people and their `exits` the expired tracks (never
more than the entries). Cameras with counting lines keep directed line counts.

### Run with Docker

The Docker image includes the CPU runtime, FFmpeg, and the detector/ReID model
files used by the API. This repository's `compose.yaml` runs only the API.
The full Tarebar stack (frontend, PostgreSQL, MediaMTX, and this service)
is started from the sibling frontend repository:

```bash
cd ../Tarebar-Smart-Monitoring-Platform
docker compose -f docker-compose.dev.yml --env-file .env.dev up --build
```

To run just this API:

```bash
docker compose up --build -d
```

Port `8000` is used by default. If it is already occupied, choose another host
port, for example `VIDEO_ANALYTICS_PORT=8001 docker compose up --build -d`.

Check the service and open its API documentation:

```bash
curl http://localhost:8000/health
docker compose logs -f api
```

The API is available at `http://localhost:8000`, and Swagger UI is at
`http://localhost:8000/docs`. Job uploads and generated artifacts persist in
the `analytics-jobs` Docker volume across container restarts. Stop the service
with `docker compose down`; add `--volumes` only when you also intend to delete
all persisted jobs.

GitHub Actions (`.github/workflows/build-image.yml`) publishes
`ghcr.io/<owner>/video-analytics:<git-sha>` and, on tags such as `v1.0.0`,
`ghcr.io/<owner>/video-analytics:v1.0.0`. Production servers pull that image
from the deployment repository; they do not build from this source tree.

ONNX weights are gitignored. Local builds need
`All_weights/Weights_final/HumanDetection_light_input_640.onnx` and
`All_weights/Weights_final/Tracking_osnet_x0_25_msmt17.onnx`. CI downloads
`weights.tgz` from the private `v1.0-models` GitHub Release (via `gh release
download` and `GITHUB_TOKEN`) when they are not in the checkout. Override with
the `VIDEO_ANALYTICS_WEIGHTS_URL` repository secret if needed.

Load the default settings:

```bash
python -c "from app.core.config import load_settings; print(load_settings())"
```

Run person detection:

```bash
python -m app.detection.cli input.jpg --output outputs/detected.jpg
python -m app.detection.cli input.mp4 --output outputs/detected.mp4
```

Run person tracking with annotated IDs and trajectories:

```bash
python -m app.tracking.cli input.mp4 --output outputs/tracked.mp4
```

OSNet appearance re-identification is optional because it adds inference cost.
Enable it when identity continuity after occlusion or a short disappearance is
more important than maximum throughput:

```bash
python -m app.tracking.cli input.mp4 \
  --enable-reid \
  --output outputs/tracked_reid.mp4
```

The default ReID model is
`All_weights/Weights_final/Tracking_osnet_x0_25_msmt17.onnx`. The dashboard
shows the ReID checkbox only to organization administrators, leaves it off by
default, stores the choice with the job, and labels ReID-enabled jobs. ReID
improves tracker-ID continuity; it is not biometric identification and can
still make mistakes when people look alike or are absent for a long time.

Count confirmed tracked people in every video frame. Without a camera YAML,
the entire image is used as one occupancy zone. The command writes both an
annotated MP4 and a CSV containing one row per frame. `confirmed_humans` is the
number of distinct confirmed tracker IDs visible in that frame, while
`total_unique_people` is the cumulative number of confirmed IDs seen since the
video run started. Both counts are drawn live on the annotated video. Polygon
columns count only foot points inside each configured zone:

```bash
python -m app.analytics.cli data/human.mp4 \
  --output outputs/human_counted.mp4 \
  --counts-csv outputs/human_counts.csv
```

To report configured polygon occupancy and line totals instead:

```bash
python -m app.analytics.cli data/human.mp4 \
  --camera-config configs/cameras/example_lobby.yaml \
  --output outputs/human_counted.mp4 \
  --counts-csv outputs/human_counts.csv
```

Restricted-area and queue processing are both runtime opt-in. Use
`--enable-restricted-area` for configured restricted zones. Use
`--enable-queue` for automatic vertical grouping, which is the default queue
mode and does not require configured queue polygons:

```bash
python -m app.analytics.cli data/human.mp4 \
  --enable-queue \
  --queue-column-distance 0.08 \
  --queue-min-people 2 \
  --output outputs/human_queues.mp4 \
  --counts-csv outputs/human_queues.csv
```

This default vertical mode preserves the Phase 8 visualization: people in the
same detected row use the same color and each row has a full-height colored
line. Phase 9 additionally estimates speed automatically in this mode. The
queue label at the top of the frame displays average member speed in `px/s`
and, when a calibrated camera YAML is supplied, `m/s`. The per-frame CSV adds
`vertical_queue_speeds_pixels_per_second` and
`vertical_queue_speeds_metres_per_second` as `row_ID:value` lists. These same
values are typed fields on each `VerticalQueueRow` for later dashboard use.

Without `--enable-queue`, no queue state is accumulated and no queue overlays,
queue CSV columns, or queue events are produced. Vertical mode adds
`vertical_queue_rows`, `vertical_queue_people`, `vertical_queue_counts`, and
the two speed columns to the CSV. To retain the original manual polygon heuristic, use
`--queue-mode configured` together with a camera YAML containing an enabled
configured queue. To enable both configured restricted areas and automatic
vertical queues:

```bash
python -m app.analytics.cli data/human.mp4 \
  --camera-config configs/cameras/example_lobby.yaml \
  --enable-restricted-area \
  --enable-queue \
  --output outputs/human_analytics.mp4
```

Without `--enable-restricted-area`, no restricted-area state, overlay, CSV
columns, or restricted-area events are produced.

Heatmap processing is runtime opt-in even when the camera YAML lists the module.
Pass `--enable-heatmap` to produce evolving occupancy and dwell overlay videos,
plus separate final CSV grids, colorized PNGs, and first-frame overlays.
Ground-plane CSVs and PNGs are also produced when calibration exists. Use the
camera YAML's `outputs.heatmap_directory`, allow the default output directory,
or override it explicitly:

```bash
python -m app.analytics.cli data/human.mp4 \
  --camera-config configs/cameras/example_lobby.yaml \
  --enable-heatmap \
  --heatmap-dir outputs/human_heatmaps
```

Without `--enable-heatmap`, no heatmap state is accumulated and no heatmap
files or videos are created. Videos are streamed frame by frame, so enabling
them does not retain an unbounded collection of video frames in memory.

Grid sizes, aggregation window, maximum sample gap, idle-state timeout, color
map, overlay opacity, and rendering smoothing are configured under `heatmap`.
CSV rows follow image
or ground Y and columns follow X. Image-space PNG and overlay dimensions match
the source frame; ground PNG dimensions match the configured ground grid.

Trajectory trails are shown by default. Hide only the trails while continuing
to collect trajectory history for later analytics:

```bash
python -m app.tracking.cli input.mp4 \
  --output outputs/tracked.mp4 \
  --no-trajectories
```

The tracking CLI reports average detection, tracking, and total-frame time.
Tracker activation threshold, lost-track buffer, IoU match threshold, and
history size are configured under `tracker` in `configs/default.yaml`.

For a bounded smoke run:

```bash
python -m app.detection.cli input.mp4 --max-frames 5
```

The CLI prints the selected model/providers, frame and detection counts, and
average preprocessing, inference, post-processing, and total detector time.
CUDA or another available execution provider can be requested first while
retaining CPU fallback:

```bash
python -m app.detection.cli input.mp4 \
  --providers CUDAExecutionProvider CPUExecutionProvider
```

Set `VIDEO_ANALYTICS_CONFIG` to load another YAML file. Individual overrides
are available through `VIDEO_ANALYTICS_LOG_LEVEL`,
`VIDEO_ANALYTICS_OUTPUT_DIR`, `VIDEO_ANALYTICS_DATABASE_PATH`,
`VIDEO_ANALYTICS_DETECTOR_MODEL`, and comma-separated
`VIDEO_ANALYTICS_ONNX_PROVIDERS`.

## Camera geometry configuration

The complete example at `configs/cameras/example_lobby.yaml` includes
occupancy and restricted polygons, a directed line, queue/service geometry,
heatmap settings, and a four-point calibration. Load it independently of the
application settings:

```bash
python -c "from app.geometry import load_camera_config; print(load_camera_config('configs/cameras/example_lobby.yaml'))"
```

To draw geometry on a reference image and save validated YAML:

```bash
python scripts/configure_camera.py data/human.jpg \
  --camera-id lobby_east \
  --name "East lobby" \
  --source data/lobby.mp4 \
  --output configs/cameras/lobby_east.yaml
```

Select a polygon, two line endpoints, an optional queue service point, and at
least four calibration image points. Each calibration click prompts for the
matching ground-plane coordinate. The selector uses optional system Tk; it is
not imported by the headless runtime.

## Supported model

Phase 2 supports only:

```text
All_weights/Weights_final/HumanDetection_light_input_640.onnx
input:  [batch, 3, 640, 640]
output: [batch, 5, 8400]
```

The dynamic batch metadata is accepted, but the frame API intentionally runs a
single frame per call. The output is interpreted as one-class YOLO-style
`center_x, center_y, width, height, confidence` candidates.

`HumanDetection_input_640.onnx` is excluded because it references
`model.onnx.data`, which is missing beside the model in `Weights_final`. A
possible sidecar elsewhere in the experimental repository has not been copied
or assumed to match.

`HumanDetection_server_input_640.onnx` is also excluded in Phase 2. Its primary
`[batch, 300, 6]` output and auxiliary outputs have not yet had their exact box,
score, class, and suppression semantics verified. The light-contract validator
rejects it with a clear shape error instead of guessing.

## Planned commands

The following interfaces are planned and are not available in Phase 8:

```bash
video-analytics api --config configs/default.yaml
video-analytics dashboard --config configs/default.yaml
```

Model paths and detector thresholds are configurable. No weights are copied,
changed, or repaired by this application.
