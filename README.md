# Video Analytics MVP

This is the isolated application scaffold for a medium-complexity, video-based
people analytics MVP. Existing computer-vision experiments and model weights
remain outside this directory and are treated as read-only inputs.

Phase 9 adds timestamp-window movement speed in explicitly labelled pixels per
second and, only with valid metre-based ground calibration, metres per second.
It also adds queue progress toward each configured service point. A FastAPI
integration layer now exposes recorded-video upload jobs and their generated
artifacts to the Tarebar dashboard.

The event-driven video-insight API reuses the optimized Qwen2.5-VL flow from
`Action_recognition/test_vlm_qwen_optimized_video.py`: eight uniformly sampled
RGB frames, bounded processor resolution, 4-bit NF4 weights with FP16 compute
on CUDA, and deterministic generation. `POST /api/v1/video-insights` accepts a
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
it only needs to be uploaded once. Set `VIDEO_ANALYTICS_CAMERA_CONFIG_PATH` to
choose a different persistent path.

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
```

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
