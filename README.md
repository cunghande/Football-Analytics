# Football Analytics

**Bài tập lớn phát triển từ [alina-letzien/football-analytics](https://github.com/alina-letzien/football-analytics) theo giấy phép MIT.** Mã gốc cung cấp pipeline YOLO, ByteTrack, phân cụm màu áo, gán bóng và ước tính chuyển động. Phần phát triển của bài làm gồm kiểm tra và chia dataset bốn lớp, fine-tune YOLO26x trên Modal, so sánh ba thiết lập, cải tiến chia đội, xuất CSV, heatmap/minimap, dashboard Streamlit và ứng viên đường chuyền. Nguồn ảnh là [FSDM detection-player v4](https://universe.roboflow.com/fsdm-olss1/detection-player/dataset/4), CC BY 4.0.

**Báo cáo có thể nộp:** [bản Word](report/BAO_CAO_HOAN_CHINH.docx), [bản Markdown](report/BAO_CAO_HOAN_CHINH.md), [hình và dữ liệu thực nghiệm](report/), [hướng dẫn dataset](docs/DATASET_GUIDE.md). [Hướng dẫn chạy thử PDF buổi tập](docs/HUONG_DAN_TEST_BAO_CAO.md) mô tả đầu ra sinh tự động theo video. Các bảng và hình tách kết quả đo thực từ hình minh họa kỹ thuật. Chưa có video demo độc lập nên chưa có phép đo độ chính xác đầu cuối cho tracking, gán bóng, quãng đường và đường chuyền.

Computer vision football analysis pipeline with YOLO, ByteTrack, OpenCV, and Python.

## About the project

This project analyzes a football video and produces an annotated output video with:

- tracked players, referees, and ball
- team assignment (based on jersey colors)
- ball possession per frame
- team ball-control percentage
- player speed and distance estimates


### Context

The project was created to learn and apply practical computer vision concepts in a real-world scenario.
It focuses on building an end-to-end pipeline, integrating model and tracker into a cohesive video analytics workflow, with the goal of providing an analytics baseline that can be extended and retrained on custom data.

### How It Works (High Level)

1. Detect objects in each frame with YOLO.
2. Track detected objects over time with ByteTrack.
3. Estimate camera movement and adjust object positions.
4. Assign players to teams using jersey color clustering (KMeans).
5. Associate ball possession with nearest player.
6. Estimate speed and distance from tracked movement.
7. Render overlays and export the final analysis video.

## Features

- YOLO detection for players, referees, and ball
- ByteTrack multi-object tracking with persistent track IDs
- team assignment from jersey color (KMeans)
- ball interpolation across missed detections
- ball possession assignment to nearest player
- team ball-control overlay in output video
- camera motion estimation and position adjustment
- perspective-transform integration (with fallback scaling)
- player speed and distance estimation
- optional stub caching for faster repeated runs

## Project Structure

```text
football-analytics/
├── input/                        # Input videos (expected: DFL-Scoutingfeed.mp4)
├── output/                       # Annotated output videos
├── stubs/                        # Cached intermediate files from analysis
├── datasets/                     # Local training datasets
├── src/
│   ├── yolo_detector.py
│   ├── tracker.py                # ByteTrackTracker
│   ├── team_assigner.py
│   ├── player_ball_assigner.py
│   ├── camera_motion.py
│   ├── perspective_transformer.py
│   └── speed_distance_calculator.py
├── tests/
│   ├── conftest.py               # Shared fixtures (synthetic tracks and frames)
│   ├── test_player_ball_assigner.py
│   ├── test_speed_distance.py
│   ├── test_pipeline_integration.py
│   └── test_smoke.py             # Full pipeline smoke test (marked slow)
├── config.py
├── main.py                       # Main analysis entrypoint
├── train.py                      # Local YOLO training entrypoint
├── pytest.ini
└── requirements.txt
```

## Pipeline Architecture

- `main.py`: orchestrates full analysis run
- `src/yolo_detector.py`: YOLO inference and object parsing
- `src/tracker.py`: ByteTrack integration and track state
- `src/team_assigner.py`: team clustering and team ID assignment
- `src/player_ball_assigner.py`: nearest-player possession assignment
- `src/camera_motion.py`: optical-flow camera motion estimation
- `src/perspective_transformer.py`: coordinate transformation utilities
- `src/speed_distance_calculator.py`: speed/distance computation and overlays

## Installation

### Prerequisites

- Python 3.13+

### Setup

Run the following commands within a terminal:

```bash
git clone https://github.com/alina-letzien/football-analytics.git
cd football-analytics
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run Analysis

1. Put an input video inside the `input` folder (or use the default `input/DFL-Scoutingfeed.mp4`).
2. Open a terminal and run `./venv/bin/python main.py` inside the project root.
    - The YOLO Model is downloaded automatically on first run if not present locally.

### Output and generated files

- `output/analysis.mp4`
- `output/analysis_summary.json` — sanity metrics used by the smoke test's tolerance-band assertions (see Testing below)
- `stubs/track_stubs.pkl`
- `stubs/camera_movement_stubs.pkl`

## Testing

Fast tests run automatically on every push and pull request via GitHub Actions (`.github/workflows/tests.yml`). 

The smoke test runs weekly on a schedule, can be triggered manually from the Actions tab (`workflow_dispatch`), and also runs automatically on any pull request that changes `requirements.txt`, `config.py`, or `src/{yolo_detector,tracker,camera_motion}.py` — the files where fast tests alone can't catch a regression.

The smoke test also asserts tolerance-band sanity metrics on `output/analysis_summary.json` (ball detection rate, player track-ID count, team balance, speed bounds — see `tests/test_smoke.py` and `calibrate_smoke_thresholds.py`), so a detector/tracker regression fails CI instead of silently passing "didn't crash." These bounds don't catch everything a human eye would (e.g. a team-color mixup that still balances two teams), so `.github/workflows/video-review-gate.yml` additionally blocks merging any sensitive-file PR until it carries a `video-reviewed` label — download the `smoke-test-output` artifact from the Tests run (kept for 14 days), watch `output/analysis.mp4`, then add the label. The label is stripped automatically on every new commit, so re-review is required after each push.

The test suite has three layers with different speed and dependency requirements.

### Fast tests — unit and integration (no ML, no video)

```bash
pytest
```

Runs 25 tests in ~2 seconds. No YOLO model or input video required.

| Layer | File | What it covers |
|---|---|---|
| Unit | `tests/test_player_ball_assigner.py` | Ball-to-player assignment logic, foot-corner distance |
| Unit | `tests/test_speed_distance.py` | Speed formula, distance accumulation, Pythagorean math |
| Unit | `tests/test_pipeline_summary.py` | Sanity-metric computation from the tracks dict (`src/pipeline_summary.py`) |
| Unit | `tests/test_calibrate_thresholds.py` | Tolerance-band derivation formula (`calibrate_smoke_thresholds.py`) |
| Integration | `tests/test_pipeline_integration.py` | All post-detection stages on synthetic data — position enrichment, camera motion, team assignment, possession, speed/distance |

### Smoke test — full pipeline (requires input video and YOLO model)

```bash
pytest -m slow
```

Runs `main.py` end-to-end and asserts `output/analysis.mp4` is written, plus the tolerance-band sanity checks described above. Skipped automatically if `input/DFL-Scoutingfeed.mp4` is not present.

> **Before running the smoke test after a dependency bump:** delete `stubs/track_stubs.pkl` and `stubs/camera_movement_stubs.pkl` first. Without this, the pipeline loads cached detections and never calls YOLO or torch — the updated packages are not actually exercised.

> **Recalibrating the tolerance bands:** after any intentional change to detection/tracking behavior (model version, confidence thresholds, an `ultralytics`/`torch`/`supervision` dependency bump), run `python calibrate_smoke_thresholds.py output/analysis_summary.json` against a fresh baseline run and paste the printed constants into `tests/test_smoke.py`, keeping the accompanying explanatory comments the script prints alongside `AVG_REFEREES_HIGH` and `SPEED_KMH_MAX_CEILING`.

```bash
rm stubs/*.pkl
pytest -m slow
```

### Run only fast tests explicitly

```bash
pytest -m "not slow"
```

## Example Outcome

After a full run, the output video includes:

- bounding boxes and track IDs
- team labels and ball possession markers
- camera motion info
- team ball-control percentages
- speed and distance text near tracked players

## Configuration

All tunable settings are in `config.py` — it is the single place to adjust the pipeline without touching source code:

- `YOLO_CONFIG` — model path, device, confidence and IoU thresholds
- `TRACKER_CONFIG` — ByteTrack parameters
- `TEAM_CONFIG` — number of teams, KMeans random state
- `POSSESSION_CONFIG` — max distance to assign ball to a player
- `FIELD_CONFIG` — field dimensions in meters
- `SPEED_CONFIG` — FPS, meters-per-pixel scale, speed averaging window
- `CAMERA_CONFIG` — motion threshold, Farnebäck optical flow parameters
- `VIDEO_CONFIG` — input/output paths, codec, stub caching toggle
- `COLORS` — bounding box colors for players, ball, and referees (BGR format)

### Note
If logic changed, delete files in `stubs/` and rerun to recompute cached data.

## Training

The experiment uses [FSDM `detection-player` v4](https://universe.roboflow.com/fsdm-olss1/detection-player/dataset/4). The prepared YOLO dataset is stored locally in `datasets/football_4class_clip_split/` and is excluded from Git. See the [dataset guide](docs/DATASET_GUIDE.md) for download, audit, split and license details. The COCO pretrained `yolo26x.pt` only maps `person` and `sports ball` into two of the four target classes; the fine-tuned checkpoints predict all four.

Run the two training settings on Modal from PowerShell. The named runs already exist in the original experiment, so **choose fresh run names** to repeat them:

```powershell
python -m pip install modal
python -X utf8 -m modal setup
python -X utf8 -m modal volume create football-dataset
python -X utf8 -m modal volume put football-dataset .\datasets\football_4class_clip_split /football_4class_clip_split
python -X utf8 -m modal run modal_train.py --dataset football_4class_clip_split --model yolo26x.pt --epochs 50 --imgsz 640 --batch 8 --freeze 10 --name freeze10_repeat
python -X utf8 -m modal run modal_train.py --dataset football_4class_clip_split --model yolo26x.pt --epochs 50 --imgsz 640 --batch 8 --name full_repeat
```

`modal_train.py` stores checkpoint, `results.csv` and plots in the `football-training-runs` Volume. The completed comparison is in [the full report](report/BAO_CAO_HOAN_CHINH.md) and [raw summary JSON](report/data/comparison_b_summary.json). The default dashboard looks for the downloaded `freeze=10` checkpoint under `runs/modal/detection_player_freeze10_v1/weights/best.pt`; download it from your Modal Volume when setting up another computer. Dataset and checkpoints are intentionally not pushed to GitHub.

## Current Scope and Limitations

- optimized for single-camera broadcast-style footage
- all frames are loaded into memory before processing — not suitable for long videos
- team assignment remains sensitive to kit color, lighting, and camera angle; temporal reclustering reduced accuracy on the reviewed 50-frame clip
- track ID resets (after occlusion or track loss) break per-player speed and distance accumulation
- speed and distance estimates are approximations — accurate values require manual perspective calibration via `PerspectiveTransformer.set_calibration_points()`
- heavy occlusions can reduce tracking quality
- CSV movement and possession values are estimates; without manual field calibration, meters use the original 0.1 m/pixel assumption
- coaches and bench staff are detected as players — the base model has no concept of in-play vs. out-of-play persons
- player detection quality depends on model — the base model can miss players under occlusion or crowding; a custom-trained model improves this (see Training section)
- pass maps contain unverified candidate events only; formation analysis and xG are not implemented

## Roadmap Ideas

- evaluate CSV movement and possession against manually checked video annotations
- validate spatial heatmaps against annotated video and calibration points
- validate pass candidates against manually annotated events; explore shot detection
- add formation detection from player position clusters
- auto field calibration via pitch line detection (no manual calibration points)
- add player re-identification across track ID resets

## Resources

### Detection & Tracking

- [Ultralytics YOLO — Docs](https://docs.ultralytics.com/) — full reference for model loading, inference, training, and export; covers all YOLO variants
- [Ultralytics YOLO — GitHub](https://github.com/ultralytics/ultralytics) — source, issues, and model release notes; useful when upgrading `ultralytics`
- [ByteTrack Paper](https://arxiv.org/abs/2110.06864) — original multi-object tracking algorithm used under the hood via supervision
- [Supervision — Docs](https://supervision.roboflow.com/) — API reference for `ByteTrack`, `Detections`, annotators, and all utilities wrapping the tracker
- [Supervision — GitHub](https://github.com/roboflow/supervision) — changelog and migration guides; check here after supervision version bumps

### Video & Image Processing

- [OpenCV — Docs](https://docs.opencv.org/) — reference for video I/O, drawing, and the optical flow used in `CameraMotionAnalyzer`

### Machine Learning

- [PyTorch — Docs](https://pytorch.org/docs/stable/) — framework used by Ultralytics for inference and training
- [scikit-learn KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) — team color clustering used in `TeamAssigner`

### Datasets

- [Roboflow Universe](https://universe.roboflow.com/) — browse and export football datasets in YOLO format; search "football players detection" to find the dataset referenced in `train.py`
- [Roboflow Export Guide](https://docs.roboflow.com/exporting-data) — how to export a dataset in Ultralytics/YOLO format compatible with `train.py`

## License

MIT License. See `LICENSE`.

## Academic extension: four-class football detector

This repository is based on [alina-letzien/football-analytics](https://github.com/alina-letzien/football-analytics) (MIT). The original video pipeline, tracking, team-color clustering, possession logic, and speed/distance estimates are inherited from that project. The student extension currently adds dataset auditing and clip-group preparation, YOLO26x training on Modal, a three-setting detector evaluation, temporal team-color assignment with separate goalkeeper/referee roles, corrected consecutive-frame distance accounting, structured CSV export, spatial heatmap/minimap views, a local Streamlit dashboard, and a conservative pass-candidate map. The default detector still uses `yolo26x.pt`; set `YOLO_MODEL_PATH` to the four-class checkpoint to activate role-specific predictions.

| Component | Status and source |
|---|---|
| Video analytics pipeline | Inherited from the original MIT project |
| `detection-player` v4 dataset | FSDM on [Roboflow Universe](https://universe.roboflow.com/fsdm-olss1/detection-player/dataset/4), CC BY 4.0; cleaned and split by clip for this experiment |
| Four-class YOLO fine-tuning | Added here; pretrained, `freeze=10`, and full fine-tune comparison on Modal |
| Dataset audit and experimental figures | Added here; scripts in `scripts/`, reports and figures in `report/` |
| Team assignment and role separation | Added here; 50 reviewed validation frames, 200 player samples, ablation results in step C. An optional rare-yellow referee rule reduces role flicker on the reviewed video and records its method in JSON. |
| CSV export | Added here; player, team, possession timeline and method metadata. No real-video movement results yet. |
| Heatmap/minimap | Added here; two explicit modes: calibrated field or normalized image space. Dataset illustration in step E. |
| Streamlit | Added here; local video upload, annotated H.264 playback, charts/pass map, CSV tables, and downloadable report ZIP. |
| Session PDF | Added here; automatically summarizes metrics by temporary track ID and team, without ratings. Uncalibrated distance and speed are hidden. |
| Pass map (optional G) | Added here as an unverified same-team candidate heuristic; no pass labels or accuracy metrics yet. |

See the [dataset guide](docs/DATASET_GUIDE.md), [step A report](report/BAO_CAO_BUOC_A.md), [step B report](report/BAO_CAO_BUOC_B.md), [step C report](report/BAO_CAO_BUOC_C.md), [step D report](report/BAO_CAO_BUOC_D.md), [step E report](report/BAO_CAO_BUOC_E.md), [step F report](report/BAO_CAO_BUOC_F.md), [step G report](report/BAO_CAO_BUOC_G.md), and [combined report outline](report/DE_CUONG_BAO_CAO.md). The reports include experimental protocols, figures, and limitations of the clip-level split. Large datasets and checkpoints remain local or in Modal Volumes and are excluded from Git.

### Run the local dashboard

```powershell
python -m pip install -r requirements.txt
powershell -ExecutionPolicy Bypass -File .\run_dashboard.ps1
```

Open the Streamlit URL printed in PowerShell (usually `http://127.0.0.1:8501`). The single video flow uses `runs/modal/detection_player_freeze10_v1/weights/best.pt` and automatically uses a detected local CUDA GPU. On this Windows machine, `run_dashboard.ps1` starts the CUDA-enabled Python 3.14 even when the CPU-only `venv` is active. Browser uploads can be up to 500 MB; a same-machine file path can be used for larger recordings. The app automatically analyzes up to 900 consecutive frames and adjusts resolution to bound memory use. The PDF and CSV describe only that segment. The separate academic-results tab was removed; video PDF and ZIP downloads remain available. Without manual field calibration, distances and speeds remain approximate. Pass-map arrows are unverified candidates.

The single pipeline automatically applies role consistency and an image-space pitch-region filter. The yellow-shirt cue activates only when the clip provides supporting evidence; otherwise model role labels remain. These heuristics have not been evaluated against independent role and off-field ground truth. See the [750-frame failure analysis](report/PHAN_TICH_LOI_VAI_TRO_VA_NGUOI_NGOAI_SAN.md).

Each run also produces a printable `analysis_session_report.pdf`. The PDF shows per-track observation time, image-space foot displacement, ball-assigned frames, and unverified pass-candidate send/receive counts; it does not infer player names or rate performance. Pixel displacement is not running distance. Distance/speed in physical units are shown only when field coordinates have been manually calibrated. The video minimap overlay is smaller than the standalone minimap figure. See the [test guide](docs/HUONG_DAN_TEST_BAO_CAO.md).
