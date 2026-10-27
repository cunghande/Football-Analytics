"""Isolated input/output handling for the local Streamlit dashboard."""

import csv
import json
import math
from pathlib import Path
import shutil
import subprocess
import uuid
import zipfile

import cv2


ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
MAX_FRAMES = 450
MAX_EXTENDED_FRAMES = 900
MAX_FRAME_PIXELS = 1280 * 720


def create_job(uploaded, jobs_root="output/dashboard_jobs"):
    """Store uploaded bytes under an app-generated path, never its client name."""
    suffix = Path(uploaded.name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError("Chỉ nhận video MP4, MOV, AVI hoặc MKV.")
    if uploaded.size <= 0 or uploaded.size > MAX_UPLOAD_BYTES:
        raise ValueError(f"Video tải lên phải có dung lượng từ 1 byte đến {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    job = Path(jobs_root) / uuid.uuid4().hex
    job.mkdir(parents=True, exist_ok=False)
    input_path = job / f"input{suffix}"
    uploaded.seek(0)
    with input_path.open("wb") as target:
        shutil.copyfileobj(uploaded, target, length=1024 * 1024)
    if input_path.stat().st_size != uploaded.size:
        raise IOError("Tệp tải lên chưa được lưu đầy đủ.")
    return job, input_path


def create_job_from_local_file(path, jobs_root="output/dashboard_jobs"):
    """Use a same-machine video without loading its full contents into Streamlit."""
    source = Path(path).expanduser()
    if not source.is_file() or source.stat().st_size <= 0:
        raise ValueError("Đường dẫn video trên máy không tồn tại hoặc tệp rỗng.")
    if source.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError("Chỉ nhận video MP4, MOV, AVI hoặc MKV.")
    job = Path(jobs_root) / uuid.uuid4().hex
    job.mkdir(parents=True, exist_ok=False)
    return job, source.resolve()


def probe_video(path, *, enforce_limits=True, max_frames=None, max_frame_pixels=None):
    """Validate a small clip before a memory-heavy full pipeline run."""
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("Không đọc được video đã tải lên.")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ok, _ = capture.read()
        if not ok or fps <= 0 or frames <= 0 or width <= 0 or height <= 0:
            raise ValueError("Video thiếu frame hoặc metadata FPS/kích thước hợp lệ.")
        frame_limit = MAX_FRAMES if max_frames is None else max_frames
        pixel_limit = MAX_FRAME_PIXELS if max_frame_pixels is None else max_frame_pixels
        if enforce_limits and frames > frame_limit:
            raise ValueError(f"Đoạn phân tích tối đa {frame_limit} frame.")
        if enforce_limits and width * height > pixel_limit:
            raise ValueError("Độ phân giải đoạn phân tích vượt ngân sách RAM.")
        return {"fps": fps, "frames": frames, "width": width, "height": height,
                "duration_seconds": frames / fps}
    finally:
        capture.release()


def prepare_analysis_clip(source, job, details, start_seconds=0.0, max_frames=None):
    """Bound total decoded pixels while preserving source FPS and time order."""
    from imageio_ffmpeg import get_ffmpeg_exe

    fps = details["fps"]
    duration = details["duration_seconds"]
    max_frames = MAX_FRAMES if max_frames is None else max_frames
    if not 1 <= max_frames <= MAX_EXTENDED_FRAMES:
        raise ValueError(f"Chỉ hỗ trợ từ 1 đến {MAX_EXTENDED_FRAMES} frame mỗi lần.")
    if not 0 <= start_seconds < duration:
        raise ValueError("Thời điểm bắt đầu phải nằm trong thời lượng video.")
    remaining = max(1, math.ceil((duration - start_seconds) * fps))
    selected_frames = min(max_frames, remaining)
    pixel_budget = min(MAX_FRAME_PIXELS,
                       (MAX_FRAMES * MAX_FRAME_PIXELS) // selected_frames)
    needs_clip = (start_seconds > 0 or details["frames"] > selected_frames
                  or details["width"] > 1280 or details["height"] > 720
                  or details["width"] * details["height"] > pixel_budget)
    if not needs_clip:
        return Path(source), None

    scale = min(1.0, 1280 / details["width"], 720 / details["height"],
                math.sqrt(pixel_budget / (details["width"] * details["height"])))
    width = max(2, int(details["width"] * scale) // 2 * 2)
    height = max(2, int(details["height"] * scale) // 2 * 2)
    destination = Path(job) / "analysis_input.mp4"
    command = [get_ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
               "-ss", f"{start_seconds:.6f}", "-i", str(source), "-an",
               "-frames:v", str(selected_frames), "-vf", f"scale={width}:{height}",
               "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination)]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    processed = probe_video(destination, max_frames=max_frames,
                            max_frame_pixels=pixel_budget)
    clip_info = {
        "source_frames": details["frames"],
        "source_width": details["width"],
        "source_height": details["height"],
        "requested_max_frames": max_frames,
        "pixel_budget_per_frame": pixel_budget,
        "start_seconds": start_seconds,
        "end_seconds": start_seconds + processed["duration_seconds"],
        "analyzed_frames": processed["frames"],
        "analyzed_width": processed["width"],
        "analyzed_height": processed["height"],
    }
    return destination, clip_info


def transcode_for_browser(source, destination):
    """Convert OpenCV's mp4v output to browser-compatible H.264 MP4."""
    from imageio_ffmpeg import get_ffmpeg_exe

    command = [
        get_ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(source), "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(destination),
    ]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    destination = Path(destination)
    if not destination.exists() or destination.stat().st_size == 0:
        raise RuntimeError("Không tạo được video H.264 để xem trong trình duyệt.")
    return destination


def analysis_files(stem):
    stem = Path(stem)
    base = str(stem)
    return {
        "video": Path(base + "_browser.mp4"),
        "raw_video": Path(base + ".mp4"),
        "players": Path(base + "_players.csv"),
        "teams": Path(base + "_teams.csv"),
        "timeline": Path(base + "_possession_timeline.csv"),
        "summary": Path(base + "_summary.json"),
        "csv_metadata": Path(base + "_csv_metadata.json"),
        "role_refinement": Path(base + "_role_refinement.json"),
        "heatmaps": Path(base + "_heatmaps.png"),
        "heatmaps_pdf": Path(base + "_heatmaps.pdf"),
        "heatmaps_metadata": Path(base + "_heatmaps_metadata.json"),
        "minimap": Path(base + "_minimap_last.png"),
        "chart": Path(base + "_report_chart.png"),
        "chart_pdf": Path(base + "_report_chart.pdf"),
        "pass_candidates": Path(base + "_pass_candidates.csv"),
        "pass_map": Path(base + "_pass_map.png"),
        "pass_map_pdf": Path(base + "_pass_map.pdf"),
        "pass_map_metadata": Path(base + "_pass_map_metadata.json"),
        "session_report": Path(base + "_session_report.pdf"),
    }


def run_job(input_path, job, model_path, device, source_name=None, clip_info=None,
            role_refinement_enabled=True):
    """Call the same analysis pipeline used by main.py, then render charts."""
    from main import FootballAnalyzer
    from config import YOLO_CONFIG
    from scripts.plot_csv_report import plot_csv_report

    stem = Path(job) / "analysis"
    analyzer = FootballAnalyzer(
        video_path=str(input_path),
        output_path=str(stem) + ".mp4",
        model_path=str(model_path),
        device=device,
        read_from_stub=False,
        stub_dir=str(Path(job) / "stubs"),
        role_refinement_enabled=role_refinement_enabled,
    )
    analyzer.analyze_video()
    files = analysis_files(stem)
    needed = ("raw_video", "players", "teams", "timeline", "role_refinement", "heatmaps", "minimap",
              "pass_candidates", "pass_map", "session_report")
    missing = [key for key in needed if not files[key].exists()]
    if missing:
        raise RuntimeError(f"Pipeline chưa tạo đủ kết quả: {', '.join(missing)}")
    plot_csv_report(str(stem))
    transcode_for_browser(files["raw_video"], files["video"])
    run_metadata = {
        "model_path": str(Path(model_path).resolve()),
        "device": device,
        "source_name": Path(source_name or input_path.name).name,
        "player_confidence_threshold": YOLO_CONFIG["confidence_threshold"],
        "ball_confidence_threshold": YOLO_CONFIG["ball_confidence_threshold"],
        "ball_assignment_direct_detection_only": True,
    }
    if clip_info is not None:
        run_metadata["clip"] = clip_info
    (Path(job) / "dashboard_run.json").write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    from src.session_report import generate_session_report
    generate_session_report(stem)
    return files


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def create_report_zip(job):
    """Bundle only descriptive report data, not the large video or raw upload."""
    job = Path(job)
    files = analysis_files(job / "analysis")
    include = [path for key, path in files.items() if key not in ("video", "raw_video")]
    include.append(job / "dashboard_run.json")
    protocol = job / "dataset_demo_protocol.json"
    if protocol.exists():
        include.append(protocol)
    missing = [path.name for path in include if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Thiếu tệp báo cáo: {', '.join(missing)}")
    destination = job / "analysis_report.zip"
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in include:
            archive.write(path, arcname=path.name)
    return destination


def create_academic_zip(root):
    """Bundle the existing A-G reports and clearly labeled figures."""
    root = Path(root)
    reports = [root / "report" / f"BAO_CAO_BUOC_{step}.md" for step in "ABCDEFG"]
    reports.append(root / "report" / "DE_CUONG_BAO_CAO.md")
    figure_names = (
        "dataset_distribution.png", "comparison_b_metrics.png",
        "comparison_b_per_class.png", "team_assignment_comparison.png",
        "team_assignment_frame_heatmap.png", "dataset_spatial_preview_50frames.png",
        "dataset_spatial_preview_metadata.json", "dashboard_pipeline_smoke.png",
        "pass_map_schematic_pass_map.png", "pass_map_schematic_pass_map_metadata.json",
        "pass_map_schematic_pass_candidates.csv",
    )
    figures = [root / "report" / "figures" / name for name in figure_names]
    files = reports + figures
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Thiếu tư liệu: {', '.join(missing)}")
    destination = root / "output" / "academic_reports_A_E.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, arcname=str(path.relative_to(root)))
    return destination
