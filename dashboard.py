"""Local Streamlit interface for the existing football analysis pipeline."""

import json
from pathlib import Path

import streamlit as st

from src.dashboard_io import (
    MAX_FRAMES, MAX_UPLOAD_BYTES, analysis_files, create_academic_zip,
    create_job, create_report_zip, probe_video, read_csv, run_job,
)


ROOT = Path(__file__).resolve().parent
CHECKPOINTS = {
    "Fine-tune freeze=10 (khuyến nghị)": ROOT / "runs/modal/detection_player_freeze10_v1/weights/best.pt",
    "Fine-tune toàn bộ": ROOT / "runs/modal/detection_player_full_v1/weights/best.pt",
    "Model gốc COCO": ROOT / "yolo26x.pt",
}


def _device_options():
    try:
        import torch
        return (["0", "cpu"] if torch.cuda.is_available() else ["cpu"]), None
    except (ImportError, OSError) as exc:
        return ["cpu"], f"PyTorch không tải được trong Python hiện tại: {exc}"


def _show_job(job):
    files = analysis_files(Path(job) / "analysis")
    if not files["video"].exists():
        st.warning("Kết quả trước chưa có video để xem.")
        return
    st.subheader("Kết quả video")
    st.video(str(files["video"]), format="video/mp4")
    st.caption("Video H.264 được chuyển mã từ output OpenCV để xem trong trình duyệt.")

    teams = read_csv(files["teams"])
    players = read_csv(files["players"])
    st.subheader("Thống kê mô tả")
    columns = st.columns(3)
    for col, row, title in zip(columns, teams, ("Đội 1", "Đội 2", "Chưa rõ đội")):
        col.metric(title, f"{float(row['possession_pct_video']):.1f}%")
    st.caption("Tỷ lệ frame được gán bóng, không phải nhãn kiểm soát bóng chuẩn. Hai đội là ID cụm màu áo.")
    st.image(str(files["chart"]), caption="Gán bóng theo frame và mức quan sát track; quãng đường chỉ hiện khi đã hiệu chỉnh sân.")
    st.image(str(files["heatmaps"]), caption="Mật độ vị trí quan sát; xem metadata để biết chế độ tọa độ.")
    st.image(str(files["pass_map"]), caption="Đường chuyền ứng viên theo quy tắc; chưa có nhãn chuẩn để xác nhận.")
    st.image(str(files["minimap"]), caption="Minimap của frame cuối.", width=480)
    if files["session_report"].exists():
        st.download_button("Tải báo cáo buổi tập (PDF)",
                           data=files["session_report"].read_bytes(),
                           file_name="bao_cao_buoi_tap.pdf", mime="application/pdf")
        st.caption("Báo cáo theo track ID tạm thời; chỉ hiển thị chỉ số, không chấm điểm cầu thủ.")
    with st.expander("Bảng dữ liệu cầu thủ và đội"):
        st.dataframe(teams, width="stretch")
        st.dataframe(players, width="stretch")
    with st.expander("Các đường chuyền ứng viên"):
        candidates = read_csv(files["pass_candidates"])
        if candidates:
            st.dataframe(candidates, width="stretch")
        else:
            st.write("Không có ứng viên đáp ứng tất cả điều kiện trong clip này.")

    report_path = create_report_zip(job)
    st.download_button("Tải báo cáo phân tích (ZIP)",
                       data=report_path.read_bytes(), file_name=report_path.name,
                       mime="application/zip")
    st.caption("ZIP gồm CSV, biểu đồ, metadata và thống kê; video không nằm trong ZIP.")


def _video_tab():
    st.write("Tải một clip ngắn để chạy YOLO, ByteTrack, chia đội và xuất báo cáo trên máy này.")
    st.caption(f"Giới hạn: {MAX_UPLOAD_BYTES // (1024 * 1024)} MB, {MAX_FRAMES} frame, 1280×720. "
               "Pipeline hiện nạp toàn bộ frame vào RAM.")
    available = {label: path for label, path in CHECKPOINTS.items() if path.is_file()}
    if not available:
        st.error("Chưa tìm thấy checkpoint. Hãy tải best.pt từ kết quả Modal vào runs/modal/ hoặc đặt yolo26x.pt tại thư mục gốc.")
        return
    checkpoint_name = st.selectbox("Checkpoint", list(available))
    if checkpoint_name == "Model gốc COCO":
        st.warning("Model gốc không dự đoán riêng goalkeeper/referee; báo cáo vai trò sẽ thiếu.")
    devices, torch_error = _device_options()
    device = st.selectbox("Thiết bị suy luận", devices)
    if torch_error:
        st.error(torch_error)
        st.caption("Nếu Windows chặn torch_python.dll trong venv, hãy thoát venv rồi chạy "
                   "dashboard bằng Python hệ thống đã cài PyTorch.")
    uploaded = st.file_uploader("Video bóng đá", type=["mp4", "mov", "avi", "mkv"],
                                max_upload_size=100)
    technical_only = st.checkbox("Clip ghép từ ảnh hoặc FPS giả định - chỉ kiểm tra kỹ thuật")
    if st.button("Phân tích video", type="primary",
                 disabled=uploaded is None or torch_error is not None):
        try:
            with st.spinner("Đang xử lý video và tạo báo cáo. Clip CPU có thể mất vài phút..."):
                job, input_path = create_job(uploaded, ROOT / "output/dashboard_jobs")
                details = probe_video(input_path)
                if technical_only:
                    (job / "dataset_demo_protocol.json").write_text(json.dumps({
                        "purpose": "technical_demo_only",
                        "warning": "FPS hoặc thứ tự thời gian chưa được xác minh; không dùng làm số liệu buổi tập.",
                    }, ensure_ascii=False, indent=2), encoding="utf-8")
                st.write(f"Đầu vào: {details['frames']} frame, {details['fps']:.2f} FPS, "
                         f"{details['width']}×{details['height']} pixel.")
                run_job(input_path, job, available[checkpoint_name], device,
                        source_name=uploaded.name)
                st.session_state["dashboard_job"] = str(job)
                st.session_state["dashboard_source"] = uploaded.name
            st.success("Đã hoàn thành phân tích.")
        except Exception as exc:
            st.error(f"Không xử lý được video: {exc}")
    job = st.session_state.get("dashboard_job")
    if job:
        st.divider()
        st.caption(f"Kết quả lần chạy gần nhất: {st.session_state.get('dashboard_source', 'video')}.")
        _show_job(job)


def _academic_tab():
    st.write("Các hình dưới đây là kết quả của bước A–E, dùng nguồn dữ liệu và phương pháp ghi trong từng báo cáo.")
    figures = [
        ("So sánh detector trên validation", "comparison_b_metrics.png",
         "Cùng 599 frame validation; xem báo cáo B để biết ánh xạ lớp của baseline."),
        ("Thí nghiệm chia đội trên 50 frame", "team_assignment_comparison.png",
         "200 crop đã kiểm tra thủ công, dùng box GT; không phải độ chính xác toàn pipeline."),
        ("Minimap và mật độ vị trí từ dataset", "dataset_spatial_preview_50frames.png",
         "50 frame và 594 box player GT; tọa độ ảnh, không có track ID hay mét thật."),
        ("Pass map minh họa quy tắc", "pass_map_schematic_pass_map.png",
         "Ví dụ hoàn toàn giả lập; mũi tên là ứng viên, chưa được kiểm chứng bằng nhãn pass."),
    ]
    for title, filename, caption in figures:
        path = ROOT / "report/figures" / filename
        if path.exists():
            st.subheader(title)
            st.image(str(path), caption=caption)
    try:
        bundle = create_academic_zip(ROOT)
        st.download_button("Tải báo cáo A–G và hình (ZIP)",
                           data=bundle.read_bytes(), file_name=bundle.name,
                           mime="application/zip")
    except FileNotFoundError as exc:
        st.warning(str(exc))


def main():
    st.set_page_config(page_title="Football Analytics", layout="wide")
    st.title("Football Analytics")
    st.caption("Dự án phát triển từ alina-letzien/football-analytics (MIT); "
               "dataset detection-player v4 của FSDM (CC BY 4.0).")
    video_tab, academic_tab = st.tabs(["Phân tích video", "Kết quả cho báo cáo"])
    with video_tab:
        _video_tab()
    with academic_tab:
        _academic_tab()


if __name__ == "__main__":
    main()
