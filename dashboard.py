"""Local Streamlit interface for the existing football analysis pipeline."""

import json
from pathlib import Path

import streamlit as st

from src.dashboard_io import (
    MAX_EXTENDED_FRAMES, MAX_UPLOAD_BYTES, analysis_files,
    create_job, create_job_from_local_file, create_report_zip, prepare_analysis_clip,
    probe_video, read_csv, run_job,
)


ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "runs/modal/detection_player_role_ft_full_v1/weights/best.pt"


def _inference_device():
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return "0", f"GPU 0 — {name}", None
        return "cpu", "CPU", None
    except (ImportError, OSError) as exc:
        return "cpu", "CPU", f"PyTorch không tải được trong Python hiện tại: {exc}"


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
    if files["role_refinement"].exists():
        role_method = json.loads(files["role_refinement"].read_text(encoding="utf-8"))
        if role_method.get("method") in {"rare_yellow_kit_temporal_vote", "supported_yellow_referee_tracks"}:
            st.caption("Vai trò cầu thủ/trọng tài được ổn định theo áo vàng, quần tối và 5 quan sát gần nhau; "
                       "đây vẫn là ước đoán, chưa đối chiếu nhãn thủ công.")
    if files["field_person_filter"].exists():
        field_method = json.loads(files["field_person_filter"].read_text(encoding="utf-8"))
        if field_method.get("enabled"):
            st.caption(f"Đã bỏ {field_method['excluded_observations']} quan sát người có dấu hiệu ở ngoài sân; "
                       "kiểm tra video vì đây là quy tắc theo ảnh, không phải nhãn thủ công.")
    st.subheader("Gán bóng theo đội")
    st.caption(f"Đã phân nhóm màu áo: Đội 1 có {teams[0]['tracked_people']} track, "
               f"Đội 2 có {teams[1]['tracked_people']} track. "
               "Track không phải số cầu thủ duy nhất.")
    assigned_frames = sum(int(row["possession_frames"]) for row in teams
                          if row["team_id"] in {"1", "2"})
    if assigned_frames == 0:
        st.warning("Đã chia được hai nhóm màu áo, nhưng không gán được bóng cho đội nào. "
                   "'Không đo được' bên dưới chỉ nói về gán bóng, không phải phát hiện đội.")
    columns = st.columns(3)
    for col, row, title in zip(columns, teams, ("Đội 1", "Đội 2", "Không gán bóng")):
        col.metric(title, f"{float(row['possession_pct_video']):.1f}%"
                   if assigned_frames else "Không đo được")
    st.caption("Tỷ lệ frame được gán bóng, không phải nhãn kiểm soát bóng chuẩn. Hai đội là ID cụm màu áo.")
    chart_metadata = json.loads(files["csv_metadata"].read_text(encoding="utf-8"))
    chart_caption = ("Gán bóng theo frame và quãng đường ước tính trên sân đã hiệu chỉnh."
                     if chart_metadata.get("coordinate_method") == "four_point_field_homography" else
                     "Gán bóng theo frame và chuyển động tương đối trên ảnh (px/s); đây chưa phải tốc độ chạy km/h."
                     if any(row.get("mean_camera_relative_motion_px_s") for row in players) else
                     "Gán bóng theo frame và số frame quan sát theo track.")
    st.image(str(files["chart"]), caption=chart_caption)
    st.image(str(files["heatmaps"]), caption="Mật độ vị trí quan sát; xem metadata để biết chế độ tọa độ.")
    st.image(str(files["pass_map"]), caption="Đường chuyền ứng viên theo quy tắc; chưa có nhãn chuẩn để xác nhận.")
    st.image(str(files["minimap"]), caption="Minimap của frame cuối.", width=480)
    if files["session_report"].exists():
        st.download_button("Tải báo cáo buổi tập (PDF)",
                           data=files["session_report"].read_bytes(),
                           file_name="bao_cao_buoi_tap.pdf", mime="application/pdf")
        st.caption("Báo cáo theo track ID tạm thời; chỉ hiển thị chỉ số, không chấm điểm cầu thủ.")
    st.subheader("Chỉ số theo từng track")
    columns = [
        "role", "track_id", "team_id", "observed_seconds", "observed_path_px",
        "mean_camera_relative_motion_px_s",
        "possession_frames", "pass_candidate_sent", "pass_candidate_received",
    ]
    st.dataframe([{key: row.get(key, "") for key in columns} for row in players], width="stretch")
    st.caption("mean_camera_relative_motion_px_s là chuyển động tương đối trên ảnh, chưa phải tốc độ km/h. "
               "Chuyền/nhận là ứng viên chưa xác minh thành công; track ID có thể đổi khi mất dấu.")
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
    st.write("Chọn video để chạy YOLO, ByteTrack, chia đội và xuất báo cáo trên máy này.")
    st.caption(f"Upload tối đa {MAX_UPLOAD_BYTES // (1024 * 1024)} MB; "
               "video lớn hơn có thể đọc từ đường dẫn trên chính máy chạy dashboard. "
               "Báo cáo chỉ tính cho đoạn đã chọn.")
    if not CHECKPOINT.is_file():
        st.error("Chưa tìm thấy checkpoint role_ft_v1 tại runs/modal/detection_player_role_ft_full_v1/weights/best.pt.")
        return
    st.caption("Model: detection_player_role_ft_full_v1 (4 lớp; bổ sung 60 frame sửa vai trò).")
    device, device_label, torch_error = _inference_device()
    st.write(f"Thiết bị suy luận: **{device_label}**")
    st.caption("YOLO chạy trên GPU; đọc video, theo dõi, thống kê và tạo báo cáo vẫn sử dụng CPU."
               if device == "0" else "YOLO và các bước xử lý còn lại chạy trên CPU.")
    if device == "cpu" and not torch_error:
        st.warning("Python hiện tại không nhận GPU CUDA. Trên máy này, dừng Streamlit và chạy "
                   "`powershell -ExecutionPolicy Bypass -File .\\run_dashboard.ps1` để dùng Python có GPU.")
    if torch_error:
        st.error(torch_error)
        st.caption("Nếu Windows chặn torch_python.dll trong venv, hãy thoát venv rồi chạy "
                   "dashboard bằng Python hệ thống đã cài PyTorch.")
    input_mode = st.radio("Nguồn video", ["Tải lên từ trình duyệt", "Đường dẫn trên máy này"],
                          horizontal=True)
    uploaded = None
    local_path = ""
    if input_mode == "Tải lên từ trình duyệt":
        uploaded = st.file_uploader("Video bóng đá", type=["mp4", "mov", "avi", "mkv"],
                                    max_upload_size=MAX_UPLOAD_BYTES // (1024 * 1024))
    else:
        local_path = st.text_input("Đường dẫn video trên máy chạy dashboard",
                                   placeholder=r"D:\Downloads\video.mp4")
        st.caption("Chỉ dùng khi video nằm trên chính máy chạy Streamlit; tệp nguồn không bị sao chép.")
    local_source = local_path.strip().strip('"')
    st.caption(f"Tự lấy tối đa {MAX_EXTENDED_FRAMES} frame liên tiếp và điều chỉnh độ phân giải "
               "theo giới hạn bộ nhớ; nếu video dài hơn, báo cáo ghi rõ đoạn đã phân tích.")
    start_seconds = st.number_input("Bắt đầu phân tích từ giây thứ", min_value=0.0,
                                    value=0.0, step=1.0)
    if st.button("Phân tích video", type="primary",
                 disabled=(uploaded is None and not local_source) or torch_error is not None):
        try:
            status = ("YOLO đang chạy trên GPU; các bước còn lại dùng CPU. "
                      "Đang xử lý video và tạo báo cáo..." if device == "0" else
                      "Đang xử lý video trên CPU và tạo báo cáo; có thể mất vài phút...")
            with st.spinner(status):
                if uploaded is not None:
                    job, input_path = create_job(uploaded, ROOT / "output/dashboard_jobs")
                else:
                    job, input_path = create_job_from_local_file(
                        local_source, ROOT / "output/dashboard_jobs")
                source_name = input_path.name if uploaded is None else uploaded.name
                original = probe_video(input_path, enforce_limits=False)
                analysis_input, clip_info = prepare_analysis_clip(
                    input_path, job, original, start_seconds=start_seconds,
                    max_frames=MAX_EXTENDED_FRAMES)
                details = probe_video(analysis_input, enforce_limits=False)
                if clip_info:
                    st.info(f"Chỉ phân tích đoạn {clip_info['start_seconds']:.1f}–"
                            f"{clip_info['end_seconds']:.1f} giây; "
                            f"{details['frames']} frame, {details['width']}×{details['height']} pixel. "
                            "Các chỉ số trong báo cáo thuộc đoạn này, không thuộc cả video gốc.")
                st.write(f"Đầu vào: {details['frames']} frame, {details['fps']:.2f} FPS, "
                         f"{details['width']}×{details['height']} pixel.")
                run_job(analysis_input, job, CHECKPOINT, device,
                        source_name=source_name, clip_info=clip_info)
                st.session_state["dashboard_job"] = str(job)
                st.session_state["dashboard_source"] = source_name
            st.success("Đã hoàn thành phân tích.")
        except Exception as exc:
            st.error(f"Không xử lý được video: {exc}")
    job = st.session_state.get("dashboard_job")
    current_name = (uploaded.name if uploaded is not None else Path(local_source).name
                    if local_source else None)
    if job and current_name and current_name != st.session_state.get("dashboard_source"):
        st.info("Kết quả đang lưu thuộc video trước. Hãy bấm Phân tích video để tạo báo cáo cho tệp mới.")
    elif job:
        st.divider()
        st.caption(f"Kết quả lần chạy gần nhất: {st.session_state.get('dashboard_source', 'video')}.")
        _show_job(job)


def main():
    st.set_page_config(page_title="Football Analytics", layout="wide")
    st.title("Football Analytics")
    st.caption("Dự án phát triển từ alina-letzien/football-analytics (MIT); "
               "dataset detection-player v4 của FSDM (CC BY 4.0).")
    _video_tab()


if __name__ == "__main__":
    main()
