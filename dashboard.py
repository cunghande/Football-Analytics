"""Local Streamlit interface for the existing football analysis pipeline."""

import json
from pathlib import Path

import streamlit as st

from src.dashboard_io import (
    MAX_EXTENDED_FRAMES, MAX_FRAMES, MAX_UPLOAD_BYTES, analysis_files, create_academic_zip,
    create_job, create_job_from_local_file, create_report_zip, prepare_analysis_clip,
    probe_video, read_csv, run_job,
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
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return {f"GPU 0 — {name} (khuyến nghị)": "0", "CPU": "cpu"}, None
        return {"CPU": "cpu"}, None
    except (ImportError, OSError) as exc:
        return {"CPU": "cpu"}, f"PyTorch không tải được trong Python hiện tại: {exc}"


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
        if role_method.get("method") == "rare_yellow_kit_temporal_vote":
            st.caption("Vai trò cầu thủ/trọng tài được ổn định theo áo vàng, quần tối hiếm và 5 quan sát gần nhau; "
                       "đây vẫn là ước đoán, chưa đối chiếu nhãn thủ công.")
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
    st.image(str(files["chart"]), caption="Gán bóng theo frame và mức quan sát track; quãng đường chỉ hiện khi đã hiệu chỉnh sân.")
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
        "possession_frames", "pass_candidate_sent", "pass_candidate_received",
    ]
    st.dataframe([{key: row.get(key, "") for key in columns} for row in players], width="stretch")
    st.caption("observed_path_px là dịch chuyển trên ảnh, chưa phải quãng đường chạy thực tế. "
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
    available = {label: path for label, path in CHECKPOINTS.items() if path.is_file()}
    if not available:
        st.error("Chưa tìm thấy checkpoint. Hãy tải best.pt từ kết quả Modal vào runs/modal/ hoặc đặt yolo26x.pt tại thư mục gốc.")
        return
    checkpoint_name = st.selectbox("Checkpoint", list(available))
    if checkpoint_name == "Model gốc COCO":
        st.warning("Model gốc không dự đoán riêng goalkeeper/referee; báo cáo vai trò sẽ thiếu.")
    devices, torch_error = _device_options()
    device_label = st.selectbox("Thiết bị suy luận", list(devices))
    device = devices[device_label]
    st.caption("YOLO chạy trên GPU đã chọn; đọc video, theo dõi, thống kê và tạo báo cáo vẫn sử dụng CPU."
               if device == "0" else "YOLO và các bước xử lý còn lại chạy trên CPU.")
    if len(devices) == 1 and not torch_error:
        st.caption("Python hiện tại không nhận GPU CUDA; dashboard dùng CPU. GPU Modal chỉ dùng lúc huấn luyện.")
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
    max_frames = st.selectbox("Độ dài đoạn phân tích",
                              [MAX_FRAMES, MAX_EXTENDED_FRAMES],
                              format_func=lambda count: (f"{count} frame - tối đa 720p"
                                                         if count == MAX_FRAMES else
                                                         f"{count} frame - tự giảm độ phân giải"))
    st.caption("Chế độ 900 frame dùng độ phân giải thấp hơn để giữ lượng ảnh trong RAM gần chế độ 450 frame. "
               "Các bước theo dõi và tạo báo cáo có thể mất lâu hơn.")
    start_seconds = st.number_input("Bắt đầu phân tích từ giây thứ", min_value=0.0,
                                    value=0.0, step=1.0)
    technical_only = st.checkbox("Clip ghép từ ảnh hoặc FPS giả định - chỉ kiểm tra kỹ thuật")
    role_refinement_enabled = st.checkbox(
        "Sửa nhãn trọng tài áo vàng/quần tối khi màu này hiếm (thử nghiệm)", value=True)
    st.caption("Quy tắc chỉ bật khi bộ áo vàng/quần tối chiếm phần nhỏ trong clip và model có bằng chứng trọng tài. "
               "Bỏ chọn nếu trọng tài không mặc vàng hoặc bạn muốn dùng nguyên nhãn model.")
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
                    max_frames=max_frames)
                details = probe_video(analysis_input, enforce_limits=False)
                if clip_info:
                    st.info(f"Chỉ phân tích đoạn {clip_info['start_seconds']:.1f}–"
                            f"{clip_info['end_seconds']:.1f} giây; "
                            f"{details['frames']} frame, {details['width']}×{details['height']} pixel. "
                            "Các chỉ số trong báo cáo thuộc đoạn này, không thuộc cả video gốc.")
                if technical_only:
                    (job / "dataset_demo_protocol.json").write_text(json.dumps({
                        "purpose": "technical_demo_only",
                        "warning": "FPS hoặc thứ tự thời gian chưa được xác minh; không dùng làm số liệu buổi tập.",
                    }, ensure_ascii=False, indent=2), encoding="utf-8")
                st.write(f"Đầu vào: {details['frames']} frame, {details['fps']:.2f} FPS, "
                         f"{details['width']}×{details['height']} pixel.")
                run_job(analysis_input, job, available[checkpoint_name], device,
                        source_name=source_name, clip_info=clip_info,
                        role_refinement_enabled=role_refinement_enabled)
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
