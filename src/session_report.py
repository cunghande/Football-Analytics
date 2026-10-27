"""Create a descriptive, printable coaching report from one analysis job.

Track IDs are temporary identifiers. This module intentionally does not score,
rank, or infer the identity of a player.
"""

import csv
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

from matplotlib import get_data_path
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)


def _load_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _number(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Non-finite analysis metric")
    return number


def _fmt(value, digits=1):
    return f"{_number(value):,.{digits}f}".replace(",", " ")


def _font_name():
    font_name = "FootballReportDejaVu"
    if font_name not in pdfmetrics.getRegisteredFontNames():
        font_path = Path(get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
        pdfmetrics.registerFont(TTFont(font_name, str(font_path)))
    return font_name


def _table(rows, widths, font, *, header=True):
    table = Table(rows, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    rules = [
        ("FONTNAME", (0, 0), (-1, -1), font),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#9CA3AF")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F7FA")]),
    ]
    if header:
        rules.extend([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EDF2")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#17212B")),
        ])
    table.setStyle(TableStyle(rules))
    return table


def _footer(canvas, document):
    canvas.saveState()
    canvas.setFont(_font_name(), 8)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.drawString(17 * mm, 12 * mm, "Báo cáo mô tả từ video - track ID chưa xác minh danh tính")
    canvas.drawRightString(A4[0] - 17 * mm, 12 * mm, f"Trang {document.page}")
    canvas.restoreState()


def generate_session_report(stem, destination=None, source_name=None):
    """Render PDF from CSV/metadata produced by FootballAnalyzer.

    Raises on missing or inconsistent inputs, so the dashboard cannot silently
    offer a plausible-looking partial report.
    """
    stem = Path(stem)
    base = str(stem)
    destination = Path(destination or base + "_session_report.pdf")
    players = _load_csv(base + "_players.csv")
    teams = _load_csv(base + "_teams.csv")
    metadata = _load_json(base + "_csv_metadata.json")
    summary = _load_json(base + "_summary.json")
    run_path = stem.parent / "dashboard_run.json"
    run = _load_json(run_path) if run_path.exists() else {}
    demo_path = stem.parent / "dataset_demo_protocol.json"
    technical_demo = demo_path.exists()

    frames = int(metadata["frames"])
    fps = _number(metadata["fps"])
    if frames <= 0 or fps <= 0 or int(summary["frames"]) != frames:
        raise ValueError("Invalid or inconsistent video frame metadata")
    if {str(row["team_id"]) for row in teams} != {"0", "1", "2"}:
        raise ValueError("Team CSV must contain team IDs 0, 1 and 2")
    if sum(int(row["possession_frames"]) for row in teams) != frames:
        raise ValueError("Team frame totals do not match video frames")
    calibrated = metadata.get("coordinate_method") == "four_point_field_homography"
    font = _font_name()
    styles = getSampleStyleSheet()
    title = ParagraphStyle("ReportTitle", parent=styles["Title"], fontName=font,
                           fontSize=16, leading=21, textColor=colors.HexColor("#17212B"),
                           alignment=TA_LEFT, spaceAfter=8)
    heading = ParagraphStyle("ReportHeading", parent=styles["Heading2"], fontName=font,
                             fontSize=11, leading=15, textColor=colors.HexColor("#17212B"),
                             spaceBefore=13, spaceAfter=6, keepWithNext=True)
    body = ParagraphStyle("ReportBody", parent=styles["BodyText"], fontName=font,
                          fontSize=8.5, leading=13, textColor=colors.HexColor("#273340"),
                          spaceAfter=5)
    muted = ParagraphStyle("ReportMuted", parent=body, fontSize=7.5, leading=11,
                           textColor=colors.HexColor("#59687A"))
    center = ParagraphStyle("ReportCenter", parent=muted, alignment=TA_CENTER)

    story = [Paragraph("BÁO CÁO BUỔI TẬP", title),
             Paragraph("Số liệu mô tả từ video; không chấm điểm hoặc xếp loại cầu thủ.", body)]
    if technical_demo:
        story.append(Paragraph(
            "<b>CLIP KIỂM TRA KỸ THUẬT:</b> ảnh tĩnh từ dataset được ghép với FPS giả định. "
            "Không diễn giải thời gian, tốc độ, quãng đường, gán bóng hoặc đường chuyền "
            "như số liệu một buổi tập thật.", body))
    source = Path(source_name or run.get("source_name") or "video đầu vào").name
    model_name = Path(run.get("model_path") or "không ghi nhận").name
    facts = [
        ["Video", Paragraph(escape(source), body)],
        ["Model", Paragraph(escape(model_name), body)],
        ["Số frame", str(frames)],
        ["FPS", _fmt(fps, 2) + (" (giả định)" if technical_demo else "")],
        ["Thời lượng", "Không xác định" if technical_demo else _fmt(frames / fps, 1) + " giây"],
        ["Tọa độ", "Sân đã hiệu chỉnh thủ công" if calibrated else "Ảnh, chưa hiệu chỉnh sân"],
    ]
    clip = run.get("clip")
    if clip:
        facts.insert(1, ["Đoạn phân tích",
                         f"{_fmt(clip['start_seconds'], 1)}–{_fmt(clip['end_seconds'], 1)} giây "
                         "của video gốc; các chỉ số chỉ thuộc đoạn này"])
    story.extend([Paragraph("1. Thông tin đầu vào", heading),
                  _table(facts, [95, 400], font, header=False)])

    ball_rate = 100 * _number(summary.get("ball_detection_rate", 0))
    unknown = next(row for row in teams if str(row["team_id"]) == "0")
    story.extend([
        Paragraph("2. Mức độ quan sát", heading),
        Paragraph(
            f"Có <b>{sum(row['role'] == 'player' for row in players)} track cầu thủ</b> "
            f"và <b>{sum(row['role'] == 'goalkeeper' for row in players)} track thủ môn</b>. "
            "Một người có thể sinh nhiều track ID sau khi mất dấu. "
            f"Hệ thống phát hiện bóng ở <b>{_fmt(ball_rate)}%</b> frame; "
            f"<b>{int(unknown['possession_frames'])}/{frames}</b> frame không gán bóng "
            "cho một trong hai đội.", body),
        ])

    player_data = [["Track ID", "Vai trò", "Đội", "Frame thấy", "Gán bóng\n(frame)"]]
    for row in sorted(players, key=lambda r: (r["role"], int(r["track_id"]))):
        prefix = "GK" if row["role"] == "goalkeeper" else "P"
        team = int(row["team_id"])
        player_data.append([
            prefix + str(row["track_id"]),
            "Thủ môn" if prefix == "GK" else "Cầu thủ",
            "Chưa rõ" if team == 0 else f"Đội {team}",
            str(int(row["observed_frames"])),
            str(int(row["possession_frames"])),
        ])
    if calibrated and not technical_demo and len(player_data) > 1:
        story.append(Paragraph("3. Chuyển động ước tính", heading))
        movement = [["Track ID", "Quãng đường (m)", "Tốc độ TB (km/h)", "Tốc độ cao nhất (km/h)"]]
        for row in sorted(players, key=lambda r: (r["role"], int(r["track_id"]))):
            prefix = "GK" if row["role"] == "goalkeeper" else "P"
            movement.append([
                prefix + str(row["track_id"]),
                _fmt(row["distance_m_estimate"]) if row["distance_m_estimate"] else "-",
                _fmt(row["mean_speed_kmh_estimate"]) if row["mean_speed_kmh_estimate"] else "-",
                _fmt(row["max_speed_kmh_estimate"]) if row["max_speed_kmh_estimate"] else "-",
            ])
        story.append(_table(movement, [80, 125, 145, 145], font))
        story.append(Paragraph(
            "Ước tính từ đường đi quan sát được sau hiệu chỉnh thủ công; không nội suy "
            "quãng đường lúc mất dấu. Cần đối chiếu thiết bị tham chiếu trước khi dùng "
            "cho quyết định thể lực.", muted))
    else:
        story.append(Paragraph("3. Chuyển động", heading))
        story.append(Paragraph(
            "Không hiển thị quãng đường và tốc độ trong PDF vì video chưa có hiệu chỉnh "
            "sân đáng tin cậy hoặc chỉ là clip kiểm tra kỹ thuật. CSV thô vẫn giữ các "
            "giá trị theo giả định để kiểm tra công thức; không dùng chúng như số đo thực.", body))

    story.append(Paragraph("4. Tổng hợp theo đội", heading))
    team_data = [["Nhóm màu áo", "Số track", "Frame gán bóng", "% tổng frame"]]
    for row in sorted(teams, key=lambda r: (int(r["team_id"]) == 0, int(r["team_id"]))):
        team = int(row["team_id"])
        team_data.append([
            "Chưa rõ" if team == 0 else f"Đội {team}",
            str(int(row["tracked_people"])),
            str(int(row["possession_frames"])),
            _fmt(row["possession_pct_video"]) + "%",
        ])
    story.append(_table(team_data, [125, 100, 130, 140], font))
    story.append(Paragraph(
        "Đội 1 và đội 2 là ID cụm áo, chưa phải tên đội. Nhóm chưa rõ gồm frame "
        "không gán bóng; số track không phải số người duy nhất.", muted))

    heatmap = Path(base + "_heatmaps.png")
    heatmap_meta = Path(base + "_heatmaps_metadata.json")
    if frames >= 10 and heatmap.exists() and heatmap_meta.exists():
        mode = _load_json(heatmap_meta).get("coordinate_mode")
        image = Image(str(heatmap), width=172 * mm, height=54 * mm, kind="proportional")
        caption = ("Tọa độ sân đã hiệu chỉnh" if mode == "calibrated_field"
                   else "Tọa độ ảnh chưa hiệu chỉnh")
        story.append(KeepTogether([
            Paragraph("5. Mật độ vị trí theo đội", heading), image,
            Paragraph(f"Mỗi quan sát cầu thủ ở một frame đóng góp một lần. {caption}. "
                      "Hình tổng hợp theo đội, không phải heatmap riêng của từng người.", center),
        ]))
    else:
        story.append(Paragraph("5. Mật độ vị trí theo đội", heading))
        reason = ("Clip dưới 10 frame nên hình mật độ không có nhiều ý nghĩa."
                  if frames < 10 else "Chưa có hình mật độ vị trí cho video này.")
        story.append(Paragraph(reason, body))

    story.append(Paragraph("6. Số liệu theo track", heading))
    if len(player_data) == 1:
        story.append(Paragraph("Không có track cầu thủ hoặc thủ môn trong video.", body))
    else:
        story.append(_table(player_data, [76, 89, 95, 100, 135], font))
    story.append(Paragraph(
        "Frame thấy là số frame track xuất hiện trong vùng camera, không phải thời gian thi đấu. "
        "Gán bóng là quy tắc gần bóng theo ảnh, không phải nhãn kiểm soát bóng đã xác minh. "
        "Track ID chưa được nối thành danh tính cầu thủ.", muted))

    if frames >= 10:
        story.append(Paragraph("7. Cách đọc báo cáo", heading))
        story.append(Paragraph(
            "Báo cáo không nhận dạng họ tên hoặc số áo. Track ID có thể đổi khi cầu thủ bị che "
            "hoặc rời khung hình. Số liệu bóng phụ thuộc phát hiện bóng và quy tắc gán gần nhất. "
            "Không có điểm số hay nhận xét chất lượng cầu thủ. File CSV và JSON trong ZIP "
            "lưu số gốc, FPS và giả định tính toán để kiểm tra lại.", body))

    destination.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(str(destination), pagesize=A4, leftMargin=17 * mm,
                                 rightMargin=17 * mm, topMargin=16 * mm,
                                 bottomMargin=19 * mm, title="Báo cáo buổi tập",
                                 author="Football Analytics")
    document.build(story, onFirstPage=_footer, onLaterPages=_footer)
    if not destination.exists() or destination.stat().st_size < 1000:
        raise RuntimeError("Không tạo được PDF báo cáo buổi tập")
    return destination
