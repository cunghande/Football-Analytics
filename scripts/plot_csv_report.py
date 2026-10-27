"""Render report-ready charts after a real video has produced analysis CSV files."""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_csv_report(stem):
    people = _read(stem + "_players.csv")
    teams = _read(stem + "_teams.csv")
    labels = {"1": "Team 1", "2": "Team 2", "0": "Unknown"}
    team_by_id = {row["team_id"]: row for row in teams}
    for team_id in labels:
        if team_id not in team_by_id:
            raise ValueError(f"Missing team {team_id} in CSV")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    ids = ["1", "2", "0"]
    colors = ["#2877b8", "#d35247", "#a0a0a0"]
    possession = [float(team_by_id[i]["possession_pct_video"]) for i in ids]
    axes[0].bar([labels[i] for i in ids], possession, color=colors)
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Percent of all video frames (%)")
    axes[0].set_title("Ball assignment by kit cluster")
    for x, value in enumerate(possession):
        axes[0].text(x, value + 1.2, f"{value:.1f}%", ha="center", fontsize=9)

    ranked = sorted(
        (row for row in people if row["distance_m_estimate"] != ""),
        key=lambda row: float(row["distance_m_estimate"]), reverse=True,
    )[:10]
    if ranked:
        names = [f"{row['role']} {row['track_id']}" for row in ranked][::-1]
        distances = [float(row["distance_m_estimate"]) for row in ranked][::-1]
        axes[1].barh(names, distances, color="#478c70")
        axes[1].set_xlabel("Estimated distance (m)")
    else:
        axes[1].text(0.5, 0.5, "No measured tracks", ha="center", va="center",
                     transform=axes[1].transAxes)
        axes[1].set_xticks([])
        axes[1].set_yticks([])
    axes[1].set_title("Top tracked people by estimated distance")
    fig.suptitle("Descriptive video statistics (not ground-truth performance)", fontsize=11)
    paths = [stem + "_report_chart.png", stem + "_report_chart.pdf"]
    for path in paths:
        fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stem", help="Output/video stem, e.g. output/analysis")
    args = parser.parse_args()
    for filename in plot_csv_report(args.stem):
        print(os.path.abspath(filename))
