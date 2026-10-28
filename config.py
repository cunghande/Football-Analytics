"""
Configuration file for football analysis system
"""

import os

# YOLO Configuration
YOLO_CONFIG = {
    "model_path": os.environ.get("YOLO_MODEL_PATH", "yolo26x.pt"),
    "confidence_threshold": 0.5,
    "ball_confidence_threshold": 0.25,
    "iou_threshold": 0.45,
    # "cpu", "mps" (Apple Silicon), or GPU index like "0"; override with YOLO_DEVICE env var (used in CI, which has no mps)
    "device": os.environ.get("YOLO_DEVICE", "cpu"),  # "cpu", "0" (NVIDIA GPU), or "mps" (Apple Silicon)
}

# Tracker Configuration (ByteTrack)
TRACKER_CONFIG = {
    "track_activation_threshold": 0.25,
    "lost_track_buffer": 30,        # Frames to keep a lost track alive
    "minimum_matching_threshold": 0.8,
    "frame_rate": 30,
    "max_ball_interpolation_gap": 15,  # Frames; longer gaps between ball detections are left as no ball position instead of guessed
}

# Team Assignment Configuration
TEAM_CONFIG = {
    "n_teams": 2,                     # Number of teams to identify
    "kmeans_random_state": 42,
    "color_sample_frames": 10,        # Number of early frames to aggregate jersey colors from
    "min_confidence_threshold": 0.15, # Relative margin below which assignment is UNKNOWN (team 0)
    "recluster_interval": 30,         # Refit every 30 frames; set 0 for stable kits (higher score on the reviewed clip)
    "recluster_window": 90,           # Use a recent 90-frame window for each refit
    "team_vote_window": 7,            # Smooth each tracked player's recent confident votes
    "suppress_pitch_green": True,      # Disable when a field team wears green jerseys
    # Override the rendering color for a team (BGR tuple). Leave empty — KMeans picks colors
    # automatically from jersey crops and works well when kits are visually distinct.
    #
    # Use this when:
    #   1. The team label is swapped (T1 draws yellow boxes on white-kit players) — run once
    #      without overrides to see which cluster ID maps to which team, then remap.
    #   2. You want a more vivid display color than the dull average KMeans found.
    #
    # Example — white kits for T1, yellow kits for T2 (BGR):
    #   "team_color_overrides": {1: (230, 230, 230), 2: (0, 215, 255)}
    #
    # Note: overrides only affect bounding-box rendering color, not which cluster a player
    # is assigned to. For genuinely similar kits where players are being misclassified,
    # raise min_confidence_threshold (more players get UNKNOWN instead of a wrong team)
    # or raise color_sample_frames to give KMeans more data to work with.
    "team_color_overrides": {},
}

# Ball Possession Configuration
POSSESSION_CONFIG = {
    "max_player_ball_distance": 70,  # Max pixel distance to assign ball to a player
}

# Candidate passes are deliberately conservative and are not verified events.
PASS_CONFIG = {
    "min_stable_frames": 2,
    "max_gap_seconds": 1.5,
    "min_distance_norm": 0.03,
    "max_distance_norm": 0.75,
}

# Perspective Transformation Configuration
FIELD_CONFIG = {
    "field_width": 120.0,   # Meters
    "field_height": 80.0,   # Meters
    # Optional manual calibration: {"src": [(x,y), ... 4 image points],
    # "dst": [(x_m,y_m), ... 4 matching field points]}. No automatic calibration.
    "calibration_points": None,
}

# Speed and Distance Configuration
SPEED_CONFIG = {
    "fps": 30.0,            # Frames per second of video
    "meters_per_pixel": 0.1,  # Conversion factor (requires calibration)
    "frame_window": 5,      # Frames over which speed is averaged
}

# Camera Motion Configuration
CAMERA_CONFIG = {
    "minimum_distance": 0.15,   # Ignore only near-zero global optical flow
    # Farnebäck optical flow parameters
    "farneback_pyr_scale": 0.5,
    "farneback_levels": 3,
    "farneback_winsize": 15,
    "farneback_iterations": 3,
    "farneback_poly_n": 5,
    "farneback_poly_sigma": 1.2,
}

# Video Processing Configuration
VIDEO_CONFIG = {
    "input_path": "input/DFL-Scoutingfeed.mp4",
    "output_path": "output/analysis.mp4",
    "output_codec": "mp4v",  # Options: "mp4v", "XVID", "MJPG"
    "read_from_stub": True,
    "stub_dir": "stubs",
}

# Visualization Colors (BGR format)
COLORS = {
    "player": (255, 0, 0),      # Blue (fallback when team color is unknown)
    "ball": (0, 255, 0),        # Green
    "referee": (0, 255, 255),   # Yellow
    "goalkeeper": (0, 165, 255), # Orange
}
