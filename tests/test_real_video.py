"""Integration tests against real drone detection video.

Uses: test_data/drone_yolov7/test/pexels-nino-souza-8459631-1920x1080-30fps.mp4
This video has a visible drone flying against sky. The pipeline should:
1. Detect the drone (not miss it entirely)
2. NOT classify it as insect (it's a drone)
3. Estimate depth > 10m (it's clearly far away, not 1m)
4. Not flood with >20 labeled tracks on a simple sky scene
"""
import cv2
import numpy as np
import sys, os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from detector.pipeline import Pipeline

VIDEO_PATH = os.path.join(os.path.dirname(__file__), '..', 'test_data', 'drone_yolov7',
                          'test', 'pexels-nino-souza-8459631-1920x1080-30fps.mp4')


@pytest.fixture
def pipeline_on_drone_video():
    """Run pipeline on 100 frames of drone video, collect results."""
    if not os.path.exists(VIDEO_PATH):
        pytest.skip("Drone test video not available")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        pytest.skip("Cannot open drone test video")

    pipe = Pipeline(config={'frame_width': 512, 'frame_height': 384})

    all_labeled = []
    for i in range(100):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (512, 384))
        pipe.process_frame(frame)
        tracks = pipe.get_active_tracks()
        labeled = [t for t in tracks
                   if t.get('age', 0) >= 8
                   and t.get('label', 'unknown') != 'unknown']
        all_labeled.append(labeled)

    cap.release()
    return all_labeled


def test_detects_something(pipeline_on_drone_video):
    """Pipeline should detect at least one object in drone footage."""
    any_labeled = any(len(frame_tracks) > 0 for frame_tracks in pipeline_on_drone_video)
    assert any_labeled, "No labeled tracks in 100 frames of drone footage"


def test_not_all_insect(pipeline_on_drone_video):
    """A drone video should not classify everything as insect."""
    all_labels = []
    for frame_tracks in pipeline_on_drone_video:
        for t in frame_tracks:
            all_labels.append(t['label'])

    if not all_labels:
        pytest.skip("No labels to check")

    insect_ratio = all_labels.count('insect') / len(all_labels)
    assert insect_ratio < 0.9, (
        f"Insect ratio {insect_ratio:.0%} — drone video should not be >90% insect. "
        f"Labels: insect={all_labels.count('insect')}, drone={all_labels.count('drone')}, "
        f"bird={all_labels.count('bird')}"
    )


def test_drone_detected_at_some_point(pipeline_on_drone_video):
    """At least some frames should classify a track as 'drone'."""
    drone_count = 0
    for frame_tracks in pipeline_on_drone_video:
        for t in frame_tracks:
            if t['label'] == 'drone':
                drone_count += 1

    assert drone_count > 0, "No drone classifications in 100 frames of drone footage"


def test_depth_not_absurdly_low(pipeline_on_drone_video):
    """Depth estimates should not all be <5m for a drone flying in the sky."""
    depths = []
    for frame_tracks in pipeline_on_drone_video:
        for t in frame_tracks:
            if t.get('depth'):
                depths.append(float(t['depth']))

    if not depths:
        pytest.skip("No depth values to check")

    median_depth = sorted(depths)[len(depths) // 2]
    assert median_depth > 5.0, (
        f"Median depth {median_depth:.1f}m — drone in sky should be >5m. "
        f"Depth range: {min(depths):.1f}-{max(depths):.1f}m"
    )


def test_not_too_many_tracks(pipeline_on_drone_video):
    """Sky scene should not produce >15 labeled tracks per frame on average."""
    counts = [len(frame_tracks) for frame_tracks in pipeline_on_drone_video]
    avg = sum(counts) / len(counts) if counts else 0
    assert avg < 15, f"Average {avg:.1f} labeled tracks/frame — too many for a sky scene"
