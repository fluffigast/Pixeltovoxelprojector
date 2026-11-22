#!/usr/bin/env python3
"""Ultimate insect detection optimized for flying insects in outdoor environments."""

import cv2
import numpy as np
import time
import subprocess
import argparse
import math
from collections import deque, defaultdict
from typing import List, Tuple, Dict, Optional

class UltimateInsectDetector:
    """Flying object detection using ego-motion-compensated frame differencing.

    Single-pass detection on downsampled frames. Estimates camera motion via
    sparse optical flow, warps previous frame to compensate, then detects
    only independently-moving objects in the residual.

    Inspired by TrackCraft3r: work in compressed space, single forward pass.
    """

    def __init__(self):
        # Detection parameters
        self.min_area = 2
        self.max_area = 500
        self.min_perimeter = 4
        self.motion_threshold = 20

        # Downsampling factor for detection (work at 1/4 resolution)
        self.detect_scale = 0.25

        # Frame buffer for differencing
        self.prev_gray_full = None      # full-res for ego-motion estimation
        self.prev_gray_small = None     # downsampled for detection
        self.processed_frames = 0

        # Tracking with prediction
        self.tracks = {}
        self.next_track_id = 0
        self.max_tracks = 50

        # Insect behavior patterns
        self.min_speed = 1.5
        self.max_speed = 80.0
        self.erratic_threshold = 25
        self.hover_frames = 8

        # Adaptive thresholding
        self.adaptive_enabled = True
        self.recent_detection_counts = deque(maxlen=60)
        self.target_detections_per_second = 5

        # Ego-motion state
        self.camera_moving = False
        self.warp_matrix = None

    def detect_flying_insects(self, frame: np.ndarray) -> List[Dict]:
        """Single-pass compensated detection."""
        self.processed_frames += 1

        # Full-res grayscale for ego-motion estimation
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Downsampled for detection (compressed space)
        h, w = frame.shape[:2]
        small_w = int(w * self.detect_scale)
        small_h = int(h * self.detect_scale)
        gray_small = cv2.resize(gray_full, (small_w, small_h))
        gray_small = cv2.GaussianBlur(gray_small, (3, 3), 0)

        detections = []

        if self.prev_gray_full is not None and self.prev_gray_small is not None:
            # Estimate ego-motion from full-res (more features)
            self.warp_matrix = self._estimate_ego_motion(self.prev_gray_full, gray_full)

            # Compensate and detect on downsampled (fast)
            detections = self._detect_compensated(gray_small, self.prev_gray_small,
                                                   small_w, small_h, w, h)

        self.prev_gray_full = gray_full
        self.prev_gray_small = gray_small

        # Quality filter
        detections = self._filter_detections(detections, frame)

        if self.adaptive_enabled:
            self._adapt_sensitivity(len(detections))

        return detections

    def _estimate_ego_motion(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> Optional[np.ndarray]:
        """Estimate camera motion as 2x3 affine via sparse optical flow."""
        features = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01,
                                           minDistance=30, blockSize=7)
        if features is None or len(features) < 8:
            self.camera_moving = False
            return None

        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, features, None,
            winSize=(21, 21), maxLevel=3)

        if new_pts is None:
            self.camera_moving = False
            return None

        good = status.flatten() == 1
        if np.sum(good) < 6:
            self.camera_moving = False
            return None

        matrix, inliers = cv2.estimateAffinePartial2D(
            features[good], new_pts[good],
            method=cv2.RANSAC, ransacReprojThreshold=3.0)

        if matrix is None:
            self.camera_moving = False
            return None

        tx, ty = matrix[0, 2], matrix[1, 2]
        self.camera_moving = math.sqrt(tx*tx + ty*ty) > 1.5
        return matrix

    def _detect_compensated(self, curr_small: np.ndarray, prev_small: np.ndarray,
                             sw: int, sh: int, fw: int, fh: int) -> List[Dict]:
        """Ego-motion-compensated frame differencing on downsampled frames.

        Single method replaces the four redundant detection passes.
        """
        # Warp previous frame to align with current (compensate camera motion)
        if self.warp_matrix is not None:
            # Scale the affine matrix for the downsampled resolution
            scale_matrix = self.warp_matrix.copy()
            scale_matrix[0, 2] *= self.detect_scale
            scale_matrix[1, 2] *= self.detect_scale
            aligned_prev = cv2.warpAffine(prev_small, scale_matrix, (sw, sh),
                                           borderMode=cv2.BORDER_REFLECT)
        else:
            aligned_prev = prev_small

        # Frame difference — only independently-moving objects survive
        diff = cv2.absdiff(curr_small, aligned_prev)

        # Adaptive threshold: mean + 2*std eliminates residual noise from
        # imperfect compensation (parallax, rolling shutter, etc.)
        mean_val = np.mean(diff)
        std_val = np.std(diff)
        threshold = max(self.motion_threshold, mean_val + 2.0 * std_val)
        _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

        # Morphology to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Find contours in downsampled space
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Scale detections back to full resolution
        scale_inv = 1.0 / self.detect_scale
        detections = []
        for contour in contours:
            if len(contour) < 3:
                continue

            area_small = cv2.contourArea(contour)
            area_full = area_small * scale_inv * scale_inv

            if area_full < self.min_area or area_full > self.max_area:
                continue

            x, y, bw, bh = cv2.boundingRect(contour)
            # Scale bbox to full resolution
            fx = int(x * scale_inv)
            fy = int(y * scale_inv)
            fbw = int(bw * scale_inv)
            fbh = int(bh * scale_inv)
            center_x = fx + fbw // 2
            center_y = fy + fbh // 2

            perimeter = cv2.arcLength(contour, True) * scale_inv
            if perimeter < self.min_perimeter:
                continue

            aspect_ratio = fbw / fbh if fbh > 0 else 0
            if aspect_ratio < 0.1 or aspect_ratio > 10.0:
                continue

            compactness = 4 * np.pi * area_full / (perimeter * perimeter) if perimeter > 0 else 0

            detections.append({
                'center': (center_x, center_y),
                'bbox': (fx, fy, fbw, fbh),
                'area': area_full,
                'perimeter': perimeter,
                'aspect_ratio': aspect_ratio,
                'compactness': compactness,
                'method': 'compensated',
                'timestamp': self.processed_frames
            })

        # Cap detections
        if len(detections) > 30:
            detections.sort(key=lambda d: d['area'], reverse=True)
            detections = detections[:30]

        return detections

    def _filter_detections(self, detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        """Quality filter on full-resolution frame."""
        h, w = frame.shape[:2]
        filtered = []
        for det in detections:
            cx, cy = det['center']
            if cx < 5 or cy < 5 or cx >= w - 5 or cy >= h - 5:
                continue
            if det['compactness'] < 0.05 or det['compactness'] > 1.2:
                continue

            # Contrast check on full-res ROI
            x, y, bw, bh = det['bbox']
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w, x + bw), min(h, y + bh)
            if x2 <= x1 or y2 <= y1:
                continue
            roi = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
            if roi.size == 0 or np.ptp(roi) < 10:
                continue

            filtered.append(det)
        return filtered

    def _adapt_sensitivity(self, current_detections: int):
        """Adapt detection sensitivity based on recent performance."""
        self.recent_detection_counts.append(current_detections)

        if len(self.recent_detection_counts) >= 30:
            recent_avg = np.mean(list(self.recent_detection_counts)[-30:])

            if recent_avg > self.target_detections_per_second * 2:
                self.motion_threshold = min(50, self.motion_threshold + 2)
                self.min_area = min(20, self.min_area + 1)

            elif recent_avg < self.target_detections_per_second * 0.3:
                self.motion_threshold = max(8, self.motion_threshold - 1)
                self.min_area = max(2, self.min_area - 1)
    
    def update_tracks_with_prediction(self, detections: List[Dict]) -> List[Dict]:
        """Prediction-first tracking.

        For each track, predict where it should be (from velocity), then search
        only near that prediction. Like temporal RoPE alignment: tell the matcher
        where to look before it searches.
        """
        updated_tracks = {}
        assigned_detections = set()

        # Build predictions for all tracks
        predictions = {}
        for track_id, track in self.tracks.items():
            positions = track['positions']
            if len(positions) >= 4:
                # Quadratic fit on last 6 points
                recent = positions[-min(6, len(positions)):]
                t = np.arange(len(recent), dtype=np.float64)
                px = np.polyfit(t, [p[0] for p in recent], deg=2)
                py = np.polyfit(t, [p[1] for p in recent], deg=2)
                next_t = float(len(recent))
                predictions[track_id] = (np.polyval(px, next_t), np.polyval(py, next_t))
            elif len(positions) >= 2:
                last, prev = positions[-1], positions[-2]
                predictions[track_id] = (2*last[0] - prev[0], 2*last[1] - prev[1])
            else:
                predictions[track_id] = track['last_center']

        # Score all (track, detection) pairs — only near predictions
        scores = []
        for track_id, pred in predictions.items():
            track = self.tracks[track_id]
            search_radius = min(50, max(15, track.get('speed', 5) * 3))
            for i, det in enumerate(detections):
                cx, cy = det['center']
                dist = math.sqrt((cx - pred[0])**2 + (cy - pred[1])**2)
                if dist < search_radius:
                    score = 1.0 - dist / search_radius
                    scores.append((score, track_id, i))

        # Greedy assign highest scores first
        scores.sort(reverse=True)
        assigned_tracks = set()
        for score, track_id, det_idx in scores:
            if track_id in assigned_tracks or det_idx in assigned_detections:
                continue
            track = self.tracks[track_id]
            det = detections[det_idx]
            track['positions'].append(det['center'])
            track['last_center'] = det['center']
            track['lost_frames'] = 0
            track['detection'] = det
            track['age'] += 1
            self._analyze_insect_behavior(track)
            updated_tracks[track_id] = track
            assigned_tracks.add(track_id)
            assigned_detections.add(det_idx)

        # Unmatched tracks: keep briefly with prediction
        for track_id, track in self.tracks.items():
            if track_id in assigned_tracks:
                continue
            track['lost_frames'] += 1
            if track['lost_frames'] < 15:  # Longer persistence via prediction
                pred = predictions[track_id]
                track['positions'].append((pred[0], pred[1]))
                track['last_center'] = (pred[0], pred[1])
                updated_tracks[track_id] = track

        # New tracks for unmatched detections (under cap)
        for i, det in enumerate(detections):
            if i not in assigned_detections and len(updated_tracks) < self.max_tracks:
                new_track = {
                    'id': self.next_track_id,
                    'positions': [det['center']],
                    'last_center': det['center'],
                    'lost_frames': 0,
                    'detection': det,
                    'age': 1,
                    'speed': 0,
                    'direction_changes': 0,
                    'is_hovering': False,
                    'is_fast': False,
                    'is_erratic': False,
                    'confidence': 0.4
                }
                updated_tracks[self.next_track_id] = new_track
                self.next_track_id += 1

        self.tracks = updated_tracks

        return [t for t in self.tracks.values()
                if t['lost_frames'] == 0 and t['confidence'] > 0.3]
    
    def _analyze_insect_behavior(self, track: Dict):
        """Analyze track for insect-like behavior patterns."""
        positions = track['positions']
        
        if len(positions) < 2:
            return
        
        # Calculate speed
        if len(positions) >= 2:
            recent_positions = positions[-min(8, len(positions)):]
            total_distance = 0
            
            for i in range(1, len(recent_positions)):
                dx = recent_positions[i][0] - recent_positions[i-1][0]
                dy = recent_positions[i][1] - recent_positions[i-1][1]
                total_distance += math.sqrt(dx*dx + dy*dy)
            
            track['speed'] = total_distance / (len(recent_positions) - 1) if len(recent_positions) > 1 else 0
        
        # Analyze direction changes (erratic flight)
        if len(positions) >= 4:
            recent_pos = positions[-min(6, len(positions)):]
            direction_changes = 0
            
            for i in range(2, len(recent_pos)):
                # Calculate direction vectors
                v1 = (recent_pos[i-1][0] - recent_pos[i-2][0], recent_pos[i-1][1] - recent_pos[i-2][1])
                v2 = (recent_pos[i][0] - recent_pos[i-1][0], recent_pos[i][1] - recent_pos[i-1][1])
                
                # Calculate angle between vectors
                if (v1[0] != 0 or v1[1] != 0) and (v2[0] != 0 or v2[1] != 0):
                    dot_product = v1[0]*v2[0] + v1[1]*v2[1]
                    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
                    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
                    
                    if mag1 > 0 and mag2 > 0:
                        cos_angle = dot_product / (mag1 * mag2)
                        cos_angle = max(-1, min(1, cos_angle))  # Clamp to valid range
                        angle_deg = math.degrees(math.acos(cos_angle))
                        
                        if angle_deg > self.erratic_threshold:
                            direction_changes += 1
            
            track['direction_changes'] = direction_changes
            track['is_erratic'] = direction_changes > 1
        
        # Detect hovering
        if len(positions) >= self.hover_frames:
            recent_pos = positions[-self.hover_frames:]
            center_x = sum(pos[0] for pos in recent_pos) / len(recent_pos)
            center_y = sum(pos[1] for pos in recent_pos) / len(recent_pos)
            
            max_distance = max(math.sqrt((pos[0] - center_x)**2 + (pos[1] - center_y)**2) 
                             for pos in recent_pos)
            track['is_hovering'] = max_distance < 6
        
        # Detect fast movement
        track['is_fast'] = track['speed'] > 20
        
        # Calculate confidence
        confidence = 0.4
        
        # Size preference (smaller more likely insects in outdoor video)
        area = track['detection']['area']
        if 1 <= area <= 15:
            confidence += 0.3
        elif 15 < area <= 40:
            confidence += 0.2
        elif 40 < area <= 80:
            confidence += 0.1
        
        # Speed bonus
        if self.min_speed <= track['speed'] <= self.max_speed:
            confidence += 0.2
        
        # Behavior bonuses
        if track['is_erratic']:
            confidence += 0.15
        if track['is_hovering']:
            confidence += 0.1
        if track['is_fast']:
            confidence += 0.1
        
        # Age bonus
        if track['age'] > 5:
            confidence += 0.1
        
        # Method bonus
        method = track['detection'].get('method', '')
        if 'merged' in method:
            confidence += 0.1
        
        track['confidence'] = min(1.0, confidence)


class UltimateInsectTracker:
    """Ultimate insect tracking with 3D voxel visualization."""
    
    def __init__(self, stream_url: str):
        self.stream_url = stream_url
        self.detector = UltimateInsectDetector()
        
        # Video parameters
        self.cap = None
        self.width = 1024  # Higher resolution
        self.height = 768
        
        # Ultra-high resolution voxel system
        self.voxel_dims = (150, 150, 75)
        self.voxel_size = 0.01  # Very fine voxels
        self.insect_voxels = np.zeros(self.voxel_dims, dtype=np.float32)
        self.voxel_colors = np.zeros((*self.voxel_dims, 3), dtype=np.float32)
        self.voxel_decay = 0.75  # Fast decay — voxels fade in ~10 frames
        
        # Camera parameters
        self.camera_fov = 80  # Wide field of view for outdoor scenes
        self.camera_z = 0.5
        
        # Display settings
        self.show_all_detections = False
        self.min_confidence_display = 0.5
        self.show_trails = True
        self.show_behavior = True
        self.show_stats = True
        
        # Performance
        self.frame_times = deque(maxlen=60)
        self.total_insects_detected = 0
        
    def get_stream_url(self):
        """Get high-quality stream URL."""
        if 'youtube.com' in self.stream_url or 'youtu.be' in self.stream_url:
            try:
                # Try for highest quality
                cmd = ['yt-dlp', '-f', 'best[height<=1080]', '--get-url', self.stream_url]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
                if result.returncode == 0:
                    return result.stdout.strip()
            except:
                pass
        return self.stream_url
        
    def start(self):
        """Start ultimate insect detection."""
        url = self.get_stream_url()
        print(f"🦟 Ultimate Insect Detection: {url}")
        
        self.cap = cv2.VideoCapture(url)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open stream")
        
        print("🚀 Ultimate Insect Detection System Started!")
        print("\n🎯 Optimized for:")
        print("  • Flying insects in outdoor environments")
        print("  • Ultra-sensitive detection (1-150 pixels)")
        print("  • 4-layer detection algorithms")
        print("  • Adaptive sensitivity")
        print("  • Behavioral analysis")
        print("\n🎮 Controls:")
        print("  q/ESC - Quit")
        print("  r - Reset")
        print("  a - Toggle all detections")
        print("  t - Toggle trails")
        print("  b - Toggle behavior labels")
        print("  s - Toggle statistics")
        print("  = - Lower confidence threshold")
        print("  - - Raise confidence threshold")
        print("  space - Pause/Resume")
        print("  h - Help")
        
        cv2.namedWindow('Ultimate Insect Detector', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Ultimate Insect Detector', 1400, 900)
        
        paused = False
        frame_count = 0
        
        while True:
            frame_start = time.time()
            
            if not paused:
                ret, frame = self.cap.read()
                if not ret:
                    print("End of stream")
                    break
                
                frame_count += 1
                
                # Resize maintaining aspect ratio
                frame = cv2.resize(frame, (self.width, self.height))
                
                # Ultimate detection
                detections = self.detector.detect_flying_insects(frame)
                
                # Advanced tracking
                active_tracks = self.detector.update_tracks_with_prediction(detections)
                
                # Update voxels
                self._update_ultimate_voxels(active_tracks)
                
                # Count insects
                confident_tracks = [t for t in active_tracks if t['confidence'] >= self.min_confidence_display]
                self.total_insects_detected += len(confident_tracks)
            
            # Create visualization
            vis_frame = self._create_ultimate_visualization(frame, detections if not paused else [], active_tracks if not paused else [])
            
            cv2.imshow('Ultimate Insect Detector', vis_frame)
            
            # Performance tracking
            if not paused:
                self.frame_times.append(time.time() - frame_start)
            
            # Handle controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord(' '):  # Space for pause
                paused = not paused
                print("PAUSED" if paused else "RESUMED")
            elif key == ord('r'):
                self._reset_ultimate_system()
            elif key == ord('a'):
                self.show_all_detections = not self.show_all_detections
                print(f"Show all detections: {self.show_all_detections}")
            elif key == ord('t'):
                self.show_trails = not self.show_trails
                print(f"Show trails: {self.show_trails}")
            elif key == ord('b'):
                self.show_behavior = not self.show_behavior
                print(f"Show behavior: {self.show_behavior}")
            elif key == ord('s'):
                self.show_stats = not self.show_stats
                print(f"Show statistics: {self.show_stats}")
            elif key == ord('=') or key == ord('+'):
                self.min_confidence_display = max(0.1, self.min_confidence_display - 0.1)
                print(f"Confidence threshold: {self.min_confidence_display:.1f}")
            elif key == ord('-'):
                self.min_confidence_display = min(0.9, self.min_confidence_display + 0.1)
                print(f"Confidence threshold: {self.min_confidence_display:.1f}")
            elif key == ord('h'):
                self._show_ultimate_help()
                
    def _update_ultimate_voxels(self, tracks: List[Dict]):
        """Update ultra-high resolution voxels."""
        # Apply decay
        self.insect_voxels *= self.voxel_decay
        self.voxel_colors *= self.voxel_decay
        
        for track in tracks:
            if track['confidence'] < 0.4:
                continue
                
            center_x, center_y = track['last_center']
            
            # Advanced depth estimation
            area = track['detection']['area']
            speed = track['speed']
            age = track['age']
            
            # Multi-factor depth estimation
            # Smaller objects assumed further away
            size_factor = max(0.2, min(2.0, 25.0 / (area + 2)))
            # Faster objects assumed closer
            speed_factor = max(0.5, min(1.5, (speed + 5) / 15.0))
            # Older tracks more reliable depth
            age_factor = min(1.2, 1.0 + age * 0.02)
            
            estimated_depth = size_factor * speed_factor * age_factor
            
            # Convert to world coordinates
            norm_x = (center_x / self.width - 0.5) * 2.0
            norm_y = (center_y / self.height - 0.5) * 2.0
            
            fov_scale = math.tan(math.radians(self.camera_fov / 2))
            world_x = norm_x * fov_scale * estimated_depth
            world_y = -norm_y * fov_scale * estimated_depth
            world_z = self.camera_z + estimated_depth
            
            # Convert to voxel indices
            vx = int(world_x / self.voxel_size + self.voxel_dims[0] / 2)
            vy = int(world_y / self.voxel_size + self.voxel_dims[1] / 2)
            vz = int(world_z / self.voxel_size)
            
            if (0 <= vx < self.voxel_dims[0] and 
                0 <= vy < self.voxel_dims[1] and 
                0 <= vz < self.voxel_dims[2]):
                
                # Color based on behavior and confidence
                if track['confidence'] > 0.9:
                    color = np.array([1.0, 0.0, 0.0])  # Red - very confident
                elif track['is_hovering']:
                    color = np.array([1.0, 1.0, 0.0])  # Yellow - hovering
                elif track['is_erratic']:
                    color = np.array([1.0, 0.0, 1.0])  # Magenta - erratic
                elif track['is_fast']:
                    color = np.array([0.0, 1.0, 1.0])  # Cyan - fast
                else:
                    color = np.array([0.0, 1.0, 0.0])  # Green - normal
                
                intensity = track['confidence']

                # Single voxel per detection — no spreading
                self.insect_voxels[vx, vy, vz] += intensity
                self.voxel_colors[vx, vy, vz] += color * intensity
                                
    def _create_ultimate_visualization(self, frame: np.ndarray, detections: List[Dict], tracks: List[Dict]) -> np.ndarray:
        """Create ultimate visualization."""
        vis = frame.copy()
        
        # Filter tracks for display
        if self.show_all_detections:
            display_tracks = tracks
        else:
            display_tracks = [t for t in tracks if t['confidence'] >= self.min_confidence_display]
        
        # Draw detection boxes
        for track in display_tracks:
            detection = track['detection']
            x, y, w, h = detection['bbox']
            center_x, center_y = track['last_center']
            confidence = track['confidence']
            
            # Dynamic color and thickness
            if confidence > 0.9:
                color = (0, 0, 255)    # Red - very confident
                thickness = 3
            elif track.get('is_hovering'):
                color = (0, 255, 255)  # Yellow - hovering
                thickness = 2
            elif track.get('is_erratic'):
                color = (255, 0, 255)  # Magenta - erratic
                thickness = 2
            elif track.get('is_fast'):
                color = (255, 255, 0)  # Cyan - fast
                thickness = 2
            elif confidence > 0.7:
                color = (0, 165, 255)  # Orange - good
                thickness = 2
            else:
                color = (0, 255, 0)    # Green - possible
                thickness = 1
            
            # Draw bounding box
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, thickness)
            cv2.circle(vis, (center_x, center_y), 2, color, -1)
            
            # Labels
            label_parts = [f"{confidence:.2f}"]
            if self.show_behavior:
                if track.get('is_hovering'):
                    label_parts.append("HOVER")
                if track.get('is_erratic'):
                    label_parts.append("ERRATIC")
                if track.get('is_fast'):
                    label_parts.append("FAST")
            
            label = " ".join(label_parts)
            cv2.putText(vis, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Track ID for high confidence
            if confidence > 0.8:
                cv2.putText(vis, f"#{track['id']}", (x + w + 2, y + 12), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Draw trails
        if self.show_trails:
            for track in display_tracks:
                if track['confidence'] > 0.6:
                    positions = track['positions'][-20:]  # Last 20 positions
                    if len(positions) > 1:
                        # Trail color based on behavior
                        if track.get('is_hovering'):
                            trail_color = (100, 255, 255)
                        elif track.get('is_erratic'):
                            trail_color = (200, 100, 255)
                        elif track.get('is_fast'):
                            trail_color = (255, 255, 100)
                        else:
                            trail_color = (150, 255, 150)
                        
                        pts = np.array(positions, np.int32)
                        cv2.polylines(vis, [pts], False, trail_color, 2)
        
        # Draw voxel overlay
        self._draw_ultimate_voxel_overlay(vis)
        
        # Add comprehensive stats
        if self.show_stats:
            self._add_ultimate_stats(vis, detections, tracks)
        
        return vis
        
    def _draw_ultimate_voxel_overlay(self, frame: np.ndarray):
        """Draw ultra-high resolution voxel overlay."""
        active_mask = self.insect_voxels > 0.05
        
        if not np.any(active_mask):
            return
        
        coords = np.argwhere(active_mask)
        intensities = self.insect_voxels[active_mask]
        colors = self.voxel_colors[active_mask]
        
        # Normalize colors
        colors = colors / (intensities[:, np.newaxis] + 0.001)
        colors = np.clip(colors, 0, 1)
        
        # Limit for performance but allow more for high resolution
        if len(coords) > 3000:
            indices = np.argsort(intensities)[-3000:]
            coords = coords[indices]
            intensities = intensities[indices]
            colors = colors[indices]
        
        for coord, intensity, color in zip(coords, intensities, colors):
            # Project to screen
            world_x = (coord[0] - self.voxel_dims[0] / 2) * self.voxel_size
            world_y = (coord[1] - self.voxel_dims[1] / 2) * self.voxel_size
            world_z = coord[2] * self.voxel_size
            
            if world_z > 0.05:
                distance = world_z - self.camera_z
                if distance > 0:
                    fov_scale = math.tan(math.radians(self.camera_fov / 2))
                    screen_x = int((world_x / (distance * fov_scale) + 1) * self.width / 2)
                    screen_y = int((-world_y / (distance * fov_scale) + 1) * self.height / 2)
                    
                    if 0 <= screen_x < self.width and 0 <= screen_y < self.height:
                        size = max(1, min(3, int(2 * intensity / distance)))
                        bgr_color = tuple(map(int, color[::-1] * 255))
                        cv2.circle(frame, (screen_x, screen_y), size, bgr_color, -1)
                        
    def _add_ultimate_stats(self, frame: np.ndarray, detections: List[Dict], tracks: List[Dict]):
        """Add comprehensive statistics display."""
        # Performance
        fps = 1.0 / np.mean(self.frame_times) if self.frame_times else 0
        fps_color = (0, 255, 0) if fps > 20 else (0, 255, 255) if fps > 15 else (0, 0, 255)
        
        # Confidence distribution
        very_high = sum(1 for t in tracks if t['confidence'] > 0.9)
        high = sum(1 for t in tracks if 0.8 <= t['confidence'] <= 0.9)
        medium = sum(1 for t in tracks if 0.6 <= t['confidence'] < 0.8)
        low = sum(1 for t in tracks if 0.4 <= t['confidence'] < 0.6)
        
        # Behavior analysis
        hovering = sum(1 for t in tracks if t.get('is_hovering', False))
        erratic = sum(1 for t in tracks if t.get('is_erratic', False))
        fast = sum(1 for t in tracks if t.get('is_fast', False))
        
        # Detection methods
        method_counts = defaultdict(int)
        for det in detections:
            method_counts[det.get('method', 'unknown')] += 1
        
        # Voxel statistics
        total_voxels = np.sum(self.insect_voxels > 0.01)
        bright_voxels = np.sum(self.insect_voxels > 0.5)
        max_intensity = np.max(self.insect_voxels)
        
        # Create stats panel
        stats_lines = [
            f"FPS: {fps:.1f} | Frame: {self.detector.processed_frames}",
            f"Detections: {len(detections)} | Active tracks: {len(tracks)}",
            f"Confidence: VH:{very_high} H:{high} M:{medium} L:{low}",
            f"Behavior: Hover:{hovering} Erratic:{erratic} Fast:{fast}",
            f"Voxels: {total_voxels:,} ({bright_voxels:,} bright)",
            f"Sensitivity: {self.detector.motion_threshold} | Min area: {self.detector.min_area}",
            f"Threshold: {self.min_confidence_display:.1f} | Total found: {self.total_insects_detected}"
        ]
        
        # Detection methods breakdown
        if method_counts:
            methods_str = " | ".join([f"{k[:8]}:{v}" for k, v in list(method_counts.items())[:4]])
            stats_lines.append(f"Methods: {methods_str}")
        
        # Create stats background
        panel_width = 650
        panel_height = len(stats_lines) * 25 + 15
        cv2.rectangle(frame, (10, 10), (panel_width, panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (panel_width, panel_height), (0, 255, 0), 2)
        
        # Draw stats text
        for i, line in enumerate(stats_lines):
            cv2.putText(frame, line, (15, 35 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        # FPS indicator
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 150, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, fps_color, 2)
        
    def _show_ultimate_help(self):
        """Show comprehensive help."""
        print("\n" + "="*70)
        print("🦟 ULTIMATE INSECT DETECTOR - HELP")
        print("="*70)
        print("🎯 DETECTION FEATURES:")
        print("  • 4-layer detection: Background subtraction, frame diff, motion blobs, edge motion")
        print("  • Ultra-sensitive: 1-150 pixel detection range")
        print("  • Adaptive sensitivity: Auto-adjusts based on detection rate")
        print("  • Behavioral analysis: Hovering, erratic flight, speed analysis")
        print("  • High-resolution voxels: 150x150x75 grid with 0.01m precision")
        print()
        print("🎮 CONTROLS:")
        print("  q/ESC    - Quit application")
        print("  r        - Reset system (voxels, tracks, statistics)")
        print("  space    - Pause/Resume processing")
        print("  a        - Toggle show all detections vs confident only")
        print("  t        - Toggle track trails")
        print("  b        - Toggle behavior labels")
        print("  s        - Toggle statistics panel")
        print("  =        - Lower confidence threshold (show more)")
        print("  -        - Raise confidence threshold (show fewer)")
        print("  h        - Show this help")
        print()
        print("🎨 COLOR CODING:")
        print("  Red      - Very high confidence (>90%)")
        print("  Yellow   - Hovering insects")
        print("  Magenta  - Erratic flight patterns")
        print("  Cyan     - Fast-moving insects")
        print("  Orange   - Good confidence (70-90%)")
        print("  Green    - Possible insects")
        print()
        print("📊 CURRENT SETTINGS:")
        print(f"  Confidence threshold: {self.min_confidence_display:.1f}")
        print(f"  Motion sensitivity: {self.detector.motion_threshold}")
        print(f"  Min detection area: {self.detector.min_area} pixels")
        print(f"  Adaptive mode: {self.detector.adaptive_enabled}")
        print(f"  Show all detections: {self.show_all_detections}")
        print(f"  Show trails: {self.show_trails}")
        print(f"  Show behavior: {self.show_behavior}")
        print(f"  Show statistics: {self.show_stats}")
        print("="*70)
        
    def _reset_ultimate_system(self):
        """Reset everything."""
        self.insect_voxels.fill(0)
        self.voxel_colors.fill(0)
        self.detector.tracks.clear()
        self.detector.next_track_id = 0
        self.detector.frame_buffer.clear()
        self.detector.processed_frames = 0
        self.total_insects_detected = 0
        print("🔄 Ultimate system reset - all data cleared")
        
    def stop(self):
        """Clean up."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Ultimate insect detection system')
    parser.add_argument('url', nargs='?',
                       default='https://www.youtube.com/watch?v=ZdfJcN-yBIE',
                       help='Video URL')
    args = parser.parse_args()
    
    if args.url.isdigit():
        args.url = int(args.url)
        
    print("🦟 Ultimate Insect Detection System")
    print(f"📹 Processing: {args.url}")
    
    tracker = UltimateInsectTracker(args.url)
    
    try:
        tracker.start()
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        tracker.stop()


if __name__ == "__main__":
    main()