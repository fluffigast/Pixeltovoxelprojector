#!/usr/bin/env python3
"""Highly sensitive insect detection focused on small moving objects."""

import cv2
import numpy as np
import time
import subprocess
import argparse
import math
from collections import deque, defaultdict
from typing import List, Tuple, Dict, Optional

class PreciseInsectDetector:
    """Ultra-sensitive detector for small, fast-moving insects."""
    
    def __init__(self):
        # Very specific insect parameters
        self.min_area = 3           # Very small - single pixels to tiny insects
        self.max_area = 80          # Much smaller max - typical for flies/mosquitoes
        self.min_aspect_ratio = 0.2 # Allow very thin insects
        self.max_aspect_ratio = 5.0 # Allow elongated insects
        
        # Motion detection
        self.motion_threshold = 8   # Very sensitive
        self.gaussian_blur = 3      # Less blur to preserve small details
        
        # Speed filtering (insects are fast)
        self.min_speed = 3.0        # Must be moving reasonably fast
        self.max_speed = 100.0      # Very fast for small insects
        
        # Insect behavior patterns
        self.direction_change_threshold = 20  # Insects change direction frequently
        self.min_track_length = 3   # Short tracks for quick movements
        self.max_disappeared_frames = 5  # Quick reappearance
        
        # Advanced background subtraction for small objects
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200,        # Shorter history for dynamic scenes
            varThreshold=8,     # Very sensitive to changes
            detectShadows=False # Don't detect shadows
        )
        
        # Multi-frame difference for better small object detection
        self.frame_buffer = deque(maxlen=3)
        
        # Tracking
        self.tracks = {}
        self.next_track_id = 0
        
    def detect_small_moving_objects(self, frame: np.ndarray) -> List[Dict]:
        """Detect very small moving objects using multiple techniques."""
        detections = []
        
        # Method 1: Background subtraction
        bg_detections = self._detect_with_background_subtraction(frame)
        detections.extend(bg_detections)
        
        # Method 2: Multi-frame difference
        if len(self.frame_buffer) >= 2:
            diff_detections = self._detect_with_frame_difference(frame)
            detections.extend(diff_detections)
        
        # Add frame to buffer
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.frame_buffer.append(gray)
        
        # Remove duplicates and filter
        detections = self._filter_and_merge_detections(detections, frame)
        
        return detections
    
    def _detect_with_background_subtraction(self, frame: np.ndarray) -> List[Dict]:
        """Detect using background subtraction optimized for small objects."""
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Enhance small objects
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area <= area <= self.max_area:
                detection = self._create_detection_from_contour(contour, area)
                if detection:
                    detection['method'] = 'background'
                    detections.append(detection)
        
        return detections
    
    def _detect_with_frame_difference(self, current_frame: np.ndarray) -> List[Dict]:
        """Detect using frame differencing for very small, fast objects."""
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.GaussianBlur(current_gray, (self.gaussian_blur, self.gaussian_blur), 0)
        
        detections = []
        
        # Compare with multiple previous frames
        for prev_gray in list(self.frame_buffer)[-2:]:
            # Frame difference
            diff = cv2.absdiff(current_gray, prev_gray)
            
            # Very sensitive threshold for small objects
            _, thresh = cv2.threshold(diff, self.motion_threshold, 255, cv2.THRESH_BINARY)
            
            # Minimal morphological operations to preserve small objects
            kernel = np.ones((2, 2), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if self.min_area <= area <= self.max_area:
                    detection = self._create_detection_from_contour(contour, area)
                    if detection:
                        detection['method'] = 'difference'
                        detections.append(detection)
        
        return detections
    
    def _create_detection_from_contour(self, contour, area: float) -> Optional[Dict]:
        """Create detection object from contour with insect-specific filtering."""
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter by aspect ratio
        aspect_ratio = w / h if h > 0 else 0
        if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
            return None
        
        # Calculate additional features
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Compactness (area vs perimeter) - insects are reasonably compact
        perimeter = cv2.arcLength(contour, True)
        compactness = 4 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Filter very elongated or sparse objects
        if compactness < 0.1:  # Very loose filter for small objects
            return None
        
        return {
            'center': (center_x, center_y),
            'bbox': (x, y, w, h),
            'area': area,
            'aspect_ratio': aspect_ratio,
            'compactness': compactness,
            'contour': contour
        }
    
    def _filter_and_merge_detections(self, detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        """Filter and merge overlapping detections."""
        if not detections:
            return []
        
        # Remove detections too close to each other
        filtered = []
        for detection in detections:
            is_duplicate = False
            for existing in filtered:
                # Calculate distance between centers
                dist = math.sqrt(
                    (detection['center'][0] - existing['center'][0])**2 + 
                    (detection['center'][1] - existing['center'][1])**2
                )
                # If very close, keep the smaller one (more likely to be insect)
                if dist < 10:
                    if detection['area'] < existing['area']:
                        # Replace with smaller detection
                        filtered.remove(existing)
                        break
                    else:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                filtered.append(detection)
        
        return filtered
    
    def update_tracks(self, detections: List[Dict]) -> List[Dict]:
        """Update tracks with emphasis on insect-like movement."""
        current_centers = [det['center'] for det in detections]
        
        # Update existing tracks
        updated_tracks = {}
        assigned_detections = set()
        
        for track_id, track in self.tracks.items():
            best_match = None
            best_distance = float('inf')
            best_idx = -1
            
            # Find closest detection within reasonable distance
            max_distance = min(30, track.get('speed', 10) * 2)  # Dynamic based on speed
            
            for i, center in enumerate(current_centers):
                if i in assigned_detections:
                    continue
                    
                distance = math.sqrt(
                    (center[0] - track['last_center'][0])**2 + 
                    (center[1] - track['last_center'][1])**2
                )
                
                if distance < best_distance and distance < max_distance:
                    best_distance = distance
                    best_match = center
                    best_idx = i
            
            if best_match is not None:
                # Update track
                track['positions'].append(best_match)
                track['last_center'] = best_match
                track['disappeared_frames'] = 0
                track['detection'] = detections[best_idx]
                
                # Update movement analysis
                self._analyze_movement(track)
                
                updated_tracks[track_id] = track
                assigned_detections.add(best_idx)
            else:
                # Track lost
                track['disappeared_frames'] += 1
                if track['disappeared_frames'] < self.max_disappeared_frames:
                    updated_tracks[track_id] = track
        
        # Create new tracks
        for i, detection in enumerate(detections):
            if i not in assigned_detections:
                new_track = {
                    'id': self.next_track_id,
                    'positions': [detection['center']],
                    'last_center': detection['center'],
                    'disappeared_frames': 0,
                    'detection': detection,
                    'speed': 0,
                    'direction_changes': 0,
                    'is_insect_like': False,
                    'confidence': 0.1  # Start with low confidence
                }
                updated_tracks[self.next_track_id] = new_track
                self.next_track_id += 1
        
        self.tracks = updated_tracks
        
        # Return only visible tracks with insect-like behavior
        return [track for track in self.tracks.values() 
                if track['disappeared_frames'] == 0 and track['confidence'] > 0.3]
    
    def _analyze_movement(self, track: Dict):
        """Analyze movement pattern to determine if it's insect-like."""
        positions = track['positions']
        
        if len(positions) < 2:
            return
        
        # Calculate recent speed
        if len(positions) >= 2:
            recent_positions = positions[-min(5, len(positions)):]
            total_distance = 0
            for i in range(1, len(recent_positions)):
                dx = recent_positions[i][0] - recent_positions[i-1][0]
                dy = recent_positions[i][1] - recent_positions[i-1][1]
                total_distance += math.sqrt(dx*dx + dy*dy)
            track['speed'] = total_distance / (len(recent_positions) - 1)
        
        # Count direction changes (erratic movement typical of insects)
        if len(positions) >= 3:
            directions = []
            for i in range(len(positions) - 1):
                dx = positions[i+1][0] - positions[i][0]
                dy = positions[i+1][1] - positions[i][1]
                if dx != 0 or dy != 0:
                    angle = math.atan2(dy, dx)
                    directions.append(angle)
            
            direction_changes = 0
            for i in range(1, len(directions)):
                angle_diff = abs(directions[i] - directions[i-1])
                if angle_diff > math.pi:
                    angle_diff = 2 * math.pi - angle_diff
                if math.degrees(angle_diff) > self.direction_change_threshold:
                    direction_changes += 1
            
            track['direction_changes'] = direction_changes
        
        # Determine if movement is insect-like
        track['is_insect_like'] = (
            self.min_speed <= track['speed'] <= self.max_speed and
            track['direction_changes'] > 0 and
            len(positions) >= self.min_track_length
        )
        
        # Update confidence based on insect-like features
        confidence = 0.1
        
        # Size bonus (smaller = more likely insect)
        area = track['detection']['area']
        if area < 20:
            confidence += 0.3
        elif area < 40:
            confidence += 0.2
        
        # Speed bonus
        if self.min_speed <= track['speed'] <= self.max_speed:
            confidence += 0.3
        
        # Erratic movement bonus
        if track['direction_changes'] > 1:
            confidence += 0.2
        
        # Track length bonus
        if len(positions) > 5:
            confidence += 0.2
        
        track['confidence'] = min(1.0, confidence)


class PreciseInsectVoxelTracker:
    """3D voxel tracker optimized for small, precise insect detection."""
    
    def __init__(self, stream_url: str):
        self.stream_url = stream_url
        self.detector = PreciseInsectDetector()
        
        # Video
        self.cap = None
        self.width = 640
        self.height = 480
        
        # High-resolution voxel grid for precise tracking
        self.voxel_dims = (100, 100, 50)
        self.voxel_size = 0.02  # Smaller voxels for precision
        self.insect_voxels = np.zeros(self.voxel_dims, dtype=np.float32)
        self.voxel_colors = np.zeros((*self.voxel_dims, 3), dtype=np.float32)
        self.voxel_decay = 0.92  # Faster decay for quick movements
        
        # Camera model
        self.camera_fov = 70
        self.camera_z = 1.0
        
        # Visualization settings
        self.show_all_detections = False  # Only show confident insects
        self.min_confidence_display = 0.5
        
    def get_stream_url(self):
        """Get stream URL."""
        if 'youtube.com' in self.stream_url or 'youtu.be' in self.stream_url:
            try:
                cmd = ['yt-dlp', '-f', 'best[height<=720]', '--get-url', self.stream_url]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
                if result.returncode == 0:
                    return result.stdout.strip()
            except:
                pass
        return self.stream_url
        
    def start(self):
        """Start precise insect detection."""
        url = self.get_stream_url()
        print(f"Opening stream for precise insect detection: {url}")
        
        self.cap = cv2.VideoCapture(url)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open stream")
            
        print("Precise Insect Detection Started!")
        print("\nOptimized for:")
        print("  - Very small objects (3-80 pixels)")
        print("  - Fast, erratic movement")
        print("  - High sensitivity detection")
        print("\nControls:")
        print("  q/ESC - Quit")
        print("  r - Reset system")
        print("  a - Toggle show all detections")
        print("  = - Lower confidence threshold (show more)")
        print("  - - Raise confidence threshold (show less)")
        print("  s - Increase motion sensitivity")
        print("  d - Decrease motion sensitivity")
        print("  h - Show help")
        print("\nPress 'h' during operation for help")
        
        cv2.namedWindow('Precise Insect Detector', cv2.WINDOW_NORMAL)
        
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Resize for processing
            frame = cv2.resize(frame, (self.width, self.height))
            
            # Detect small moving objects
            detections = self.detector.detect_small_moving_objects(frame)
            
            # Update tracks
            active_tracks = self.detector.update_tracks(detections)
            
            # Update voxels for confident detections only
            confident_tracks = [t for t in active_tracks if t['confidence'] >= self.min_confidence_display]
            self._update_precise_voxels(confident_tracks)
            
            # Create visualization
            vis_frame = self._create_precise_visualization(frame, detections, active_tracks)
            
            cv2.imshow('Precise Insect Detector', vis_frame)
            
            # Handle input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q or ESC
                break
            elif key == ord('r') or key == ord('R'):
                self._reset_system()
            elif key == ord('a') or key == ord('A'):
                self.show_all_detections = not self.show_all_detections
                print(f"Show all detections: {self.show_all_detections}")
            elif key == ord('=') or key == ord('+'):  # = or +
                self.min_confidence_display = max(0.1, self.min_confidence_display - 0.1)
                print(f"Confidence threshold: {self.min_confidence_display:.1f}")
            elif key == ord('-') or key == ord('_'):  # - or _
                self.min_confidence_display = min(0.9, self.min_confidence_display + 0.1)
                print(f"Confidence threshold: {self.min_confidence_display:.1f}")
            elif key == ord('s') or key == ord('S'):  # Sensitivity
                self.detector.motion_threshold = max(3, self.detector.motion_threshold - 2)
                print(f"Motion sensitivity: {self.detector.motion_threshold}")
            elif key == ord('d') or key == ord('D'):  # Decrease sensitivity
                self.detector.motion_threshold = min(30, self.detector.motion_threshold + 2)
                print(f"Motion sensitivity: {self.detector.motion_threshold}")
            elif key == ord('h') or key == ord('H'):  # Help
                self._show_help()
                
    def _update_precise_voxels(self, tracks: List[Dict]):
        """Update voxels with precise insect positions."""
        # Apply decay
        self.insect_voxels *= self.voxel_decay
        self.voxel_colors *= self.voxel_decay
        
        for track in tracks:
            center_x, center_y = track['last_center']
            
            # Convert to normalized coordinates
            norm_x = (center_x / self.width - 0.5) * 2.0
            norm_y = (center_y / self.height - 0.5) * 2.0
            
            # Estimate depth based on size and movement
            area = track['detection']['area']
            speed = track['speed']
            
            # Smaller, faster objects assumed closer
            estimated_depth = max(0.1, min(1.0, (area / 30.0) + (10.0 / (speed + 1))))
            
            # Project to 3D
            fov_scale = math.tan(math.radians(self.camera_fov / 2))
            world_x = norm_x * fov_scale * estimated_depth
            world_y = -norm_y * fov_scale * estimated_depth
            world_z = self.camera_z + estimated_depth
            
            # Convert to voxel coordinates
            vx = int(world_x / self.voxel_size + self.voxel_dims[0] / 2)
            vy = int(world_y / self.voxel_size + self.voxel_dims[1] / 2)
            vz = int(world_z / self.voxel_size)
            
            # Update voxel if in bounds
            if (0 <= vx < self.voxel_dims[0] and 
                0 <= vy < self.voxel_dims[1] and 
                0 <= vz < self.voxel_dims[2]):
                
                # Color based on confidence and movement
                if track['confidence'] > 0.8:
                    color = np.array([1.0, 0.0, 0.0])  # Red - high confidence
                elif track['is_insect_like']:
                    color = np.array([1.0, 1.0, 0.0])  # Yellow - insect-like
                else:
                    color = np.array([0.0, 1.0, 0.0])  # Green - detected movement
                
                intensity = track['confidence'] * 1.2
                
                # Update voxel with small spread for precision
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nvx, nvy = vx + dx, vy + dy
                        if (0 <= nvx < self.voxel_dims[0] and 0 <= nvy < self.voxel_dims[1]):
                            falloff = 1.0 / (1.0 + abs(dx) + abs(dy))
                            self.insect_voxels[nvx, nvy, vz] += intensity * falloff
                            self.voxel_colors[nvx, nvy, vz] += color * intensity * falloff
                            
    def _create_precise_visualization(self, frame: np.ndarray, detections: List[Dict], tracks: List[Dict]) -> np.ndarray:
        """Create visualization focused on precise insect detection."""
        vis = frame.copy()
        
        # Show all detections if enabled, otherwise only confident ones
        display_tracks = tracks if self.show_all_detections else [t for t in tracks if t['confidence'] >= self.min_confidence_display]
        
        # Draw detection boxes
        for track in display_tracks:
            detection = track['detection']
            x, y, w, h = detection['bbox']
            center_x, center_y = track['last_center']
            confidence = track['confidence']
            
            # Color based on confidence
            if confidence > 0.8:
                color = (0, 0, 255)    # Red - very confident
            elif confidence > 0.6:
                color = (0, 255, 255)  # Yellow - confident
            elif track['is_insect_like']:
                color = (255, 255, 0)  # Cyan - insect-like movement
            else:
                color = (0, 255, 0)    # Green - possible
                
            # Draw very small box
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 1)
            cv2.circle(vis, (center_x, center_y), 2, color, -1)
            
            # Show info for high confidence detections
            if confidence > 0.6:
                info = f"{confidence:.2f}"
                if track['is_insect_like']:
                    info += " BUG"
                cv2.putText(vis, info, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Draw movement trails
        for track in display_tracks:
            if track['confidence'] > 0.5:
                positions = track['positions'][-10:]  # Last 10 positions
                if len(positions) > 1:
                    pts = np.array(positions, np.int32)
                    cv2.polylines(vis, [pts], False, (255, 255, 255), 1)
        
        # Draw voxel overlay
        self._draw_precise_voxel_overlay(vis)
        
        # Add detailed UI
        self._add_precise_ui(vis, detections, tracks)
        
        return vis
        
    def _draw_precise_voxel_overlay(self, frame: np.ndarray):
        """Draw precise voxel overlay."""
        active_voxels = self.insect_voxels > 0.05
        
        if not np.any(active_voxels):
            return
            
        voxel_coords = np.argwhere(active_voxels)
        intensities = self.insect_voxels[active_voxels]
        colors = self.voxel_colors[active_voxels]
        
        # Normalize colors
        colors = colors / (intensities[:, np.newaxis] + 0.001)
        colors = np.clip(colors, 0, 1)
        
        for coord, intensity, color in zip(voxel_coords, intensities, colors):
            # Convert to world position
            world_x = (coord[0] - self.voxel_dims[0] / 2) * self.voxel_size
            world_y = (coord[1] - self.voxel_dims[1] / 2) * self.voxel_size
            world_z = coord[2] * self.voxel_size
            
            # Project to screen
            if world_z > 0.1:
                distance = world_z - self.camera_z
                if distance > 0:
                    fov_scale = math.tan(math.radians(self.camera_fov / 2))
                    screen_x = int((world_x / (distance * fov_scale) + 1) * self.width / 2)
                    screen_y = int((-world_y / (distance * fov_scale) + 1) * self.height / 2)
                    
                    if 0 <= screen_x < self.width and 0 <= screen_y < self.height:
                        # Draw small, bright voxel
                        size = max(1, int(4 * intensity))
                        bgr_color = tuple(map(int, color[::-1] * 255))
                        
                        cv2.circle(frame, (screen_x, screen_y), size + 1, (0, 0, 0), -1)  # Black outline
                        cv2.circle(frame, (screen_x, screen_y), size, bgr_color, -1)
                        
    def _add_precise_ui(self, frame: np.ndarray, detections: List[Dict], tracks: List[Dict]):
        """Add precise detection UI."""
        # Count by confidence
        high_conf = sum(1 for t in tracks if t['confidence'] > 0.8)
        med_conf = sum(1 for t in tracks if 0.5 <= t['confidence'] <= 0.8)
        low_conf = sum(1 for t in tracks if t['confidence'] < 0.5)
        
        # Count insect-like behavior
        insect_like = sum(1 for t in tracks if t['is_insect_like'])
        
        # Active voxels
        active_voxels = np.sum(self.insect_voxels > 0.05)
        
        # Total detections
        total_detections = len(detections)
        
        info_lines = [
            f"Detections: {total_detections} | Tracks: H:{high_conf} M:{med_conf} L:{low_conf}",
            f"Insect-like: {insect_like} | Voxels: {active_voxels}",
            f"Confidence threshold: {self.min_confidence_display:.1f}",
            f"Show all: {self.show_all_detections}"
        ]
        
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (10, 25 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    def _show_help(self):
        """Show help information."""
        print("\n" + "="*50)
        print("PRECISE INSECT DETECTOR - CONTROLS")
        print("="*50)
        print("BASIC CONTROLS:")
        print("  q/ESC     - Quit application")
        print("  r         - Reset voxels and tracking")
        print("  h         - Show this help")
        print()
        print("DETECTION CONTROLS:")
        print("  a         - Toggle show all detections")
        print("  =         - Lower confidence threshold (show more)")
        print("  -         - Raise confidence threshold (show less)")
        print("  s         - Increase motion sensitivity")
        print("  d         - Decrease motion sensitivity")
        print()
        print("CURRENT SETTINGS:")
        print(f"  Confidence threshold: {self.min_confidence_display:.1f}")
        print(f"  Motion sensitivity: {self.detector.motion_threshold}")
        print(f"  Show all detections: {self.show_all_detections}")
        print()
        print("DETECTION INFO:")
        print("  Red boxes    - Very confident insects (>80%)")
        print("  Yellow boxes - Confident detections (60-80%)")
        print("  Cyan boxes   - Insect-like movement")
        print("  'BUG' label  - Confirmed insect behavior")
        print("="*50)
                       
    def _reset_system(self):
        """Reset the detection system."""
        self.insect_voxels.fill(0)
        self.voxel_colors.fill(0)
        self.detector.tracks.clear()
        self.detector.next_track_id = 0
        self.detector.frame_buffer.clear()
        print("System reset")
        
    def stop(self):
        """Clean up."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Precise insect detection for small, fast-moving objects')
    parser.add_argument('url', nargs='?',
                       default='https://www.youtube.com/watch?v=ipf7ifVSeDU',
                       help='Video URL')
    args = parser.parse_args()
    
    if args.url.isdigit():
        args.url = int(args.url)
        
    print("Starting Precise Insect Detection System")
    print(f"Source: {args.url}")
    print("Optimized for small (3-80 pixel), fast-moving insects")
    
    tracker = PreciseInsectVoxelTracker(args.url)
    
    try:
        tracker.start()
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        tracker.stop()


if __name__ == "__main__":
    main()