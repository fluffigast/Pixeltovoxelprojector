#!/usr/bin/env python3
"""Insect detection with 3D voxel tracking and highlighting."""

import cv2
import numpy as np
import time
import subprocess
import argparse
from collections import deque, defaultdict
from typing import List, Tuple, Dict, Optional
import math

class InsectDetector:
    """Advanced insect detection with behavioral analysis."""
    
    def __init__(self):
        # Detection parameters
        self.min_area = 8          # Minimum insect area (pixels)
        self.max_area = 500        # Maximum insect area (pixels)
        self.min_aspect_ratio = 0.3 # Minimum width/height ratio
        self.max_aspect_ratio = 3.0 # Maximum width/height ratio
        self.motion_threshold = 15  # Motion sensitivity
        
        # Behavioral filters
        self.min_speed = 2.0       # Minimum movement speed (pixels/frame)
        self.max_speed = 50.0      # Maximum movement speed (pixels/frame)
        self.erratic_threshold = 15 # Direction change threshold for erratic movement
        self.hover_radius = 8      # Radius for detecting hovering behavior
        
        # Background subtraction
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=True
        )
        
        # Tracking
        self.tracks = {}
        self.next_track_id = 0
        self.max_track_age = 30    # Frames to keep lost tracks
        
    def detect_insects(self, frame: np.ndarray) -> List[Dict]:
        """Detect insects in frame and return detection info."""
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Remove shadows
        fg_mask[fg_mask == 127] = 0  # Remove shadow pixels
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        
        for contour in contours:
            # Filter by area
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                continue
                
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by aspect ratio
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                continue
                
            # Calculate center
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Additional insect-like features
            solidity = self._calculate_solidity(contour)
            extent = area / (w * h) if w * h > 0 else 0
            
            # Insect-like filter (compact, solid objects)
            if solidity > 0.3 and extent > 0.3:
                detection = {
                    'center': (center_x, center_y),
                    'bbox': (x, y, w, h),
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'solidity': solidity,
                    'extent': extent,
                    'contour': contour
                }
                detections.append(detection)
                
        return detections
    
    def _calculate_solidity(self, contour) -> float:
        """Calculate solidity (area/convex_hull_area) of contour."""
        area = cv2.contourArea(contour)
        if area == 0:
            return 0
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        return area / hull_area if hull_area > 0 else 0
    
    def update_tracks(self, detections: List[Dict]) -> List[Dict]:
        """Update insect tracks with behavioral analysis."""
        # Simple nearest neighbor tracking
        current_centers = [det['center'] for det in detections]
        
        # Update existing tracks
        updated_tracks = {}
        assigned_detections = set()
        
        for track_id, track in self.tracks.items():
            best_match = None
            best_distance = float('inf')
            best_idx = -1
            
            # Find closest detection
            for i, center in enumerate(current_centers):
                if i in assigned_detections:
                    continue
                    
                distance = math.sqrt((center[0] - track['last_center'][0])**2 + 
                                   (center[1] - track['last_center'][1])**2)
                
                if distance < best_distance and distance < 50:  # Max association distance
                    best_distance = distance
                    best_match = center
                    best_idx = i
            
            if best_match is not None:
                # Update track
                track['positions'].append(best_match)
                track['last_center'] = best_match
                track['age'] = 0
                track['detection'] = detections[best_idx]
                
                # Calculate behavioral features
                self._update_behavior(track)
                
                updated_tracks[track_id] = track
                assigned_detections.add(best_idx)
            else:
                # Lost track
                track['age'] += 1
                if track['age'] < self.max_track_age:
                    updated_tracks[track_id] = track
        
        # Create new tracks for unassigned detections
        for i, detection in enumerate(detections):
            if i not in assigned_detections:
                new_track = {
                    'id': self.next_track_id,
                    'positions': [detection['center']],
                    'last_center': detection['center'],
                    'age': 0,
                    'detection': detection,
                    'speed': 0,
                    'direction_changes': 0,
                    'is_hovering': False,
                    'is_erratic': False,
                    'confidence': 0.5
                }
                updated_tracks[self.next_track_id] = new_track
                self.next_track_id += 1
        
        self.tracks = updated_tracks
        
        # Return only active tracks with detections
        return [track for track in self.tracks.values() if track['age'] == 0]
    
    def _update_behavior(self, track: Dict):
        """Update behavioral analysis for track."""
        positions = track['positions']
        
        if len(positions) < 2:
            return
            
        # Calculate speed
        recent_positions = positions[-5:]  # Last 5 positions
        if len(recent_positions) >= 2:
            total_distance = 0
            for i in range(1, len(recent_positions)):
                dx = recent_positions[i][0] - recent_positions[i-1][0]
                dy = recent_positions[i][1] - recent_positions[i-1][1]
                total_distance += math.sqrt(dx*dx + dy*dy)
            track['speed'] = total_distance / (len(recent_positions) - 1)
        
        # Detect direction changes (erratic movement)
        if len(positions) >= 3:
            directions = []
            for i in range(len(positions) - 1):
                dx = positions[i+1][0] - positions[i][0]
                dy = positions[i+1][1] - positions[i][1]
                if dx != 0 or dy != 0:
                    angle = math.atan2(dy, dx)
                    directions.append(angle)
            
            # Count significant direction changes
            direction_changes = 0
            for i in range(1, len(directions)):
                angle_diff = abs(directions[i] - directions[i-1])
                if angle_diff > math.pi:
                    angle_diff = 2 * math.pi - angle_diff
                if math.degrees(angle_diff) > self.erratic_threshold:
                    direction_changes += 1
            
            track['direction_changes'] = direction_changes
            track['is_erratic'] = direction_changes > 2
        
        # Detect hovering
        if len(positions) >= 10:
            recent_pos = positions[-10:]
            center_x = sum(pos[0] for pos in recent_pos) / len(recent_pos)
            center_y = sum(pos[1] for pos in recent_pos) / len(recent_pos)
            
            max_distance = max(math.sqrt((pos[0] - center_x)**2 + (pos[1] - center_y)**2) 
                             for pos in recent_pos)
            track['is_hovering'] = max_distance < self.hover_radius
        
        # Update confidence based on behavior
        confidence = 0.5
        
        # Insect-like speed
        if self.min_speed <= track['speed'] <= self.max_speed:
            confidence += 0.2
            
        # Erratic movement (typical of insects)
        if track['is_erratic']:
            confidence += 0.2
            
        # Size and shape
        detection = track['detection']
        if 10 <= detection['area'] <= 200:
            confidence += 0.1
            
        track['confidence'] = min(1.0, confidence)


class InsectVoxelTracker:
    """3D voxel tracking system for detected insects."""
    
    def __init__(self, stream_url: str):
        self.stream_url = stream_url
        self.detector = InsectDetector()
        
        # Video
        self.cap = None
        self.width = 640
        self.height = 480
        
        # Voxel system
        self.voxel_dims = (80, 80, 40)
        self.voxel_size = 0.04
        self.insect_voxels = np.zeros(self.voxel_dims, dtype=np.float32)
        self.voxel_colors = np.zeros((*self.voxel_dims, 3), dtype=np.float32)
        self.voxel_decay = 0.96
        
        # Camera model
        self.camera_fov = 65
        self.camera_z = 1.2
        
        # Visualization
        self.show_detections = True
        self.show_tracks = True
        self.show_confidence = True
        
    def get_stream_url(self):
        """Get stream URL."""
        if 'youtube.com' in self.stream_url or 'youtu.be' in self.stream_url:
            try:
                cmd = ['yt-dlp', '-f', 'best[height<=720]', '--get-url', self.stream_url]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    return result.stdout.strip()
            except:
                pass
        return self.stream_url
        
    def start(self):
        """Start insect detection and voxel tracking."""
        url = self.get_stream_url()
        print(f"Opening stream for insect detection: {url}")
        
        self.cap = cv2.VideoCapture(url)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open stream")
            
        print("Insect Detection & Voxel Tracking Started!")
        print("\nControls:")
        print("  q - Quit")
        print("  r - Reset voxels and tracking")
        print("  d - Toggle detection boxes")
        print("  t - Toggle track trails")
        print("  c - Toggle confidence display")
        print("  +/- - Adjust motion sensitivity")
        
        cv2.namedWindow('Insect Voxel Tracker', cv2.WINDOW_NORMAL)
        
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Resize to working resolution
            frame = cv2.resize(frame, (self.width, self.height))
            
            # Detect insects
            detections = self.detector.detect_insects(frame)
            
            # Update tracks
            active_tracks = self.detector.update_tracks(detections)
            
            # Update voxels for high-confidence insects
            self._update_insect_voxels(active_tracks)
            
            # Create visualization
            vis_frame = self._create_visualization(frame, active_tracks)
            
            cv2.imshow('Insect Voxel Tracker', vis_frame)
            
            # Handle input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self._reset_system()
            elif key == ord('d'):
                self.show_detections = not self.show_detections
            elif key == ord('t'):
                self.show_tracks = not self.show_tracks
            elif key == ord('c'):
                self.show_confidence = not self.show_confidence
            elif key == ord('+'):
                self.detector.motion_threshold = max(5, self.detector.motion_threshold - 2)
                print(f"Motion sensitivity: {self.detector.motion_threshold}")
            elif key == ord('-'):
                self.detector.motion_threshold = min(50, self.detector.motion_threshold + 2)
                print(f"Motion sensitivity: {self.detector.motion_threshold}")
                
    def _update_insect_voxels(self, tracks: List[Dict]):
        """Update 3D voxel representation of insects."""
        # Apply decay
        self.insect_voxels *= self.voxel_decay
        self.voxel_colors *= self.voxel_decay
        
        for track in tracks:
            if track['confidence'] < 0.6:  # Only track confident detections
                continue
                
            center_x, center_y = track['last_center']
            
            # Convert to normalized coordinates
            norm_x = (center_x / self.width - 0.5) * 2.0
            norm_y = (center_y / self.height - 0.5) * 2.0
            
            # Project into 3D space
            fov_scale = math.tan(math.radians(self.camera_fov / 2))
            ray_x = norm_x * fov_scale
            ray_y = -norm_y * fov_scale
            
            # Estimate depth based on insect size
            detection = track['detection']
            estimated_depth = max(0.2, min(1.5, 50.0 / detection['area']))
            
            # Calculate 3D position
            world_x = ray_x * estimated_depth
            world_y = ray_y * estimated_depth
            world_z = self.camera_z + estimated_depth
            
            # Convert to voxel coordinates
            vx = int(world_x / self.voxel_size + self.voxel_dims[0] / 2)
            vy = int(world_y / self.voxel_size + self.voxel_dims[1] / 2)
            vz = int(world_z / self.voxel_size)
            
            # Update voxel if in bounds
            if (0 <= vx < self.voxel_dims[0] and 
                0 <= vy < self.voxel_dims[1] and 
                0 <= vz < self.voxel_dims[2]):
                
                # Color based on insect behavior
                if track['is_hovering']:
                    color = np.array([1.0, 1.0, 0.0])  # Yellow for hovering
                elif track['is_erratic']:
                    color = np.array([1.0, 0.0, 1.0])  # Magenta for erratic
                else:
                    color = np.array([0.0, 1.0, 0.0])  # Green for normal
                    
                intensity = track['confidence'] * 0.8
                
                # Update voxel with spreading
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        for dz in range(-1, 2):
                            nvx, nvy, nvz = vx + dx, vy + dy, vz + dz
                            if (0 <= nvx < self.voxel_dims[0] and 
                                0 <= nvy < self.voxel_dims[1] and 
                                0 <= nvz < self.voxel_dims[2]):
                                
                                falloff = 1.0 / (1.0 + abs(dx) + abs(dy) + abs(dz))
                                self.insect_voxels[nvx, nvy, nvz] += intensity * falloff
                                self.voxel_colors[nvx, nvy, nvz] += color * intensity * falloff
                                
    def _create_visualization(self, frame: np.ndarray, tracks: List[Dict]) -> np.ndarray:
        """Create visualization with insect detection and voxel overlay."""
        vis = frame.copy()
        
        # Draw detection boxes
        if self.show_detections:
            for track in tracks:
                detection = track['detection']
                x, y, w, h = detection['bbox']
                center_x, center_y = track['last_center']
                confidence = track['confidence']
                
                # Color based on confidence
                if confidence > 0.8:
                    color = (0, 255, 0)  # Green - high confidence
                elif confidence > 0.6:
                    color = (0, 255, 255)  # Yellow - medium confidence
                else:
                    color = (0, 128, 255)  # Orange - low confidence
                    
                # Draw bounding box
                cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
                
                # Draw center point
                cv2.circle(vis, (center_x, center_y), 3, color, -1)
                
                # Draw behavior indicators
                if track['is_hovering']:
                    cv2.putText(vis, "HOVER", (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                elif track['is_erratic']:
                    cv2.putText(vis, "ERRATIC", (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                    
                # Show confidence
                if self.show_confidence:
                    cv2.putText(vis, f"{confidence:.2f}", (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw track trails
        if self.show_tracks:
            for track in tracks:
                positions = track['positions'][-20:]  # Last 20 positions
                if len(positions) > 1:
                    pts = np.array(positions, np.int32)
                    cv2.polylines(vis, [pts], False, (255, 255, 0), 1)
        
        # Draw voxel overlay
        self._draw_voxel_overlay(vis)
        
        # Add UI info
        self._add_ui_info(vis, tracks)
        
        return vis
        
    def _draw_voxel_overlay(self, frame: np.ndarray):
        """Draw 3D voxels projected onto 2D frame."""
        active_voxels = self.insect_voxels > 0.02
        
        if not np.any(active_voxels):
            return
            
        voxel_coords = np.argwhere(active_voxels)
        intensities = self.insect_voxels[active_voxels]
        colors = self.voxel_colors[active_voxels]
        
        # Normalize colors
        colors = colors / (intensities[:, np.newaxis] + 0.001)
        colors = np.clip(colors, 0, 1)
        
        for coord, intensity, color in zip(voxel_coords, intensities, colors):
            # Convert voxel to world position
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
                        # Draw glowing voxel
                        size = max(2, int(8 * intensity / distance))
                        bgr_color = tuple(map(int, color[::-1] * 255))  # RGB to BGR
                        
                        # Glow effect
                        cv2.circle(frame, (screen_x, screen_y), size + 2, (50, 50, 50), -1)
                        cv2.circle(frame, (screen_x, screen_y), size, bgr_color, -1)
                        
    def _add_ui_info(self, frame: np.ndarray, tracks: List[Dict]):
        """Add UI information overlay."""
        # Count insects by confidence
        high_conf = sum(1 for t in tracks if t['confidence'] > 0.8)
        med_conf = sum(1 for t in tracks if 0.6 <= t['confidence'] <= 0.8)
        low_conf = sum(1 for t in tracks if t['confidence'] < 0.6)
        
        # Count behaviors
        hovering = sum(1 for t in tracks if t.get('is_hovering', False))
        erratic = sum(1 for t in tracks if t.get('is_erratic', False))
        
        # Active voxels
        active_voxels = np.sum(self.insect_voxels > 0.02)
        
        # Display info
        info_lines = [
            f"Insects: High:{high_conf} Med:{med_conf} Low:{low_conf}",
            f"Behavior: Hover:{hovering} Erratic:{erratic}",
            f"Voxels: {active_voxels:,}",
            f"Sensitivity: {self.detector.motion_threshold}"
        ]
        
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (10, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                       
    def _reset_system(self):
        """Reset voxels and tracking."""
        self.insect_voxels.fill(0)
        self.voxel_colors.fill(0)
        self.detector.tracks.clear()
        self.detector.next_track_id = 0
        print("System reset - voxels and tracks cleared")
        
    def stop(self):
        """Clean up."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Insect detection with 3D voxel tracking')
    parser.add_argument('url', nargs='?',
                       default='http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4',
                       help='Video URL (use 0 for webcam)')
    args = parser.parse_args()
    
    # Convert webcam number
    if args.url.isdigit():
        args.url = int(args.url)
        
    print("Starting Insect Detection & Voxel Tracking System")
    print(f"Source: {args.url}")
    
    tracker = InsectVoxelTracker(args.url)
    
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