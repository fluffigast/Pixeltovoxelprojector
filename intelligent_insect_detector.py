#!/usr/bin/env python3
"""Intelligent insect detector with temporal pattern analysis to distinguish insects from vegetation."""

import cv2
import numpy as np
import time
import subprocess
import argparse
import math
from collections import deque, defaultdict
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass, field

@dataclass
class DetectionPattern:
    """Track detection patterns in specific locations."""
    location: Tuple[int, int]
    detection_times: deque = field(default_factory=lambda: deque(maxlen=50))
    last_detection: float = 0.0
    total_detections: int = 0
    pattern_score: float = 0.0
    is_vegetation: bool = False
    avg_interval: float = 0.0
    
class TemporalIntelligence:
    """Analyze temporal patterns to distinguish insects from vegetation."""
    
    def __init__(self):
        # Spatial grid for tracking detection patterns
        self.grid_size = 20  # 20x20 pixel grid cells
        self.location_patterns: Dict[Tuple[int, int], DetectionPattern] = {}
        
        # Pattern analysis parameters
        self.vegetation_detection_threshold = 8  # Min detections to analyze pattern
        self.rhythmic_threshold = 0.3  # Variance threshold for rhythmic motion
        self.repetition_distance = 25  # Max distance for same location
        self.time_window = 30.0  # Seconds to analyze patterns
        
        # Vegetation characteristics
        self.vegetation_interval_range = (0.5, 3.0)  # Seconds between leaf movements
        self.vegetation_persistence_threshold = 0.7  # How often same spot triggers
        
        # Insect characteristics  
        self.insect_movement_threshold = 50  # Pixels insects typically move
        self.insect_randomness_threshold = 0.8  # Expected randomness in timing
        
    def analyze_detection(self, center: Tuple[int, int], timestamp: float) -> float:
        """Analyze a detection and return vegetation probability (0=insect, 1=vegetation)."""
        # Convert to grid coordinates
        grid_x = center[0] // self.grid_size
        grid_y = center[1] // self.grid_size
        grid_location = (grid_x, grid_y)
        
        # Get or create pattern for this location
        if grid_location not in self.location_patterns:
            self.location_patterns[grid_location] = DetectionPattern(
                location=grid_location,
                detection_times=deque(maxlen=50)
            )
        
        pattern = self.location_patterns[grid_location]
        
        # Update pattern data
        pattern.detection_times.append(timestamp)
        pattern.last_detection = timestamp
        pattern.total_detections += 1
        
        # Clean old detections outside time window
        cutoff_time = timestamp - self.time_window
        while pattern.detection_times and pattern.detection_times[0] < cutoff_time:
            pattern.detection_times.popleft()
        
        # Analyze pattern if we have enough data
        if len(pattern.detection_times) >= self.vegetation_detection_threshold:
            pattern.pattern_score = self._calculate_vegetation_probability(pattern, timestamp)
            pattern.is_vegetation = pattern.pattern_score > 0.6
        
        return pattern.pattern_score
    
    def _calculate_vegetation_probability(self, pattern: DetectionPattern, current_time: float) -> float:
        """Calculate probability that this pattern represents vegetation."""
        if len(pattern.detection_times) < 3:
            return 0.0
        
        detection_times = list(pattern.detection_times)
        vegetation_score = 0.0
        
        # 1. Analyze timing intervals (vegetation has rhythmic patterns)
        intervals = [detection_times[i] - detection_times[i-1] 
                    for i in range(1, len(detection_times))]
        
        if intervals:
            avg_interval = np.mean(intervals)
            interval_variance = np.var(intervals)
            interval_consistency = 1.0 / (1.0 + interval_variance)
            
            pattern.avg_interval = avg_interval
            
            # Vegetation tends to have consistent intervals (wind patterns)
            if interval_consistency > 0.5:  # Consistent timing
                vegetation_score += 0.3
            
            # Check if intervals match vegetation characteristics
            if self.vegetation_interval_range[0] <= avg_interval <= self.vegetation_interval_range[1]:
                vegetation_score += 0.2
        
        # 2. Analyze detection frequency (vegetation appears repeatedly)
        recent_window = 10.0  # Last 10 seconds
        recent_detections = sum(1 for t in detection_times 
                              if current_time - t <= recent_window)
        detection_rate = recent_detections / recent_window
        
        # High detection rate in same location suggests vegetation
        if detection_rate > 0.5:  # More than 0.5 detections per second
            vegetation_score += 0.3
        
        # 3. Analyze spatial persistence (same exact location)
        location_consistency = len(pattern.detection_times) / pattern.total_detections
        if location_consistency > self.vegetation_persistence_threshold:
            vegetation_score += 0.2
        
        # 4. Check for rhythmic patterns (FFT analysis for periodicity)
        if len(intervals) >= 10:
            periodicity_score = self._analyze_periodicity(intervals)
            vegetation_score += periodicity_score * 0.2
        
        return min(1.0, vegetation_score)
    
    def _analyze_periodicity(self, intervals: List[float]) -> float:
        """Analyze periodicity in detection intervals using autocorrelation."""
        if len(intervals) < 10:
            return 0.0
        
        # Simple autocorrelation to detect periodicity
        intervals_array = np.array(intervals)
        mean_interval = np.mean(intervals_array)
        
        # Calculate autocorrelation at different lags
        max_lag = min(len(intervals) // 2, 10)
        autocorr_scores = []
        
        for lag in range(1, max_lag):
            if lag >= len(intervals):
                break
            
            # Correlation between intervals and shifted intervals
            corr = np.corrcoef(intervals_array[:-lag], intervals_array[lag:])[0, 1]
            if not np.isnan(corr):
                autocorr_scores.append(abs(corr))
        
        if autocorr_scores:
            max_autocorr = max(autocorr_scores)
            return max_autocorr  # Higher correlation = more periodic = more likely vegetation
        
        return 0.0
    
    def get_vegetation_locations(self) -> Set[Tuple[int, int]]:
        """Get grid locations identified as vegetation."""
        return {loc for loc, pattern in self.location_patterns.items() 
                if pattern.is_vegetation}
    
    def cleanup_old_patterns(self, current_time: float):
        """Remove old patterns that are no longer active."""
        cutoff_time = current_time - self.time_window * 2
        
        patterns_to_remove = []
        for location, pattern in self.location_patterns.items():
            if pattern.last_detection < cutoff_time:
                patterns_to_remove.append(location)
        
        for location in patterns_to_remove:
            del self.location_patterns[location]


class MovementIntelligence:
    """Analyze movement patterns to distinguish insects from vegetation."""
    
    def __init__(self):
        self.track_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=20))
        self.movement_patterns: Dict[int, Dict] = {}
        
    def analyze_track_movement(self, track_id: int, position: Tuple[int, int], 
                              timestamp: float) -> Dict[str, float]:
        """Analyze movement pattern for a track."""
        history = self.track_history[track_id]
        history.append((position, timestamp))
        
        if len(history) < 3:
            return {'is_insect_like': 0.5, 'movement_score': 0.5}
        
        # Calculate movement characteristics
        positions = [pos for pos, _ in history]
        timestamps = [t for _, t in history]
        
        # 1. Total displacement (insects move more than swaying leaves)
        total_displacement = self._calculate_total_displacement(positions)
        
        # 2. Movement randomness (insects move unpredictably)
        randomness_score = self._calculate_randomness(positions)
        
        # 3. Speed analysis (insects have varied speeds)
        speed_variation = self._calculate_speed_variation(positions, timestamps)
        
        # 4. Direction changes (insects change direction frequently)
        direction_changes = self._calculate_direction_changes(positions)
        
        # 5. Return tendency (vegetation returns to original position)
        return_tendency = self._calculate_return_tendency(positions)
        
        # Combine scores
        insect_score = 0.0
        
        # High displacement suggests insect
        if total_displacement > 100:
            insect_score += 0.25
        elif total_displacement > 50:
            insect_score += 0.15
        
        # High randomness suggests insect
        insect_score += randomness_score * 0.2
        
        # Speed variation suggests insect
        insect_score += speed_variation * 0.2
        
        # Direction changes suggest insect
        insect_score += direction_changes * 0.2
        
        # Low return tendency suggests insect (vegetation returns to center)
        insect_score += (1.0 - return_tendency) * 0.15
        
        movement_analysis = {
            'is_insect_like': min(1.0, insect_score),
            'movement_score': insect_score,
            'total_displacement': total_displacement,
            'randomness': randomness_score,
            'speed_variation': speed_variation,
            'direction_changes': direction_changes,
            'return_tendency': return_tendency
        }
        
        self.movement_patterns[track_id] = movement_analysis
        return movement_analysis
    
    def _calculate_total_displacement(self, positions: List[Tuple[int, int]]) -> float:
        """Calculate total displacement from start to current position."""
        if len(positions) < 2:
            return 0.0
        
        start_pos = positions[0]
        end_pos = positions[-1]
        
        return math.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
    
    def _calculate_randomness(self, positions: List[Tuple[int, int]]) -> float:
        """Calculate movement randomness (0=predictable, 1=random)."""
        if len(positions) < 4:
            return 0.5
        
        # Calculate expected position based on linear trend
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        
        # Simple linear regression to predict movement
        n = len(positions)
        if n < 3:
            return 0.5
        
        # Calculate mean squared error from linear trend
        x_mean = np.mean(x_coords)
        y_mean = np.mean(y_coords)
        
        deviations = []
        for i, (x, y) in enumerate(positions):
            expected_x = x_mean + (i - n/2) * (x_coords[-1] - x_coords[0]) / n
            expected_y = y_mean + (i - n/2) * (y_coords[-1] - y_coords[0]) / n
            
            deviation = math.sqrt((x - expected_x)**2 + (y - expected_y)**2)
            deviations.append(deviation)
        
        avg_deviation = np.mean(deviations)
        # Normalize to 0-1 range (higher deviation = more random = more insect-like)
        return min(1.0, avg_deviation / 50.0)
    
    def _calculate_speed_variation(self, positions: List[Tuple[int, int]], 
                                  timestamps: List[float]) -> float:
        """Calculate speed variation (insects have varied speeds)."""
        if len(positions) < 3:
            return 0.0
        
        speeds = []
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            dt = timestamps[i] - timestamps[i-1]
            
            if dt > 0:
                speed = math.sqrt(dx*dx + dy*dy) / dt
                speeds.append(speed)
        
        if len(speeds) < 2:
            return 0.0
        
        speed_std = np.std(speeds)
        speed_mean = np.mean(speeds)
        
        # Coefficient of variation (higher = more varied = more insect-like)
        if speed_mean > 0:
            cv = speed_std / speed_mean
            return min(1.0, cv)
        
        return 0.0
    
    def _calculate_direction_changes(self, positions: List[Tuple[int, int]]) -> float:
        """Calculate frequency of direction changes."""
        if len(positions) < 4:
            return 0.0
        
        direction_changes = 0
        prev_direction = None
        
        for i in range(2, len(positions)):
            # Calculate direction vector
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            
            if dx == 0 and dy == 0:
                continue
            
            current_direction = math.atan2(dy, dx)
            
            if prev_direction is not None:
                # Calculate angle difference
                angle_diff = abs(current_direction - prev_direction)
                if angle_diff > math.pi:
                    angle_diff = 2 * math.pi - angle_diff
                
                # Significant direction change (> 45 degrees)
                if angle_diff > math.pi / 4:
                    direction_changes += 1
            
            prev_direction = current_direction
        
        # Normalize by number of possible changes
        max_changes = len(positions) - 2
        return direction_changes / max_changes if max_changes > 0 else 0.0
    
    def _calculate_return_tendency(self, positions: List[Tuple[int, int]]) -> float:
        """Calculate tendency to return to original position (vegetation characteristic)."""
        if len(positions) < 5:
            return 0.0
        
        start_pos = positions[0]
        
        # Calculate average distance from start position
        distances_from_start = []
        for pos in positions[1:]:
            dist = math.sqrt((pos[0] - start_pos[0])**2 + (pos[1] - start_pos[1])**2)
            distances_from_start.append(dist)
        
        # Check if recent positions are closer to start than middle positions
        recent_distances = distances_from_start[-3:]
        middle_distances = distances_from_start[len(distances_from_start)//3:2*len(distances_from_start)//3]
        
        if middle_distances and recent_distances:
            avg_recent = np.mean(recent_distances)
            avg_middle = np.mean(middle_distances)
            
            # If recent positions are closer to start, higher return tendency
            if avg_middle > 0:
                return max(0.0, 1.0 - avg_recent / avg_middle)
        
        return 0.0


class IntelligentInsectDetector:
    """Intelligent insect detector with temporal and movement analysis."""
    
    def __init__(self):
        # Intelligence modules
        self.temporal_intelligence = TemporalIntelligence()
        self.movement_intelligence = MovementIntelligence()
        
        # Background subtraction
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=16, detectShadows=True
        )
        
        # Detection parameters
        self.min_area = 3
        self.max_area = 100
        self.motion_threshold = 8
        
        # Tracking
        self.tracks = {}
        self.next_track_id = 0
        self.frame_count = 0
        
        # Frame buffer for temporal analysis
        self.frame_buffer = deque(maxlen=3)
        
    def detect_insects(self, frame: np.ndarray) -> List[Dict]:
        """Detect insects with intelligence filtering."""
        self.frame_count += 1
        current_time = time.time()
        
        # Basic detection
        detections = self._basic_detection(frame)
        
        # Apply temporal intelligence
        intelligent_detections = []
        for detection in detections:
            center = detection['center']
            
            # Analyze temporal pattern
            vegetation_probability = self.temporal_intelligence.analyze_detection(center, current_time)
            detection['vegetation_probability'] = vegetation_probability
            detection['is_likely_vegetation'] = vegetation_probability > 0.6
            
            # Filter out likely vegetation
            if not detection['is_likely_vegetation']:
                intelligent_detections.append(detection)
        
        # Update tracks with movement analysis
        tracked_detections = self._update_tracks_with_intelligence(intelligent_detections, current_time)
        
        # Cleanup old patterns
        if self.frame_count % 100 == 0:  # Every 100 frames
            self.temporal_intelligence.cleanup_old_patterns(current_time)
        
        return tracked_detections
    
    def _basic_detection(self, frame: np.ndarray) -> List[Dict]:
        """Basic insect detection using background subtraction and frame differencing."""
        detections = []
        
        # Background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Remove shadows
        fg_mask[fg_mask == 127] = 0
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            detection = self._analyze_contour(contour)
            if detection:
                detection['method'] = 'background'
                detections.append(detection)
        
        # Frame differencing for fast insects
        if len(self.frame_buffer) >= 1:
            frame_diff_detections = self._frame_difference_detection(frame)
            detections.extend(frame_diff_detections)
        
        # Add frame to buffer
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.frame_buffer.append(gray)
        
        return detections
    
    def _analyze_contour(self, contour) -> Optional[Dict]:
        """Analyze contour for insect characteristics."""
        area = cv2.contourArea(contour)
        if area < self.min_area or area > self.max_area:
            return None
        
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        
        if aspect_ratio < 0.1 or aspect_ratio > 6.0:
            return None
        
        # Solidity check
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        if solidity < 0.2:
            return None
        
        center = (x + w // 2, y + h // 2)
        
        return {
            'center': center,
            'bbox': (x, y, w, h),
            'area': area,
            'aspect_ratio': aspect_ratio,
            'solidity': solidity,
            'contour': contour
        }
    
    def _frame_difference_detection(self, frame: np.ndarray) -> List[Dict]:
        """Detect using frame differencing."""
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = self.frame_buffer[-1]
        
        diff = cv2.absdiff(current_gray, prev_gray)
        _, thresh = cv2.threshold(diff, self.motion_threshold, 255, cv2.THRESH_BINARY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            detection = self._analyze_contour(contour)
            if detection:
                detection['method'] = 'frame_diff'
                detections.append(detection)
        
        return detections
    
    def _update_tracks_with_intelligence(self, detections: List[Dict], current_time: float) -> List[Dict]:
        """Update tracks with movement intelligence analysis."""
        # Simple tracking for demonstration
        updated_tracks = {}
        assigned_detections = set()
        
        # Update existing tracks
        for track_id, track in self.tracks.items():
            best_detection = None
            best_distance = float('inf')
            best_idx = -1
            
            for i, detection in enumerate(detections):
                if i in assigned_detections:
                    continue
                
                dist = math.sqrt((detection['center'][0] - track['last_center'][0])**2 + 
                               (detection['center'][1] - track['last_center'][1])**2)
                
                if dist < best_distance and dist < 50:
                    best_distance = dist
                    best_detection = detection
                    best_idx = i
            
            if best_detection:
                # Update track
                track['positions'].append(best_detection['center'])
                track['last_center'] = best_detection['center']
                track['lost_frames'] = 0
                track['detection'] = best_detection
                track['age'] += 1
                
                # Analyze movement pattern
                movement_analysis = self.movement_intelligence.analyze_track_movement(
                    track_id, best_detection['center'], current_time
                )
                track.update(movement_analysis)
                
                # Calculate final confidence
                vegetation_prob = best_detection.get('vegetation_probability', 0.0)
                movement_score = movement_analysis.get('is_insect_like', 0.5)
                
                # Combine temporal and movement intelligence
                track['confidence'] = (1.0 - vegetation_prob) * movement_score
                track['is_insect'] = track['confidence'] > 0.6
                
                updated_tracks[track_id] = track
                assigned_detections.add(best_idx)
            else:
                # Track lost
                track['lost_frames'] += 1
                if track['lost_frames'] < 8:
                    updated_tracks[track_id] = track
        
        # Create new tracks
        for i, detection in enumerate(detections):
            if i not in assigned_detections:
                new_track = {
                    'id': self.next_track_id,
                    'positions': [detection['center']],
                    'last_center': detection['center'],
                    'lost_frames': 0,
                    'detection': detection,
                    'age': 1,
                    'confidence': 0.5,
                    'is_insect': True,  # Assume insect until proven otherwise
                    'vegetation_probability': detection.get('vegetation_probability', 0.0)
                }
                updated_tracks[self.next_track_id] = new_track
                self.next_track_id += 1
        
        self.tracks = updated_tracks
        
        # Return only confident insect tracks
        return [track for track in self.tracks.values() 
                if track['lost_frames'] == 0 and track['is_insect'] and track['confidence'] > 0.5]


class IntelligentInsectVisualizer:
    """Visualizer for intelligent insect detection."""
    
    def __init__(self, stream_url: str):
        self.stream_url = stream_url
        self.detector = IntelligentInsectDetector()
        
        # Visualization settings
        self.show_vegetation_zones = True
        self.show_movement_analysis = True
        self.show_temporal_patterns = True
        
    def get_stream_url(self):
        """Get stream URL."""
        if 'youtube.com' in self.stream_url or 'youtu.be' in self.stream_url:
            try:
                cmd = ['yt-dlp', '-f', 'best[height<=1080]', '--get-url', self.stream_url]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
                if result.returncode == 0:
                    return result.stdout.strip()
            except:
                pass
        return self.stream_url
    
    def start(self):
        """Start intelligent detection visualization."""
        url = self.get_stream_url()
        print(f"🧠 Intelligent Insect Detector: {url}")
        
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            raise RuntimeError("Failed to open stream")
        
        print("🎯 Intelligent Detection Started!")
        print("\n🧠 Intelligence Features:")
        print("  • Temporal pattern analysis (repetitive vegetation filtering)")
        print("  • Movement intelligence (insect vs vegetation movement)")
        print("  • Location memory (same-spot detection filtering)")
        print("  • Rhythmic motion detection (wind pattern recognition)")
        print("\n🎮 Controls:")
        print("  q - Quit")
        print("  v - Toggle vegetation zones")
        print("  m - Toggle movement analysis")
        print("  t - Toggle temporal patterns")
        print("  r - Reset intelligence memory")
        
        cv2.namedWindow('Intelligent Insect Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Intelligent Insect Detection', 1400, 900)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame
            frame = cv2.resize(frame, (1200, 800))
            
            # Detect insects with intelligence
            tracks = self.detector.detect_insects(frame)
            
            # Create visualization
            vis_frame = self._create_intelligent_visualization(frame, tracks)
            
            cv2.imshow('Intelligent Insect Detection', vis_frame)
            
            # Handle controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('v'):
                self.show_vegetation_zones = not self.show_vegetation_zones
                print(f"Vegetation zones: {'ON' if self.show_vegetation_zones else 'OFF'}")
            elif key == ord('m'):
                self.show_movement_analysis = not self.show_movement_analysis
                print(f"Movement analysis: {'ON' if self.show_movement_analysis else 'OFF'}")
            elif key == ord('t'):
                self.show_temporal_patterns = not self.show_temporal_patterns
                print(f"Temporal patterns: {'ON' if self.show_temporal_patterns else 'OFF'}")
            elif key == ord('r'):
                # Reset intelligence
                self.detector.temporal_intelligence = TemporalIntelligence()
                self.detector.movement_intelligence = MovementIntelligence()
                print("Intelligence memory reset")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def _create_intelligent_visualization(self, frame: np.ndarray, tracks: List[Dict]) -> np.ndarray:
        """Create visualization with intelligence indicators."""
        vis = frame.copy()
        
        # Draw vegetation zones
        if self.show_vegetation_zones:
            self._draw_vegetation_zones(vis)
        
        # Draw insect tracks
        for track in tracks:
            center = track['last_center']
            confidence = track['confidence']
            vegetation_prob = track.get('vegetation_probability', 0.0)
            
            # Color based on confidence and intelligence
            if confidence > 0.8:
                color = (0, 0, 255)  # Red - high confidence insect
            elif confidence > 0.6:
                color = (0, 165, 255)  # Orange - likely insect
            else:
                color = (0, 255, 255)  # Yellow - possible insect
            
            # Draw detection
            cv2.circle(vis, center, 4, color, -1)
            cv2.circle(vis, center, 8, color, 1)
            
            # Draw bounding box
            bbox = track['detection']['bbox']
            x, y, w, h = bbox
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 1)
            
            # Add intelligence labels
            labels = [f"C:{confidence:.2f}"]
            
            if self.show_movement_analysis and 'movement_score' in track:
                labels.append(f"M:{track['movement_score']:.2f}")
            
            if self.show_temporal_patterns:
                labels.append(f"V:{vegetation_prob:.2f}")
            
            label_text = " ".join(labels)
            cv2.putText(vis, label_text, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Draw movement trail
            if 'positions' in track and len(track['positions']) > 1:
                positions = track['positions'][-10:]  # Last 10 positions
                if len(positions) > 1:
                    pts = np.array(positions, np.int32)
                    cv2.polylines(vis, [pts], False, color, 2)
        
        # Add intelligence statistics
        self._add_intelligence_stats(vis, tracks)
        
        return vis
    
    def _draw_vegetation_zones(self, frame: np.ndarray):
        """Draw detected vegetation zones."""
        vegetation_locations = self.detector.temporal_intelligence.get_vegetation_locations()
        grid_size = self.detector.temporal_intelligence.grid_size
        
        for grid_x, grid_y in vegetation_locations:
            # Convert grid coordinates back to pixel coordinates
            x1 = grid_x * grid_size
            y1 = grid_y * grid_size
            x2 = x1 + grid_size
            y2 = y1 + grid_size
            
            # Draw vegetation zone
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 100, 100), 1)  # Dark yellow
            cv2.putText(frame, "V", (x1 + 2, y1 + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 100, 100), 1)
    
    def _add_intelligence_stats(self, frame: np.ndarray, tracks: List[Dict]):
        """Add intelligence statistics overlay."""
        # Count vegetation zones
        vegetation_zones = len(self.detector.temporal_intelligence.get_vegetation_locations())
        
        # Movement analysis stats
        total_tracks = len(self.detector.movement_intelligence.movement_patterns)
        
        # Detection confidence distribution
        high_conf = sum(1 for t in tracks if t['confidence'] > 0.8)
        medium_conf = sum(1 for t in tracks if 0.6 <= t['confidence'] <= 0.8)
        low_conf = sum(1 for t in tracks if t['confidence'] < 0.6)
        
        stats = [
            f"Intelligence Stats:",
            f"Active Insects: {len(tracks)}",
            f"Vegetation Zones: {vegetation_zones}",
            f"Total Tracks: {total_tracks}",
            f"Confidence: H:{high_conf} M:{medium_conf} L:{low_conf}",
            "",
            f"Temporal Patterns: {'ON' if self.show_temporal_patterns else 'OFF'}",
            f"Movement Analysis: {'ON' if self.show_movement_analysis else 'OFF'}",
            f"Vegetation Zones: {'ON' if self.show_vegetation_zones else 'OFF'}"
        ]
        
        # Draw stats panel
        panel_width = 300
        panel_height = len(stats) * 20 + 20
        
        cv2.rectangle(frame, (10, 10), (panel_width, panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (panel_width, panel_height), (0, 255, 0), 2)
        
        for i, stat in enumerate(stats):
            cv2.putText(frame, stat, (15, 30 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


def main():
    parser = argparse.ArgumentParser(description='Intelligent insect detector with temporal analysis')
    parser.add_argument('url', nargs='?',
                       default='https://www.youtube.com/watch?v=ZdfJcN-yBIE',
                       help='Video URL')
    args = parser.parse_args()
    
    if args.url.isdigit():
        args.url = int(args.url)
    
    print("🧠 Intelligent Insect Detector")
    print(f"📹 Source: {args.url}")
    
    visualizer = IntelligentInsectVisualizer(args.url)
    
    try:
        visualizer.start()
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()