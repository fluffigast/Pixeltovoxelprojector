#!/usr/bin/env python3
"""Clean insect analyzer with minimal, elegant overlays and analytical dashboard."""

import cv2
import numpy as np
import time
import subprocess
import argparse
import math
from collections import deque, defaultdict
from typing import List, Tuple, Dict, Optional

class CleanInsectAnalyzer:
    """Clean, minimal insect detection with analytical focus."""
    
    def __init__(self):
        # Detection parameters - optimized for clarity
        self.min_area = 2
        self.max_area = 100
        self.motion_threshold = 6
        self.confidence_threshold = 0.6
        
        # Background subtraction
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=100, varThreshold=8, detectShadows=False
        )
        
        # Tracking
        self.tracks = {}
        self.next_track_id = 0
        self.frame_buffer = deque(maxlen=3)
        
        # Analytics
        self.detection_history = deque(maxlen=300)  # 10 seconds at 30fps
        self.behavior_stats = defaultdict(int)
        self.size_distribution = deque(maxlen=1000)
        self.speed_distribution = deque(maxlen=1000)
        
        # Visualization modes
        self.overlay_mode = 'minimal'  # 'minimal', 'analytical', 'scientific'
        self.show_trails = True
        self.trail_length = 15
        self.show_zones = False
        
    def detect_insects(self, frame: np.ndarray) -> List[Dict]:
        """Clean, focused insect detection."""
        # Background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            detection = self._analyze_contour(contour)
            if detection:
                detections.append(detection)
        
        # Frame differencing for fast insects
        if len(self.frame_buffer) >= 1:
            diff_detections = self._frame_difference_detection(frame)
            detections.extend(diff_detections)
        
        # Add frame to buffer
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.frame_buffer.append(gray)
        
        # Filter and merge
        return self._filter_merge_detections(detections)
    
    def _analyze_contour(self, contour) -> Optional[Dict]:
        """Analyze contour for insect characteristics."""
        area = cv2.contourArea(contour)
        if area < self.min_area or area > self.max_area:
            return None
        
        # Basic shape analysis
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        
        if aspect_ratio < 0.1 or aspect_ratio > 8.0:
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
        """Detect fast-moving insects via frame differencing."""
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = self.frame_buffer[-1]
        
        # Frame difference
        diff = cv2.absdiff(current_gray, prev_gray)
        _, thresh = cv2.threshold(diff, self.motion_threshold, 255, cv2.THRESH_BINARY)
        
        # Find small moving objects
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            detection = self._analyze_contour(contour)
            if detection:
                detection['method'] = 'frame_diff'
                detections.append(detection)
        
        return detections
    
    def _filter_merge_detections(self, detections: List[Dict]) -> List[Dict]:
        """Filter and merge nearby detections."""
        if not detections:
            return []
        
        # Simple duplicate removal
        filtered = []
        for detection in detections:
            is_duplicate = False
            for existing in filtered:
                dist = math.sqrt((detection['center'][0] - existing['center'][0])**2 + 
                               (detection['center'][1] - existing['center'][1])**2)
                if dist < 8:
                    # Keep the larger one
                    if detection['area'] > existing['area']:
                        filtered.remove(existing)
                        break
                    else:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                filtered.append(detection)
        
        return filtered
    
    def update_tracks(self, detections: List[Dict]) -> List[Dict]:
        """Simple, clean tracking."""
        # Update existing tracks
        updated_tracks = {}
        assigned_detections = set()
        
        for track_id, track in self.tracks.items():
            best_detection = None
            best_distance = float('inf')
            best_idx = -1
            
            for i, detection in enumerate(detections):
                if i in assigned_detections:
                    continue
                
                dist = math.sqrt((detection['center'][0] - track['last_center'][0])**2 + 
                               (detection['center'][1] - track['last_center'][1])**2)
                
                if dist < best_distance and dist < 30:
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
                
                # Simple behavior analysis
                self._analyze_behavior(track)
                
                updated_tracks[track_id] = track
                assigned_detections.add(best_idx)
            else:
                # Track lost
                track['lost_frames'] += 1
                if track['lost_frames'] < 5:
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
                    'speed': 0,
                    'behavior': 'normal',
                    'confidence': 0.5
                }
                updated_tracks[self.next_track_id] = new_track
                self.next_track_id += 1
        
        self.tracks = updated_tracks
        
        # Update analytics
        self._update_analytics()
        
        # Return active, confident tracks
        return [track for track in self.tracks.values() 
                if track['lost_frames'] == 0 and track['confidence'] >= self.confidence_threshold]
    
    def _analyze_behavior(self, track: Dict):
        """Simple behavior analysis."""
        positions = track['positions']
        
        if len(positions) < 2:
            return
        
        # Calculate speed
        if len(positions) >= 2:
            recent = positions[-5:]
            total_dist = 0
            for i in range(1, len(recent)):
                dx = recent[i][0] - recent[i-1][0]
                dy = recent[i][1] - recent[i-1][1]
                total_dist += math.sqrt(dx*dx + dy*dy)
            track['speed'] = total_dist / (len(recent) - 1) if len(recent) > 1 else 0
        
        # Determine behavior
        if track['speed'] < 2:
            track['behavior'] = 'hovering'
        elif track['speed'] > 15:
            track['behavior'] = 'fast'
        else:
            track['behavior'] = 'normal'
        
        # Simple confidence based on track length and consistency
        track['confidence'] = min(1.0, 0.3 + track['age'] * 0.05 + 
                                (1.0 if 2 <= track['speed'] <= 25 else 0))
    
    def _update_analytics(self):
        """Update analytical data."""
        active_tracks = [t for t in self.tracks.values() if t['lost_frames'] == 0]
        
        # Detection count history
        self.detection_history.append(len(active_tracks))
        
        # Behavior statistics
        for track in active_tracks:
            self.behavior_stats[track['behavior']] += 1
            self.size_distribution.append(track['detection']['area'])
            self.speed_distribution.append(track['speed'])


class CleanInsectVisualizer:
    """Clean, elegant visualization system."""
    
    def __init__(self, stream_url: str):
        self.stream_url = stream_url
        self.analyzer = CleanInsectAnalyzer()
        
        # Video parameters
        self.cap = None
        self.width = 1200
        self.height = 800
        
        # Visualization settings
        self.overlay_opacity = 0.7
        self.show_dashboard = True
        self.dashboard_size = 300  # Width of side dashboard
        self.highlight_color = (0, 255, 100)  # Bright green
        self.trail_colors = {
            'normal': (100, 255, 100),
            'hovering': (100, 255, 255),
            'fast': (255, 100, 100)
        }
        
        # Analytics
        self.frame_count = 0
        self.start_time = time.time()
        self.fps_history = deque(maxlen=30)
        
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
        """Start clean visualization."""
        url = self.get_stream_url()
        print(f"🔍 Clean Insect Analyzer: {url}")
        
        self.cap = cv2.VideoCapture(url)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open stream")
        
        print("✨ Clean Insect Analysis Started!")
        print("\n🎯 Features:")
        print("  • Minimal, elegant overlays")
        print("  • Real-time analytics dashboard")
        print("  • Behavioral analysis")
        print("  • Clean visual design")
        print("\n🎮 Controls:")
        print("  q - Quit")
        print("  d - Toggle dashboard")
        print("  t - Toggle trails")
        print("  z - Toggle zones")
        print("  1/2/3 - Overlay modes (minimal/analytical/scientific)")
        print("  +/- - Adjust confidence threshold")
        print("  space - Pause")
        
        # Create windows
        if self.show_dashboard:
            cv2.namedWindow('Clean Insect Analysis', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Clean Insect Analysis', 1500, 900)
        else:
            cv2.namedWindow('Clean Insect Analysis', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Clean Insect Analysis', 1200, 800)
        
        paused = False
        
        while True:
            frame_start = time.time()
            
            if not paused:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                self.frame_count += 1
                
                # Resize maintaining aspect ratio
                frame = self._resize_maintain_aspect(frame, self.width, self.height)
                
                # Analyze insects
                detections = self.analyzer.detect_insects(frame)
                active_tracks = self.analyzer.update_tracks(detections)
                
                # Create clean visualization
                vis_frame = self._create_clean_visualization(frame, active_tracks)
            else:
                # Show paused frame
                vis_frame = self._add_pause_overlay(vis_frame)
            
            cv2.imshow('Clean Insect Analysis', vis_frame)
            
            # Performance tracking
            if not paused:
                frame_time = time.time() - frame_start
                self.fps_history.append(1.0 / frame_time if frame_time > 0 else 0)
            
            # Handle controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord(' '):
                paused = not paused
                print("PAUSED" if paused else "RESUMED")
            elif key == ord('d'):
                self.show_dashboard = not self.show_dashboard
                print(f"Dashboard: {'ON' if self.show_dashboard else 'OFF'}")
            elif key == ord('t'):
                self.analyzer.show_trails = not self.analyzer.show_trails
                print(f"Trails: {'ON' if self.analyzer.show_trails else 'OFF'}")
            elif key == ord('z'):
                self.analyzer.show_zones = not self.analyzer.show_zones
                print(f"Zones: {'ON' if self.analyzer.show_zones else 'OFF'}")
            elif key == ord('1'):
                self.analyzer.overlay_mode = 'minimal'
                print("Overlay mode: Minimal")
            elif key == ord('2'):
                self.analyzer.overlay_mode = 'analytical'
                print("Overlay mode: Analytical")
            elif key == ord('3'):
                self.analyzer.overlay_mode = 'scientific'
                print("Overlay mode: Scientific")
            elif key == ord('=') or key == ord('+'):
                self.analyzer.confidence_threshold = max(0.1, self.analyzer.confidence_threshold - 0.1)
                print(f"Confidence threshold: {self.analyzer.confidence_threshold:.1f}")
            elif key == ord('-'):
                self.analyzer.confidence_threshold = min(0.9, self.analyzer.confidence_threshold + 0.1)
                print(f"Confidence threshold: {self.analyzer.confidence_threshold:.1f}")
    
    def _resize_maintain_aspect(self, frame: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
        """Resize frame maintaining aspect ratio."""
        h, w = frame.shape[:2]
        aspect = w / h
        
        if aspect > target_width / target_height:
            # Width is limiting factor
            new_width = target_width
            new_height = int(target_width / aspect)
        else:
            # Height is limiting factor
            new_height = target_height
            new_width = int(target_height * aspect)
        
        return cv2.resize(frame, (new_width, new_height))
    
    def _create_clean_visualization(self, frame: np.ndarray, tracks: List[Dict]) -> np.ndarray:
        """Create clean, elegant visualization."""
        if self.show_dashboard:
            # Create combined frame with dashboard
            total_width = frame.shape[1] + self.dashboard_size
            total_height = max(frame.shape[0], 800)
            
            combined = np.zeros((total_height, total_width, 3), dtype=np.uint8)
            combined[:frame.shape[0], :frame.shape[1]] = frame
            
            # Add minimal overlays to video
            video_section = combined[:frame.shape[0], :frame.shape[1]]
            self._add_minimal_overlays(video_section, tracks)
            
            # Create dashboard
            dashboard = combined[:, frame.shape[1]:]
            self._create_dashboard(dashboard, tracks)
            
            return combined
        else:
            # Just video with overlays
            vis = frame.copy()
            self._add_minimal_overlays(vis, tracks)
            return vis
    
    def _add_minimal_overlays(self, frame: np.ndarray, tracks: List[Dict]):
        """Add minimal, elegant overlays to video."""
        for track in tracks:
            center_x, center_y = track['last_center']
            confidence = track['confidence']
            behavior = track['behavior']
            
            # Color based on behavior
            if behavior == 'hovering':
                color = (0, 255, 255)  # Yellow
            elif behavior == 'fast':
                color = (0, 100, 255)  # Orange-red
            else:
                color = self.highlight_color  # Green
            
            # Overlay mode determines what to show
            if self.analyzer.overlay_mode == 'minimal':
                # Just a small dot
                cv2.circle(frame, (center_x, center_y), 2, color, -1)
                
            elif self.analyzer.overlay_mode == 'analytical':
                # Dot + confidence
                cv2.circle(frame, (center_x, center_y), 3, color, -1)
                cv2.circle(frame, (center_x, center_y), 6, color, 1)
                
                if confidence > 0.8:
                    cv2.putText(frame, f"{confidence:.1f}", (center_x + 8, center_y - 8),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                
            elif self.analyzer.overlay_mode == 'scientific':
                # Full annotation
                bbox = track['detection']['bbox']
                x, y, w, h = bbox
                
                # Bounding box (very thin)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
                cv2.circle(frame, (center_x, center_y), 2, color, -1)
                
                # ID and stats
                label = f"#{track['id']} {confidence:.2f}"
                cv2.putText(frame, label, (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
            # Draw trails if enabled
            if self.analyzer.show_trails and len(track['positions']) > 1:
                trail_color = self.trail_colors.get(behavior, (150, 150, 150))
                positions = track['positions'][-self.analyzer.trail_length:]
                
                if len(positions) > 1:
                    # Draw trail with fading alpha
                    for i in range(1, len(positions)):
                        alpha = i / len(positions)
                        pt1 = tuple(positions[i-1])
                        pt2 = tuple(positions[i])
                        
                        # Simple line (OpenCV doesn't support alpha easily)
                        cv2.line(frame, pt1, pt2, trail_color, 1)
        
        # Add zones if enabled
        if self.analyzer.show_zones:
            self._draw_analysis_zones(frame)
    
    def _draw_analysis_zones(self, frame: np.ndarray):
        """Draw analysis zones."""
        h, w = frame.shape[:2]
        
        # Grid zones
        zone_color = (50, 50, 50)
        
        # Vertical lines
        for i in range(1, 4):
            x = w * i // 4
            cv2.line(frame, (x, 0), (x, h), zone_color, 1)
        
        # Horizontal lines
        for i in range(1, 3):
            y = h * i // 3
            cv2.line(frame, (0, y), (w, y), zone_color, 1)
        
        # Zone labels
        cv2.putText(frame, "Analysis Zones", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, zone_color, 1)
    
    def _create_dashboard(self, dashboard: np.ndarray, tracks: List[Dict]):
        """Create analytical dashboard."""
        dashboard.fill(0)  # Black background
        
        h, w = dashboard.shape[:2]
        
        # Dashboard title
        cv2.putText(dashboard, "INSECT ANALYSIS", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        y_pos = 60
        line_height = 25
        
        # Performance stats
        fps = np.mean(self.fps_history) if self.fps_history else 0
        runtime = time.time() - self.start_time
        
        stats = [
            f"Runtime: {runtime:.0f}s",
            f"FPS: {fps:.1f}",
            f"Frame: {self.frame_count}",
            "",
            f"Active Tracks: {len(tracks)}",
            f"Total Detected: {self.analyzer.next_track_id}",
            ""
        ]
        
        for stat in stats:
            cv2.putText(dashboard, stat, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_pos += line_height
        
        # Behavior distribution
        cv2.putText(dashboard, "BEHAVIOR:", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_pos += line_height + 5
        
        behavior_counts = defaultdict(int)
        for track in tracks:
            behavior_counts[track['behavior']] += 1
        
        for behavior, count in behavior_counts.items():
            color = self.trail_colors.get(behavior, (150, 150, 150))
            cv2.putText(dashboard, f"{behavior}: {count}", (20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            y_pos += line_height
        
        y_pos += 10
        
        # Detection history graph
        if len(self.analyzer.detection_history) > 10:
            cv2.putText(dashboard, "DETECTION RATE:", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_pos += line_height
            
            self._draw_simple_graph(dashboard, list(self.analyzer.detection_history)[-100:],
                                  (10, y_pos), (w-20, 60), (0, 255, 100))
            y_pos += 80
        
        # Speed distribution
        if len(self.analyzer.speed_distribution) > 10:
            cv2.putText(dashboard, "SPEED DIST:", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_pos += line_height
            
            speeds = list(self.analyzer.speed_distribution)[-100:]
            self._draw_histogram(dashboard, speeds, (10, y_pos), (w-20, 60), (100, 200, 255))
            y_pos += 80
        
        # Controls reminder
        y_pos = max(y_pos + 20, h - 150)
        cv2.putText(dashboard, "CONTROLS:", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_pos += line_height
        
        controls = [
            "d - Dashboard",
            "t - Trails",
            "1/2/3 - Modes",
            "+/- - Confidence",
            "space - Pause"
        ]
        
        for control in controls:
            cv2.putText(dashboard, control, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            y_pos += line_height
    
    def _draw_simple_graph(self, img: np.ndarray, data: List[float], pos: Tuple[int, int], 
                          size: Tuple[int, int], color: Tuple[int, int, int]):
        """Draw a simple line graph."""
        if len(data) < 2:
            return
        
        x_start, y_start = pos
        width, height = size
        
        # Background
        cv2.rectangle(img, pos, (x_start + width, y_start + height), (30, 30, 30), -1)
        cv2.rectangle(img, pos, (x_start + width, y_start + height), (100, 100, 100), 1)
        
        # Normalize data
        max_val = max(data) if max(data) > 0 else 1
        normalized = [d / max_val for d in data]
        
        # Draw line
        points = []
        for i, val in enumerate(normalized):
            x = x_start + int(i * width / len(data))
            y = y_start + height - int(val * height)
            points.append((x, y))
        
        if len(points) > 1:
            for i in range(1, len(points)):
                cv2.line(img, points[i-1], points[i], color, 1)
    
    def _draw_histogram(self, img: np.ndarray, data: List[float], pos: Tuple[int, int],
                       size: Tuple[int, int], color: Tuple[int, int, int]):
        """Draw a simple histogram."""
        if len(data) < 2:
            return
        
        x_start, y_start = pos
        width, height = size
        
        # Background
        cv2.rectangle(img, pos, (x_start + width, y_start + height), (30, 30, 30), -1)
        cv2.rectangle(img, pos, (x_start + width, y_start + height), (100, 100, 100), 1)
        
        # Create histogram bins
        bins = 10
        max_val = max(data) if data else 1
        bin_width = max_val / bins if max_val > 0 else 1
        
        hist = [0] * bins
        for val in data:
            if bin_width > 0:
                bin_idx = min(int(val / bin_width), bins - 1)
                hist[bin_idx] += 1
        
        # Draw bars
        max_count = max(hist) if max(hist) > 0 else 1
        bar_width = width // bins
        
        for i, count in enumerate(hist):
            bar_height = int((count / max_count) * height)
            bar_x = x_start + i * bar_width
            bar_y = y_start + height - bar_height
            
            cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width - 1, y_start + height),
                         color, -1)
    
    def _add_pause_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Add pause overlay."""
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(frame, 0.7, overlay, 0.3, 0, frame)
        
        # Pause text
        cv2.putText(frame, "PAUSED - Press SPACE to resume", (w//2 - 200, h//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return frame
    
    def stop(self):
        """Clean up."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Clean insect analyzer with minimal overlays')
    parser.add_argument('url', nargs='?',
                       default='https://www.youtube.com/watch?v=ZdfJcN-yBIE',
                       help='Video URL')
    args = parser.parse_args()
    
    if args.url.isdigit():
        args.url = int(args.url)
    
    print("🔍 Clean Insect Analyzer")
    print(f"📹 Source: {args.url}")
    
    visualizer = CleanInsectVisualizer(args.url)
    
    try:
        visualizer.start()
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        visualizer.stop()


if __name__ == "__main__":
    main()