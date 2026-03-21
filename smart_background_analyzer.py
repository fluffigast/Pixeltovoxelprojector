#!/usr/bin/env python3
"""Smart background-aware insect analyzer with advanced filtering."""

import cv2
import numpy as np
import time
import subprocess
import argparse
import math
from collections import deque, defaultdict
from typing import List, Tuple, Dict, Optional

class SmartBackgroundModel:
    """Advanced background modeling for outdoor scenes with vegetation."""
    
    def __init__(self):
        # Multiple background models for robustness
        self.bg_mog2 = cv2.createBackgroundSubtractorMOG2(
            history=500,        # Longer history for stable backgrounds
            varThreshold=20,    # Higher threshold to ignore small movements
            detectShadows=True  # Detect shadows to ignore them
        )
        
        self.bg_knn = cv2.createBackgroundSubtractorKNN(
            history=500,
            dist2Threshold=800, # Higher threshold for outdoor scenes
            detectShadows=True
        )
        
        # Adaptive background for dynamic scenes
        self.learning_rate = 0.01  # Slow learning for stable background
        self.fast_learning_rate = 0.1  # Fast learning for initialization
        
        # Motion stabilization
        self.prev_frame = None
        self.motion_vectors = deque(maxlen=10)
        self.global_motion_threshold = 2.0
        
        # Environmental filters
        self.wind_compensation = True
        self.shadow_suppression = True
        self.texture_filtering = True
        
        # Background quality assessment
        self.background_stability = 0.0
        self.frame_count = 0
        self.initialization_frames = 100
        
    def update_background(self, frame: np.ndarray) -> np.ndarray:
        """Update background model with environmental awareness."""
        self.frame_count += 1
        
        # Use fast learning during initialization
        current_learning_rate = (self.fast_learning_rate if self.frame_count < self.initialization_frames 
                                else self.learning_rate)
        
        # Detect global motion (camera shake, wind)
        global_motion = self._detect_global_motion(frame)
        
        # Adjust learning rate based on global motion
        if global_motion > self.global_motion_threshold:
            current_learning_rate *= 0.1  # Slower learning during motion
        
        # Apply background subtraction
        fg_mog2 = self.bg_mog2.apply(frame, learningRate=current_learning_rate)
        fg_knn = self.bg_knn.apply(frame, learningRate=current_learning_rate)
        
        # Combine foreground masks
        combined_fg = self._combine_foreground_masks(fg_mog2, fg_knn, frame)
        
        # Apply environmental filters
        filtered_fg = self._apply_environmental_filters(combined_fg, frame)
        
        self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        return filtered_fg
    
    def _detect_global_motion(self, frame: np.ndarray) -> float:
        """Detect global motion (camera shake, wind) using optical flow."""
        if self.prev_frame is None:
            return 0.0
        
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate sparse optical flow on corner features
        corners = cv2.goodFeaturesToTrack(self.prev_frame, maxCorners=100, 
                                         qualityLevel=0.3, minDistance=7, blockSize=7)
        
        if corners is not None and len(corners) > 10:
            # Track features
            new_corners, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_frame, current_gray, corners, None,
                winSize=(15, 15), maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )
            
            # Calculate motion vectors
            good_old = corners[status == 1]
            good_new = new_corners[status == 1]
            
            if len(good_old) > 5:
                motion_vectors = good_new - good_old
                global_motion = np.mean(np.linalg.norm(motion_vectors, axis=1))
                self.motion_vectors.append(global_motion)
                return global_motion
        
        return 0.0
    
    def _combine_foreground_masks(self, fg_mog2: np.ndarray, fg_knn: np.ndarray, 
                                 frame: np.ndarray) -> np.ndarray:
        """Intelligently combine foreground masks."""
        # Remove shadows (value 127 in MOG2)
        if self.shadow_suppression:
            fg_mog2[fg_mog2 == 127] = 0
            fg_knn[fg_knn == 127] = 0
        
        # Combine with weighted average based on background stability
        if self.frame_count < self.initialization_frames:
            # During initialization, trust both equally
            combined = cv2.bitwise_or(fg_mog2, fg_knn)
        else:
            # After initialization, be more conservative
            combined = cv2.bitwise_and(fg_mog2, fg_knn)  # Only where both agree
            
            # Add areas where at least one strongly detects
            strong_mog2 = cv2.threshold(fg_mog2, 200, 255, cv2.THRESH_BINARY)[1]
            strong_knn = cv2.threshold(fg_knn, 200, 255, cv2.THRESH_BINARY)[1]
            strong_combined = cv2.bitwise_or(strong_mog2, strong_knn)
            
            combined = cv2.bitwise_or(combined, strong_combined)
        
        return combined
    
    def _apply_environmental_filters(self, fg_mask: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """Apply filters specific to outdoor environments."""
        filtered = fg_mask.copy()
        
        # 1. Texture-based filtering (leaves have high texture, insects don't)
        if self.texture_filtering:
            filtered = self._filter_high_texture_areas(filtered, frame)
        
        # 2. Wind compensation (ignore rhythmic motion)
        if self.wind_compensation:
            filtered = self._compensate_wind_motion(filtered)
        
        # 3. Size-based morphological filtering
        filtered = self._morphological_filtering(filtered)
        
        # 4. Temporal consistency filtering
        filtered = self._temporal_consistency_filter(filtered)
        
        return filtered
    
    def _filter_high_texture_areas(self, fg_mask: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """Filter out high-texture areas (leaves, branches)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate local texture using Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_map = cv2.GaussianBlur(np.abs(laplacian), (9, 9), 0)
        
        # High texture threshold (leaves have high texture)
        high_texture_threshold = np.percentile(texture_map, 85)
        high_texture_mask = texture_map > high_texture_threshold
        
        # Reduce foreground confidence in high texture areas
        fg_mask[high_texture_mask] = (fg_mask[high_texture_mask] * 0.3).astype(np.uint8)
        
        return fg_mask
    
    def _compensate_wind_motion(self, fg_mask: np.ndarray) -> np.ndarray:
        """Compensate for wind-induced motion patterns."""
        if len(self.motion_vectors) < 5:
            return fg_mask
        
        # Check for consistent global motion (wind)
        recent_motion = list(self.motion_vectors)[-5:]
        motion_std = np.std(recent_motion)
        motion_mean = np.mean(recent_motion)
        
        # If motion is consistent (low std), it's likely wind
        if motion_std < 2.0 and motion_mean > 1.0:
            # Reduce sensitivity during windy conditions
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        return fg_mask
    
    def _morphological_filtering(self, fg_mask: np.ndarray) -> np.ndarray:
        """Apply size-appropriate morphological filtering."""
        # Small kernel for preserving insect-sized objects
        small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        medium_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        
        # Remove noise while preserving small objects
        filtered = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, small_kernel, iterations=1)
        
        # Close small gaps
        filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, medium_kernel, iterations=1)
        
        return filtered
    
    def _temporal_consistency_filter(self, fg_mask: np.ndarray) -> np.ndarray:
        """Filter based on temporal consistency."""
        # This is a placeholder - in a full implementation, you'd track
        # pixel-level changes over time to filter out inconsistent detections
        return fg_mask


class SmartInsectDetector:
    """Smart insect detector with advanced background handling."""
    
    def __init__(self):
        # Enhanced background model
        self.background_model = SmartBackgroundModel()
        
        # Detection parameters optimized for outdoor scenes
        self.min_area = 3
        self.max_area = 120
        self.min_aspect_ratio = 0.1
        self.max_aspect_ratio = 6.0
        
        # Advanced shape filtering
        self.min_solidity = 0.2
        self.max_solidity = 1.0
        self.min_extent = 0.15
        self.min_compactness = 0.1
        
        # Motion characteristics
        self.min_motion_intensity = 15
        self.motion_history = deque(maxlen=5)
        
        # Multi-frame validation
        self.frame_buffer = deque(maxlen=4)
        self.detection_buffer = deque(maxlen=3)
        
        # Environmental adaptation
        self.auto_adapt = True
        self.adaptation_history = deque(maxlen=100)
        
    def detect_insects(self, frame: np.ndarray) -> List[Dict]:
        """Advanced insect detection with smart background handling."""
        # Update background model
        fg_mask = self.background_model.update_background(frame)
        
        # Primary detection from background subtraction
        bg_detections = self._detect_from_background(fg_mask, frame)
        
        # Secondary detection from frame differencing
        frame_detections = []
        if len(self.frame_buffer) >= 2:
            frame_detections = self._detect_from_frame_difference(frame)
        
        # Add frame to buffer
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.frame_buffer.append(gray)
        
        # Combine and validate detections
        all_detections = bg_detections + frame_detections
        validated_detections = self._validate_detections(all_detections, frame)
        
        # Store detection history for adaptation
        self.detection_buffer.append(validated_detections)
        
        # Adapt parameters if enabled
        if self.auto_adapt:
            self._adapt_parameters()
        
        return validated_detections
    
    def _detect_from_background(self, fg_mask: np.ndarray, frame: np.ndarray) -> List[Dict]:
        """Detect insects from background subtraction."""
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            detection = self._analyze_contour_advanced(contour, frame, 'background')
            if detection:
                detections.append(detection)
        
        return detections
    
    def _detect_from_frame_difference(self, frame: np.ndarray) -> List[Dict]:
        """Detect fast insects using frame differencing."""
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.GaussianBlur(current_gray, (3, 3), 0)
        
        detections = []
        
        # Compare with multiple previous frames
        for i, prev_gray in enumerate(list(self.frame_buffer)[-2:]):
            # Frame difference
            diff = cv2.absdiff(current_gray, prev_gray)
            
            # Adaptive threshold
            threshold = max(self.min_motion_intensity, np.percentile(diff, 95) * 0.4)
            _, motion_mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
            
            # Clean up motion mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                detection = self._analyze_contour_advanced(contour, frame, f'motion_{i}')
                if detection:
                    detections.append(detection)
        
        return detections
    
    def _analyze_contour_advanced(self, contour, frame: np.ndarray, method: str) -> Optional[Dict]:
        """Advanced contour analysis with environmental context."""
        if len(contour) < 5:
            return None
        
        # Basic measurements
        area = cv2.contourArea(contour)
        if area < self.min_area or area > self.max_area:
            return None
        
        perimeter = cv2.arcLength(contour, True)
        if perimeter < 6:  # Very small perimeter
            return None
        
        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        center = (x + w // 2, y + h // 2)
        
        # Aspect ratio
        aspect_ratio = w / h if h > 0 else 0
        if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
            return None
        
        # Shape analysis
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        if not (self.min_solidity <= solidity <= self.max_solidity):
            return None
        
        # Extent (how much of bounding box is filled)
        extent = area / (w * h) if w * h > 0 else 0
        if extent < self.min_extent:
            return None
        
        # Compactness (circularity measure)
        compactness = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        if compactness < self.min_compactness:
            return None
        
        # Advanced shape features
        moments = cv2.moments(contour)
        if moments['m00'] == 0:
            return None
        
        # Eccentricity (elongation measure)
        try:
            ellipse = cv2.fitEllipse(contour)
            major_axis = max(ellipse[1])
            minor_axis = min(ellipse[1])
            eccentricity = np.sqrt(1 - (minor_axis / major_axis)**2) if major_axis > 0 else 0
        except:
            eccentricity = 0
        
        # Environmental context analysis
        context_score = self._analyze_environmental_context(contour, frame)
        
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(area, solidity, compactness, 
                                                     eccentricity, context_score)
        
        if quality_score < 0.3:  # Quality threshold
            return None
        
        return {
            'center': center,
            'bbox': (x, y, w, h),
            'area': area,
            'perimeter': perimeter,
            'aspect_ratio': aspect_ratio,
            'solidity': solidity,
            'extent': extent,
            'compactness': compactness,
            'eccentricity': eccentricity,
            'quality_score': quality_score,
            'method': method,
            'contour': contour
        }
    
    def _analyze_environmental_context(self, contour, frame: np.ndarray) -> float:
        """Analyze environmental context around detection."""
        # Get region around contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Expand region for context analysis
        margin = max(10, max(w, h))
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(frame.shape[1], x + w + margin)
        y2 = min(frame.shape[0], y + h + margin)
        
        context_region = frame[y1:y2, x1:x2]
        
        if context_region.size == 0:
            return 0.5
        
        # Analyze context
        gray_context = cv2.cvtColor(context_region, cv2.COLOR_BGR2GRAY)
        
        # Edge density (insects stand out from smooth backgrounds)
        edges = cv2.Canny(gray_context, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Color variation (insects often contrast with background)
        color_std = np.std(gray_context)
        
        # Calculate context score
        context_score = 0.5
        
        # Moderate edge density is good (not too cluttered, not too smooth)
        if 0.02 <= edge_density <= 0.1:
            context_score += 0.2
        
        # Good contrast indicates potential insect
        if color_std > 20:
            context_score += 0.2
        
        # Too much texture suggests vegetation
        if edge_density > 0.15:
            context_score -= 0.3
        
        return np.clip(context_score, 0, 1)
    
    def _calculate_quality_score(self, area: float, solidity: float, compactness: float,
                                eccentricity: float, context_score: float) -> float:
        """Calculate overall detection quality score."""
        score = 0.0
        
        # Size score (prefer insect-sized objects)
        if 5 <= area <= 50:
            score += 0.3
        elif 3 <= area <= 80:
            score += 0.2
        else:
            score += 0.1
        
        # Shape score
        if 0.3 <= solidity <= 0.8:
            score += 0.2
        
        if 0.2 <= compactness <= 0.8:
            score += 0.2
        
        # Eccentricity score (insects can be elongated)
        if 0.2 <= eccentricity <= 0.9:
            score += 0.1
        
        # Context score
        score += context_score * 0.2
        
        return score
    
    def _validate_detections(self, detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        """Multi-stage validation of detections."""
        if not detections:
            return []
        
        # Stage 1: Remove duplicates
        filtered = self._remove_duplicates(detections)
        
        # Stage 2: Cross-validation between methods
        validated = self._cross_validate_methods(filtered)
        
        # Stage 3: Temporal validation
        if len(self.detection_buffer) >= 2:
            validated = self._temporal_validation(validated)
        
        # Stage 4: Final quality filter
        final = [det for det in validated if det['quality_score'] >= 0.4]
        
        return final
    
    def _remove_duplicates(self, detections: List[Dict]) -> List[Dict]:
        """Remove duplicate detections."""
        if len(detections) <= 1:
            return detections
        
        filtered = []
        for detection in detections:
            is_duplicate = False
            center = detection['center']
            
            for existing in filtered:
                existing_center = existing['center']
                distance = math.sqrt((center[0] - existing_center[0])**2 + 
                                   (center[1] - existing_center[1])**2)
                
                if distance < 10:  # Duplicate threshold
                    # Keep the one with higher quality
                    if detection['quality_score'] > existing['quality_score']:
                        filtered.remove(existing)
                        break
                    else:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                filtered.append(detection)
        
        return filtered
    
    def _cross_validate_methods(self, detections: List[Dict]) -> List[Dict]:
        """Cross-validate detections from different methods."""
        # Group detections by method
        method_groups = defaultdict(list)
        for det in detections:
            method_groups[det['method']].append(det)
        
        # If multiple methods detect similar objects, increase confidence
        validated = []
        for detection in detections:
            confidence_boost = 0
            center = detection['center']
            
            # Check if other methods detected similar object
            for method, method_detections in method_groups.items():
                if method != detection['method']:
                    for other_det in method_detections:
                        other_center = other_det['center']
                        distance = math.sqrt((center[0] - other_center[0])**2 + 
                                           (center[1] - other_center[1])**2)
                        if distance < 15:
                            confidence_boost += 0.1
            
            detection['quality_score'] = min(1.0, detection['quality_score'] + confidence_boost)
            validated.append(detection)
        
        return validated
    
    def _temporal_validation(self, current_detections: List[Dict]) -> List[Dict]:
        """Validate detections based on temporal consistency."""
        # This is a simplified temporal validation
        # In practice, you'd track objects across frames
        return current_detections
    
    def _adapt_parameters(self):
        """Automatically adapt detection parameters."""
        if len(self.detection_buffer) < 5:
            return
        
        # Calculate recent detection rate
        recent_detections = sum(len(dets) for dets in list(self.detection_buffer)[-5:])
        avg_detections = recent_detections / 5
        
        self.adaptation_history.append(avg_detections)
        
        if len(self.adaptation_history) >= 20:
            long_term_avg = np.mean(list(self.adaptation_history)[-20:])
            
            # If too many detections, increase selectivity
            if long_term_avg > 15:
                self.min_area = min(8, self.min_area + 1)
                self.min_solidity = min(0.4, self.min_solidity + 0.05)
            
            # If too few detections, decrease selectivity
            elif long_term_avg < 2:
                self.min_area = max(2, self.min_area - 1)
                self.min_solidity = max(0.15, self.min_solidity - 0.05)


class SmartInsectTracker:
    """Smart insect tracker with background awareness."""
    
    def __init__(self):
        self.detector = SmartInsectDetector()
        self.tracks = {}
        self.next_track_id = 0
        
    def update(self, frame: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        """Update tracking with new frame."""
        # Detect insects
        detections = self.detector.detect_insects(frame)
        
        # Update tracks (simplified for this example)
        active_tracks = self._update_tracks_simple(detections)
        
        # Get background mask for visualization
        bg_mask = self.detector.background_model.bg_mog2.getBackgroundImage()
        
        return active_tracks, bg_mask
    
    def _update_tracks_simple(self, detections: List[Dict]) -> List[Dict]:
        """Simple tracking update."""
        # This is a placeholder - implement full tracking logic here
        for detection in detections:
            detection['id'] = self.next_track_id
            detection['confidence'] = detection['quality_score']
            detection['behavior'] = 'normal'
            self.next_track_id += 1
        
        return detections


class SmartBackgroundVisualizer:
    """Visualizer with background analysis capabilities."""
    
    def __init__(self, stream_url: str):
        self.stream_url = stream_url
        self.tracker = SmartInsectTracker()
        
        # Visualization settings
        self.show_background = False
        self.show_foreground = False
        self.show_debug = True
        self.overlay_opacity = 0.8
        
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
        """Start smart background analysis."""
        url = self.get_stream_url()
        print(f"🧠 Smart Background-Aware Insect Analyzer: {url}")
        
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            raise RuntimeError("Failed to open stream")
        
        print("🎯 Smart Background Analysis Started!")
        print("\n✨ Features:")
        print("  • Advanced background modeling")
        print("  • Environmental context analysis")
        print("  • Wind/shadow compensation")
        print("  • Multi-method validation")
        print("  • Adaptive parameter tuning")
        print("\n🎮 Controls:")
        print("  q - Quit")
        print("  b - Toggle background view")
        print("  f - Toggle foreground mask")
        print("  d - Toggle debug info")
        print("  r - Reset background model")
        
        cv2.namedWindow('Smart Insect Analysis', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Smart Insect Analysis', 1400, 900)
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Resize frame
            frame = cv2.resize(frame, (1200, 800))
            
            # Process frame
            tracks, bg_image = self.tracker.update(frame)
            
            # Create visualization
            vis_frame = self._create_smart_visualization(frame, tracks, bg_image)
            
            cv2.imshow('Smart Insect Analysis', vis_frame)
            
            # Handle controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('b'):
                self.show_background = not self.show_background
                print(f"Background view: {'ON' if self.show_background else 'OFF'}")
            elif key == ord('f'):
                self.show_foreground = not self.show_foreground
                print(f"Foreground mask: {'ON' if self.show_foreground else 'OFF'}")
            elif key == ord('d'):
                self.show_debug = not self.show_debug
                print(f"Debug info: {'ON' if self.show_debug else 'OFF'}")
            elif key == ord('r'):
                # Reset background model
                self.tracker.detector.background_model = SmartBackgroundModel()
                print("Background model reset")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def _create_smart_visualization(self, frame: np.ndarray, tracks: List[Dict], 
                                   bg_image: Optional[np.ndarray]) -> np.ndarray:
        """Create smart visualization with background analysis."""
        if self.show_background and bg_image is not None:
            # Show background model
            bg_resized = cv2.resize(bg_image, (frame.shape[1], frame.shape[0]))
            if len(bg_resized.shape) == 2:
                bg_resized = cv2.cvtColor(bg_resized, cv2.COLOR_GRAY2BGR)
            
            # Split screen: original on left, background on right
            combined = np.hstack([frame, bg_resized])
        else:
            combined = frame.copy()
        
        # Draw detections
        for track in tracks:
            center = track['center']
            bbox = track['bbox']
            quality = track['quality_score']
            method = track['method']
            
            # Color based on quality and method
            if quality > 0.8:
                color = (0, 0, 255)  # Red - high quality
            elif quality > 0.6:
                color = (0, 165, 255)  # Orange - good quality
            elif 'background' in method:
                color = (0, 255, 0)  # Green - background method
            else:
                color = (255, 255, 0)  # Cyan - motion method
            
            # Draw detection
            cv2.circle(combined, center, 4, color, -1)
            cv2.circle(combined, center, 8, color, 1)
            
            # Draw bounding box for high quality detections
            if quality > 0.7:
                x, y, w, h = bbox
                cv2.rectangle(combined, (x, y), (x + w, y + h), color, 1)
                
                # Quality label
                cv2.putText(combined, f"{quality:.2f}", (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Add debug information
        if self.show_debug:
            self._add_debug_info(combined, tracks)
        
        return combined
    
    def _add_debug_info(self, frame: np.ndarray, tracks: List[Dict]):
        """Add debug information overlay."""
        # Background model info
        bg_model = self.tracker.detector.background_model
        
        debug_info = [
            f"Tracks: {len(tracks)}",
            f"BG Learning: {bg_model.learning_rate:.3f}",
            f"Global Motion: {np.mean(bg_model.motion_vectors) if bg_model.motion_vectors else 0:.1f}",
            f"Frame Count: {bg_model.frame_count}",
            f"Initialization: {'Yes' if bg_model.frame_count < bg_model.initialization_frames else 'No'}"
        ]
        
        # Method breakdown
        method_counts = defaultdict(int)
        for track in tracks:
            method_counts[track['method']] += 1
        
        debug_info.append("")
        debug_info.append("Detection Methods:")
        for method, count in method_counts.items():
            debug_info.append(f"  {method}: {count}")
        
        # Draw debug panel
        panel_width = 300
        panel_height = len(debug_info) * 25 + 20
        
        cv2.rectangle(frame, (10, 10), (panel_width, panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (panel_width, panel_height), (0, 255, 0), 2)
        
        for i, info in enumerate(debug_info):
            cv2.putText(frame, info, (15, 35 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


def main():
    parser = argparse.ArgumentParser(description='Smart background-aware insect analyzer')
    parser.add_argument('url', nargs='?',
                       default='https://www.youtube.com/watch?v=ZdfJcN-yBIE',
                       help='Video URL')
    args = parser.parse_args()
    
    if args.url.isdigit():
        args.url = int(args.url)
    
    print("🧠 Smart Background-Aware Insect Analyzer")
    print(f"📹 Source: {args.url}")
    
    visualizer = SmartBackgroundVisualizer(args.url)
    
    try:
        visualizer.start()
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pass


if __name__ == "__main__":
    main()