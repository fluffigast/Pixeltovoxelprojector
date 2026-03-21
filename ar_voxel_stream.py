#!/usr/bin/env python3
"""Augmented Reality voxel overlay with realistic projection."""

import cv2
import numpy as np
import time
import subprocess
import argparse
from collections import deque

class ARVoxelStream:
    """Advanced AR voxel overlay with realistic depth and lighting."""
    
    def __init__(self, stream_url: str, voxel_dims=(96, 96, 96)):
        self.stream_url = stream_url
        self.voxel_dims = voxel_dims
        self.voxel_size = 0.03
        
        # Video properties
        self.cap = None
        self.fps = 30
        self.width = 640
        self.height = 480
        
        # Voxel data - intensity and color
        self.voxel_intensity = np.zeros(voxel_dims, dtype=np.float32)
        self.voxel_color = np.zeros((*voxel_dims, 3), dtype=np.float32)
        self.voxel_age = np.zeros(voxel_dims, dtype=np.float32)
        
        # Processing parameters
        self.voxel_decay = 0.98
        self.motion_threshold = 20
        self.last_frame = None
        
        # Camera model
        self.camera_fov = 55.0
        self.camera_pos = np.array([0.0, 0.0, 1.8])
        self.focal_length = None
        
        # AR visualization settings
        self.show_particles = True
        self.show_trails = True
        self.particle_alpha = 0.8
        self.glow_enabled = True
        
        # Performance tracking
        self.frame_times = deque(maxlen=30)
        self.process_every = 2  # Process every Nth frame
        self.frame_counter = 0
        
    def get_stream_url(self) -> str:
        """Extract direct stream URL."""
        if 'youtube.com' in self.stream_url or 'youtu.be' in self.stream_url:
            try:
                cmd = ['yt-dlp', '-f', 'best[height<=720]', '--get-url', '--no-playlist', self.stream_url]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
                if result.returncode == 0:
                    return result.stdout.strip()
            except:
                pass
        return self.stream_url
        
    def start(self):
        """Start AR processing."""
        stream_url = self.get_stream_url()
        print("Opening stream for AR voxel overlay...")
        
        self.cap = cv2.VideoCapture(stream_url)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open stream")
            
        # Get actual video dimensions
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        
        # Calculate focal length for projection
        self.focal_length = self.width / (2.0 * np.tan(np.radians(self.camera_fov) / 2.0))
        
        print(f"Stream: {self.width}x{self.height} @ {self.fps:.1f} fps")
        print(f"Focal length: {self.focal_length:.1f}")
        print("\nAR Voxel Controls:")
        print("  q - Quit")
        print("  r - Reset voxels")
        print("  p - Toggle particles")
        print("  t - Toggle trails")
        print("  g - Toggle glow effect")
        print("  +/- - Adjust opacity")
        print("  [ ] - Adjust motion sensitivity")
        print("  space - Pause/resume")
        
        cv2.namedWindow('AR Voxel Stream', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('AR Voxel Stream', min(1280, self.width), min(720, self.height))
        
        self._main_loop()
        
    def _main_loop(self):
        """Main processing loop."""
        paused = False
        
        while True:
            start_time = time.time()
            
            if not paused:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Stream ended")
                    break
                    
                self.frame_counter += 1
                
                # Process motion every few frames
                if self.frame_counter % self.process_every == 0:
                    self._process_frame(frame)
                    
                # Age voxels
                self._age_voxels()
                
            # Create AR overlay
            ar_frame = self._create_ar_frame(frame if not paused else None)
            
            # Add UI
            self._add_ui(ar_frame)
            
            # Display
            cv2.imshow('AR Voxel Stream', ar_frame)
            
            # Performance tracking
            self.frame_times.append(time.time() - start_time)
            
            # Handle input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
                print("Paused" if paused else "Resumed")
            elif key == ord('r'):
                self._reset_voxels()
            elif key == ord('p'):
                self.show_particles = not self.show_particles
            elif key == ord('t'):
                self.show_trails = not self.show_trails
            elif key == ord('g'):
                self.glow_enabled = not self.glow_enabled
            elif key == ord('+'):
                self.particle_alpha = min(1.0, self.particle_alpha + 0.1)
            elif key == ord('-'):
                self.particle_alpha = max(0.1, self.particle_alpha - 0.1)
            elif key == ord('['):
                self.motion_threshold = min(100, self.motion_threshold + 5)
            elif key == ord(']'):
                self.motion_threshold = max(5, self.motion_threshold - 5)
                
    def _process_frame(self, frame: np.ndarray):
        """Process frame for motion and update voxels."""
        # Detect motion
        motion_mask = self._detect_motion(frame)
        
        # Update voxels from motion
        if np.any(motion_mask):
            self._update_voxels_from_motion(frame, motion_mask)
            
    def _detect_motion(self, frame: np.ndarray) -> np.ndarray:
        """Advanced motion detection."""
        # Convert to grayscale and blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.last_frame is None:
            self.last_frame = gray
            return np.zeros_like(gray)
            
        # Frame difference
        diff = cv2.absdiff(self.last_frame, gray)
        
        # Adaptive threshold
        thresh = cv2.threshold(diff, self.motion_threshold, 255, cv2.THRESH_BINARY)[1]
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.dilate(thresh, kernel, iterations=2)
        
        self.last_frame = gray
        return thresh
        
    def _update_voxels_from_motion(self, frame: np.ndarray, motion_mask: np.ndarray):
        """Update voxel grid from detected motion."""
        height, width = frame.shape[:2]
        
        # Get motion pixel coordinates
        motion_coords = np.column_stack(np.where(motion_mask > 0))
        
        if len(motion_coords) == 0:
            return
            
        # Sample points for performance
        max_points = 400
        if len(motion_coords) > max_points:
            indices = np.random.choice(len(motion_coords), max_points, replace=False)
            motion_coords = motion_coords[indices]
            
        # Process each motion pixel
        for y, x in motion_coords:
            # Get pixel color
            pixel_color = frame[y, x].astype(np.float32) / 255.0
            intensity = np.mean(pixel_color)
            
            # Skip very dark pixels
            if intensity < 0.1:
                continue
                
            # Generate ray from camera through pixel
            ray_direction = self._pixel_to_ray(x, y, width, height)
            
            # Cast ray and update voxels
            self._cast_ray_with_color(self.camera_pos, ray_direction, intensity, pixel_color)
            
    def _pixel_to_ray(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        """Convert pixel coordinates to ray direction."""
        # Convert to normalized camera coordinates
        cam_x = (x - width / 2.0) / self.focal_length
        cam_y = (y - height / 2.0) / self.focal_length
        
        # Ray direction in camera space (z = -1 forward)
        ray_dir = np.array([cam_x, cam_y, -1.0])
        
        # Normalize
        return ray_dir / np.linalg.norm(ray_dir)
        
    def _cast_ray_with_color(self, origin: np.ndarray, direction: np.ndarray, intensity: float, color: np.ndarray):
        """Cast ray through voxel grid and update with color."""
        # Ray marching parameters
        max_distance = 2.5
        num_steps = 25
        step_size = max_distance / num_steps
        
        for step in range(num_steps):
            # Current position along ray
            t = (step + 0.5) * step_size
            world_pos = origin + direction * t
            
            # Convert to voxel grid coordinates
            voxel_coord = self._world_to_voxel(world_pos)
            
            if self._is_valid_voxel(voxel_coord):
                # Distance falloff
                falloff = 1.0 / (1.0 + step * 0.08)
                
                # Update voxel
                update_amount = intensity * 0.15 * falloff
                
                self.voxel_intensity[voxel_coord] += update_amount
                self.voxel_color[voxel_coord] += color * update_amount
                self.voxel_age[voxel_coord] = 1.0  # Reset age
                
    def _world_to_voxel(self, world_pos: np.ndarray) -> tuple:
        """Convert world position to voxel grid indices."""
        # Center the grid around origin
        grid_center = np.array(self.voxel_dims) / 2.0
        voxel_pos = world_pos / self.voxel_size + grid_center
        
        return tuple(voxel_pos.astype(int))
        
    def _is_valid_voxel(self, voxel_coord: tuple) -> bool:
        """Check if voxel coordinates are within grid bounds."""
        return all(0 <= coord < dim for coord, dim in zip(voxel_coord, self.voxel_dims))
        
    def _age_voxels(self):
        """Apply aging and decay to voxels."""
        # Decay intensity
        self.voxel_intensity *= self.voxel_decay
        self.voxel_color *= self.voxel_decay
        
        # Age voxels
        self.voxel_age *= 0.99
        
    def _reset_voxels(self):
        """Reset all voxel data."""
        self.voxel_intensity.fill(0)
        self.voxel_color.fill(0)
        self.voxel_age.fill(0)
        print("Voxels reset")
        
    def _create_ar_frame(self, frame: np.ndarray) -> np.ndarray:
        """Create augmented reality frame with voxel overlay."""
        if frame is None:
            # Create black frame if paused
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
        ar_frame = frame.copy()
        
        # Get active voxels
        active_mask = self.voxel_intensity > 0.01
        if not np.any(active_mask):
            return ar_frame
            
        # Get voxel data
        active_coords = np.argwhere(active_mask)
        intensities = self.voxel_intensity[active_mask]
        colors = self.voxel_color[active_mask]
        ages = self.voxel_age[active_mask]
        
        # Normalize colors by intensity
        colors = colors / (intensities[:, np.newaxis] + 0.001)
        colors = np.clip(colors, 0, 1)
        
        # Limit number of voxels for performance
        max_voxels = 1500
        if len(active_coords) > max_voxels:
            # Keep the brightest voxels
            indices = np.argsort(intensities)[-max_voxels:]
            active_coords = active_coords[indices]
            intensities = intensities[indices]
            colors = colors[indices]
            ages = ages[indices]
            
        # Project voxels to screen
        self._project_voxels_to_screen(ar_frame, active_coords, intensities, colors, ages)
        
        return ar_frame
        
    def _project_voxels_to_screen(self, frame: np.ndarray, voxel_coords: np.ndarray, 
                                 intensities: np.ndarray, colors: np.ndarray, ages: np.ndarray):
        """Project 3D voxels onto 2D screen with realistic depth."""
        height, width = frame.shape[:2]
        
        for coord, intensity, color, age in zip(voxel_coords, intensities, colors, ages):
            # Convert voxel to world position
            grid_center = np.array(self.voxel_dims) / 2.0
            world_pos = (coord - grid_center) * self.voxel_size
            
            # Transform to camera space
            cam_pos = world_pos - self.camera_pos
            
            # Skip if behind camera
            if cam_pos[2] >= -0.1:
                continue
                
            # Project to screen coordinates
            screen_x = int(-cam_pos[0] * self.focal_length / cam_pos[2] + width / 2)
            screen_y = int(-cam_pos[1] * self.focal_length / cam_pos[2] + height / 2)
            
            # Skip if outside screen
            if not (0 <= screen_x < width and 0 <= screen_y < height):
                continue
                
            # Calculate size based on distance and intensity
            distance = abs(cam_pos[2])
            base_size = max(1, int(30 * intensity / distance))
            
            # Calculate final color with age effect  
            final_color = color * 255 * age
            
            # Draw voxel based on mode
            if self.show_particles:
                self._draw_ar_particle(frame, screen_x, screen_y, base_size, final_color, intensity * age)
                
    def _draw_ar_particle(self, frame: np.ndarray, x: int, y: int, size: int, color: np.ndarray, intensity: float):
        """Draw AR particle with glow effect."""
        height, width = frame.shape[:2]
        
        # Create particle with glow
        max_radius = min(size + 5, 20)
        
        for radius in range(max_radius, 0, -1):
            # Calculate alpha based on radius and intensity
            alpha = (radius / max_radius) ** 2 * intensity * self.particle_alpha
            
            if alpha < 0.01:
                continue
                
            # Draw circle with blending
            overlay = frame.copy()
            # Ensure color is BGR format (3 values)
            bgr_color = tuple(map(int, color[:3]))
            cv2.circle(overlay, (x, y), radius, bgr_color, -1)
            
            # Blend with original
            cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0, frame)
            
        # Add bright center if glow enabled
        if self.glow_enabled and intensity > 0.5:
            bright_color = np.minimum(color * 1.5, 255)
            bright_bgr = tuple(map(int, bright_color[:3]))
            cv2.circle(frame, (x, y), max(1, size // 3), bright_bgr, -1)
            
    def _add_ui(self, frame: np.ndarray):
        """Add UI overlay with information."""
        height = frame.shape[0]
        
        # Performance info
        if self.frame_times:
            fps = 1.0 / np.mean(self.frame_times)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        # Voxel stats
        active_voxels = np.sum(self.voxel_intensity > 0.01)
        cv2.putText(frame, f"Active Voxels: {active_voxels:,}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Settings
        settings = [
            f"Particles: {'ON' if self.show_particles else 'OFF'}",
            f"Glow: {'ON' if self.glow_enabled else 'OFF'}",
            f"Alpha: {int(self.particle_alpha * 100)}%",
            f"Sensitivity: {self.motion_threshold}"
        ]
        
        for i, setting in enumerate(settings):
            cv2.putText(frame, setting, (10, height - 120 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
    def stop(self):
        """Clean up resources."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        

def main():
    parser = argparse.ArgumentParser(description='AR voxel overlay on video stream')
    parser.add_argument('url', nargs='?',
                       default='http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4',
                       help='Video URL (YouTube, HTTP, file, or webcam number)')
    parser.add_argument('--dims', '-d', type=int, default=96,
                       help='Voxel grid dimension (default: 96)')
    
    args = parser.parse_args()
    
    # Check if URL is a webcam number
    if args.url.isdigit():
        args.url = int(args.url)
        
    print("=== AR Voxel Stream ===")
    print(f"Stream: {args.url}")
    print(f"Voxel grid: {args.dims}³")
    
    viewer = ARVoxelStream(args.url, voxel_dims=(args.dims, args.dims, args.dims))
    
    try:
        viewer.start()
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        viewer.stop()
        

if __name__ == "__main__":
    main()