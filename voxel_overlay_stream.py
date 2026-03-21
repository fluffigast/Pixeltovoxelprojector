#!/usr/bin/env python3
"""Overlay 3D voxels directly onto the video stream."""

import cv2
import numpy as np
import time
import subprocess
import argparse
from collections import deque

class VoxelOverlayStream:
    """Augmented reality voxel overlay on video stream."""
    
    def __init__(self, stream_url: str, voxel_dims=(128, 128, 128)):
        self.stream_url = stream_url
        self.voxel_dims = voxel_dims
        self.voxel_size = 0.02
        
        # Video
        self.cap = None
        self.fps = 30
        
        # Voxel grid - now stores color too
        self.voxel_grid = np.zeros(voxel_dims, dtype=np.float32)
        self.voxel_colors = np.zeros((*voxel_dims, 3), dtype=np.float32)
        self.voxel_decay = 0.98
        
        # Motion detection
        self.last_frame = None
        self.motion_threshold = 25
        
        # Camera parameters
        self.camera_fov = 60.0
        self.camera_pos = np.array([0, 0, 2.0])
        
        # Visualization
        self.overlay_mode = 'particles'  # 'particles', 'wireframe', 'solid'
        self.show_motion = True
        self.voxel_alpha = 0.7
        
        # Performance
        self.frame_times = deque(maxlen=30)
        
    def get_stream_url(self) -> str:
        """Get direct stream URL."""
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
        """Start processing."""
        stream_url = self.get_stream_url()
        print("Opening stream...")
        self.cap = cv2.VideoCapture(stream_url)
        
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open stream")
            
        # Get properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Stream: {self.width}x{self.height} @ {self.fps:.1f} fps")
        print("\nControls:")
        print("  q - Quit")
        print("  r - Reset voxels")
        print("  m - Toggle motion overlay")
        print("  1/2/3 - Change overlay mode (particles/wireframe/solid)")
        print("  +/- - Adjust voxel opacity")
        print("  [ ] - Adjust motion sensitivity")
        
        # Create window
        cv2.namedWindow('Voxel AR Stream', cv2.WINDOW_NORMAL)
        
        # Process stream
        self._process_stream()
        
    def _process_stream(self):
        """Main processing loop."""
        frame_count = 0
        
        while True:
            start_time = time.time()
            
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Process for motion (on smaller frame)
            small_frame = cv2.resize(frame, (640, 480))
            motion_mask = self._detect_motion(small_frame)
            
            # Update voxels
            if np.any(motion_mask):
                self._update_voxels(small_frame, motion_mask)
                
            # Apply decay
            self.voxel_grid *= self.voxel_decay
            self.voxel_colors *= self.voxel_decay
            
            # Create AR overlay
            ar_frame = self._create_ar_overlay(frame)
            
            # Add UI overlay
            self._add_ui_overlay(ar_frame, frame_count)
            
            # Display
            cv2.imshow('Voxel AR Stream', ar_frame)
            
            # Track performance
            self.frame_times.append(time.time() - start_time)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.voxel_grid.fill(0)
                self.voxel_colors.fill(0)
                print("Voxels reset")
            elif key == ord('m'):
                self.show_motion = not self.show_motion
            elif key == ord('1'):
                self.overlay_mode = 'particles'
            elif key == ord('2'):
                self.overlay_mode = 'wireframe'
            elif key == ord('3'):
                self.overlay_mode = 'solid'
            elif key == ord('+'):
                self.voxel_alpha = min(1.0, self.voxel_alpha + 0.1)
            elif key == ord('-'):
                self.voxel_alpha = max(0.1, self.voxel_alpha - 0.1)
            elif key == ord('['):
                self.motion_threshold = min(100, self.motion_threshold + 5)
            elif key == ord(']'):
                self.motion_threshold = max(5, self.motion_threshold - 5)
                
    def _detect_motion(self, frame: np.ndarray) -> np.ndarray:
        """Detect motion in frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.last_frame is None:
            self.last_frame = gray
            return np.zeros_like(gray)
            
        diff = cv2.absdiff(self.last_frame, gray)
        thresh = cv2.threshold(diff, self.motion_threshold, 255, cv2.THRESH_BINARY)[1]
        
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        self.last_frame = gray
        return thresh
        
    def _update_voxels(self, frame: np.ndarray, motion_mask: np.ndarray):
        """Update voxel grid from motion."""
        height, width = frame.shape[:2]
        
        # Get motion pixels
        motion_y, motion_x = np.where(motion_mask > 0)
        
        if len(motion_x) == 0:
            return
            
        # Sample pixels
        if len(motion_x) > 500:
            indices = np.random.choice(len(motion_x), 500, replace=False)
            motion_x = motion_x[indices]
            motion_y = motion_y[indices]
            
        # Process each pixel
        for x, y in zip(motion_x, motion_y):
            # Get color from original frame
            color = frame[y, x].astype(np.float32) / 255.0
            intensity = np.mean(color)
            
            # Cast ray through pixel
            ray_dir = self._pixel_to_ray(x, y, width, height)
            self._cast_ray(self.camera_pos, ray_dir, intensity, color)
            
    def _pixel_to_ray(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        """Convert pixel to ray direction."""
        aspect = width / height
        fov_rad = np.radians(self.camera_fov)
        
        px = (2.0 * x / width - 1.0) * aspect * np.tan(fov_rad / 2)
        py = -(2.0 * y / height - 1.0) * np.tan(fov_rad / 2)
        
        ray = np.array([px, py, -1.0])
        return ray / np.linalg.norm(ray)
        
    def _cast_ray(self, origin: np.ndarray, direction: np.ndarray, intensity: float, color: np.ndarray):
        """Cast ray and update voxels with color."""
        max_dist = 3.0
        steps = 30
        
        for i in range(steps):
            t = (i + 1) * max_dist / steps
            pos = origin + direction * t
            
            # Convert to voxel indices
            grid_center = np.array(self.voxel_dims) / 2
            voxel_pos = pos / self.voxel_size + grid_center
            voxel_idx = voxel_pos.astype(int)
            
            if all(0 <= idx < dim for idx, dim in zip(voxel_idx, self.voxel_dims)):
                # Update voxel intensity and color
                falloff = 1.0 / (1.0 + i * 0.1)
                update = intensity * 0.2 * falloff
                
                self.voxel_grid[voxel_idx[0], voxel_idx[1], voxel_idx[2]] += update
                self.voxel_colors[voxel_idx[0], voxel_idx[1], voxel_idx[2]] += color * update
                
    def _create_ar_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Create augmented reality overlay on frame."""
        ar_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # Get active voxels
        active = self.voxel_grid > 0.01
        if not np.any(active):
            return ar_frame
            
        # Get voxel data
        positions = np.argwhere(active)
        intensities = self.voxel_grid[active]
        colors = self.voxel_colors[active]
        
        # Normalize colors
        colors = colors / (intensities[:, np.newaxis] + 0.001)
        
        # Limit for performance
        if len(positions) > 2000:
            indices = np.argsort(intensities)[-2000:]
            positions = positions[indices]
            intensities = intensities[indices]
            colors = colors[indices]
            
        # Project voxels back to image
        overlay = np.zeros_like(frame, dtype=np.float32)
        depth_buffer = np.full((height, width), np.inf)
        
        for pos, intensity, color in zip(positions, intensities, colors):
            # Convert voxel to world position
            grid_center = np.array(self.voxel_dims) / 2
            world_pos = (pos - grid_center) * self.voxel_size
            
            # Project to camera space
            cam_pos = world_pos - self.camera_pos
            
            # Skip if behind camera
            if cam_pos[2] >= 0:
                continue
                
            # Project to image
            f = width / (2 * np.tan(np.radians(self.camera_fov) / 2))
            px = int(-cam_pos[0] * f / cam_pos[2] + width / 2)
            py = int(-cam_pos[1] * f / cam_pos[2] + height / 2)
            
            # Calculate size based on distance
            size = int(max(1, min(50, 300 * intensity / abs(cam_pos[2]))))
            
            # Draw voxel
            if 0 <= px < width and 0 <= py < height:
                if self.overlay_mode == 'particles':
                    # Draw as glowing particle
                    self._draw_particle(overlay, px, py, size, color * 255, intensity)
                elif self.overlay_mode == 'wireframe':
                    # Draw as wireframe cube
                    self._draw_wireframe_cube(overlay, px, py, size, color * 255)
                elif self.overlay_mode == 'solid':
                    # Draw as solid cube
                    self._draw_solid_cube(overlay, px, py, size, color * 255, depth_buffer, abs(cam_pos[2]))
                    
        # Blend overlay with original frame
        overlay_uint8 = np.clip(overlay, 0, 255).astype(np.uint8)
        ar_frame = cv2.addWeighted(ar_frame, 1.0, overlay_uint8, self.voxel_alpha, 0)
        
        return ar_frame
        
    def _draw_particle(self, img: np.ndarray, x: int, y: int, size: int, color: np.ndarray, intensity: float):
        """Draw glowing particle."""
        # Create gradient circle
        for r in range(size, 0, -1):
            alpha = (r / size) ** 2 * intensity
            circle_color = color * alpha
            cv2.circle(img, (x, y), r, circle_color.tolist(), -1)
            
    def _draw_wireframe_cube(self, img: np.ndarray, x: int, y: int, size: int, color: np.ndarray):
        """Draw wireframe cube."""
        half = size // 2
        quarter = size // 4
        
        # Define cube vertices in 2D projection
        vertices = [
            (x - half + quarter, y - half + quarter),
            (x + half - quarter, y - half + quarter),
            (x + half - quarter, y + half - quarter),
            (x - half + quarter, y + half - quarter),
            (x - half, y - half),
            (x + half, y - half),
            (x + half, y + half),
            (x - half, y + half)
        ]
        
        # Draw edges
        edges = [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)]
        for i, j in edges:
            cv2.line(img, vertices[i], vertices[j], color.tolist(), 1)
            
    def _draw_solid_cube(self, img: np.ndarray, x: int, y: int, size: int, color: np.ndarray, depth_buffer: np.ndarray, depth: float):
        """Draw solid cube with depth testing."""
        half = size // 2
        
        # Simple filled rectangle with depth test
        y1, y2 = max(0, y - half), min(img.shape[0], y + half)
        x1, x2 = max(0, x - half), min(img.shape[1], x + half)
        
        if y2 > y1 and x2 > x1:
            # Check depth
            if np.min(depth_buffer[y1:y2, x1:x2]) > depth:
                img[y1:y2, x1:x2] = color
                depth_buffer[y1:y2, x1:x2] = depth
                
    def _add_ui_overlay(self, frame: np.ndarray, frame_count: int):
        """Add UI information overlay."""
        height = frame.shape[0]
        
        # FPS
        if len(self.frame_times) > 0:
            fps = 1.0 / np.mean(self.frame_times)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        # Active voxels
        active_voxels = np.sum(self.voxel_grid > 0.01)
        cv2.putText(frame, f"Voxels: {active_voxels:,}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Mode
        cv2.putText(frame, f"Mode: {self.overlay_mode}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Settings
        cv2.putText(frame, f"Opacity: {int(self.voxel_alpha * 100)}%", (10, height - 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, f"Sensitivity: {self.motion_threshold}", (10, height - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
    def stop(self):
        """Clean up."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        

def main():
    parser = argparse.ArgumentParser(description='AR voxel overlay on video stream')
    parser.add_argument('url', nargs='?',
                       default='http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4',
                       help='Video URL')
    parser.add_argument('--dims', '-d', type=int, default=128,
                       help='Voxel grid dimension')
    
    args = parser.parse_args()
    
    viewer = VoxelOverlayStream(args.url, voxel_dims=(args.dims, args.dims, args.dims))
    
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