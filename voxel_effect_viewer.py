#!/usr/bin/env python3
"""Real-time voxel effect visualization with OpenCV."""

import cv2
import numpy as np
import time
import subprocess
import argparse
from collections import deque

class VoxelEffectViewer:
    """Shows video stream with live voxel effect visualization."""
    
    def __init__(self, stream_url: str, voxel_dims=(128, 128, 128)):
        self.stream_url = stream_url
        self.voxel_dims = voxel_dims
        self.voxel_size = 0.02
        
        # Video
        self.cap = None
        self.fps = 30
        
        # Voxel grid
        self.voxel_grid = np.zeros(voxel_dims, dtype=np.float32)
        self.voxel_decay = 0.97
        
        # Motion detection
        self.last_frame = None
        self.motion_threshold = 20
        
        # Camera
        self.camera_fov = 70.0
        self.camera_pos = np.array([0, 0, 1.5])
        
        # Visualization parameters
        self.view_angle = 0
        self.view_distance = 2.0
        
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
        """Start processing and visualization."""
        # Open stream
        stream_url = self.get_stream_url()
        print(f"Opening stream...")
        self.cap = cv2.VideoCapture(stream_url)
        
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open stream")
            
        # Get properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Stream: {width}x{height} @ {self.fps:.1f} fps")
        print("\nControls:")
        print("  q - Quit")
        print("  r - Reset voxels")
        print("  +/- - Adjust sensitivity")
        print("  arrow keys - Rotate view")
        
        # Create windows
        cv2.namedWindow('Original Stream', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Motion Detection', cv2.WINDOW_NORMAL)
        cv2.namedWindow('3D Voxel View', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Voxel Projection', cv2.WINDOW_NORMAL)
        
        # Process stream
        self._process_stream()
        
    def _process_stream(self):
        """Main processing loop."""
        frame_count = 0
        skip_frames = 2
        
        while True:
            start_time = time.time()
            
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Skip frames for performance
            if frame_count % skip_frames != 0:
                continue
                
            # Resize for processing
            process_frame = frame
            if frame.shape[0] > 480:
                scale = 480 / frame.shape[0]
                new_width = int(frame.shape[1] * scale)
                process_frame = cv2.resize(frame, (new_width, 480))
                
            # Detect motion
            motion_mask = self._detect_motion(process_frame)
            
            # Update voxels
            if np.any(motion_mask):
                self._update_voxels(process_frame, motion_mask)
                
            # Apply decay
            self.voxel_grid *= self.voxel_decay
            
            # Create visualizations
            motion_viz = self._create_motion_viz(process_frame, motion_mask)
            voxel_3d_viz = self._create_3d_voxel_viz()
            voxel_proj_viz = self._create_voxel_projection()
            
            # Display
            display_frame = cv2.resize(frame, (640, 480))
            cv2.imshow('Original Stream', display_frame)
            cv2.imshow('Motion Detection', motion_viz)
            cv2.imshow('3D Voxel View', voxel_3d_viz)
            cv2.imshow('Voxel Projection', voxel_proj_viz)
            
            # Update view angle
            self.view_angle += 1
            
            # Track performance
            frame_time = time.time() - start_time
            self.frame_times.append(frame_time)
            
            # Show stats every second
            if frame_count % 30 == 0:
                avg_time = np.mean(self.frame_times)
                fps = 1.0 / avg_time if avg_time > 0 else 0
                active_voxels = np.sum(self.voxel_grid > 0.01)
                print(f"\rFPS: {fps:.1f} | Active voxels: {active_voxels:,} | Sensitivity: {self.motion_threshold}", end='', flush=True)
                
            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.voxel_grid.fill(0)
                print("\nVoxels reset")
            elif key == ord('+'):
                self.motion_threshold = max(5, self.motion_threshold - 5)
            elif key == ord('-'):
                self.motion_threshold = min(100, self.motion_threshold + 5)
                
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
        thresh = cv2.dilate(thresh, kernel, iterations=2)
        
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
        max_pixels = 500
        if len(motion_x) > max_pixels:
            indices = np.random.choice(len(motion_x), max_pixels, replace=False)
            motion_x = motion_x[indices]
            motion_y = motion_y[indices]
            
        # Process pixels
        for x, y in zip(motion_x, motion_y):
            # Get color
            color = frame[y, x]
            intensity = np.mean(color) / 255.0
            
            # Cast ray
            ray_dir = self._pixel_to_ray(x, y, width, height)
            self._cast_ray(self.camera_pos, ray_dir, intensity, color)
            
    def _pixel_to_ray(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        """Convert pixel to ray."""
        aspect = width / height
        fov_rad = np.radians(self.camera_fov)
        
        px = (2.0 * x / width - 1.0) * aspect * np.tan(fov_rad / 2)
        py = -(2.0 * y / height - 1.0) * np.tan(fov_rad / 2)
        
        ray = np.array([px, py, -1.0])
        return ray / np.linalg.norm(ray)
        
    def _cast_ray(self, origin: np.ndarray, direction: np.ndarray, intensity: float, color: np.ndarray):
        """Cast ray and update voxels."""
        max_dist = 3.0
        steps = 30
        
        for i in range(steps):
            t = (i + 1) * max_dist / steps
            pos = origin + direction * t
            
            # Convert to voxel
            grid_center = np.array(self.voxel_dims) / 2
            voxel_pos = pos / self.voxel_size + grid_center
            voxel_idx = voxel_pos.astype(int)
            
            if all(0 <= idx < dim for idx, dim in zip(voxel_idx, self.voxel_dims)):
                # Update with distance falloff
                falloff = 1.0 / (1.0 + i * 0.1)
                self.voxel_grid[voxel_idx[0], voxel_idx[1], voxel_idx[2]] += intensity * 0.15 * falloff
                
    def _create_motion_viz(self, frame: np.ndarray, motion_mask: np.ndarray) -> np.ndarray:
        """Create motion visualization."""
        viz = frame.copy()
        
        # Create colored overlay
        overlay = np.zeros_like(viz)
        overlay[:, :, 1] = motion_mask  # Green for motion
        
        # Blend
        viz = cv2.addWeighted(viz, 0.7, overlay, 0.3, 0)
        
        # Add text
        cv2.putText(viz, f"Motion Threshold: {self.motion_threshold}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return viz
        
    def _create_3d_voxel_viz(self) -> np.ndarray:
        """Create 3D voxel visualization."""
        # Create canvas
        size = 480
        viz = np.zeros((size, size, 3), dtype=np.uint8)
        
        # Get active voxels
        active = self.voxel_grid > 0.01
        if not np.any(active):
            cv2.putText(viz, "No active voxels", (size//2 - 80, size//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            return viz
            
        # Get positions and values
        positions = np.argwhere(active)
        values = self.voxel_grid[active]
        
        # Limit points
        if len(positions) > 2000:
            indices = np.argsort(values)[-2000:]
            positions = positions[indices]
            values = values[indices]
            
        # Center positions
        center = np.array(self.voxel_dims) / 2
        positions = positions - center
        
        # Rotate view
        angle_rad = np.radians(self.view_angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Project to 2D
        for pos, val in zip(positions, values):
            # Rotate around Y axis
            x = pos[0] * cos_a - pos[2] * sin_a
            z = pos[0] * sin_a + pos[2] * cos_a
            y = pos[1]
            
            # Perspective projection
            if z < -0.1:
                continue
                
            scale = 200 / (z + self.view_distance)
            px = int(x * scale + size/2)
            py = int(y * scale + size/2)
            
            if 0 <= px < size and 0 <= py < size:
                # Draw voxel
                radius = max(1, int(5 * val))
                color_intensity = int(255 * val)
                color = (color_intensity, color_intensity//2, 0)  # Orange glow
                cv2.circle(viz, (px, py), radius, color, -1)
                
        # Add text
        active_count = len(positions)
        cv2.putText(viz, f"Active Voxels: {active_count}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return viz
        
    def _create_voxel_projection(self) -> np.ndarray:
        """Create front projection of voxels."""
        size = 480
        viz = np.zeros((size, size, 3), dtype=np.uint8)
        
        # Sum voxels along Z axis
        projection = np.sum(self.voxel_grid, axis=2)
        
        if np.max(projection) > 0:
            # Normalize
            projection = projection / np.max(projection)
            
            # Resize to display size
            projection_resized = cv2.resize(projection.T, (size, size))
            
            # Convert to color
            heatmap = (projection_resized * 255).astype(np.uint8)
            colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
            
            # Blend with black background
            viz = cv2.addWeighted(viz, 0.3, colored, 0.7, 0)
            
        cv2.putText(viz, "Z-Axis Projection", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return viz
        
    def stop(self):
        """Clean up."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        

def main():
    parser = argparse.ArgumentParser(description='Real-time voxel effect visualization')
    parser.add_argument('url', nargs='?',
                       default='http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4',
                       help='Video URL')
    parser.add_argument('--dims', '-d', type=int, default=128,
                       help='Voxel grid dimension')
    
    args = parser.parse_args()
    
    viewer = VoxelEffectViewer(args.url, voxel_dims=(args.dims, args.dims, args.dims))
    
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