#!/usr/bin/env python3
"""Live voxel visualization while processing video streams."""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.backends.backend_agg import FigureCanvasAgg
import time
import threading
import queue
import subprocess
import argparse

class LiveVoxelViewer:
    """Shows video stream and 3D voxel visualization side by side."""
    
    def __init__(self, stream_url: str, voxel_dims=(64, 64, 64)):
        self.stream_url = stream_url
        self.voxel_dims = voxel_dims
        self.voxel_size = 0.05
        
        # Video capture
        self.cap = None
        self.fps = 30
        self.frame_count = 0
        
        # Voxel grid
        self.voxel_grid = np.zeros(voxel_dims, dtype=np.float32)
        self.voxel_decay = 0.98
        
        # Motion detection
        self.last_frame = None
        self.motion_threshold = 25
        
        # Camera parameters
        self.camera_fov = 60.0
        self.camera_pos = np.array([0, 0, 2])
        
        # Visualization
        self.fig = None
        self.ax = None
        self.scatter = None
        
        # Threading
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=5)
        self.voxel_lock = threading.Lock()
        
    def get_stream_url(self) -> str:
        """Extract direct stream URL for YouTube videos."""
        if 'youtube.com' in self.stream_url or 'youtu.be' in self.stream_url:
            print("Extracting YouTube stream URL...")
            try:
                cmd = ['yt-dlp', '-f', 'best[height<=720]', '--get-url', self.stream_url]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    return result.stdout.strip()
            except:
                pass
        return self.stream_url
        
    def start(self):
        """Start processing and visualization."""
        # Open video stream
        stream_url = self.get_stream_url()
        self.cap = cv2.VideoCapture(stream_url)
        
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open video stream")
            
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Stream opened: {width}x{height} @ {self.fps:.1f} fps")
        
        self.is_running = True
        
        # Start capture thread
        capture_thread = threading.Thread(target=self._capture_loop)
        capture_thread.daemon = True
        capture_thread.start()
        
        # Start visualization
        self._run_visualization()
        
    def _capture_loop(self):
        """Capture and process frames in background."""
        skip_counter = 0
        
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            self.frame_count += 1
            skip_counter += 1
            
            # Process every 3rd frame for performance
            if skip_counter < 3:
                continue
            skip_counter = 0
            
            # Resize for processing
            if frame.shape[0] > 480:
                scale = 480 / frame.shape[0]
                new_width = int(frame.shape[1] * scale)
                frame = cv2.resize(frame, (new_width, 480))
                
            # Detect motion
            motion_mask = self._detect_motion(frame)
            
            # Update voxels
            if np.any(motion_mask):
                self._update_voxels(frame, motion_mask)
                
            # Add to queue for display
            try:
                self.frame_queue.put((frame, motion_mask), timeout=0.01)
            except queue.Full:
                pass
                
    def _detect_motion(self, frame: np.ndarray) -> np.ndarray:
        """Detect motion between frames."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.last_frame is None:
            self.last_frame = gray
            return np.zeros_like(gray)
            
        diff = cv2.absdiff(self.last_frame, gray)
        thresh = cv2.threshold(diff, self.motion_threshold, 255, cv2.THRESH_BINARY)[1]
        
        kernel = np.ones((5, 5), np.uint8)
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
        if len(motion_x) > 300:
            indices = np.random.choice(len(motion_x), 300, replace=False)
            motion_x = motion_x[indices]
            motion_y = motion_y[indices]
            
        with self.voxel_lock:
            # Apply decay
            self.voxel_grid *= self.voxel_decay
            
            # Process each pixel
            for x, y in zip(motion_x, motion_y):
                # Get color intensity
                intensity = np.mean(frame[y, x]) / 255.0
                
                # Generate ray
                ray_dir = self._pixel_to_ray(x, y, width, height)
                
                # Cast ray
                self._cast_ray(self.camera_pos, ray_dir, intensity)
                
    def _pixel_to_ray(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        """Convert pixel to ray direction."""
        aspect = width / height
        fov_rad = np.radians(self.camera_fov)
        
        px = (2.0 * x / width - 1.0) * aspect * np.tan(fov_rad / 2)
        py = -(2.0 * y / height - 1.0) * np.tan(fov_rad / 2)
        
        ray = np.array([px, py, -1.0])
        return ray / np.linalg.norm(ray)
        
    def _cast_ray(self, origin: np.ndarray, direction: np.ndarray, intensity: float):
        """Cast ray through voxel grid."""
        max_dist = 4.0
        steps = 40
        
        for i in range(steps):
            t = i * max_dist / steps
            pos = origin + direction * t
            
            # Convert to voxel indices
            grid_center = np.array(self.voxel_dims) / 2
            voxel_pos = pos / self.voxel_size + grid_center
            voxel_idx = voxel_pos.astype(int)
            
            # Check bounds
            if all(0 <= idx < dim for idx, dim in zip(voxel_idx, self.voxel_dims)):
                # Update voxel with distance falloff
                falloff = 1.0 / (1.0 + i * 0.05)
                self.voxel_grid[voxel_idx[0], voxel_idx[1], voxel_idx[2]] += intensity * 0.2 * falloff
                
    def _run_visualization(self):
        """Run the live visualization."""
        print("\nShowing live voxel visualization...")
        print("Press 'q' to quit\n")
        
        # Create figure with subplots
        plt.ion()
        self.fig = plt.figure(figsize=(16, 8))
        
        # Video display (converted to matplotlib)
        self.ax_video = self.fig.add_subplot(121)
        self.ax_video.set_title("Video Stream with Motion")
        self.ax_video.axis('off')
        self.im_video = None
        
        # 3D voxel display
        self.ax = self.fig.add_subplot(122, projection='3d')
        self.ax.set_title("3D Voxel Visualization")
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        
        # Set 3D view limits
        mid = self.voxel_dims[0] // 2
        span = self.voxel_dims[0] // 3
        self.ax.set_xlim(mid - span, mid + span)
        self.ax.set_ylim(mid - span, mid + span)
        self.ax.set_zlim(mid - span, mid + span)
        
        # Animation update
        def update_plot(frame_num):
            # Get latest frame
            try:
                frame, motion_mask = self.frame_queue.get_nowait()
                
                # Update video display
                display_frame = frame.copy()
                # Overlay motion in green
                motion_overlay = np.zeros_like(display_frame)
                motion_overlay[:, :, 1] = motion_mask  # Green channel
                display_frame = cv2.addWeighted(display_frame, 0.7, motion_overlay, 0.3, 0)
                
                # Convert BGR to RGB for matplotlib
                display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                
                if self.im_video is None:
                    self.im_video = self.ax_video.imshow(display_frame)
                else:
                    self.im_video.set_data(display_frame)
                    
            except queue.Empty:
                pass
                
            # Update voxel display
            with self.voxel_lock:
                active_voxels = self.voxel_grid > 0.01
                
                if np.any(active_voxels):
                    # Clear previous scatter
                    self.ax.clear()
                    
                    # Get voxel positions and values
                    positions = np.argwhere(active_voxels)
                    values = self.voxel_grid[active_voxels]
                    
                    # Limit number of points for performance
                    if len(positions) > 5000:
                        indices = np.random.choice(len(positions), 5000, replace=False)
                        positions = positions[indices]
                        values = values[indices]
                        
                    # Create scatter plot
                    self.scatter = self.ax.scatter(
                        positions[:, 0], positions[:, 1], positions[:, 2],
                        c=values, cmap='hot', s=values * 100, alpha=0.6,
                        vmin=0, vmax=1
                    )
                    
                    # Reset view
                    self.ax.set_xlabel('X')
                    self.ax.set_ylabel('Y')
                    self.ax.set_zlabel('Z')
                    self.ax.set_title(f"Active Voxels: {len(positions)}")
                    
                    # Set view angle
                    self.ax.view_init(elev=20, azim=frame_num * 0.5)
                    
        # Create animation
        ani = animation.FuncAnimation(self.fig, update_plot, interval=50, blit=False)
        
        try:
            plt.show()
        except KeyboardInterrupt:
            pass
        finally:
            self.is_running = False
            
    def stop(self):
        """Stop processing."""
        self.is_running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        plt.close('all')
        

def main():
    parser = argparse.ArgumentParser(description='Live 3D voxel visualization from video streams')
    parser.add_argument('url', nargs='?', 
                       default='http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4',
                       help='Video URL (YouTube, HTTP, or file path)')
    parser.add_argument('--dims', '-d', type=int, default=64, 
                       help='Voxel grid dimension (default: 64)')
    
    args = parser.parse_args()
    
    print("=== Live Voxel Viewer ===")
    print(f"Stream: {args.url}")
    print(f"Voxel dimensions: {args.dims}x{args.dims}x{args.dims}")
    
    # Create and run viewer
    viewer = LiveVoxelViewer(args.url, voxel_dims=(args.dims, args.dims, args.dims))
    
    try:
        viewer.start()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        viewer.stop()
        

if __name__ == "__main__":
    main()