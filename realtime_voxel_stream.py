import cv2
import numpy as np
import pyvista as pv
import threading
import queue
import time
from typing import Optional, Tuple
import ctypes
import os
from pathlib import Path

# Try to import the compiled C++ module
try:
    import process_image_cpp
    HAS_CPP_MODULE = True
except ImportError:
    HAS_CPP_MODULE = False
    print("Warning: C++ module not found. Using Python implementation.")

class RealtimeVoxelStream:
    """Real-time video to voxel streaming with live visualization."""
    
    def __init__(self, source=0, voxel_dims=(256, 256, 256), voxel_size=0.1):
        self.source = source
        self.voxel_dims = voxel_dims
        self.voxel_size = voxel_size
        
        # Video capture
        self.cap = None
        self.is_running = False
        self.fps = 30
        
        # Motion detection
        self.last_frame = None
        self.motion_threshold = 25
        self.min_motion_area = 100
        
        # Voxel grid
        self.voxel_grid = np.zeros(voxel_dims, dtype=np.float32)
        self.voxel_decay = 0.99  # Decay factor for temporal fading
        
        # Camera parameters
        self.camera_fov = 60.0  # degrees
        self.camera_position = np.array([0.0, 0.0, 2.0])
        self.camera_rotation = np.eye(3)
        
        # Visualization
        self.plotter = None
        self.mesh = None
        self.update_interval = 0.1  # seconds
        
        # Threading
        self.frame_queue = queue.Queue(maxsize=30)
        self.capture_thread = None
        self.process_thread = None
        
    def start(self):
        """Start all processing threads."""
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {self.source}")
            
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        
        self.is_running = True
        
        # Start capture thread
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        # Start processing thread
        self.process_thread = threading.Thread(target=self._process_loop)
        self.process_thread.daemon = True
        self.process_thread.start()
        
        # Initialize visualization
        self._init_visualization()
        
    def stop(self):
        """Stop all threads and cleanup."""
        self.is_running = False
        
        if self.capture_thread:
            self.capture_thread.join()
        if self.process_thread:
            self.process_thread.join()
            
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
    def _capture_loop(self):
        """Continuously capture frames."""
        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                try:
                    self.frame_queue.put(frame, timeout=0.01)
                except queue.Full:
                    # Drop frame if queue is full
                    pass
                    
    def _process_loop(self):
        """Process frames and update voxel grid."""
        while self.is_running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                
                # Detect motion
                motion_mask = self._detect_motion(frame)
                
                # Process motion pixels
                if np.any(motion_mask):
                    self._update_voxels(frame, motion_mask)
                    
                # Apply temporal decay
                self.voxel_grid *= self.voxel_decay
                
                # Show debug windows
                cv2.imshow('Video Stream', frame)
                cv2.imshow('Motion', motion_mask)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.is_running = False
                    
            except queue.Empty:
                continue
                
    def _detect_motion(self, frame: np.ndarray) -> np.ndarray:
        """Detect motion using frame differencing."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.last_frame is None:
            self.last_frame = gray
            return np.zeros_like(gray)
            
        # Frame difference
        diff = cv2.absdiff(self.last_frame, gray)
        thresh = cv2.threshold(diff, self.motion_threshold, 255, cv2.THRESH_BINARY)[1]
        
        # Dilate to fill gaps
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Find contours and filter small areas
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_mask = np.zeros_like(thresh)
        for contour in contours:
            if cv2.contourArea(contour) > self.min_motion_area:
                cv2.drawContours(motion_mask, [contour], -1, 255, -1)
                
        self.last_frame = gray
        return motion_mask
        
    def _update_voxels(self, frame: np.ndarray, motion_mask: np.ndarray):
        """Update voxel grid based on motion pixels."""
        height, width = frame.shape[:2]
        
        # Get motion pixel coordinates
        motion_y, motion_x = np.where(motion_mask > 0)
        
        if len(motion_x) == 0:
            return
            
        # Sample motion pixels (process every nth pixel for performance)
        sample_step = max(1, len(motion_x) // 1000)
        motion_x = motion_x[::sample_step]
        motion_y = motion_y[::sample_step]
        
        # Process each motion pixel
        for x, y in zip(motion_x, motion_y):
            # Get pixel intensity
            if len(frame.shape) == 3:
                intensity = np.mean(frame[y, x]) / 255.0
            else:
                intensity = frame[y, x] / 255.0
                
            # Generate ray for this pixel
            ray_dir = self._pixel_to_ray(x, y, width, height)
            
            # Cast ray and update voxels
            self._cast_ray(self.camera_position, ray_dir, intensity)
            
    def _pixel_to_ray(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        """Convert pixel to ray direction."""
        # Normalized device coordinates
        aspect = width / height
        fov_rad = np.radians(self.camera_fov)
        
        # Convert to camera space
        px = (2.0 * x / width - 1.0) * aspect * np.tan(fov_rad / 2)
        py = -(2.0 * y / height - 1.0) * np.tan(fov_rad / 2)
        
        # Ray in camera space
        ray_cam = np.array([px, py, -1.0])
        ray_cam = ray_cam / np.linalg.norm(ray_cam)
        
        # Transform to world space
        ray_world = self.camera_rotation @ ray_cam
        
        return ray_world
        
    def _cast_ray(self, origin: np.ndarray, direction: np.ndarray, intensity: float):
        """Cast ray through voxel grid using DDA algorithm."""
        # Voxel grid bounds
        grid_min = -np.array(self.voxel_dims) * self.voxel_size / 2
        grid_max = np.array(self.voxel_dims) * self.voxel_size / 2
        
        # Ray-box intersection
        t_min = (grid_min - origin) / (direction + 1e-10)
        t_max = (grid_max - origin) / (direction + 1e-10)
        
        t_enter = np.max(np.minimum(t_min, t_max))
        t_exit = np.min(np.maximum(t_min, t_max))
        
        if t_enter > t_exit or t_exit < 0:
            return
            
        # Start position
        t_start = max(0, t_enter)
        pos = origin + direction * t_start
        
        # DDA traversal
        step_size = self.voxel_size * 0.5
        max_steps = int((t_exit - t_start) / step_size)
        
        for _ in range(max_steps):
            # Convert to voxel indices
            voxel_pos = (pos - grid_min) / self.voxel_size
            voxel_idx = voxel_pos.astype(int)
            
            # Check bounds
            if np.all(voxel_idx >= 0) and np.all(voxel_idx < self.voxel_dims):
                # Update voxel
                self.voxel_grid[voxel_idx[0], voxel_idx[1], voxel_idx[2]] += intensity * 0.1
                
            # Step along ray
            pos += direction * step_size
            
    def _init_visualization(self):
        """Initialize PyVista visualization."""
        self.plotter = pv.Plotter()
        self.plotter.add_axes()
        
        # Create initial mesh
        self._update_mesh()
        
        # Set camera
        self.plotter.camera_position = [(5, 5, 5), (0, 0, 0), (0, 1, 0)]
        
        # Add update callback
        self.plotter.add_timer_callback(self._update_visualization, self.update_interval * 1000)
        
    def _update_mesh(self):
        """Update visualization mesh from voxel grid."""
        # Get active voxels
        active_voxels = self.voxel_grid > 0.01
        
        if not np.any(active_voxels):
            return
            
        # Get voxel positions
        voxel_positions = np.argwhere(active_voxels)
        values = self.voxel_grid[active_voxels]
        
        # Convert to world coordinates
        grid_center = np.array(self.voxel_dims) / 2
        world_positions = (voxel_positions - grid_center) * self.voxel_size
        
        # Create point cloud
        point_cloud = pv.PolyData(world_positions)
        point_cloud["intensity"] = values
        
        # Remove old mesh
        if self.mesh is not None:
            self.plotter.remove_actor(self.mesh)
            
        # Add new mesh
        self.mesh = self.plotter.add_mesh(
            point_cloud,
            scalars="intensity",
            point_size=5,
            render_points_as_spheres=True,
            cmap="hot",
            clim=[0, 1]
        )
        
    def _update_visualization(self):
        """Update visualization callback."""
        self._update_mesh()
        
    def run(self):
        """Run the visualization loop."""
        self.plotter.show()
        
    def save_voxels(self, filename: str):
        """Save current voxel grid."""
        np.save(filename, self.voxel_grid)
        

def main():
    """Run real-time voxel streaming."""
    # Create processor
    stream = RealtimeVoxelStream(source=0)  # 0 for webcam, or path to video file
    
    try:
        # Start processing
        stream.start()
        print("Real-time voxel streaming started.")
        print("Press 'q' in the video window to quit.")
        
        # Run visualization
        stream.run()
        
    finally:
        # Cleanup
        stream.stop()
        
        # Save final voxel grid
        stream.save_voxels("realtime_voxels.npy")
        print("Voxel grid saved.")
        

if __name__ == "__main__":
    main()