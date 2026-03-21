import cv2
import numpy as np
import threading
import queue
import time
from typing import Optional, Tuple, Callable
import json
from pathlib import Path

class VideoStreamProcessor:
    """Handles video stream capture and processing for voxel projection."""
    
    def __init__(self, source=0, buffer_size=10):
        """
        Initialize video stream processor.
        
        Args:
            source: Video source (0 for webcam, or path to video file)
            buffer_size: Size of frame buffer for processing
        """
        self.source = source
        self.cap = None
        self.frame_buffer = queue.Queue(maxsize=buffer_size)
        self.is_running = False
        self.capture_thread = None
        self.process_thread = None
        self.frame_count = 0
        self.fps = 30
        self.last_frame = None
        self.motion_threshold = 30
        
        # Camera parameters (to be calibrated)
        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_position = np.array([0.0, 0.0, 0.0])
        self.camera_orientation = np.array([0.0, 0.0, 0.0])  # Euler angles
        
        # Voxel grid parameters
        self.voxel_dims = (256, 256, 256)
        self.voxel_size = 0.1  # meters
        self.voxel_grid = np.zeros(self.voxel_dims, dtype=np.float32)
        
    def start(self):
        """Start video capture and processing threads."""
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {self.source}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.is_running = True
        
        self.capture_thread = threading.Thread(target=self._capture_frames)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
    def stop(self):
        """Stop video capture and processing."""
        self.is_running = False
        if self.capture_thread:
            self.capture_thread.join()
        if self.cap:
            self.cap.release()
            
    def _capture_frames(self):
        """Continuously capture frames from video source."""
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                if isinstance(self.source, str):  # Video file ended
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
                    continue
                else:
                    break
                    
            try:
                self.frame_buffer.put(frame, timeout=0.1)
                self.frame_count += 1
            except queue.Full:
                # Drop frame if buffer is full
                pass
                
    def get_frame(self) -> Optional[np.ndarray]:
        """Get the next frame from the buffer."""
        try:
            return self.frame_buffer.get(timeout=0.1)
        except queue.Empty:
            return None
            
    def detect_motion(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect motion between current and previous frame.
        
        Returns:
            Binary mask of motion areas
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.last_frame is None:
            self.last_frame = gray
            return np.zeros_like(gray)
            
        # Calculate frame difference
        diff = cv2.absdiff(self.last_frame, gray)
        
        # Threshold to get binary mask
        _, motion_mask = cv2.threshold(diff, self.motion_threshold, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to clean up noise
        kernel = np.ones((5, 5), np.uint8)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
        
        self.last_frame = gray
        return motion_mask
        
    def process_frame_to_voxels(self, frame: np.ndarray, motion_mask: Optional[np.ndarray] = None):
        """
        Process a frame and update voxel grid.
        
        Args:
            frame: Input frame
            motion_mask: Optional mask indicating which pixels to process
        """
        height, width = frame.shape[:2]
        
        # If no motion mask provided, process all pixels
        if motion_mask is None:
            motion_mask = np.ones((height, width), dtype=np.uint8) * 255
            
        # Get pixel coordinates where motion occurred
        motion_pixels = np.column_stack(np.where(motion_mask > 0))
        
        # Process each motion pixel
        for y, x in motion_pixels:
            # Get pixel color/intensity
            if len(frame.shape) == 3:
                intensity = np.mean(frame[y, x])
            else:
                intensity = frame[y, x]
                
            # Cast ray from camera through pixel
            ray_direction = self._pixel_to_ray(x, y, width, height)
            
            # Update voxels along ray
            self._cast_ray(self.camera_position, ray_direction, intensity)
            
    def _pixel_to_ray(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        """
        Convert pixel coordinates to ray direction in world space.
        
        Args:
            x, y: Pixel coordinates
            width, height: Image dimensions
            
        Returns:
            Normalized ray direction vector
        """
        # Simple pinhole camera model (can be improved with calibration)
        fov = np.radians(60)  # Field of view
        aspect_ratio = width / height
        
        # Normalize pixel coordinates to [-1, 1]
        px = (2.0 * x / width - 1.0) * aspect_ratio
        py = 1.0 - 2.0 * y / height
        
        # Calculate ray direction in camera space
        ray_cam = np.array([
            px * np.tan(fov / 2),
            py * np.tan(fov / 2),
            -1.0
        ])
        
        # Normalize
        ray_cam = ray_cam / np.linalg.norm(ray_cam)
        
        # Transform to world space using camera orientation
        # (Simplified - should use proper rotation matrix)
        ray_world = ray_cam  # TODO: Apply camera rotation
        
        return ray_world
        
    def _cast_ray(self, origin: np.ndarray, direction: np.ndarray, intensity: float):
        """
        Cast a ray through the voxel grid and accumulate intensity.
        
        Args:
            origin: Ray origin
            direction: Ray direction (normalized)
            intensity: Intensity value to accumulate
        """
        # Simple DDA algorithm for voxel traversal
        max_distance = 10.0  # Maximum ray distance
        step_size = self.voxel_size * 0.5
        
        current_pos = origin.copy()
        
        for _ in range(int(max_distance / step_size)):
            # Convert world position to voxel indices
            voxel_idx = self._world_to_voxel(current_pos)
            
            # Check if inside voxel grid
            if (0 <= voxel_idx[0] < self.voxel_dims[0] and
                0 <= voxel_idx[1] < self.voxel_dims[1] and
                0 <= voxel_idx[2] < self.voxel_dims[2]):
                
                # Accumulate intensity
                self.voxel_grid[voxel_idx[0], voxel_idx[1], voxel_idx[2]] += intensity * 0.01
                
            # Step along ray
            current_pos += direction * step_size
            
    def _world_to_voxel(self, pos: np.ndarray) -> Tuple[int, int, int]:
        """Convert world position to voxel indices."""
        # Center voxel grid at origin
        offset = np.array(self.voxel_dims) * self.voxel_size / 2
        voxel_pos = (pos + offset) / self.voxel_size
        
        return tuple(np.clip(voxel_pos.astype(int), 0, np.array(self.voxel_dims) - 1))
        
    def save_voxel_grid(self, filename: str):
        """Save voxel grid to binary file."""
        with open(filename, 'wb') as f:
            # Write dimensions
            np.array(self.voxel_dims, dtype=np.int32).tofile(f)
            # Write voxel data
            self.voxel_grid.astype(np.float32).tofile(f)
            
    def set_camera_calibration(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
        """Set camera calibration parameters."""
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        
    def set_camera_pose(self, position: np.ndarray, orientation: np.ndarray):
        """Set camera position and orientation."""
        self.camera_position = position
        self.camera_orientation = orientation


def main():
    """Example usage of video stream processor."""
    processor = VideoStreamProcessor(source=0)  # Use webcam
    
    try:
        processor.start()
        print("Video stream started. Press 'q' to quit.")
        
        while True:
            frame = processor.get_frame()
            if frame is None:
                continue
                
            # Detect motion
            motion_mask = processor.detect_motion(frame)
            
            # Process frame to voxels
            processor.process_frame_to_voxels(frame, motion_mask)
            
            # Display frame with motion overlay
            display_frame = frame.copy()
            display_frame[motion_mask > 0] = [0, 255, 0]  # Green for motion
            
            cv2.imshow('Video Stream', display_frame)
            cv2.imshow('Motion Mask', motion_mask)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        # Save voxel grid
        processor.save_voxel_grid('video_voxels.bin')
        print(f"Saved voxel grid with {np.sum(processor.voxel_grid > 0)} active voxels")
        
    finally:
        processor.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()