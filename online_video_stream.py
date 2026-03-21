#!/usr/bin/env python3
"""Process online video streams to 3D voxels."""

import cv2
import numpy as np
import threading
import queue
import time
import argparse
from urllib.parse import urlparse
from typing import Optional, Union
import os

class OnlineVideoStreamProcessor:
    """Process online video streams (YouTube, RTSP, HTTP streams) to voxels."""
    
    def __init__(self, stream_url: str, voxel_dims=(256, 256, 256), voxel_size=0.1):
        """
        Initialize online video stream processor.
        
        Args:
            stream_url: URL of the video stream
                - YouTube: URL of YouTube video
                - RTSP: rtsp://username:password@ip:port/path
                - HTTP: http://example.com/stream.m3u8
                - Local file: path/to/video.mp4
            voxel_dims: Dimensions of voxel grid
            voxel_size: Size of each voxel in meters
        """
        self.stream_url = stream_url
        self.voxel_dims = voxel_dims
        self.voxel_size = voxel_size
        
        # Video capture
        self.cap = None
        self.is_running = False
        self.fps = 30
        self.frame_count = 0
        self.dropped_frames = 0
        
        # Processing parameters
        self.motion_threshold = 25
        self.last_frame = None
        self.skip_frames = 0  # Process every Nth frame for performance
        
        # Voxel grid
        self.voxel_grid = np.zeros(voxel_dims, dtype=np.float32)
        self.voxel_decay = 0.995  # Slower decay for online streams
        
        # Camera parameters (assuming fixed camera)
        self.camera_fov = 60.0
        self.camera_position = np.array([0.0, 0.0, 2.0])
        
        # Threading
        self.frame_buffer = queue.Queue(maxsize=60)
        self.capture_thread = None
        self.process_thread = None
        self.stats_thread = None
        
        # Performance stats
        self.start_time = None
        self.processed_frames = 0
        
    def _get_stream_url(self) -> str:
        """Process the stream URL for OpenCV."""
        # Check if it's a YouTube URL
        if 'youtube.com' in self.stream_url or 'youtu.be' in self.stream_url:
            # For YouTube, we need to extract the direct stream URL
            # In production, you'd use youtube-dl or yt-dlp
            print("Note: YouTube URLs require youtube-dl/yt-dlp for direct stream access")
            return self.stream_url
            
        # For RTSP and HTTP streams, use directly
        return self.stream_url
        
    def start(self):
        """Start video capture and processing."""
        print(f"Connecting to video stream: {self.stream_url}")
        
        # Open video stream
        stream_url = self._get_stream_url()
        self.cap = cv2.VideoCapture(stream_url)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video stream: {self.stream_url}")
            
        # Get stream properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Stream opened successfully!")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {self.fps}")
        
        # Configure buffer size based on stream type
        if self.stream_url.startswith('rtsp://'):
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency for live streams
            
        self.is_running = True
        self.start_time = time.time()
        
        # Start threads
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        self.process_thread = threading.Thread(target=self._process_loop)
        self.process_thread.daemon = True
        self.process_thread.start()
        
        self.stats_thread = threading.Thread(target=self._stats_loop)
        self.stats_thread.daemon = True
        self.stats_thread.start()
        
    def stop(self):
        """Stop all processing."""
        print("\nStopping stream processing...")
        self.is_running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2)
        if self.process_thread:
            self.process_thread.join(timeout=2)
        if self.stats_thread:
            self.stats_thread.join(timeout=1)
            
        if self.cap:
            self.cap.release()
            
        cv2.destroyAllWindows()
        
    def _capture_loop(self):
        """Continuously capture frames from stream."""
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                # Stream might have ended or disconnected
                print("Stream ended or disconnected")
                break
                
            self.frame_count += 1
            
            # Skip frames for performance if needed
            if self.skip_frames > 0 and self.frame_count % (self.skip_frames + 1) != 0:
                continue
                
            try:
                self.frame_buffer.put(frame, timeout=0.001)
            except queue.Full:
                self.dropped_frames += 1
                
    def _process_loop(self):
        """Process frames and update voxel grid."""
        while self.is_running:
            try:
                frame = self.frame_buffer.get(timeout=0.5)
                
                # Resize frame for faster processing if needed
                if frame.shape[0] > 720:
                    scale = 720 / frame.shape[0]
                    new_width = int(frame.shape[1] * scale)
                    frame = cv2.resize(frame, (new_width, 720))
                    
                # Detect motion
                motion_mask = self._detect_motion(frame)
                
                # Process to voxels
                if np.any(motion_mask):
                    self._update_voxels(frame, motion_mask)
                    
                # Apply temporal decay
                self.voxel_grid *= self.voxel_decay
                
                self.processed_frames += 1
                
                # Optional: Display frame
                if self.processed_frames % 10 == 0:  # Show every 10th frame
                    display_frame = cv2.resize(frame, (640, 480))
                    cv2.imshow('Stream', display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.is_running = False
                        
            except queue.Empty:
                continue
                
    def _stats_loop(self):
        """Print processing statistics."""
        while self.is_running:
            time.sleep(5)  # Update every 5 seconds
            
            if self.start_time:
                elapsed = time.time() - self.start_time
                capture_fps = self.frame_count / elapsed
                process_fps = self.processed_frames / elapsed
                
                active_voxels = np.sum(self.voxel_grid > 0.01)
                
                print(f"\rStats: Captured {capture_fps:.1f} fps | "
                      f"Processed {process_fps:.1f} fps | "
                      f"Dropped {self.dropped_frames} | "
                      f"Active voxels {active_voxels:,}", end='', flush=True)
                      
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
        
        # Morphological operations to clean up
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.dilate(thresh, kernel, iterations=2)
        
        self.last_frame = gray
        return thresh
        
    def _update_voxels(self, frame: np.ndarray, motion_mask: np.ndarray):
        """Update voxel grid based on motion."""
        height, width = frame.shape[:2]
        
        # Sample motion pixels (process subset for performance)
        motion_y, motion_x = np.where(motion_mask > 0)
        
        if len(motion_x) == 0:
            return
            
        # Randomly sample up to 1000 pixels
        if len(motion_x) > 1000:
            indices = np.random.choice(len(motion_x), 1000, replace=False)
            motion_x = motion_x[indices]
            motion_y = motion_y[indices]
            
        # Process each motion pixel
        for x, y in zip(motion_x, motion_y):
            # Get pixel intensity
            intensity = np.mean(frame[y, x]) / 255.0
            
            # Generate ray
            ray_dir = self._pixel_to_ray(x, y, width, height)
            
            # Cast ray
            self._cast_ray(self.camera_position, ray_dir, intensity)
            
    def _pixel_to_ray(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        """Convert pixel to ray direction."""
        aspect = width / height
        fov_rad = np.radians(self.camera_fov)
        
        # Normalized device coordinates
        px = (2.0 * x / width - 1.0) * aspect * np.tan(fov_rad / 2)
        py = -(2.0 * y / height - 1.0) * np.tan(fov_rad / 2)
        
        # Ray in camera space
        ray = np.array([px, py, -1.0])
        return ray / np.linalg.norm(ray)
        
    def _cast_ray(self, origin: np.ndarray, direction: np.ndarray, intensity: float):
        """Cast ray through voxel grid."""
        # Simple ray marching
        max_distance = 5.0
        step_size = self.voxel_size
        steps = int(max_distance / step_size)
        
        for i in range(steps):
            # Current position along ray
            pos = origin + direction * (i * step_size)
            
            # Convert to voxel indices
            voxel_idx = self._world_to_voxel(pos)
            
            # Check bounds
            if (0 <= voxel_idx[0] < self.voxel_dims[0] and
                0 <= voxel_idx[1] < self.voxel_dims[1] and
                0 <= voxel_idx[2] < self.voxel_dims[2]):
                
                # Update voxel with distance falloff
                distance_factor = 1.0 / (1.0 + i * 0.1)
                self.voxel_grid[voxel_idx[0], voxel_idx[1], voxel_idx[2]] += intensity * 0.05 * distance_factor
                
    def _world_to_voxel(self, pos: np.ndarray) -> tuple:
        """Convert world position to voxel indices."""
        # Center grid at origin
        grid_center = np.array(self.voxel_dims) / 2
        voxel_pos = pos / self.voxel_size + grid_center
        
        return tuple(np.clip(voxel_pos.astype(int), 0, np.array(self.voxel_dims) - 1))
        
    def save_voxels(self, filename: str):
        """Save voxel grid to file."""
        np.save(filename, self.voxel_grid)
        print(f"\nVoxel grid saved to {filename}")
        
        # Save metadata
        metadata = {
            'stream_url': self.stream_url,
            'voxel_dims': self.voxel_dims,
            'voxel_size': self.voxel_size,
            'total_frames': self.frame_count,
            'processed_frames': self.processed_frames,
            'duration': time.time() - self.start_time if self.start_time else 0
        }
        
        import json
        with open(filename.replace('.npy', '_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
            

def main():
    """Run online video stream processor."""
    parser = argparse.ArgumentParser(description='Process online video streams to 3D voxels')
    parser.add_argument('url', nargs='?', default=None, 
                       help='Video stream URL (YouTube, RTSP, HTTP, or local file)')
    parser.add_argument('--voxel-dims', type=int, nargs=3, default=[128, 128, 128],
                       help='Voxel grid dimensions (default: 128 128 128)')
    parser.add_argument('--skip-frames', type=int, default=0,
                       help='Skip N frames between processing (0 = process all)')
    parser.add_argument('--duration', type=int, default=0,
                       help='Process for N seconds (0 = until stopped)')
    parser.add_argument('--output', type=str, default='stream_voxels.npy',
                       help='Output filename for voxel grid')
    
    args = parser.parse_args()
    
    # Example streams if no URL provided
    if not args.url:
        print("No URL provided. Here are some example streams you can use:")
        print("\n1. Local video file:")
        print("   python online_video_stream.py path/to/video.mp4")
        print("\n2. HTTP stream:")
        print("   python online_video_stream.py http://example.com/stream.m3u8")
        print("\n3. RTSP stream:")
        print("   python online_video_stream.py rtsp://username:password@ip:port/path")
        print("\n4. Sample public test stream:")
        print("   python online_video_stream.py http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4")
        return
        
    # Create processor
    processor = OnlineVideoStreamProcessor(
        args.url,
        voxel_dims=tuple(args.voxel_dims),
        voxel_size=0.05
    )
    
    if args.skip_frames > 0:
        processor.skip_frames = args.skip_frames
        
    try:
        # Start processing
        processor.start()
        
        print("\nProcessing stream... Press 'q' in video window or Ctrl+C to stop")
        
        # Run for specified duration or until stopped
        if args.duration > 0:
            time.sleep(args.duration)
            processor.stop()
        else:
            # Wait for user to stop
            while processor.is_running:
                time.sleep(1)
                
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        processor.stop()
        
        # Save results
        processor.save_voxels(args.output)
        
        # Print final statistics
        active_voxels = np.sum(processor.voxel_grid > 0.01)
        max_intensity = np.max(processor.voxel_grid)
        
        print(f"\nFinal Statistics:")
        print(f"  Total frames captured: {processor.frame_count}")
        print(f"  Frames processed: {processor.processed_frames}")
        print(f"  Active voxels: {active_voxels:,}")
        print(f"  Max voxel intensity: {max_intensity:.4f}")
        

if __name__ == "__main__":
    main()