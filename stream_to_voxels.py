#!/usr/bin/env python3
"""Convert any video stream to 3D voxels with YouTube support."""

import cv2
import numpy as np
import subprocess
import json
import sys
import time
import threading
import queue
from typing import Optional, Dict, Any
import argparse

class StreamToVoxels:
    """Convert video streams to 3D voxel representation."""
    
    def __init__(self, stream_url: str, voxel_dims=(128, 128, 128)):
        self.stream_url = stream_url
        self.voxel_dims = voxel_dims
        self.voxel_size = 0.05
        
        # Video parameters
        self.cap = None
        self.direct_url = None
        self.fps = 30
        self.frame_count = 0
        
        # Voxel grid
        self.voxel_grid = np.zeros(voxel_dims, dtype=np.float32)
        self.voxel_decay = 0.99
        
        # Motion detection
        self.last_frame = None
        self.motion_threshold = 30
        
        # Camera model
        self.camera_fov = 60.0
        self.camera_pos = np.array([0, 0, 2])
        
        # Processing
        self.is_running = False
        self.process_every_n_frames = 2  # Process every 2nd frame for performance
        
    def get_stream_url(self) -> str:
        """Extract direct stream URL using yt-dlp."""
        if self.stream_url.startswith(('http://', 'https://')):
            if 'youtube.com' in self.stream_url or 'youtu.be' in self.stream_url:
                print("Extracting YouTube stream URL...")
                try:
                    # Use yt-dlp to get stream URL
                    cmd = [
                        'yt-dlp', 
                        '-f', 'best[height<=720]',  # Limit to 720p for performance
                        '--get-url',
                        '--no-playlist',
                        self.stream_url
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode == 0:
                        self.direct_url = result.stdout.strip()
                        print("✓ Stream URL extracted successfully")
                        return self.direct_url
                    else:
                        print(f"Error: {result.stderr}")
                        return self.stream_url
                except Exception as e:
                    print(f"yt-dlp error: {e}")
                    return self.stream_url
            else:
                # Direct HTTP/HTTPS stream
                return self.stream_url
        else:
            # Local file or RTSP stream
            return self.stream_url
            
    def start(self):
        """Start processing the video stream."""
        # Get direct stream URL
        stream_url = self.get_stream_url()
        
        print(f"Opening video stream...")
        self.cap = cv2.VideoCapture(stream_url)
        
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open video stream")
            
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Stream opened: {width}x{height} @ {self.fps:.1f} fps")
        
        self.is_running = True
        
        # Start processing
        self._process_stream()
        
    def stop(self):
        """Stop processing."""
        self.is_running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
    def _process_stream(self):
        """Main processing loop."""
        print("\nProcessing stream... Press 'q' to quit\n")
        
        start_time = time.time()
        frames_processed = 0
        
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                print("\nStream ended")
                break
                
            self.frame_count += 1
            
            # Skip frames for performance
            if self.frame_count % self.process_every_n_frames != 0:
                continue
                
            # Resize for faster processing
            if frame.shape[0] > 480:
                scale = 480 / frame.shape[0]
                new_width = int(frame.shape[1] * scale)
                frame = cv2.resize(frame, (new_width, 480))
                
            # Detect motion
            motion_mask = self._detect_motion(frame)
            
            # Update voxels
            if np.any(motion_mask):
                self._update_voxels(frame, motion_mask)
                
            # Apply decay
            self.voxel_grid *= self.voxel_decay
            
            frames_processed += 1
            
            # Display progress
            if frames_processed % 30 == 0:
                elapsed = time.time() - start_time
                fps = frames_processed / elapsed
                active_voxels = np.sum(self.voxel_grid > 0.01)
                print(f"\rProcessing: {fps:.1f} fps | Active voxels: {active_voxels:,} | Frame: {self.frame_count}", end='', flush=True)
                
            # Show preview (optional)
            if frames_processed % 10 == 0:
                preview = cv2.resize(frame, (640, 480))
                cv2.putText(preview, f"Frame: {self.frame_count}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Stream Preview', preview)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        print(f"\n\nProcessed {frames_processed} frames in {time.time() - start_time:.1f} seconds")
        
    def _detect_motion(self, frame: np.ndarray) -> np.ndarray:
        """Simple motion detection."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.last_frame is None:
            self.last_frame = gray
            return np.zeros_like(gray)
            
        # Frame difference
        diff = cv2.absdiff(self.last_frame, gray)
        thresh = cv2.threshold(diff, self.motion_threshold, 255, cv2.THRESH_BINARY)[1]
        
        # Clean up with morphology
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        self.last_frame = gray
        return thresh
        
    def _update_voxels(self, frame: np.ndarray, motion_mask: np.ndarray):
        """Update voxel grid from motion pixels."""
        height, width = frame.shape[:2]
        
        # Get motion pixels
        motion_y, motion_x = np.where(motion_mask > 0)
        
        if len(motion_x) == 0:
            return
            
        # Sample pixels for performance
        if len(motion_x) > 500:
            indices = np.random.choice(len(motion_x), 500, replace=False)
            motion_x = motion_x[indices]
            motion_y = motion_y[indices]
            
        # Process each pixel
        for x, y in zip(motion_x, motion_y):
            # Get intensity
            intensity = np.mean(frame[y, x]) / 255.0
            
            # Generate ray
            ray_dir = self._pixel_to_ray(x, y, width, height)
            
            # Cast ray
            self._cast_ray(self.camera_pos, ray_dir, intensity)
            
    def _pixel_to_ray(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        """Convert pixel to ray direction."""
        # Simple pinhole camera model
        aspect = width / height
        fov_rad = np.radians(self.camera_fov)
        
        # NDC coordinates
        px = (2.0 * x / width - 1.0) * aspect * np.tan(fov_rad / 2)
        py = -(2.0 * y / height - 1.0) * np.tan(fov_rad / 2)
        
        # Ray direction
        ray = np.array([px, py, -1.0])
        return ray / np.linalg.norm(ray)
        
    def _cast_ray(self, origin: np.ndarray, direction: np.ndarray, intensity: float):
        """Cast ray through voxel grid."""
        # Ray marching
        max_dist = 5.0
        steps = 50
        
        for i in range(steps):
            t = i * max_dist / steps
            pos = origin + direction * t
            
            # Convert to voxel indices
            voxel_idx = self._world_to_voxel(pos)
            
            # Check bounds
            if all(0 <= idx < dim for idx, dim in zip(voxel_idx, self.voxel_dims)):
                # Update voxel
                self.voxel_grid[voxel_idx[0], voxel_idx[1], voxel_idx[2]] += intensity * 0.1
                
    def _world_to_voxel(self, pos: np.ndarray) -> tuple:
        """Convert world position to voxel indices."""
        grid_center = np.array(self.voxel_dims) / 2
        voxel_pos = pos / self.voxel_size + grid_center
        return tuple(np.clip(voxel_pos.astype(int), 0, np.array(self.voxel_dims) - 1))
        
    def save_voxels(self, filename: str):
        """Save voxel grid."""
        np.save(filename, self.voxel_grid)
        print(f"Voxel grid saved to {filename}")
        
        # Create visualization script
        viz_code = '''import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

voxels = np.load("%s")
active = voxels > 0.01
if np.any(active):
    pos = np.argwhere(active)
    vals = voxels[active]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], 
                        c=vals, cmap='hot', s=vals*50, alpha=0.6)
    ax.set_xlabel('X')
    ax.set_ylabel('Y') 
    ax.set_zlabel('Z')
    plt.colorbar(scatter)
    plt.title("3D Voxel Visualization")
    plt.show()
else:
    print("No active voxels")
''' % filename
        
        viz_filename = filename.replace('.npy', '_viz.py')
        with open(viz_filename, 'w') as f:
            f.write(viz_code)
        print(f"Visualization script saved to {viz_filename}")
        

def main():
    parser = argparse.ArgumentParser(description='Convert video streams to 3D voxels')
    parser.add_argument('url', help='Video URL (YouTube, HTTP stream, or file path)')
    parser.add_argument('--output', '-o', default='voxels.npy', help='Output filename')
    parser.add_argument('--dims', '-d', type=int, default=128, help='Voxel grid dimension')
    parser.add_argument('--skip', '-s', type=int, default=2, help='Process every Nth frame')
    
    args = parser.parse_args()
    
    # Create processor
    processor = StreamToVoxels(args.url, voxel_dims=(args.dims, args.dims, args.dims))
    processor.process_every_n_frames = args.skip
    
    try:
        # Process stream
        processor.start()
        
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        processor.stop()
        
        # Save results
        processor.save_voxels(args.output)
        
        # Print stats
        active = np.sum(processor.voxel_grid > 0.01)
        print(f"\nFinal stats:")
        print(f"  Frames processed: {processor.frame_count}")
        print(f"  Active voxels: {active:,}")
        print(f"  Max intensity: {np.max(processor.voxel_grid):.4f}")
        

if __name__ == "__main__":
    main()