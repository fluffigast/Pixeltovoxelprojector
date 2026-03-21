#!/usr/bin/env python3
"""Demo of voxel streaming with synthetic video."""

import cv2
import numpy as np
import pyvista as pv
import time
from realtime_voxel_stream import RealtimeVoxelStream

class SyntheticVideoStream(RealtimeVoxelStream):
    """Video stream with synthetic moving objects."""
    
    def __init__(self, voxel_dims=(128, 128, 128), voxel_size=0.05):
        super().__init__(source=None, voxel_dims=voxel_dims, voxel_size=voxel_size)
        self.frame_width = 640
        self.frame_height = 480
        self.time = 0
        self.fps = 30
        
    def _capture_loop(self):
        """Generate synthetic frames instead of capturing from camera."""
        while self.is_running:
            # Create synthetic frame with moving objects
            frame = self._generate_frame()
            
            try:
                self.frame_queue.put(frame, timeout=0.01)
            except:
                pass
                
            # Control frame rate
            time.sleep(1.0 / self.fps)
            self.time += 1.0 / self.fps
            
    def _generate_frame(self):
        """Generate a synthetic frame with moving objects."""
        frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        
        # Moving circle
        cx = int(self.frame_width/2 + 150 * np.sin(self.time * 2))
        cy = int(self.frame_height/2 + 100 * np.cos(self.time * 3))
        cv2.circle(frame, (cx, cy), 30, (0, 255, 0), -1)
        
        # Rotating line
        angle = self.time * 4
        length = 100
        cx2 = self.frame_width // 2
        cy2 = self.frame_height // 2
        ex = int(cx2 + length * np.cos(angle))
        ey = int(cy2 + length * np.sin(angle))
        cv2.line(frame, (cx2, cy2), (ex, ey), (255, 0, 0), 5)
        
        # Pulsing square
        size = int(30 + 20 * np.sin(self.time * 5))
        sx = self.frame_width - 100
        sy = 100
        cv2.rectangle(frame, (sx-size, sy-size), (sx+size, sy+size), (0, 0, 255), -1)
        
        return frame
        
    def start(self):
        """Start synthetic video generation."""
        self.is_running = True
        
        # Start threads
        import threading
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        self.process_thread = threading.Thread(target=self._process_loop)
        self.process_thread.daemon = True
        self.process_thread.start()
        
        # Initialize visualization
        self._init_visualization()


def main():
    """Run synthetic video stream demo."""
    print("Starting synthetic video stream demo...")
    print("This demo shows moving objects being converted to 3D voxels in real-time.")
    print("Press 'q' in the video window to quit.")
    
    # Create synthetic stream
    stream = SyntheticVideoStream(voxel_dims=(128, 128, 128), voxel_size=0.02)
    
    try:
        # Start processing
        stream.start()
        
        # Run visualization
        stream.run()
        
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        # Cleanup
        stream.stop()
        
        # Save voxel grid
        stream.save_voxels("synthetic_demo_voxels.npy")
        print("Demo completed. Voxel grid saved to synthetic_demo_voxels.npy")
        
        # Print statistics
        active_voxels = np.sum(stream.voxel_grid > 0.01)
        print(f"Final voxel grid has {active_voxels:,} active voxels")


if __name__ == "__main__":
    main()