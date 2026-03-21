#!/usr/bin/env python3
"""Simple but effective voxel overlay on video stream."""

import cv2
import numpy as np
import time
import subprocess
import argparse

class SimpleVoxelOverlay:
    """Simple voxel overlay that definitely shows the effect."""
    
    def __init__(self, stream_url: str):
        self.stream_url = stream_url
        self.voxel_dims = (64, 64, 32)  # Smaller for performance
        self.voxel_size = 0.05
        
        # Video
        self.cap = None
        self.width = 640
        self.height = 480
        
        # Voxel grid - simple intensity
        self.voxels = np.zeros(self.voxel_dims, dtype=np.float32)
        self.voxel_decay = 0.95
        
        # Motion detection
        self.last_frame = None
        self.motion_threshold = 30
        
        # Camera parameters
        self.camera_fov = 60
        self.camera_z = 1.5  # Camera distance from origin
        
    def get_stream_url(self):
        """Get stream URL."""
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
        """Start the overlay."""
        url = self.get_stream_url()
        print(f"Opening: {url}")
        
        self.cap = cv2.VideoCapture(url)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open stream")
            
        print("Stream opened successfully!")
        print("\nControls:")
        print("  q - Quit")
        print("  r - Reset voxels")
        print("  +/- - Adjust motion sensitivity")
        
        cv2.namedWindow('Voxel Overlay Stream', cv2.WINDOW_NORMAL)
        
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Resize to working resolution
            frame = cv2.resize(frame, (self.width, self.height))
            
            # Process every 3rd frame for performance
            if frame_count % 3 == 0:
                self._process_frame(frame)
                
            # Create overlay
            overlay_frame = self._create_overlay(frame)
            
            # Show result
            cv2.imshow('Voxel Overlay Stream', overlay_frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.voxels.fill(0)
                print("Voxels reset")
            elif key == ord('+'):
                self.motion_threshold = max(5, self.motion_threshold - 5)
                print(f"Sensitivity: {self.motion_threshold}")
            elif key == ord('-'):
                self.motion_threshold = min(100, self.motion_threshold + 5)
                print(f"Sensitivity: {self.motion_threshold}")
                
    def _process_frame(self, frame):
        """Process frame for motion and update voxels."""
        # Motion detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.last_frame is not None:
            # Find motion
            diff = cv2.absdiff(self.last_frame, gray)
            motion = cv2.threshold(diff, self.motion_threshold, 255, cv2.THRESH_BINARY)[1]
            
            # Clean up motion mask
            kernel = np.ones((5, 5), np.uint8)
            motion = cv2.morphologyEx(motion, cv2.MORPH_OPEN, kernel)
            
            # Get motion pixels
            motion_y, motion_x = np.where(motion > 0)
            
            if len(motion_x) > 0:
                # Sample up to 200 pixels
                if len(motion_x) > 200:
                    indices = np.random.choice(len(motion_x), 200, replace=False)
                    motion_x = motion_x[indices]
                    motion_y = motion_y[indices]
                    
                # Update voxels
                self._update_voxels(motion_x, motion_y, frame)
                
        self.last_frame = gray
        
        # Apply decay
        self.voxels *= self.voxel_decay
        
    def _update_voxels(self, motion_x, motion_y, frame):
        """Update voxel grid from motion pixels."""
        height, width = frame.shape[:2]
        
        for x, y in zip(motion_x, motion_y):
            # Get pixel brightness
            intensity = np.mean(frame[y, x]) / 255.0
            
            # Convert pixel to normalized coordinates
            norm_x = (x / width - 0.5) * 2.0  # -1 to 1
            norm_y = (y / height - 0.5) * 2.0  # -1 to 1
            
            # Cast ray into 3D space
            # Simple perspective projection
            fov_scale = np.tan(np.radians(self.camera_fov / 2))
            
            # Ray direction
            ray_x = norm_x * fov_scale
            ray_y = -norm_y * fov_scale  # Flip Y
            ray_z = -1.0  # Forward
            
            # Normalize ray
            ray_len = np.sqrt(ray_x**2 + ray_y**2 + ray_z**2)
            ray_x /= ray_len
            ray_y /= ray_len
            ray_z /= ray_len
            
            # March along ray and update voxels
            for step in range(20):
                # Current position
                t = step * 0.1
                pos_x = ray_x * t
                pos_y = ray_y * t
                pos_z = self.camera_z + ray_z * t
                
                # Convert to voxel indices
                vx = int((pos_x / self.voxel_size + self.voxel_dims[0] / 2))
                vy = int((pos_y / self.voxel_size + self.voxel_dims[1] / 2))
                vz = int((pos_z / self.voxel_size))
                
                # Check bounds
                if (0 <= vx < self.voxel_dims[0] and 
                    0 <= vy < self.voxel_dims[1] and 
                    0 <= vz < self.voxel_dims[2]):
                    
                    # Add intensity with distance falloff
                    falloff = 1.0 / (1.0 + step * 0.1)
                    self.voxels[vx, vy, vz] += intensity * 0.3 * falloff
                    
    def _create_overlay(self, frame):
        """Create frame with voxel overlay."""
        overlay = frame.copy()
        
        # Get active voxels
        active_voxels = self.voxels > 0.01
        active_count = np.sum(active_voxels)
        
        if active_count > 0:
            # Get voxel positions and intensities
            voxel_coords = np.argwhere(active_voxels)
            intensities = self.voxels[active_voxels]
            
            # Limit for performance
            if len(voxel_coords) > 1000:
                # Keep brightest voxels
                indices = np.argsort(intensities)[-1000:]
                voxel_coords = voxel_coords[indices]
                intensities = intensities[indices]
                
            # Project voxels back to screen
            for (vx, vy, vz), intensity in zip(voxel_coords, intensities):
                # Convert voxel to world position
                world_x = (vx - self.voxel_dims[0] / 2) * self.voxel_size
                world_y = (vy - self.voxel_dims[1] / 2) * self.voxel_size
                world_z = vz * self.voxel_size
                
                # Project to screen
                if world_z > 0.1:  # In front of camera
                    # Simple perspective projection
                    fov_scale = np.tan(np.radians(self.camera_fov / 2))
                    distance = world_z - self.camera_z
                    
                    if distance > 0:  # In front of camera
                        screen_x = int((world_x / (distance * fov_scale) + 1) * self.width / 2)
                        screen_y = int((-world_y / (distance * fov_scale) + 1) * self.height / 2)
                        
                        # Draw voxel if on screen
                        if 0 <= screen_x < self.width and 0 <= screen_y < self.height:
                            # Size based on distance and intensity
                            size = max(1, int(10 * intensity / distance))
                            
                            # Color based on intensity (hot colormap)
                            color_val = min(255, int(intensity * 400))
                            if color_val > 128:
                                # Yellow to white
                                color = (color_val - 128, 255, 255)
                            else:
                                # Red to yellow
                                color = (0, color_val * 2, 255)
                                
                            # Draw glowing particle
                            cv2.circle(overlay, (screen_x, screen_y), size + 2, (0, 50, 100), -1)
                            cv2.circle(overlay, (screen_x, screen_y), size, color, -1)
                            
        # Add info text
        cv2.putText(overlay, f"Active Voxels: {active_count:,}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(overlay, f"Motion Sensitivity: {self.motion_threshold}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return overlay
        
    def stop(self):
        """Clean up."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('url', nargs='?',
                       default='http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4',
                       help='Video URL')
    args = parser.parse_args()
    
    overlay = SimpleVoxelOverlay(args.url)
    
    try:
        overlay.start()
    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        overlay.stop()


if __name__ == "__main__":
    main()