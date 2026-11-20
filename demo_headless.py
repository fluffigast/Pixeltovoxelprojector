 #!/usr/bin/env python3
"""Headless demo of video to voxel conversion."""

import numpy as np
import time
from video_stream_processor import VideoStreamProcessor

class HeadlessVideoProcessor(VideoStreamProcessor):
    """Video processor that works without display."""
    
    def __init__(self):
        super().__init__(source=None)
        self.frame_count_processed = 0
        
    def generate_synthetic_video(self, duration=5.0):
        """Generate and process synthetic video frames."""
        print(f"Generating {duration} seconds of synthetic video...")
        
        frames_to_generate = int(self.fps * duration)
        frame_width, frame_height = 640, 480
        
        for i in range(frames_to_generate):
            t = i / self.fps
            
            # Create frame with moving objects
            frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            
            # Moving circle
            cx = int(frame_width/2 + 150 * np.sin(t * 2))
            cy = int(frame_height/2 + 100 * np.cos(t * 3))
            cv2_circle(frame, cx, cy, 30, (0, 255, 0))
            
            # Rotating line
            angle = t * 4
            length = 100
            cx2 = frame_width // 2
            cy2 = frame_height // 2
            ex = int(cx2 + length * np.cos(angle))
            ey = int(cy2 + length * np.sin(angle))
            cv2_line(frame, cx2, cy2, ex, ey, (255, 0, 0), 5)
            
            # Detect motion
            motion_mask = self.detect_motion(frame)
            
            # Process to voxels
            if np.any(motion_mask):
                self.process_frame_to_voxels(frame, motion_mask)
                self.frame_count_processed += 1
                
            # Progress update
            if (i + 1) % 30 == 0:
                print(f"Processed {i+1}/{frames_to_generate} frames...")
                
        print(f"Completed! Processed {self.frame_count_processed} frames with motion.")
        
def cv2_circle(img, cx, cy, radius, color):
    """Draw a filled circle without OpenCV."""
    y, x = np.ogrid[:img.shape[0], :img.shape[1]]
    mask = (x - cx)**2 + (y - cy)**2 <= radius**2
    img[mask] = color
    
def cv2_line(img, x1, y1, x2, y2, color, thickness):
    """Draw a line without OpenCV."""
    # Simple line drawing using Bresenham's algorithm
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    
    while True:
        # Draw thick line by drawing multiple pixels
        for ty in range(-thickness//2, thickness//2 + 1):
            for tx in range(-thickness//2, thickness//2 + 1):
                py, px = y1 + ty, x1 + tx
                if 0 <= py < img.shape[0] and 0 <= px < img.shape[1]:
                    img[py, px] = color
                    
        if x1 == x2 and y1 == y2:
            break
            
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy
            
def visualize_voxel_stats(voxel_grid):
    """Print statistics about the voxel grid."""
    active_voxels = voxel_grid > 0.01
    num_active = np.sum(active_voxels)
    
    if num_active > 0:
        print(f"\nVoxel Grid Statistics:")
        print(f"  Total voxels: {voxel_grid.size:,}")
        print(f"  Active voxels: {num_active:,} ({100*num_active/voxel_grid.size:.2f}%)")
        print(f"  Max intensity: {np.max(voxel_grid):.4f}")
        print(f"  Mean intensity (active): {np.mean(voxel_grid[active_voxels]):.4f}")
        
        # Find bounding box of active voxels
        indices = np.argwhere(active_voxels)
        min_coords = indices.min(axis=0)
        max_coords = indices.max(axis=0)
        bbox_size = max_coords - min_coords + 1
        
        print(f"  Bounding box: {bbox_size[0]}x{bbox_size[1]}x{bbox_size[2]} voxels")
        print(f"  Center of mass: {indices.mean(axis=0).astype(int)}")
    else:
        print("\nNo active voxels in the grid.")
        
def main():
    """Run headless demo."""
    print("=== Headless Video to Voxel Demo ===\n")
    
    # Create processor
    processor = HeadlessVideoProcessor()
    
    # Generate and process synthetic video
    processor.generate_synthetic_video(duration=5.0)
    
    # Print statistics
    visualize_voxel_stats(processor.voxel_grid)
    
    # Save voxel grid
    filename = "headless_demo_voxels.bin"
    processor.save_voxel_grid(filename)
    print(f"\nVoxel grid saved to: {filename}")
    
    # Also save as numpy array for easy loading
    np.save("headless_demo_voxels.npy", processor.voxel_grid)
    print(f"Also saved as: headless_demo_voxels.npy")
    
    # Create a simple visualization script
    viz_script = '''#!/usr/bin/env python3
"""Visualize the saved voxel grid."""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load voxel grid
voxels = np.load("headless_demo_voxels.npy")
active = voxels > 0.01

# Get active voxel positions
positions = np.argwhere(active)
intensities = voxels[active]

if len(positions) > 0:
    # Create 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot voxels
    scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                        c=intensities, cmap='hot', s=intensities*100, alpha=0.6)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Voxel Grid Visualization')
    
    plt.colorbar(scatter, label='Intensity')
    plt.show()
else:
    print("No active voxels to display.")
'''
    
    with open("visualize_voxels.py", "w") as f:
        f.write(viz_script)
    print("\nCreated visualize_voxels.py - run it to see the voxel grid in 3D")
    

if __name__ == "__main__":
    main()
