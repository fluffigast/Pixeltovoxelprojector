# Video Stream to 3D Voxels

This system converts video streams (online or local) into 3D voxel representations by detecting motion and projecting it into 3D space.

## Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install opencv-python numpy pyvista yt-dlp streamlink
```

## Usage

### Basic Usage

```bash
# Process a YouTube video
python stream_to_voxels.py "https://www.youtube.com/watch?v=VIDEO_ID"

# Process a local video file
python stream_to_voxels.py path/to/video.mp4

# Process a public test stream
python stream_to_voxels.py "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"
```

### Command Line Options

```bash
python stream_to_voxels.py [URL] [options]

Options:
  --output, -o     Output filename (default: voxels.npy)
  --dims, -d       Voxel grid dimension (default: 128)
  --skip, -s       Process every Nth frame (default: 2)
```

### Examples

```bash
# High quality processing (slower)
python stream_to_voxels.py video.mp4 --dims 256 --skip 1

# Fast processing for real-time streams
python stream_to_voxels.py rtsp://camera.local --dims 64 --skip 5

# Process and save with custom name
python stream_to_voxels.py video.mp4 --output my_voxels.npy
```

## How It Works

1. **Video Capture**: Opens video stream using OpenCV
2. **Motion Detection**: Detects changes between frames
3. **Ray Casting**: Projects motion pixels into 3D space
4. **Voxel Accumulation**: Builds up 3D representation over time
5. **Temporal Decay**: Older voxels fade over time

## Visualizing Results

After processing, a visualization script is automatically created:

```bash
# Run the generated visualization script
python voxels_viz.py
```

Or visualize manually:

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load voxels
voxels = np.load('voxels.npy')
active = voxels > 0.01

# Plot
if np.any(active):
    pos = np.argwhere(active)
    vals = voxels[active]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], 
                        c=vals, cmap='hot', s=vals*50, alpha=0.6)
    plt.colorbar(scatter)
    plt.show()
```

## Supported Stream Types

- **YouTube Videos**: Any public YouTube video URL
- **Local Files**: MP4, AVI, MOV, etc.
- **HTTP Streams**: M3U8, direct video URLs
- **RTSP Streams**: IP cameras, streaming servers
- **Webcam**: Use `0` for default webcam

## Performance Tips

1. **Reduce Grid Size**: Use `--dims 64` for faster processing
2. **Skip Frames**: Use `--skip 5` to process every 5th frame
3. **Lower Resolution**: The system automatically downscales to 480p
4. **Motion Threshold**: Adjust in code for sensitivity

## Troubleshooting

### YouTube Videos Not Working
- Ensure yt-dlp is installed: `pip install yt-dlp`
- Try updating yt-dlp: `pip install -U yt-dlp`

### Webcam Access Denied (macOS)
- Grant camera permissions in System Preferences > Security & Privacy > Camera

### Out of Memory
- Reduce voxel dimensions: `--dims 64`
- Process shorter segments of video

## Advanced Usage

### Custom Processing

```python
from stream_to_voxels import StreamToVoxels

# Create processor
processor = StreamToVoxels("video.mp4", voxel_dims=(128, 128, 128))

# Customize parameters
processor.motion_threshold = 40  # Higher = less sensitive
processor.voxel_decay = 0.95     # Faster decay
processor.camera_fov = 90.0      # Wider field of view

# Process
processor.start()
processor.save_voxels("output.npy")
```

### Batch Processing

```bash
# Process multiple videos
for video in *.mp4; do
    python stream_to_voxels.py "$video" --output "${video%.mp4}_voxels.npy"
done
```