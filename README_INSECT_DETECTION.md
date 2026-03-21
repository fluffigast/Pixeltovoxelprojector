# 🐛 Insect Detection with 3D Voxel Tracking

This system uses advanced computer vision to detect insects in video streams and highlight them with 3D voxel overlays in real-time.

## Features

### 🎯 **Intelligent Insect Detection**
- **Size filtering**: Detects objects 8-500 pixels (typical insect sizes)
- **Shape analysis**: Filters by aspect ratio and solidity
- **Background subtraction**: Eliminates static background elements
- **Motion analysis**: Focuses on moving objects

### 🧠 **Behavioral Analysis**
- **Speed tracking**: Measures movement velocity
- **Erratic movement**: Detects zigzag flight patterns typical of insects
- **Hovering detection**: Identifies stationary flying insects
- **Direction changes**: Counts sudden turns and maneuvers

### 🎨 **3D Voxel Visualization**
- **Real-time tracking**: Projects insect positions into 3D space
- **Color coding**: Different colors for different behaviors
  - 🟢 **Green**: Normal flying insects
  - 🟡 **Yellow**: Hovering insects (bees, flies)
  - 🟣 **Magenta**: Erratic movement (mosquitoes, gnats)
- **Confidence levels**: Brightness indicates detection confidence

## Installation & Usage

```bash
# Activate environment
source venv/bin/activate

# Run with webcam (best for real insect detection)
python insect_detector_voxels.py 0

# Run with video file
python insect_detector_voxels.py path/to/video.mp4

# Run with online stream
python insect_detector_voxels.py "https://youtube.com/watch?v=VIDEO_ID"
```

## Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `r` | Reset voxels and tracking |
| `d` | Toggle detection boxes |
| `t` | Toggle track trails |
| `c` | Toggle confidence display |
| `+` | Increase sensitivity |
| `-` | Decrease sensitivity |

## What You'll See

### 📦 **Detection Boxes**
- Colored rectangles around detected insects
- Green = High confidence (>80%)
- Yellow = Medium confidence (60-80%)
- Orange = Low confidence (<60%)

### 🎯 **Behavior Labels**
- "HOVER" for stationary flying insects
- "ERRATIC" for rapid direction changes

### ✨ **3D Voxel Overlay**
- Glowing particles that follow insect movement
- Particles fade over time creating motion trails
- Color indicates behavior type

### 📊 **Live Statistics**
- Insect count by confidence level
- Behavior analysis (hovering vs erratic)
- Active voxel count
- Motion sensitivity setting

## Detection Algorithm

### 1. **Background Subtraction**
```python
# Uses MOG2 background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500, varThreshold=50, detectShadows=True
)
```

### 2. **Size & Shape Filtering**
- Area: 8-500 pixels
- Aspect ratio: 0.3-3.0
- Solidity: >0.3 (compact objects)
- Extent: >0.3 (fills bounding box)

### 3. **Behavioral Analysis**
- **Speed**: 2-50 pixels/frame
- **Erratic movement**: >15° direction changes
- **Hovering**: Movement within 8-pixel radius

### 4. **3D Projection**
- Estimates depth based on insect size
- Projects to 3D voxel grid
- Applies temporal decay for trails

## Tips for Best Results

### 🎥 **Camera Setup**
- Use good lighting (natural light preferred)
- Avoid cluttered backgrounds
- Position camera 2-6 feet from target area
- Ensure insects are 10-100 pixels in size

### ⚙️ **Sensitivity Tuning**
- **High sensitivity** (low threshold): Detects smaller movements
- **Low sensitivity** (high threshold): Reduces false positives
- Start with default (20) and adjust as needed

### 🐛 **Target Insects**
Works best with:
- **Flying insects**: Flies, mosquitoes, bees, wasps
- **Moths and butterflies**: Larger specimens
- **Gnats and midges**: In swarms

Less effective with:
- Very small insects (<5 pixels)
- Very fast insects (hummingbird moths)
- Insects on complex backgrounds

## Example Output

```
🐛 Insects: High:3 Med:1 Low:0
📊 Behavior: Hover:1 Erratic:2
🎯 Voxels: 1,247
⚙️ Sensitivity: 20
```

## Advanced Configuration

You can modify detection parameters by editing the code:

```python
# In InsectDetector.__init__()
self.min_area = 8          # Minimum size
self.max_area = 500        # Maximum size
self.motion_threshold = 15 # Motion sensitivity
self.min_speed = 2.0       # Minimum movement speed
self.erratic_threshold = 15 # Direction change sensitivity
```

## Troubleshooting

### No Detections
- Increase sensitivity with `+` key
- Check lighting conditions
- Ensure insects are moving
- Verify camera is working

### Too Many False Positives
- Decrease sensitivity with `-` key
- Remove moving objects from background
- Let background subtractor adapt (30+ seconds)

### Poor Tracking
- Ensure insects are clearly visible
- Avoid rapid camera movement
- Use consistent lighting

## Scientific Applications

This system can be used for:
- **Entomology research**: Studying insect behavior
- **Pest monitoring**: Tracking fly/mosquito activity
- **Pollinator studies**: Monitoring bee movements
- **Indoor air quality**: Detecting flying particles
- **Wildlife observation**: Non-invasive insect tracking

The 3D voxel representation provides spatial context that traditional 2D tracking cannot offer, making it valuable for understanding insect flight patterns and behaviors.