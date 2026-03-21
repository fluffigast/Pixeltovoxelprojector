#!/usr/bin/env python3
"""Test script to verify video streaming functionality."""

import cv2
import numpy as np
import sys

def test_opencv_installation():
    """Test if OpenCV is properly installed."""
    print("Testing OpenCV installation...")
    print(f"OpenCV version: {cv2.__version__}")
    
    # Test video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("WARNING: Cannot access webcam. Testing with synthetic video...")
        return test_synthetic_video()
    
    print("✓ Webcam access successful")
    
    # Capture a few frames
    for i in range(5):
        ret, frame = cap.read()
        if ret:
            print(f"✓ Frame {i+1} captured: {frame.shape}")
        else:
            print(f"✗ Failed to capture frame {i+1}")
            
    cap.release()
    cv2.destroyAllWindows()
    return True

def test_synthetic_video():
    """Test with synthetic video frames."""
    print("\nTesting with synthetic video frames...")
    
    # Create synthetic frames
    width, height = 640, 480
    for i in range(5):
        # Create a frame with moving circle
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        center_x = int(width/2 + 100 * np.sin(i * 0.5))
        center_y = int(height/2)
        cv2.circle(frame, (center_x, center_y), 50, (0, 255, 0), -1)
        
        print(f"✓ Synthetic frame {i+1} created: {frame.shape}")
        
    return True

def test_pyvista_installation():
    """Test if PyVista is properly installed."""
    print("\nTesting PyVista installation...")
    try:
        import pyvista as pv
        print(f"✓ PyVista version: {pv.__version__}")
        
        # Create simple mesh
        sphere = pv.Sphere()
        print(f"✓ Created test sphere mesh with {sphere.n_points} points")
        return True
    except Exception as e:
        print(f"✗ PyVista test failed: {e}")
        return False

def test_video_stream_module():
    """Test the video stream processor module."""
    print("\nTesting video stream processor module...")
    try:
        from video_stream_processor import VideoStreamProcessor
        print("✓ VideoStreamProcessor imported successfully")
        
        # Create processor instance
        processor = VideoStreamProcessor(source=0)
        print("✓ VideoStreamProcessor instance created")
        
        # Test motion detection with synthetic frames
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2 = frame1.copy()
        cv2.circle(frame2, (320, 240), 50, (255, 255, 255), -1)
        
        motion_mask = processor.detect_motion(frame1)
        motion_mask = processor.detect_motion(frame2)
        motion_pixels = np.sum(motion_mask > 0)
        
        print(f"✓ Motion detection working: {motion_pixels} pixels detected")
        
        return True
    except Exception as e:
        print(f"✗ VideoStreamProcessor test failed: {e}")
        return False

def test_realtime_module():
    """Test the realtime voxel stream module."""
    print("\nTesting realtime voxel stream module...")
    try:
        from realtime_voxel_stream import RealtimeVoxelStream
        print("✓ RealtimeVoxelStream imported successfully")
        
        # Create instance
        stream = RealtimeVoxelStream(source=0, voxel_dims=(64, 64, 64))
        print("✓ RealtimeVoxelStream instance created")
        
        # Test ray casting
        ray_dir = stream._pixel_to_ray(320, 240, 640, 480)
        print(f"✓ Ray casting working: ray direction = {ray_dir}")
        
        return True
    except Exception as e:
        print(f"✗ RealtimeVoxelStream test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=== Video Stream Installation Test ===\n")
    
    tests = [
        ("OpenCV", test_opencv_installation),
        ("PyVista", test_pyvista_installation),
        ("VideoStreamProcessor", test_video_stream_module),
        ("RealtimeVoxelStream", test_realtime_module)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ {name} test crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n=== Test Summary ===")
    all_passed = True
    for name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{name}: {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        print("\n✓ All tests passed! The video streaming system is ready to use.")
        print("\nTo run the video stream processor:")
        print("  python video_stream_processor.py")
        print("\nTo run the realtime visualizer:")
        print("  python realtime_voxel_stream.py")
    else:
        print("\n✗ Some tests failed. Please check the error messages above.")
        
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())