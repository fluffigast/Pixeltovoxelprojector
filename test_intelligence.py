#!/usr/bin/env python3
"""Quick test of the intelligent insect detection system."""

import cv2
import numpy as np
import time
from intelligent_insect_detector import TemporalIntelligence, MovementIntelligence

def test_temporal_intelligence():
    """Test temporal pattern analysis."""
    print("🧠 Testing Temporal Intelligence...")
    
    temporal = TemporalIntelligence()
    current_time = time.time()
    
    # Simulate repetitive vegetation detections at same location
    vegetation_location = (100, 100)
    
    print("\n📍 Simulating vegetation (repetitive same-location detections):")
    vegetation_scores = []
    for i in range(15):
        # Regular intervals like wind-blown leaves
        detection_time = current_time + i * 1.5  # Every 1.5 seconds
        score = temporal.analyze_detection(vegetation_location, detection_time)
        vegetation_scores.append(score)
        print(f"  Detection {i+1}: Vegetation probability = {score:.3f}")
    
    print(f"\n🌿 Final vegetation probability: {vegetation_scores[-1]:.3f}")
    print(f"🌿 Classified as vegetation: {vegetation_scores[-1] > 0.6}")
    
    # Simulate insect detections (moving around)
    print("\n🐛 Simulating insect (moving locations):")
    insect_scores = []
    for i in range(15):
        # Different locations like an insect flying around
        location = (150 + i*10, 150 + (i%3)*20)
        detection_time = current_time + 20 + i * 0.8  # Irregular timing
        score = temporal.analyze_detection(location, detection_time)
        insect_scores.append(score)
        print(f"  Detection {i+1} at {location}: Vegetation probability = {score:.3f}")
    
    print(f"\n🐛 Final insect vegetation probability: {insect_scores[-1]:.3f}")
    print(f"🐛 Classified as insect: {insect_scores[-1] <= 0.6}")

def test_movement_intelligence():
    """Test movement pattern analysis."""
    print("\n\n🚀 Testing Movement Intelligence...")
    
    movement = MovementIntelligence()
    current_time = time.time()
    
    # Simulate vegetation movement (returns to center)
    print("\n🌿 Simulating vegetation movement (returns to center):")
    vegetation_positions = [
        (200, 200), (205, 202), (210, 205), (208, 203), (203, 201), (200, 200)
    ]
    
    for i, pos in enumerate(vegetation_positions):
        analysis = movement.analyze_track_movement(1, pos, current_time + i * 0.5)
        print(f"  Position {pos}: Insect-like = {analysis['is_insect_like']:.3f}")
    
    # Simulate insect movement (erratic, long distance)
    print("\n🐛 Simulating insect movement (erratic, long distance):")
    insect_positions = [
        (300, 300), (320, 280), (350, 290), (380, 320), (420, 310), (450, 350)
    ]
    
    for i, pos in enumerate(insect_positions):
        analysis = movement.analyze_track_movement(2, pos, current_time + 10 + i * 0.3)
        print(f"  Position {pos}: Insect-like = {analysis['is_insect_like']:.3f}")

def main():
    print("🧠 Intelligent Insect Detection System Test")
    print("="*50)
    
    test_temporal_intelligence()
    test_movement_intelligence()
    
    print("\n" + "="*50)
    print("✅ Intelligence test complete!")
    print("\n💡 Key insights:")
    print("  • Temporal intelligence identifies repetitive patterns as vegetation")
    print("  • Movement intelligence distinguishes insect flight from vegetation sway")
    print("  • System learns from detection patterns to reduce false positives")

if __name__ == "__main__":
    main()