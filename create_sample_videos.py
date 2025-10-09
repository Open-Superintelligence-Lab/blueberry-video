"""
Create sample videos for testing the data pipeline
Generates simple synthetic videos without external data
"""

import cv2
import numpy as np
from pathlib import Path
import json


def create_moving_circle_video(output_path, num_frames=16, resolution=256):
    """Create a video with a moving circle"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, 8.0, (resolution, resolution))
    
    for i in range(num_frames):
        # Create blank frame
        frame = np.zeros((resolution, resolution, 3), dtype=np.uint8)
        
        # Draw moving circle
        center_x = int(resolution * 0.2 + (resolution * 0.6) * (i / num_frames))
        center_y = resolution // 2
        radius = 30
        color = (0, 255, 0)  # Green
        cv2.circle(frame, (center_x, center_y), radius, color, -1)
        
        out.write(frame)
    
    out.release()


def create_expanding_square_video(output_path, num_frames=16, resolution=256):
    """Create a video with an expanding square"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, 8.0, (resolution, resolution))
    
    for i in range(num_frames):
        # Create blank frame
        frame = np.zeros((resolution, resolution, 3), dtype=np.uint8)
        
        # Draw expanding square
        size = int(20 + 100 * (i / num_frames))
        center = resolution // 2
        pt1 = (center - size, center - size)
        pt2 = (center + size, center + size)
        color = (255, 0, 0)  # Blue
        cv2.rectangle(frame, pt1, pt2, color, -1)
        
        out.write(frame)
    
    out.release()


def create_rotating_line_video(output_path, num_frames=16, resolution=256):
    """Create a video with a rotating line"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, 8.0, (resolution, resolution))
    
    center = (resolution // 2, resolution // 2)
    length = resolution // 3
    
    for i in range(num_frames):
        # Create blank frame
        frame = np.zeros((resolution, resolution, 3), dtype=np.uint8)
        
        # Draw rotating line
        angle = 2 * np.pi * (i / num_frames)
        end_x = int(center[0] + length * np.cos(angle))
        end_y = int(center[1] + length * np.sin(angle))
        color = (0, 0, 255)  # Red
        cv2.line(frame, center, (end_x, end_y), color, 5)
        
        out.write(frame)
    
    out.release()


def main():
    print("Creating sample videos for testing...")
    
    # Create output directory
    output_dir = Path("data/videos/videos")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate sample videos
    videos = [
        ("sample_001_circle.mp4", create_moving_circle_video, "A green circle moving from left to right"),
        ("sample_002_square.mp4", create_expanding_square_video, "A blue square expanding from center"),
        ("sample_003_line.mp4", create_rotating_line_video, "A red line rotating around center"),
    ]
    
    metadata = {}
    
    for filename, create_func, description in videos:
        output_path = output_dir / filename
        print(f"Creating {filename}...")
        create_func(output_path, num_frames=16, resolution=256)
        
        # Add to metadata
        video_name = output_path.stem
        metadata[video_name] = description
    
    # Save metadata
    metadata_path = output_dir.parent / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Created {len(videos)} sample videos")
    print(f"📁 Location: {output_dir}")
    print(f"📝 Metadata: {metadata_path}")
    print("\nYou can now run: python train_video.py")


if __name__ == "__main__":
    main()

