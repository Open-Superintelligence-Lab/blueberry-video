"""
Load a working video dataset with actual downloadable videos.
"""

from datasets import load_dataset, DownloadMode
import json
from pathlib import Path

print("🎬 Finding and loading a video dataset with actual files...")
print("=" * 80)

# Try UCF101 with verification disabled
print("\n📦 Attempt 1: UCF101 subset (ignoring verification)...")
try:
    dataset = load_dataset(
        "sayakpaul/ucf101-subset",
        verification_mode="no_checks",
    )
    
    print("✅ Loaded successfully!")
    print(f"\nDataset: {dataset}")
    
    # Get train split
    train_data = dataset['train']
    print(f"\nTrain samples: {len(train_data)}")
    
    # Get first sample
    print("\n📹 First Sample:")
    first = train_data[0]
    for key, value in first.items():
        if key == 'video':
            if hasattr(value, 'keys'):
                print(f"  video: dict with keys {list(value.keys())}")
                if 'path' in value:
                    print(f"    path: {value['path']}")
                if 'bytes' in value:
                    print(f"    bytes: {len(value['bytes'])} bytes")
            else:
                print(f"  video: {type(value)} - {str(value)[:100]}")
        else:
            print(f"  {key}: {value}")
    
    # Try to save/access a video
    if 'video' in first:
        video_data = first['video']
        if isinstance(video_data, dict) and 'bytes' in video_data:
            print(f"\n✅ Video has bytes! Size: {len(video_data['bytes'])} bytes")
            
            # Save first video
            output_dir = Path("/root/blueberry-video/ucf101_videos")
            output_dir.mkdir(exist_ok=True)
            
            video_file = output_dir / f"sample_0_{first.get('label', 'unknown')}.mp4"
            with open(video_file, 'wb') as f:
                f.write(video_data['bytes'])
            
            print(f"💾 Saved video to: {video_file}")
            print(f"   File size: {video_file.stat().st_size} bytes")
            
            # Save a few more
            for i in range(1, min(5, len(train_data))):
                sample = train_data[i]
                if 'video' in sample and isinstance(sample['video'], dict) and 'bytes' in sample['video']:
                    video_file = output_dir / f"sample_{i}_{sample.get('label', 'unknown')}.mp4"
                    with open(video_file, 'wb') as f:
                        f.write(sample['video']['bytes'])
                    print(f"💾 Saved: {video_file.name} ({video_file.stat().st_size} bytes)")
            
            print(f"\n✅ SUCCESS! Downloaded {min(5, len(train_data))} videos!")
            print(f"📁 Location: {output_dir}")
            
        elif hasattr(video_data, 'path'):
            print(f"\n📂 Video file path: {video_data.path}")

except Exception as e:
    print(f"❌ Failed: {e}")

print("\n" + "=" * 80)

