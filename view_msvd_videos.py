"""
Script to download and display sample videos from the MSVD dataset.
"""

from datasets import load_dataset
import os
from pathlib import Path
import json

def view_msvd_videos(num_samples=5):
    """Download and display sample videos from MSVD dataset."""
    
    print("Loading MSVD dataset...")
    dataset = load_dataset("friedrichor/MSVD")
    
    # Create directory for videos
    video_dir = Path("/root/blueberry-video/msvd_videos")
    video_dir.mkdir(exist_ok=True)
    
    print(f"\n📹 Downloading {num_samples} sample videos from the training set...")
    print("=" * 80)
    
    train_data = dataset['train']
    
    # Get metadata about the dataset columns
    print("\n🔍 Dataset Features:")
    print(train_data.features)
    print()
    
    samples_info = []
    
    for i in range(min(num_samples, len(train_data))):
        sample = train_data[i]
        print(f"\n📼 Sample {i+1}/{num_samples}")
        print("-" * 80)
        print(f"Video ID: {sample['video_id']}")
        print(f"Video File: {sample['video']}")
        print(f"Number of Captions: {len(sample['caption'])}")
        print(f"\n📝 Captions (showing first 3):")
        for j, caption in enumerate(sample['caption'][:3], 1):
            print(f"  {j}. {caption}")
        if len(sample['caption']) > 3:
            print(f"  ... and {len(sample['caption']) - 3} more captions")
        
        # Store sample info
        sample_info = {
            'video_id': sample['video_id'],
            'video_file': sample['video'],
            'num_captions': len(sample['caption']),
            'captions': sample['caption'][:5],  # Store first 5 captions
            'source': sample['source']
        }
        samples_info.append(sample_info)
        
        # Check if dataset has actual video data or URLs
        print(f"\n🔎 Video field type: {type(sample['video'])}")
        print(f"   Video field value: {sample['video']}")
    
    # Save samples info to JSON
    info_file = video_dir / "samples_info.json"
    with open(info_file, 'w') as f:
        json.dump(samples_info, f, indent=2)
    
    print(f"\n\n💾 Sample information saved to: {info_file}")
    print("\n" + "=" * 80)
    print("\n📌 Note: The MSVD dataset contains video filenames (.avi files), but the actual")
    print("   video files need to be downloaded separately from the original MSVD source.")
    print("   The dataset provides metadata and captions for the videos.")
    
    # Show how to access the data
    print("\n\n🎯 How to use this dataset:")
    print("-" * 80)
    print("1. Each video has 30-50 human-annotated captions")
    print("2. Video IDs follow the format: [YouTubeID]_[start_frame]_[end_frame]")
    print("3. The dataset is perfect for:")
    print("   - Video captioning tasks")
    print("   - Text-to-video retrieval")
    print("   - Video understanding models")
    print("   - Cross-modal learning")
    
    return samples_info

if __name__ == "__main__":
    samples = view_msvd_videos(num_samples=5)
    
    print("\n\n✅ Completed! Check the samples_info.json file for detailed information.")

