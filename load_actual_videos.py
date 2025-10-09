"""
Load a dataset with actual video files that can be downloaded.
"""

from datasets import load_dataset
import os

print("🎬 Loading a dataset with actual video files...")
print("=" * 80)

# Try UCF101 - a popular action recognition dataset with actual videos
try:
    print("\n📦 Attempting to load UCF101 dataset (subset)...")
    dataset = load_dataset("UCF101-24/UCF101", trust_remote_code=True, split="train", streaming=True)
    
    print("✅ Dataset loaded! Getting first sample...")
    first = next(iter(dataset))
    
    print(f"\n📋 Sample structure:")
    for key, value in first.items():
        print(f"   - {key}: {type(value).__name__}")
        
except Exception as e:
    print(f"❌ UCF101 failed: {e}")

# Try another approach - look for datasets with 'video' in features
print("\n\n📦 Trying Kinetics dataset subset...")
try:
    dataset = load_dataset("kinetics700", split="train", streaming=True)
    print("✅ Kinetics loaded!")
    first = next(iter(dataset))
    print(f"Fields: {list(first.keys())}")
except Exception as e:
    print(f"❌ Kinetics failed: {e}")

# Try ActivityNet
print("\n\n📦 Trying ActivityNet Captions...")
try:
    dataset = load_dataset("HuggingFaceM4/ActivityNet", split="train", streaming=True)
    print("✅ ActivityNet loaded!")
    first = next(iter(dataset))
    print(f"Fields: {list(first.keys())}")
except Exception as e:
    print(f"❌ ActivityNet failed: {e}")

# Try a simple one that should work
print("\n\n📦 Trying something smaller - video-dataset-small...")
try:
    # Search for any dataset with actual videos
    dataset = load_dataset("nateraw/party-parrot", split="train")
    print("✅ Party Parrot loaded!")
    print(f"Number of samples: {len(dataset)}")
    first = dataset[0]
    print(f"Fields: {list(first.keys())}")
    
    if 'image' in first:
        print("Has image field")
    if 'video' in first:
        print("Has video field")
        
except Exception as e:
    print(f"❌ Failed: {e}")

print("\n" + "=" * 80)

