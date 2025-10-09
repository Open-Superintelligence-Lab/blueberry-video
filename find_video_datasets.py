"""
Search for Hugging Face datasets that actually contain video files (not just URLs).
"""

from datasets import load_dataset
import requests

# Try some known datasets that should have actual video files
datasets_to_try = [
    "HuggingFaceM4/WebVid",  # Web videos
    "microsoft/XTD",  # Cross-modal dataset
    "facebook/ego4d_clips",  # Ego-centric videos
    "ceyda/smithsonian_butterflies",  # Nature videos
    "clip-benchmark/wds_vtab-dmlab",  # VTAB videos
]

print("🔍 Searching for Hugging Face datasets with actual video files...")
print("=" * 80)

for dataset_name in datasets_to_try:
    print(f"\n📦 Trying: {dataset_name}")
    try:
        # Try to load just the first sample to check structure
        dataset = load_dataset(dataset_name, split="train", streaming=True)
        first_sample = next(iter(dataset))
        
        print(f"   ✅ Loaded successfully!")
        print(f"   Fields: {list(first_sample.keys())}")
        
        # Check for video field
        for key, value in first_sample.items():
            value_type = type(value).__name__
            if 'video' in key.lower() or value_type == 'bytes':
                print(f"   🎬 Found video field: '{key}' (type: {value_type})")
                if hasattr(value, 'path'):
                    print(f"      Path: {value.path}")
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)[:100]}")

print("\n" + "=" * 80)
print("Checking Hugging Face Hub for video datasets...")

# Try a simple, known working dataset
print("\n📹 Trying a small video dataset: UCF101 subset")

