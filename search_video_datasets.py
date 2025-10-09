"""
Search Hugging Face for datasets with actual video files.
"""

from datasets import load_dataset

# These are known to work and have actual videos
datasets_with_videos = [
    "HuggingFaceM4/WebVid",
    "iejMac/video-dataset-small", 
    "MCG-NJU/MultiModalQA",
    "jbarat/video_test",
]

print("🔍 Testing known video datasets on Hugging Face...")
print("=" * 80)

for dataset_name in datasets_with_videos:
    print(f"\n📦 Testing: {dataset_name}")
    try:
        # Try loading with streaming first
        dataset = load_dataset(dataset_name, split="train", streaming=True)
        first_sample = next(iter(dataset))
        
        print(f"   ✅ SUCCESS! Fields: {list(first_sample.keys())}")
        
        # Check each field
        for key, value in first_sample.items():
            if isinstance(value, bytes):
                print(f"   🎬 '{key}' contains bytes data (likely video): {len(value)} bytes")
            elif hasattr(value, 'path'):
                print(f"   🎬 '{key}' has path: {value.path}")
            elif isinstance(value, dict):
                print(f"   📋 '{key}' is dict with keys: {list(value.keys())}")
            else:
                value_str = str(value)[:80] if value else "None"
                print(f"   📝 '{key}': {value_str}")
        
        print(f"\n   🎯 This dataset works! Let's use it.")
        break
        
    except Exception as e:
        print(f"   ❌ Failed: {str(e)[:150]}")
        continue

# If none work, try UCF101 alternative
print("\n\nTrying UCF101 alternative sources...")
ucf_alternatives = [
    "sayakpaul/ucf101-subset",
    "fcakyon/ucf101-subset",
]

for dataset_name in ucf_alternatives:
    print(f"\n📦 Testing: {dataset_name}")
    try:
        dataset = load_dataset(dataset_name, streaming=True)
        first_sample = next(iter(dataset.values().__iter__()))
        print(f"   ✅ SUCCESS! Fields: {list(first_sample.keys())}")
        break
    except Exception as e:
        print(f"   ❌ Failed: {str(e)[:100]}")

