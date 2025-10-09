"""
Load UCF101 subset dataset with actual video files!
"""

from datasets import load_dataset
import json
from pathlib import Path

print("🎬 Loading UCF101 Subset Dataset...")
print("=" * 80)

try:
    # Load the dataset (without streaming to get actual files)
    print("\n📦 Loading dataset: sayakpaul/ucf101-subset")
    dataset = load_dataset("sayakpaul/ucf101-subset")
    
    print(f"\n✅ Dataset loaded successfully!")
    print(f"\n📊 Dataset Structure:")
    print(dataset)
    
    # Get splits
    print(f"\n📋 Available Splits:")
    for split_name in dataset.keys():
        print(f"  - {split_name}: {len(dataset[split_name])} samples")
    
    # Get first split
    split_name = list(dataset.keys())[0]
    data = dataset[split_name]
    
    print(f"\n🔍 Features in '{split_name}' split:")
    print(data.features)
    
    # Get first few samples
    print(f"\n" + "=" * 80)
    print("📹 SAMPLE VIDEOS")
    print("=" * 80)
    
    samples_info = []
    num_samples = min(5, len(data))
    
    for i in range(num_samples):
        sample = data[i]
        print(f"\n{'─' * 80}")
        print(f"Sample {i+1}/{num_samples}")
        print(f"{'─' * 80}")
        
        sample_info = {}
        for key, value in sample.items():
            if key == 'video':
                # This should be the actual video data
                print(f"\n🎬 Video field:")
                if hasattr(value, 'keys'):
                    print(f"   Type: dict with keys {list(value.keys())}")
                    sample_info['video'] = {k: str(v)[:100] for k, v in value.items()}
                elif hasattr(value, 'path'):
                    print(f"   Path: {value.path}")
                    sample_info['video'] = value.path
                elif isinstance(value, bytes):
                    print(f"   Bytes: {len(value)} bytes")
                    sample_info['video'] = f"<{len(value)} bytes>"
                else:
                    print(f"   Type: {type(value)}")
                    print(f"   Value: {str(value)[:200]}")
                    sample_info['video'] = str(type(value))
            elif key == 'label':
                print(f"\n🏷️  Label: {value}")
                sample_info['label'] = value
            else:
                print(f"\n📝 {key}: {str(value)[:100]}")
                sample_info[key] = str(value)[:100]
        
        samples_info.append(sample_info)
    
    # Save info
    output_dir = Path("/root/blueberry-video/ucf101_data")
    output_dir.mkdir(exist_ok=True)
    
    info_file = output_dir / "samples_info.json"
    with open(info_file, 'w') as f:
        json.dump(samples_info, f, indent=2)
    
    print(f"\n\n{'=' * 80}")
    print(f"💾 Sample information saved to: {info_file}")
    print("=" * 80)
    
    print("\n\n📝 UCF101 DATASET INFO:")
    print("─" * 80)
    print("UCF101 is an action recognition dataset with:")
    print("  • 101 action categories")
    print("  • Realistic web videos")
    print("  • Activities include: sports, playing instruments, human-object interactions")
    print("  • Perfect for: action recognition, video classification, temporal modeling")
    print("=" * 80)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

