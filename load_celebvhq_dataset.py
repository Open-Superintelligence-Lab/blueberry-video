"""
Script to load and explore the CelebV-HQ dataset from Hugging Face.
Dataset: https://huggingface.co/datasets/SwayStar123/CelebV-HQ
"""

from datasets import load_dataset
import json
from pathlib import Path

def load_celebvhq_dataset():
    """Load the CelebV-HQ dataset from Hugging Face."""
    print("Loading CelebV-HQ dataset from Hugging Face...")
    print("=" * 80)
    
    # Load the dataset
    dataset = load_dataset("SwayStar123/CelebV-HQ")
    
    # Print dataset information
    print("\n📊 Dataset Structure:")
    print(dataset)
    
    print("\n📋 Dataset Splits:")
    for split in dataset.keys():
        print(f"  - {split}: {len(dataset[split])} samples")
    
    # Get the first split available
    split_name = list(dataset.keys())[0]
    data = dataset[split_name]
    
    print(f"\n🔍 Dataset Features:")
    print(data.features)
    
    # Examine first sample
    print(f"\n🎬 First Sample from '{split_name}' split:")
    print("-" * 80)
    first_sample = data[0]
    
    for key, value in first_sample.items():
        if isinstance(value, (str, int, float)):
            print(f"{key}: {value}")
        elif isinstance(value, list):
            print(f"{key}: {type(value).__name__} with {len(value)} items")
            if value and len(value) > 0:
                print(f"  First item: {value[0]}")
        elif isinstance(value, dict):
            print(f"{key}: {type(value).__name__} with keys: {list(value.keys())}")
        else:
            print(f"{key}: {type(value).__name__}")
    
    print("\n" + "=" * 80)
    print("📈 Exploring Multiple Samples:")
    print("-" * 80)
    
    # Show several samples
    num_samples = min(5, len(data))
    samples_info = []
    
    for i in range(num_samples):
        sample = data[i]
        print(f"\n📹 Sample {i+1}/{num_samples}:")
        
        sample_info = {}
        for key, value in sample.items():
            if isinstance(value, str):
                print(f"  {key}: {value[:100]}..." if len(str(value)) > 100 else f"  {key}: {value}")
                sample_info[key] = value
            elif isinstance(value, (int, float)):
                print(f"  {key}: {value}")
                sample_info[key] = value
            elif isinstance(value, list):
                print(f"  {key}: list with {len(value)} items")
                sample_info[key] = f"list with {len(value)} items"
            elif isinstance(value, dict):
                print(f"  {key}: dict with keys: {list(value.keys())[:5]}")
                sample_info[key] = f"dict with keys: {list(value.keys())[:5]}"
            else:
                print(f"  {key}: {type(value).__name__}")
                sample_info[key] = type(value).__name__
        
        samples_info.append(sample_info)
    
    # Save sample info
    output_dir = Path("/root/blueberry-video/celebvhq_data")
    output_dir.mkdir(exist_ok=True)
    
    info_file = output_dir / "samples_info.json"
    with open(info_file, 'w') as f:
        json.dump(samples_info, f, indent=2)
    
    print(f"\n\n💾 Sample information saved to: {info_file}")
    print("=" * 80)
    
    return dataset

if __name__ == "__main__":
    dataset = load_celebvhq_dataset()
    print("\n✅ Dataset loaded successfully!")

