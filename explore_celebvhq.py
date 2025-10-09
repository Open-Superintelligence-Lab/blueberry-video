"""
Script to explore and visualize the CelebV-HQ dataset structure.
Dataset: https://huggingface.co/datasets/SwayStar123/CelebV-HQ
"""

from datasets import load_dataset
import json
from pathlib import Path

def explore_celebvhq():
    """Explore the CelebV-HQ dataset structure."""
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
    
    # Get the first split
    split_name = list(dataset.keys())[0]
    data = dataset[split_name]
    
    print(f"\n🔍 Features in '{split_name}' split:")
    print(data.features)
    
    # Examine first sample in detail
    print(f"\n" + "=" * 80)
    print("🎬 FIRST SAMPLE - DETAILED VIEW")
    print("=" * 80)
    
    first_sample = data[0]
    
    print("\n📌 Meta Info:")
    if 'meta_info' in first_sample:
        meta_info = first_sample['meta_info']
        print(f"  Type: {type(meta_info)}")
        if isinstance(meta_info, dict):
            for key in meta_info.keys():
                print(f"  - {key}:")
                val = meta_info[key]
                if isinstance(val, dict) and len(val) > 0:
                    # Show first few items
                    items = list(val.items())[:3]
                    for k, v in items:
                        print(f"      {k}: {v}")
                    if len(val) > 3:
                        print(f"      ... and {len(val) - 3} more")
                else:
                    print(f"      {val}")
    
    print("\n📹 Clips:")
    if 'clips' in first_sample:
        clips = first_sample['clips']
        print(f"  Type: {type(clips)}")
        print(f"  Number of clips: {len(clips) if hasattr(clips, '__len__') else 'N/A'}")
        
        if isinstance(clips, dict):
            print(f"\n  📂 Clip IDs (showing first 5):")
            clip_items = list(clips.items())[:5]
            for clip_id, clip_data in clip_items:
                print(f"\n    Clip ID: {clip_id}")
                print(f"    Clip data type: {type(clip_data)}")
                
                if isinstance(clip_data, dict):
                    for key, value in clip_data.items():
                        if isinstance(value, str):
                            display_val = value[:100] + "..." if len(value) > 100 else value
                            print(f"      - {key}: {display_val}")
                        elif isinstance(value, (int, float, bool)):
                            print(f"      - {key}: {value}")
                        elif isinstance(value, list):
                            print(f"      - {key}: list with {len(value)} items")
                            if len(value) > 0 and len(value) <= 3:
                                print(f"           {value}")
                        else:
                            print(f"      - {key}: {type(value).__name__}")
                elif hasattr(clip_data, '__dict__'):
                    print(f"      Attributes: {dir(clip_data)}")
                else:
                    print(f"      Data: {str(clip_data)[:200]}")
            
            if len(clips) > 5:
                print(f"\n    ... and {len(clips) - 5} more clips")
    
    # Save detailed structure
    output_dir = Path("/root/blueberry-video/celebvhq_data")
    output_dir.mkdir(exist_ok=True)
    
    # Try to save a more detailed JSON
    detailed_info = {
        'dataset_splits': list(dataset.keys()),
        'num_samples': {split: len(dataset[split]) for split in dataset.keys()},
        'features': str(data.features),
        'first_sample_keys': list(first_sample.keys()) if hasattr(first_sample, 'keys') else None,
    }
    
    # Add clip IDs from first sample
    if 'clips' in first_sample and isinstance(first_sample['clips'], dict):
        detailed_info['first_sample_clip_ids'] = list(first_sample['clips'].keys())[:10]
        
        # Get one clip as example
        first_clip_id = list(first_sample['clips'].keys())[0]
        first_clip = first_sample['clips'][first_clip_id]
        detailed_info['example_clip_structure'] = {
            'clip_id': first_clip_id,
            'clip_keys': list(first_clip.keys()) if isinstance(first_clip, dict) else None,
            'clip_type': str(type(first_clip))
        }
    
    info_file = output_dir / "dataset_structure.json"
    with open(info_file, 'w') as f:
        json.dump(detailed_info, f, indent=2)
    
    print(f"\n\n💾 Dataset structure saved to: {info_file}")
    print("=" * 80)
    
    return dataset

if __name__ == "__main__":
    dataset = explore_celebvhq()
    print("\n✅ Exploration complete!")

