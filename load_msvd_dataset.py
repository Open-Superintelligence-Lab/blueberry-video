"""
Script to load and explore the MSVD dataset from Hugging Face.
Dataset: https://huggingface.co/datasets/friedrichor/MSVD
"""

from datasets import load_dataset
import pandas as pd

def load_msvd_dataset():
    """Load the MSVD dataset from Hugging Face."""
    print("Loading MSVD dataset from Hugging Face...")
    print("=" * 60)
    
    # Load the dataset
    dataset = load_dataset("friedrichor/MSVD")
    
    # Print dataset information
    print("\n📊 Dataset Structure:")
    print(dataset)
    
    print("\n📋 Dataset Splits:")
    for split in dataset.keys():
        print(f"  - {split}: {len(dataset[split])} samples")
    
    # Examine the training set
    train_data = dataset['train']
    
    print("\n🔍 First Training Sample:")
    print("-" * 60)
    first_sample = train_data[0]
    print(f"Video ID: {first_sample['video_id']}")
    print(f"Video File: {first_sample['video']}")
    print(f"Number of captions: {len(first_sample['caption'])}")
    print(f"Source: {first_sample['source']}")
    print(f"\n📝 Sample Captions (first 5):")
    for i, caption in enumerate(first_sample['caption'][:5], 1):
        print(f"  {i}. {caption}")
    
    print("\n" + "=" * 60)
    print("📈 Dataset Statistics:")
    print("-" * 60)
    
    # Calculate statistics for each split
    for split_name in ['train', 'validation', 'test']:
        split_data = dataset[split_name]
        total_captions = sum(len(sample['caption']) for sample in split_data)
        avg_captions = total_captions / len(split_data)
        
        print(f"\n{split_name.upper()}:")
        print(f"  Videos: {len(split_data)}")
        print(f"  Total captions: {total_captions:,}")
        print(f"  Average captions per video: {avg_captions:.1f}")
    
    # Show some random samples
    print("\n" + "=" * 60)
    print("🎬 Random Samples from Training Set:")
    print("-" * 60)
    
    import random
    random_indices = random.sample(range(len(train_data)), min(3, len(train_data)))
    
    for idx in random_indices:
        sample = train_data[idx]
        print(f"\nSample #{idx}:")
        print(f"  Video ID: {sample['video_id']}")
        print(f"  First caption: {sample['caption'][0]}")
        print(f"  Total captions: {len(sample['caption'])}")
    
    print("\n" + "=" * 60)
    print("✅ Dataset loaded successfully!")
    
    return dataset

if __name__ == "__main__":
    dataset = load_msvd_dataset()
    
    # Optional: Save a sample to CSV for inspection
    print("\n💾 Saving train split sample to CSV...")
    train_df = pd.DataFrame(dataset['train'])
    train_df_sample = train_df.head(10)
    train_df_sample.to_csv('/root/blueberry-video/msvd_sample.csv', index=False)
    print(f"Sample saved to: /root/blueberry-video/msvd_sample.csv")

