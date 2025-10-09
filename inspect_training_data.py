"""
Script to inspect training data format
Shows what the dummy training dataset returns
"""

import torch
from data.video_dataset import DummyVideoDataset

def main():
    print("=" * 60)
    print("Training Data Inspector")
    print("=" * 60)
    
    # Create dataset with same params as training
    dataset = DummyVideoDataset(
        num_samples=50,
        resolution=64,
        num_frames=8,
        latent_channels=16
    )
    
    print(f"\nDataset Info:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Resolution: 64x64")
    print(f"  Frames per sample: 8")
    print(f"  Latent channels: 16")
    
    # Get a few examples
    print("\n" + "=" * 60)
    print("Sample Examples:")
    print("=" * 60)
    
    for i in range(5):
        sample = dataset[i]
        
        print(f"\nExample {i}:")
        print(f"  Latents shape: {sample['latents'].shape}")
        print(f"  Latents dtype: {sample['latents'].dtype}")
        print(f"  Latents range: [{sample['latents'].min():.3f}, {sample['latents'].max():.3f}]")
        print(f"  Prompt: '{sample['prompts']}'")
        print(f"  Video name: {sample['video_name']}")
    
    # Show what a batch would look like
    print("\n" + "=" * 60)
    print("Batch Example (as used in training):")
    print("=" * 60)
    
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(dataloader))
    
    print(f"\nBatch contents:")
    print(f"  Latents shape: {batch['latents'].shape}")
    print(f"    [batch_size, channels, frames, height, width]")
    print(f"    [{batch['latents'].shape[0]}, {batch['latents'].shape[1]}, {batch['latents'].shape[2]}, {batch['latents'].shape[3]}, {batch['latents'].shape[4]}]")
    print(f"  Prompts: {batch['prompts']}")
    print(f"  Video names: {batch['video_name']}")
    
    print("\n" + "=" * 60)
    print("Training Process:")
    print("=" * 60)
    print(f"  1. Latents are permuted to: [batch, frames, channels, height, width]")
    print(f"  2. Random noise is added (diffusion process)")
    print(f"  3. Model predicts the noise")
    print(f"  4. Loss = MSE between predicted noise and actual noise")
    print(f"  5. Backprop and optimizer step")
    
    print("\n" + "=" * 60)
    print("Notes:")
    print("=" * 60)
    print("  • These are DUMMY latents (random noise)")
    print("  • Real training would use VAE-encoded video latents")
    print("  • Text prompts are cyclical (5 different prompts repeated)")
    print("  • Training is limited to 10 batches/epoch for demo speed")

if __name__ == "__main__":
    main()

