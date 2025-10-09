"""
Demo Training Script with Dummy Data
Use this to test the training pipeline without real video data
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from models import CogVideoXTransformer3DModel
from data.video_dataset import DummyVideoDataset


def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        # Move batch to device
        latents = batch['latents'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        
        # Reshape latents to model's expected format
        # From [batch, channels, frames, height, width] to [batch, frames, channels, height, width]
        latents = latents.permute(0, 2, 1, 3, 4)
        
        # Simple diffusion loss
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, 1000, (latents.shape[0],), device=device)
        
        # Add noise to latents
        noisy_latents = latents + noise
        
        # Create dummy encoder hidden states for text conditioning
        # Shape: (batch_size, seq_length, text_embed_dim)
        batch_size = latents.shape[0]
        text_seq_len = 10  # Must match model's max_text_seq_length
        encoder_hidden_states = torch.randn(batch_size, text_seq_len, 4096, device=device)
        
        # Predict noise using the model
        # Model forward expects: (hidden_states, timestep, encoder_hidden_states)
        # hidden_states shape: [batch, frames, channels, height, width]
        model_output = model(
            hidden_states=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=True
        )
        
        # Get the predicted output
        predicted_noise = model_output.sample
        
        # Compute loss (predict the noise)
        loss = nn.functional.mse_loss(predicted_noise, noise)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
        
        # Only train on a few batches for demo
        if batch_idx >= 10:
            break
    
    avg_loss = total_loss / min(len(dataloader), 10)
    return avg_loss


def main():
    print("=" * 60)
    print("Blueberry Video - Demo Training")
    print("=" * 60)
    print("\nThis demo uses dummy data to test the training pipeline.")
    print("For real training, prepare your dataset and use train_video.py\n")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize small model for demo
    print("\nInitializing CogVideoX model (small version for demo)...")
    # Note: sample dimensions must match the input latent dimensions
    # For 64x64 resolution with patch_size=2, we get 32x32 patches
    # For 8 frames with temporal_compression_ratio=4: (8-1)*4+1 = 29
    text_seq_len = 10  # Dummy text sequence length
    model = CogVideoXTransformer3DModel(
        num_attention_heads=4,
        attention_head_dim=32,
        in_channels=16,
        out_channels=16,
        num_layers=2,  # Just 2 layers for demo
        sample_width=32,  # 64/2 (patch_size)
        sample_height=32,  # 64/2 (patch_size)
        sample_frames=29,  # (8-1)*4+1 for 8 frames
        max_text_seq_length=text_seq_len,  # Must match encoder_hidden_states seq length
    )
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Create dummy dataset
    print("\nCreating dummy dataset...")
    dataset = DummyVideoDataset(
        num_samples=50,
        resolution=64,  # Small resolution for demo
        num_frames=8,   # Few frames for demo
        latent_channels=16  # Match model's in_channels
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Training loop
    num_epochs = 3
    print(f"\nStarting demo training for {num_epochs} epochs...")
    print("(Only 10 batches per epoch for demo)\n")
    
    for epoch in range(num_epochs):
        avg_loss = train_epoch(model, dataloader, optimizer, device, epoch)
        print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Prepare your video dataset")
    print("2. Update configs/video/train/cogvideox.yaml")
    print("3. Run: python train_video.py")
    print("\nHappy researching! 🚀")


if __name__ == "__main__":
    main()

