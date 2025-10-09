"""
Overfitting Training Script on 5 Sample Videos
Simplified version that works with RGB videos directly
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from models import CogVideoXTransformer3DModel
from data.video_dataset import VideoDataset


def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        # Move batch to device - shape is [batch, T, C, H, W]
        latents = batch['latents'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        
        # Simple diffusion loss
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, 1000, (latents.shape[0],), device=device)
        
        # Add noise to latents
        noisy_latents = latents + noise
        
        # Create dummy encoder hidden states for text conditioning
        # Shape: (batch_size, seq_length, text_embed_dim)
        batch_size = latents.shape[0]
        text_seq_len = 77  # Standard CLIP sequence length
        encoder_hidden_states = torch.zeros(batch_size, text_seq_len, 768, device=device)
        
        # Predict noise using the model
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
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def main():
    print("=" * 70)
    print("Blueberry Video - Overfitting on 5 Sample Videos")
    print("=" * 70)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Configuration
    resolution = 64  # Very low resolution for faster training
    num_frames = 8
    num_epochs = 100
    batch_size = 1
    learning_rate = 0.001
    
    # Initialize small model for overfitting
    print(f"\nInitializing CogVideoX model...")
    print(f"  Resolution: {resolution}x{resolution}")
    print(f"  Frames: {num_frames}")
    print(f"  RGB channels: 3")
    
    # Calculate positional embedding dimensions
    patch_size = 2
    temporal_compression_ratio = 4
    sample_width = resolution // patch_size
    sample_height = resolution // patch_size
    sample_frames = ((num_frames - 1) * temporal_compression_ratio + 1)
    text_seq_len = 77
    
    model = CogVideoXTransformer3DModel(
        num_attention_heads=2,  # Very small for overfitting
        attention_head_dim=32,
        in_channels=3,  # RGB videos
        out_channels=3,
        num_layers=4,  # Small number of layers
        sample_width=sample_width,
        sample_height=sample_height,
        sample_frames=sample_frames,
        max_text_seq_length=text_seq_len,
        text_embed_dim=768,
        temporal_compression_ratio=temporal_compression_ratio,
    )
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {num_params:,}")
    
    # Create dataset
    print(f"\nLoading dataset...")
    dataset = VideoDataset(
        data_path='./data/videos',
        resolution=resolution,
        num_frames=num_frames
    )
    
    print(f"  Found {len(dataset)} videos")
    if len(dataset) == 0:
        print("  ERROR: No videos found!")
        return
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # 0 for debugging
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    
    # Create checkpoint directory
    checkpoint_dir = Path('./checkpoints/overfit')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print(f"\nStarting overfitting training for {num_epochs} epochs...")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Checkpoint dir: {checkpoint_dir}\n")
    
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        avg_loss = train_epoch(model, dataloader, optimizer, device, epoch)
        print(f"Epoch {epoch+1:3d}/{num_epochs}: Average Loss = {avg_loss:.6f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1:03d}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"  → Saved checkpoint to {checkpoint_path}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = checkpoint_dir / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, best_path)
    
    print("\n" + "=" * 70)
    print(f"Training completed!")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()

