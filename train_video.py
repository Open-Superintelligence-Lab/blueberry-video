"""
Simple Video Diffusion Training Script
Train video generation models (CogVideoX, Hunyuan Video) on your dataset
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import yaml
from tqdm import tqdm
import argparse

from models import CogVideoXTransformer3DModel, HunyuanVideoTransformer3DModel
from data.video_dataset import VideoDataset
from utils.training import get_optimizer, get_scheduler, save_checkpoint
from utils.config import load_config


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        # Move batch to device
        latents = batch['latents'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        
        # Simple diffusion loss (MSE between predicted and target noise)
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, 1000, (latents.shape[0],), device=device)
        
        # Add noise to latents
        noisy_latents = latents + noise
        
        # Create dummy text embeddings (batch_size, seq_len, hidden_dim)
        batch_size = latents.shape[0]
        # Typical CLIP text embedding dimension is 768, sequence length 77
        dummy_text_embeddings = torch.zeros(batch_size, 77, 768, device=device)
        
        # Predict noise
        model_output = model(
            hidden_states=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=dummy_text_embeddings,
            return_dict=True
        )
        
        predicted_noise = model_output.sample
        loss = nn.functional.mse_loss(predicted_noise, noise)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train video generation model')
    parser.add_argument('--config', type=str, default='configs/video/train/cogvideox.yaml',
                        help='Path to config file')
    args = parser.parse_args()
    
    # Load configuration
    config_path = args.config
    config = load_config(config_path)
    print(f"Loading config from: {config_path}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model based on config
    model_type = config.get('model_type', 'cogvideox')
    
    if model_type == 'cogvideox':
        # Calculate sample_frames based on num_frames and temporal compression ratio
        # formula: sample_frames = ((num_frames - 1) * temporal_compression_ratio + 1)
        temporal_compression_ratio = config.get('temporal_compression_ratio', 4)
        num_frames = config['num_frames']
        sample_frames = ((num_frames - 1) * temporal_compression_ratio + 1)
        
        model = CogVideoXTransformer3DModel(
            num_attention_heads=config['num_attention_heads'],
            attention_head_dim=config['attention_head_dim'],
            in_channels=config['in_channels'],
            out_channels=config.get('out_channels', config['in_channels']),
            num_layers=config['num_layers'],
            text_embed_dim=768,  # Standard CLIP dimension
            sample_width=config['resolution'],
            sample_height=config['resolution'],
            sample_frames=sample_frames,
            temporal_compression_ratio=temporal_compression_ratio,
        )
    elif model_type == 'hunyuan':
        model = HunyuanVideoTransformer3DModel(
            num_attention_heads=config['num_attention_heads'],
            attention_head_dim=config['attention_head_dim'],
            in_channels=config['in_channels'],
            out_channels=config.get('out_channels', config['in_channels']),
            num_layers=config['num_layers'],
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    
    # Create dataset and dataloader
    dataset = VideoDataset(
        data_path=config['data_path'],
        resolution=config['resolution'],
        num_frames=config['num_frames']
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 4)
    )
    
    # Setup optimizer and scheduler
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    
    # Training loop
    num_epochs = config.get('num_epochs', 100)
    save_dir = Path(config.get('save_dir', 'checkpoints'))
    save_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Starting training for {num_epochs} epochs")
    print(f"Model: {model_type}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Batch size: {config['batch_size']}")
    
    for epoch in range(num_epochs):
        avg_loss = train_epoch(model, dataloader, optimizer, scheduler, device, epoch)
        print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config.get('save_every', 10) == 0:
            checkpoint_path = save_dir / f"checkpoint_epoch_{epoch+1}.pt"
            save_checkpoint(model, optimizer, epoch, avg_loss, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    print("Training completed!")


if __name__ == "__main__":
    main()

