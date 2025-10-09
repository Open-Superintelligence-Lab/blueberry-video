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
        prompts = batch['prompts']
        
        # Forward pass
        optimizer.zero_grad()
        
        # Simple diffusion loss (MSE between predicted and target noise)
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, 1000, (latents.shape[0],), device=device)
        
        # Add noise to latents
        noisy_latents = latents + noise
        
        # Predict noise
        model_output = model(
            hidden_states=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=prompts,
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
    # Load configuration
    config_path = "configs/video/train/cogvideox.yaml"
    config = load_config(config_path)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model based on config
    model_type = config.get('model_type', 'cogvideox')
    
    if model_type == 'cogvideox':
        model = CogVideoXTransformer3DModel(
            num_attention_heads=config['num_attention_heads'],
            attention_head_dim=config['attention_head_dim'],
            in_channels=config['in_channels'],
            out_channels=config.get('out_channels', config['in_channels']),
            num_layers=config['num_layers'],
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

