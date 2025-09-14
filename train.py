import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import cv2
import numpy as np
from decord import VideoReader, cpu
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm
from pathlib import Path
from accelerate import Accelerator
from ema_pytorch import EMA
from einops import rearrange
import math
import random
import requests
import tempfile
import os
import pandas as pd
from torchvision import transforms
import torchvision.io as io
from PIL import Image

# Diffusers imports
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup

# Simple config
CONFIG = {
    'model': {
        'num_frames': 8,
        'image_size': 192,
        'patch_size': 4,
        'in_channels': 3,
        'hidden_size': 384,
        'depth': 4,
        'num_heads': 6,
        'mlp_ratio': 4.0,
        'learn_sigma': True,
    },
    'training': {
        'batch_size': 1,  # Single sample for overfitting
        'learning_rate': 1e-4,  # Higher LR for faster overfitting
        'num_epochs': 50,  # More epochs to ensure overfitting
        'gradient_clip': 1.0,
        'ema_decay': 0.9999,
        'mixed_precision': 'fp16',
        'gradient_accumulation_steps': 1,  # No accumulation needed for single sample
    },
    'dataset': {
        # OVERFITTING MODE: Using synthetic circle videos - only 1 video repeatedly
        'name': 'synthetic_circles',
        'use_local_videos': True,  # Using local synthetic dataset

        # Synthetic dataset paths
        'csv_path': 'dataset/synthetic_circles/synthetic_circles.csv',  # Path to synthetic CSV
        'video_folder': 'dataset/synthetic_circles/videos/',  # Path to synthetic video folder

        'num_frames': 8,
        'resolution': 256,  # Match video resolution
        'max_samples': 1,  # OVERFIT: Use only 1 video repeatedly
        'fps_target': 30,
    }
}

class SimpleVideoDiT(nn.Module):
    """Simplified Video Diffusion Transformer"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config['hidden_size']
        self.num_heads = config['num_heads']
        self.depth = config['depth']
        self.patch_size = config['patch_size']
        self.num_frames = config['num_frames']
        self.image_size = config['image_size']

        # Calculate dimensions
        self.num_patches = (self.num_frames // 2) * (self.image_size // self.patch_size) ** 2

        # Embeddings
        self.patch_embed = nn.Conv3d(3, self.hidden_size,
                                   kernel_size=(2, self.patch_size, self.patch_size),
                                   stride=(2, self.patch_size, self.patch_size))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.hidden_size))
        self.time_embed = nn.Sequential(
            nn.Linear(256, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(self.hidden_size, self.num_heads)
            for _ in range(self.depth)
        ])

        # Output
        self.final_layer = nn.Linear(self.hidden_size,
                                   2 * self.patch_size * self.patch_size * 3)  # learn_sigma=True

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.patch_embed.weight)
        nn.init.constant_(self.patch_embed.bias, 0)
        nn.init.constant_(self.final_layer.weight, 0)
        nn.init.constant_(self.final_layer.bias, 0)

    def forward(self, x, t, text_emb=None):
        # Patch embedding: (B, C, T, H, W) -> (B, N, D)
        x = self.patch_embed(x)
        x = rearrange(x, 'b c t h w -> b (t h w) c')
        x = x + self.pos_embed

        # Time embedding
        t_emb = self.time_embed(self.timestep_embedding(t, 256))

        # Apply transformer blocks with checkpointing
        from torch.utils.checkpoint import checkpoint
        for block in self.blocks:
            x = checkpoint(block, x, t_emb, text_emb, use_reentrant=False)

        # Output
        x = self.final_layer(x)

        # Unpatchify
        return self.unpatchify(x)

    def timestep_embedding(self, t, dim):
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def unpatchify(self, x):
        c = 6  # 3 channels * 2 (learn_sigma)
        pt, ph, pw = 2, self.patch_size, self.patch_size
        t = self.num_frames // 2
        h = w = self.image_size // self.patch_size

        x = x.reshape(x.shape[0], t, h, w, pt, ph, pw, c//2)
        x = rearrange(x, 'b t h w pt ph pw c -> b c (t pt) (h ph) (w pw)')
        return x


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, text_embed_dim=768):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, 
                                               kdim=text_embed_dim, vdim=text_embed_dim, batch_first=True)
        self.norm3 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size)
        )
        self.adaLN = nn.Linear(hidden_size, 6 * hidden_size)  # Updated for 3 attention layers

    def forward(self, x, t_emb, context=None):
        # Adaptive modulation
        modulation = self.adaLN(t_emb)
        shift_msa, scale_msa, shift_ca, scale_ca, shift_mlp, scale_mlp = modulation.chunk(6, dim=-1)

        # Self-attention
        x_norm = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # Cross-attention with text context (if provided)
        if context is not None:
            x_norm = self.norm2(x) * (1 + scale_ca.unsqueeze(1)) + shift_ca.unsqueeze(1)
            attn_out, _ = self.cross_attn(x_norm, context, context)
            x = x + attn_out
            norm_for_mlp = self.norm3(x)
        else:
            norm_for_mlp = self.norm2(x)

        # MLP
        x_norm = norm_for_mlp * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + self.mlp(x_norm)

        return x

class OpenVidLocalDataset(Dataset):
    """Dataset class for locally downloaded OpenVid-1M videos"""

    def __init__(self, config):
        self.config = config
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

        # Load CSV metadata
        csv_path = config['csv_path']
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}. Please download OpenVid-1M dataset first.")

        self.df = pd.read_csv(csv_path)

        # Filter to subset if specified
        if config.get('subset') and config['subset'] in csv_path:
            # If using HD subset, filter accordingly
            pass  # CSV already contains the subset

        # Limit to max_samples for testing
        if config['max_samples'] > 0:
            self.df = self.df.head(config['max_samples'])

        # Video transforms
        self.transform = transforms.Compose([
            transforms.Resize((config['resolution'], config['resolution'])),
            transforms.Lambda(lambda x: x.float() / 255.0),  # Normalize to [0, 1]
            transforms.Lambda(lambda x: (x * 2) - 1),  # Scale to [-1, 1]
        ])

        print(f"Loaded {len(self.df)} video-caption pairs from {csv_path}")

    def __len__(self):
        # For overfitting with 1 video: return large number so dataloader repeats the same video
        if len(self.df) == 1:
            return 1000  # Return same video 1000 times per epoch
        return len(self.df)

    def __getitem__(self, idx):
        # For overfitting: always use the first (and only) video
        if len(self.df) == 1:
            row = self.df.iloc[0]  # Always use first video for overfitting
        else:
            row = self.df.iloc[idx]

        caption = row['caption']
        video_name = row['video']  # Video filename from CSV
        video_path = os.path.join(self.config['video_folder'], video_name)

        # Load and process video
        video = self._load_video(video_path)

        # Tokenize caption
        text_tokens = self.tokenizer(caption, padding="max_length", max_length=77,
                                   truncation=True, return_tensors="pt")

        return {
            'video': video,
            'text_tokens': text_tokens['input_ids'].squeeze(0),
            'text_mask': text_tokens['attention_mask'].squeeze(0),
            'caption': caption
        }

    def _load_video(self, video_path):
        """Load video using torchvision.io for better performance"""
        try:
            if not os.path.exists(video_path):
                print(f"Warning: Video file not found: {video_path}")
                return self._create_dummy_video()

            # Try torchvision first, fallback to OpenCV if needed
            try:
                # Read video using torchvision
                video, _, _ = io.read_video(video_path, pts_unit='sec')
                video = video.numpy()  # Shape: (T, H, W, C)
                print(f"Loaded with torchvision: {video.shape}")
            except Exception as e:
                print(f"torchvision failed, trying OpenCV: {e}")
                # Fallback to OpenCV
                cap = cv2.VideoCapture(video_path)
                frames = []

                # Read all frames
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # Convert BGR to RGB and resize
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (self.config['resolution'], self.config['resolution']))
                    frames.append(frame)

                cap.release()

                if not frames:
                    print(f"No frames could be read from {video_path}")
                    return self._create_dummy_video()

                video = np.array(frames)  # (T, H, W, C)
                print(f"Loaded with OpenCV: {video.shape}")

            # Sample or pad frames
            num_frames = self.config['num_frames']
            total_frames = video.shape[0]

            if total_frames >= num_frames:
                # Sample evenly spaced frames
                indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
                video = video[indices]
            else:
                # Pad with last frame if video is too short
                if total_frames > 0:
                    last_frame = video[-1:]
                    padding_needed = num_frames - total_frames
                    padding = np.repeat(last_frame, padding_needed, axis=0)
                    video = np.concatenate([video, padding], axis=0)
                else:
                    # Empty video, create dummy
                    return self._create_dummy_video()

            # Apply transforms - but handle properly
            video_tensor = torch.from_numpy(video)  # (T, H, W, C)
            print(f"Before transforms: {video_tensor.shape}")

            # Apply transforms frame by frame if needed
            if self.transform:
                # Transform expects individual frames, so we process each frame
                transformed_frames = []
                for i in range(video_tensor.shape[0]):
                    frame = video_tensor[i]  # (H, W, C)
                    transformed_frame = self.transform(frame)
                    transformed_frames.append(transformed_frame)
                video_tensor = torch.stack(transformed_frames, dim=0)  # (T, ...)

            print(f"After transforms: {video_tensor.shape}")

            # Ensure correct output shape (C, T, H, W)
            if video_tensor.dim() == 4:
                if video_tensor.shape[-1] == 3:  # (T, H, W, C) -> (C, T, H, W)
                    video_tensor = video_tensor.permute(3, 0, 1, 2)
                elif video_tensor.shape[1] == 3:  # Already (T, C, H, W) -> (C, T, H, W)  
                    video_tensor = video_tensor.permute(1, 0, 2, 3)

            print(f"Final shape: {video_tensor.shape}")
            return video_tensor

        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            return self._create_dummy_video()

    def _create_dummy_video(self):
        """Create a dummy video tensor for failed video loads"""
        # Create dummy video directly in the expected shape (C, T, H, W)
        # This bypasses the transforms that were causing shape issues
        channels = 3
        time = self.config['num_frames']
        height = self.config['resolution']
        width = self.config['resolution']

        # Create video with shape (C, T, H, W) - normalized to [-1, 1]
        dummy_video = torch.zeros((channels, time, height, width), dtype=torch.float32)

        return dummy_video


class SimpleDataset(Dataset):
    """Fallback dataset using HuggingFace streaming (with dummy videos)"""

    def __init__(self, config):
        self.config = config
        # Convert streaming dataset to list for proper indexing
        streaming_dataset = load_dataset(config['name'], split="train", streaming=True).take(config['max_samples'])
        self.dataset = list(streaming_dataset)
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        # Process video - use the 'video' field directly
        video_data = sample.get('video', None)
        if video_data is None:
            print(f"Warning: No video data found in sample {idx}")
            video = self._create_dummy_video()
        else:
            video = self.process_video(video_data)

        text = sample.get('caption', sample.get('text', ''))
        text_tokens = self.tokenizer(text, padding="max_length", max_length=77, truncation=True, return_tensors="pt")

        return {
            'video': video,
            'text_tokens': text_tokens['input_ids'].squeeze(0),
            'text_mask': text_tokens['attention_mask'].squeeze(0),
            'caption': text
        }

    def process_video(self, video_data):
        # Simplified version - just create dummy videos
        return self._create_dummy_video()

    def _create_dummy_video(self):
        """Create a dummy video tensor for failed video loads"""
        # Create dummy video directly in the expected shape (C, T, H, W)
        channels = 3
        time = self.config['num_frames']
        height = self.config['resolution']
        width = self.config['resolution']

        # Create video with shape (C, T, H, W) - normalized to [-1, 1]
        dummy_video = torch.zeros((channels, time, height, width), dtype=torch.float32)

        return dummy_video


def create_dataset(config):
    """Factory function to create the appropriate dataset"""
    if config.get('use_local_videos', False):
        return OpenVidLocalDataset(config)
    else:
        return SimpleDataset(config)

# Removed SimpleDiffusion class - will use DDPMScheduler from diffusers

def safe_collate(batch):
    return torch.utils.data.dataloader.default_collate(batch)

def train():
    # Initialize
    accelerator = Accelerator(
        mixed_precision=CONFIG['training']['mixed_precision'],
        gradient_accumulation_steps=CONFIG['training']['gradient_accumulation_steps']
    )

    # Model and components
    model = SimpleVideoDiT(CONFIG['model'])
    ema = EMA(model, beta=CONFIG['training']['ema_decay'])
    
    # Use Diffusers scheduler instead of custom SimpleDiffusion
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        prediction_type="epsilon"  # predict noise
    )
    
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder.requires_grad_(False)

    # Dataset and optimizer
    dataset = create_dataset(CONFIG['dataset'])
    dataloader = DataLoader(dataset, batch_size=CONFIG['training']['batch_size'],
                          shuffle=True, num_workers=2, collate_fn=safe_collate)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['training']['learning_rate'])
    
    # Add learning rate scheduler from diffusers
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=len(dataloader) * CONFIG['training']['num_epochs']
    )

    # Prepare for distributed training
    model, text_encoder, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, text_encoder, optimizer, dataloader, lr_scheduler
    )
    ema.to(accelerator.device)

    # Training loop
    global_step = 0
    for epoch in range(CONFIG['training']['num_epochs']):
        model.train()
        progress_bar = tqdm(dataloader, disable=not accelerator.is_local_main_process)

        for batch_idx, batch in enumerate(progress_bar):
            videos = batch['video']
            text_tokens = batch['text_tokens']
            text_mask = batch['text_mask']

            # Get text embeddings
            with torch.no_grad():
                text_outputs = text_encoder(input_ids=text_tokens, attention_mask=text_mask)
                text_emb = text_outputs.last_hidden_state

            # Sample timesteps
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, 
                                    (videos.shape[0],), device=videos.device)

            # Add noise using diffusers scheduler
            noise = torch.randn_like(videos)
            noisy_videos = noise_scheduler.add_noise(videos, noise, timesteps)

            # Forward pass
            with accelerator.autocast():
                # Model predicts the noise
                noise_pred = model(noisy_videos, timesteps, text_emb)
                loss = nn.functional.mse_loss(noise_pred, noise)

            # Backward pass
            accelerator.backward(loss / accelerator.gradient_accumulation_steps)

            # Step optimizer
            if (batch_idx + 1) % accelerator.gradient_accumulation_steps == 0:
                if CONFIG['training']['gradient_clip'] > 0:
                    accelerator.clip_grad_norm_(model.parameters(), CONFIG['training']['gradient_clip'])
                optimizer.step()
                lr_scheduler.step()  # Update learning rate
                optimizer.zero_grad()
                ema.update()

                if global_step % 10 == 0 and accelerator.is_main_process:
                    current_lr = lr_scheduler.get_last_lr()[0]
                    print(f"Step {global_step}: Loss: {loss.item():.4f}, LR: {current_lr:.2e}")
                    # For overfitting: also print running loss average
                    if hasattr(accelerator, 'running_loss'):
                        accelerator.running_loss = accelerator.running_loss * 0.9 + loss.item() * 0.1
                    else:
                        accelerator.running_loss = loss.item()
                    if global_step % 100 == 0:
                        print(f"Running Loss Average: {accelerator.running_loss:.4f}")

                global_step += 1

            progress_bar.set_description(f"Epoch {epoch} | Loss: {loss.item():.4f}")

    print("Training completed!")

def generate_video(model, text_encoder, tokenizer, noise_scheduler, prompt, num_frames=8, height=192, width=192, num_inference_steps=50, device="cuda"):
    """Generate video using trained model and diffusers scheduler"""
    model.eval()
    
    # Encode text prompt
    text_inputs = tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    
    with torch.no_grad():
        text_outputs = text_encoder(**text_inputs)
        text_emb = text_outputs.last_hidden_state
    
    # Initialize random noise
    shape = (1, 3, num_frames, height, width)
    latents = torch.randn(shape, device=device)
    
    # Set scheduler timesteps
    noise_scheduler.set_timesteps(num_inference_steps, device=device)
    
    # Denoising loop
    for t in tqdm(noise_scheduler.timesteps, desc="Generating video"):
        # Prepare timestep
        timestep = t.expand(latents.shape[0]).to(device)
        
        # Predict noise
        with torch.no_grad():
            noise_pred = model(latents, timestep, text_emb)
        
        # Compute previous timestep
        latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
    
    # Convert to numpy and denormalize
    video = latents.cpu().permute(0, 2, 3, 4, 1).numpy()  # (B, T, H, W, C)
    video = (video + 1) * 127.5  # Denormalize from [-1, 1] to [0, 255]
    video = np.clip(video, 0, 255).astype(np.uint8)
    
    return video[0]  # Return first (and only) video

if __name__ == "__main__":
    train()