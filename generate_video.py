"""
Simple Video Generation Script
Generate videos using a trained model
"""

import torch
import numpy as np
from pathlib import Path
import argparse
import cv2
from tqdm import tqdm

from models import CogVideoXTransformer3DModel


def denoise_video(model, noise, text_embeds, num_steps=50, device='cuda'):
    """
    Simple denoising process to generate video from noise
    
    Args:
        model: The trained transformer model
        noise: Initial noise tensor [batch, frames, channels, height, width]
        text_embeds: Text conditioning embeddings
        num_steps: Number of denoising steps
        device: Device to run on
    """
    model.eval()
    
    # Start with pure noise
    latents = noise.clone()
    
    # Simple linear schedule for timesteps
    timesteps = torch.linspace(999, 0, num_steps, dtype=torch.long, device=device)
    
    with torch.no_grad():
        for i, t in enumerate(tqdm(timesteps, desc="Generating")):
            # Predict noise
            timestep = t.unsqueeze(0).repeat(latents.shape[0])
            
            model_output = model(
                hidden_states=latents,
                timestep=timestep,
                encoder_hidden_states=text_embeds,
                return_dict=True
            )
            
            predicted_noise = model_output.sample
            
            # Simple denoising step (simplified DDPM)
            alpha = 1 - (t / 1000.0)
            latents = latents - (1 - alpha) * predicted_noise
    
    return latents


def latents_to_video(latents, output_path, fps=8):
    """
    Convert latents to video file
    Note: In real use, you'd decode latents through a VAE first
    For demo purposes, we'll just normalize and save
    
    Args:
        latents: [batch, frames, channels, height, width]
        output_path: Where to save the video
        fps: Frames per second
    """
    # Take first batch item
    latents = latents[0]  # [frames, channels, height, width]
    
    # Permute to [frames, height, width, channels]
    latents = latents.permute(0, 2, 3, 1).cpu().numpy()
    
    # Normalize to [0, 255]
    # Take first 3 channels if more than 3
    if latents.shape[-1] > 3:
        latents = latents[..., :3]
    
    latents = (latents - latents.min()) / (latents.max() - latents.min() + 1e-8)
    latents = (latents * 255).astype(np.uint8)
    
    # Get dimensions
    num_frames, height, width, channels = latents.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Write frames
    for frame in latents:
        # Convert RGB to BGR for OpenCV
        if channels == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
    
    out.release()
    print(f"✅ Video saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate videos with trained model")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default="A beautiful video", help="Text prompt")
    parser.add_argument("--output", type=str, default="generated_video.mp4", help="Output video path")
    parser.add_argument("--num_frames", type=int, default=8, help="Number of frames to generate")
    parser.add_argument("--resolution", type=int, default=64, help="Video resolution")
    parser.add_argument("--steps", type=int, default=50, help="Number of denoising steps")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model (small version for demo)
    print("\nInitializing model...")
    text_seq_len = 10
    model = CogVideoXTransformer3DModel(
        num_attention_heads=4,
        attention_head_dim=32,
        in_channels=16,
        out_channels=16,
        num_layers=2,
        sample_width=args.resolution // 2,
        sample_height=args.resolution // 2,
        sample_frames=(args.num_frames - 1) * 4 + 1,
        max_text_seq_length=text_seq_len,
    )
    
    # Load checkpoint if provided
    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"Loading checkpoint from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("⚠️  No checkpoint provided, using randomly initialized model")
        print("   (For real generation, train a model first and provide --checkpoint)")
    
    model = model.to(device)
    model.eval()
    
    print(f"\nPrompt: {args.prompt}")
    print(f"Generating {args.num_frames} frames at {args.resolution}x{args.resolution}...")
    
    # Create random text embeddings (in real use, encode the prompt with a text encoder)
    text_embeds = torch.randn(1, text_seq_len, 4096, device=device)
    
    # Start with random noise
    noise = torch.randn(
        1,  # batch size
        args.num_frames,
        16,  # latent channels
        args.resolution,
        args.resolution,
        device=device
    )
    
    # Generate video through denoising
    generated_latents = denoise_video(model, noise, text_embeds, num_steps=args.steps, device=device)
    
    # Convert to video file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    latents_to_video(generated_latents, output_path, fps=8)
    
    print(f"\n🎉 Generation complete!")
    print(f"Note: This is a demonstration. For real video generation:")
    print(f"  1. Train the model on your dataset")
    print(f"  2. Use a proper text encoder (e.g., T5) for prompts")
    print(f"  3. Use a VAE decoder to convert latents to RGB video")


if __name__ == "__main__":
    main()

