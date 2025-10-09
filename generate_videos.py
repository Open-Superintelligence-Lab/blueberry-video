"""
Video Generation Script - Inference with Trained Model
Generate videos from the overfitted model
"""

import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

from models import CogVideoXTransformer3DModel


def denoise_video(model, device, num_frames=8, resolution=64, num_inference_steps=50):
    """
    Generate a video by denoising random noise
    
    Args:
        model: Trained transformer model
        device: CUDA or CPU
        num_frames: Number of frames to generate
        resolution: Resolution of the video (height and width)
        num_inference_steps: Number of denoising steps
    
    Returns:
        Generated video tensor [num_frames, channels, height, width]
    """
    model.eval()
    
    # Start with random noise [T, C, H, W]
    latents = torch.randn(1, num_frames, 3, resolution, resolution, device=device)
    
    # Create dummy text embeddings (same as training)
    batch_size = 1
    text_seq_len = 77
    encoder_hidden_states = torch.zeros(batch_size, text_seq_len, 768, device=device)
    
    # Denoising loop
    print(f"Generating video with {num_inference_steps} denoising steps...")
    
    with torch.no_grad():
        for step in tqdm(range(num_inference_steps), desc="Denoising"):
            # Calculate timestep
            timestep = torch.tensor([1000 - (step * 1000 // num_inference_steps)], device=device)
            
            # Predict noise
            model_output = model(
                hidden_states=latents,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=True
            )
            
            predicted_noise = model_output.sample
            
            # Simple denoising: remove predicted noise
            # Using a simple DDPM-like denoising
            alpha = 1 - (step / num_inference_steps)
            latents = latents - alpha * 0.1 * predicted_noise
    
    return latents[0]  # Return [T, C, H, W]


def save_video(video_tensor, output_path, fps=8):
    """
    Save video tensor as MP4 file
    
    Args:
        video_tensor: Tensor [T, C, H, W] in range [-1, 1]
        output_path: Path to save the video
        fps: Frames per second
    """
    # Convert from [-1, 1] to [0, 255]
    video_tensor = ((video_tensor + 1.0) * 127.5).clamp(0, 255)
    video_np = video_tensor.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    
    # Convert RGB to BGR for OpenCV
    video_np = video_np[:, :, :, ::-1]
    
    # Get video properties
    num_frames, height, width, channels = video_np.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Write frames
    for frame in video_np:
        out.write(frame)
    
    out.release()
    print(f"✅ Saved video to: {output_path}")


def load_checkpoint(checkpoint_path, model, device):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', 0.0)
    print(f"Loaded checkpoint from epoch {epoch+1} with loss {loss:.6f}")
    return model


def main():
    print("=" * 70)
    print("Video Generation - Inference with Overfitted Model")
    print("=" * 70)
    
    # Configuration (must match training)
    resolution = 64
    num_frames = 8
    num_videos = 5  # Generate 5 videos
    num_inference_steps = 50  # Number of denoising steps
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Initialize model (same config as training)
    print("\nInitializing model...")
    patch_size = 2
    temporal_compression_ratio = 4
    sample_width = resolution // patch_size
    sample_height = resolution // patch_size
    sample_frames = ((num_frames - 1) * temporal_compression_ratio + 1)
    text_seq_len = 77
    
    model = CogVideoXTransformer3DModel(
        num_attention_heads=2,
        attention_head_dim=32,
        in_channels=3,
        out_channels=3,
        num_layers=4,
        sample_width=sample_width,
        sample_height=sample_height,
        sample_frames=sample_frames,
        max_text_seq_length=text_seq_len,
        text_embed_dim=768,
        temporal_compression_ratio=temporal_compression_ratio,
    )
    model = model.to(device)
    
    # Load checkpoint
    checkpoint_path = Path("checkpoints/overfit/best_model.pt")
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    model = load_checkpoint(checkpoint_path, model, device)
    
    # Create output directory
    output_dir = Path("generated_videos")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nGenerating {num_videos} videos...")
    print(f"  Resolution: {resolution}x{resolution}")
    print(f"  Frames: {num_frames}")
    print(f"  Denoising steps: {num_inference_steps}")
    print(f"  Output directory: {output_dir}\n")
    
    # Generate videos
    for i in range(num_videos):
        print(f"\n--- Generating Video {i+1}/{num_videos} ---")
        
        # Generate video
        video = denoise_video(
            model=model,
            device=device,
            num_frames=num_frames,
            resolution=resolution,
            num_inference_steps=num_inference_steps
        )
        
        # Save video
        output_path = output_dir / f"generated_video_{i+1:03d}.mp4"
        save_video(video, output_path, fps=8)
    
    print("\n" + "=" * 70)
    print(f"✨ Generation complete!")
    print(f"Generated {num_videos} videos in: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()

