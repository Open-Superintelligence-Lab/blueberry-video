"""
Simple Video Dataset for Training
Loads video files and corresponding text prompts
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
import cv2
import numpy as np


class VideoDataset(Dataset):
    """
    Simple video dataset that loads videos and prompts
    
    Expected structure:
    data_path/
        videos/
            video_001.mp4
            video_002.mp4
            ...
        metadata.json  # Contains prompts for each video
    """
    
    def __init__(self, data_path, resolution=256, num_frames=16):
        """
        Args:
            data_path: Path to dataset directory
            resolution: Resize videos to this resolution (height=width)
            num_frames: Number of frames to sample from each video
        """
        self.data_path = Path(data_path)
        self.resolution = resolution
        self.num_frames = num_frames
        
        # Load metadata
        metadata_path = self.data_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
        
        # Find all video files
        self.video_files = sorted(list((self.data_path / "videos").glob("*.mp4")))
        
        if len(self.video_files) == 0:
            print(f"Warning: No videos found in {self.data_path / 'videos'}")
    
    def __len__(self):
        return len(self.video_files)
    
    def load_video(self, video_path):
        """Load video and sample frames"""
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frame indices evenly
        if total_frames > self.num_frames:
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            # If video has fewer frames, repeat frames
            indices = np.arange(total_frames)
            indices = np.pad(indices, (0, self.num_frames - total_frames), mode='edge')
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize
                frame = cv2.resize(frame, (self.resolution, self.resolution))
                frames.append(frame)
        
        cap.release()
        
        # Convert to tensor [T, H, W, C] -> [C, T, H, W]
        frames = np.stack(frames, axis=0)
        frames = torch.from_numpy(frames).float()
        frames = frames.permute(3, 0, 1, 2)  # [C, T, H, W]
        
        # Normalize to [-1, 1]
        frames = (frames / 127.5) - 1.0
        
        return frames
    
    def __getitem__(self, idx):
        """Get video and prompt"""
        video_path = self.video_files[idx]
        video_name = video_path.stem
        
        # Load video
        try:
            latents = self.load_video(video_path)
        except Exception as e:
            print(f"Error loading {video_path}: {e}")
            # Return dummy data
            latents = torch.randn(3, self.num_frames, self.resolution, self.resolution)
        
        # Get prompt from metadata
        prompt = self.metadata.get(video_name, "")
        
        return {
            'latents': latents,
            'prompts': prompt,
            'video_name': video_name
        }


class DummyVideoDataset(Dataset):
    """
    Dummy dataset for testing without real video data
    """
    
    def __init__(self, num_samples=100, resolution=256, num_frames=16):
        self.num_samples = num_samples
        self.resolution = resolution
        self.num_frames = num_frames
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Random latents
        latents = torch.randn(3, self.num_frames, self.resolution, self.resolution)
        
        # Dummy prompt
        prompts = [
            "A cat playing with a ball",
            "Sunset over the ocean",
            "Car driving through city",
            "Person walking in forest",
            "Bird flying in the sky"
        ]
        prompt = prompts[idx % len(prompts)]
        
        return {
            'latents': latents,
            'prompts': prompt,
            'video_name': f'dummy_{idx:04d}'
        }

