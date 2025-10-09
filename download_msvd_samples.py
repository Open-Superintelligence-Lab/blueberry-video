"""
Script to download actual video clips from MSVD dataset using YouTube.
The MSVD video IDs follow the format: [YouTubeID]_[start_frame]_[end_frame]
"""

from datasets import load_dataset
import subprocess
import os
from pathlib import Path
import cv2
import json

def extract_youtube_id_and_frames(video_id):
    """Extract YouTube ID and frame numbers from MSVD video ID."""
    parts = video_id.rsplit('_', 2)
    if len(parts) == 3:
        youtube_id = parts[0]
        start_frame = int(parts[1])
        end_frame = int(parts[2])
        return youtube_id, start_frame, end_frame
    return None, None, None

def download_and_clip_video(sample, output_dir, max_duration=10):
    """Download YouTube video and extract the specific clip."""
    
    video_id = sample['video_id']
    youtube_id, start_frame, end_frame = extract_youtube_id_and_frames(video_id)
    
    if not youtube_id:
        print(f"⚠️  Could not parse video ID: {video_id}")
        return None
    
    print(f"\n📹 Processing: {video_id}")
    print(f"   YouTube ID: {youtube_id}")
    print(f"   Frames: {start_frame} to {end_frame}")
    
    # Calculate time range (assuming 30 fps)
    fps = 30
    start_time = start_frame / fps
    duration = (end_frame - start_frame) / fps
    
    print(f"   Time: {start_time:.2f}s to {start_time + duration:.2f}s (duration: {duration:.2f}s)")
    
    output_file = output_dir / f"{video_id}.mp4"
    temp_file = output_dir / f"temp_{youtube_id}.mp4"
    
    # Skip if already downloaded
    if output_file.exists():
        print(f"   ✅ Already exists: {output_file}")
        return str(output_file)
    
    try:
        # Download the full video first (low quality for quick download)
        youtube_url = f"https://www.youtube.com/watch?v={youtube_id}"
        print(f"   ⬇️  Downloading from YouTube...")
        
        download_cmd = [
            'yt-dlp',
            '-f', 'worst',  # Download lowest quality for speed
            '-o', str(temp_file),
            '--no-playlist',
            '--quiet',
            youtube_url
        ]
        
        result = subprocess.run(download_cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0 or not temp_file.exists():
            print(f"   ❌ Download failed: {result.stderr}")
            return None
        
        # Extract the specific clip using ffmpeg
        print(f"   ✂️  Extracting clip...")
        clip_cmd = [
            'ffmpeg',
            '-i', str(temp_file),
            '-ss', str(start_time),
            '-t', str(duration),
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-y',
            '-loglevel', 'error',
            str(output_file)
        ]
        
        subprocess.run(clip_cmd, check=True, timeout=30)
        
        # Clean up temp file
        if temp_file.exists():
            temp_file.unlink()
        
        print(f"   ✅ Saved: {output_file}")
        return str(output_file)
        
    except subprocess.TimeoutExpired:
        print(f"   ⏱️  Timeout - skipping this video")
        if temp_file.exists():
            temp_file.unlink()
        return None
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        if temp_file.exists():
            temp_file.unlink()
        return None

def main():
    print("Loading MSVD dataset...")
    dataset = load_dataset("friedrichor/MSVD")
    
    # Create output directory
    output_dir = Path("/root/blueberry-video/msvd_videos")
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "=" * 80)
    print("📺 Downloading Sample Videos from MSVD Dataset")
    print("=" * 80)
    
    train_data = dataset['train']
    num_samples = 3  # Download just 3 samples for demonstration
    
    downloaded_videos = []
    
    for i in range(min(num_samples, len(train_data))):
        sample = train_data[i]
        
        print(f"\n🎬 Sample {i+1}/{num_samples}")
        print("-" * 80)
        print(f"📝 Captions (first 3):")
        for j, caption in enumerate(sample['caption'][:3], 1):
            print(f"   {j}. {caption}")
        
        video_path = download_and_clip_video(sample, output_dir)
        
        if video_path:
            downloaded_videos.append({
                'video_id': sample['video_id'],
                'video_path': video_path,
                'captions': sample['caption'][:5],
                'total_captions': len(sample['caption'])
            })
    
    # Save metadata
    metadata_file = output_dir / "downloaded_videos.json"
    with open(metadata_file, 'w') as f:
        json.dump(downloaded_videos, f, indent=2)
    
    print("\n\n" + "=" * 80)
    print(f"✅ Downloaded {len(downloaded_videos)} videos")
    print(f"📁 Videos saved to: {output_dir}")
    print(f"📄 Metadata saved to: {metadata_file}")
    print("=" * 80)
    
    # List downloaded files
    if downloaded_videos:
        print("\n📹 Downloaded Videos:")
        for video in downloaded_videos:
            print(f"   - {video['video_path']}")
            print(f"     Caption: {video['captions'][0]}")

if __name__ == "__main__":
    main()

