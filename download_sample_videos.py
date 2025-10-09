"""
Download some sample videos from open sources that don't require authentication.
"""

import requests
from pathlib import Path
import subprocess

print("🎬 Downloading Sample Videos from Open Sources")
print("=" * 80)

output_dir = Path("/root/blueberry-video/sample_videos")
output_dir.mkdir(parents=True, exist_ok=True)

# Sample videos from various open sources
sample_videos = [
    {
        "name": "sample_beach.mp4",
        "url": "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4",
        "description": "Big Buck Bunny sample (360p, 10s)"
    },
    {
        "name": "sample_nature.mp4", 
        "url": "https://test-videos.co.uk/vids/jellyfish/mp4/h264/360/Jellyfish_360_10s_1MB.mp4",
        "description": "Jellyfish sample (360p, 10s)"
    },
    {
        "name": "sample_sintel.mp4",
        "url": "https://test-videos.co.uk/vids/sintel/mp4/h264/360/Sintel_360_10s_1MB.mp4",
        "description": "Sintel sample (360p, 10s)"
    },
    {
        "name": "sample_elephants.mp4",
        "url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4",
        "description": "Google sample video - Elephants Dream"
    },
    {
        "name": "sample_countdown.mp4",
        "url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4",
        "description": "Google sample video - For Bigger Blazes"
    },
]

print(f"\n📁 Download directory: {output_dir}")

downloaded_files = []

for i, video in enumerate(sample_videos, 1):
    print(f"\n📹 Downloading {i}/{len(sample_videos)}: {video['description']}")
    output_file = output_dir / video['name']
    
    try:
        # Download with requests
        response = requests.get(video['url'], timeout=30, stream=True)
        response.raise_for_status()
        
        with open(output_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"   ✅ Downloaded: {output_file.name} ({size_mb:.2f} MB)")
        downloaded_files.append(output_file)
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")

# Try Pexels videos (public domain)
print(f"\n\n📹 Attempting to download from Pexels (public domain)...")
pexels_videos = [
    "https://player.vimeo.com/external/373680801.sd.mp4?s=05f6c85d1d14ab2976fde23cd09064a8e83eb5b7&profile_id=164&oauth2_token_id=57447761",
]

for i, url in enumerate(pexels_videos, 1):
    print(f"\n   Video {i}...")
    output_file = output_dir / f"pexels_sample_{i}.mp4"
    
    try:
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()
        
        with open(output_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"   ✅ Downloaded: {output_file.name} ({size_mb:.2f} MB)")
        downloaded_files.append(output_file)
        
    except Exception as e:
        print(f"   ❌ Failed: {str(e)[:100]}")

print("\n" + "=" * 80)
print(f"📊 SUMMARY")
print("=" * 80)
print(f"\n✅ Successfully downloaded {len(downloaded_files)} videos")
print(f"📁 Location: {output_dir}\n")

if downloaded_files:
    print("📹 Downloaded files:")
    for file in downloaded_files:
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"   - {file.name} ({size_mb:.2f} MB)")
    
    print("\n🎉 Success! You now have sample videos to work with!")
else:
    print("⚠️  No videos were downloaded. Try providing Kaggle credentials.")

print("\n" + "=" * 80)

