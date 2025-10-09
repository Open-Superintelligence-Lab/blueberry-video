"""
Download the Kaggle Short Videos dataset.
Dataset: https://www.kaggle.com/datasets/mistag/short-videos
"""

import os
import subprocess
from pathlib import Path

print("🎬 Downloading Kaggle Short Videos Dataset")
print("=" * 80)

# Check for Kaggle credentials
kaggle_dir = Path.home() / ".kaggle"
kaggle_json = kaggle_dir / "kaggle.json"

if not kaggle_json.exists():
    print("\n⚠️  Kaggle API credentials not found!")
    print("=" * 80)
    print("\nTo download from Kaggle, you need to:")
    print("\n1. Go to: https://www.kaggle.com/settings")
    print("2. Scroll to 'API' section")
    print("3. Click 'Create New Token'")
    print("4. Download kaggle.json")
    print("5. Run these commands:\n")
    print("   mkdir -p ~/.kaggle")
    print("   # Upload your kaggle.json to ~/.kaggle/")
    print("   chmod 600 ~/.kaggle/kaggle.json")
    print("\n" + "=" * 80)
    print("\nAlternatively, I can create a placeholder and you can add your credentials:")
    
    # Create directory
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    
    placeholder_file = kaggle_dir / "kaggle.json.example"
    with open(placeholder_file, 'w') as f:
        f.write('{"username":"YOUR_USERNAME","key":"YOUR_API_KEY"}')
    
    print(f"\n📝 Created example file: {placeholder_file}")
    print("   Edit this file with your credentials and rename to kaggle.json")
    
else:
    print(f"\n✅ Kaggle credentials found at: {kaggle_json}")
    
    # Set permissions
    os.chmod(kaggle_json, 0o600)
    
    # Download the dataset
    print("\n📦 Downloading dataset: mistag/short-videos")
    output_dir = Path("/root/blueberry-video/kaggle_videos")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download dataset
        cmd = [
            "kaggle", "datasets", "download",
            "-d", "mistag/short-videos",
            "-p", str(output_dir),
            "--unzip"
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"\n✅ Download successful!")
            print(f"📁 Files saved to: {output_dir}")
            
            # List downloaded files
            print("\n📹 Downloaded files:")
            video_files = list(output_dir.glob("**/*"))
            for i, file in enumerate(video_files[:10], 1):
                if file.is_file():
                    size_mb = file.stat().st_size / (1024 * 1024)
                    print(f"   {i}. {file.name} ({size_mb:.2f} MB)")
            
            if len(video_files) > 10:
                print(f"   ... and {len(video_files) - 10} more files")
            
            print("\n" + "=" * 80)
            print("✅ Dataset ready to use!")
            
        else:
            print(f"\n❌ Download failed!")
            print(f"Error: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("\n⏱️  Download timed out (large dataset)")
    except Exception as e:
        print(f"\n❌ Error: {e}")

print("\n" + "=" * 80)

