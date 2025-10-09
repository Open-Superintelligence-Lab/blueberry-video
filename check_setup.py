"""
Simple script to verify the setup is correct
Run this after installation to check if everything is ready
"""

import sys
from pathlib import Path


def check_imports():
    """Check if all required modules can be imported"""
    print("Checking imports...")
    
    errors = []
    
    # Check core dependencies
    dependencies = [
        ("torch", "PyTorch"),
        ("cv2", "OpenCV"),
        ("yaml", "PyYAML"),
        ("numpy", "NumPy"),
        ("tqdm", "tqdm"),
    ]
    
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError as e:
            print(f"  ❌ {name} - {e}")
            errors.append(name)
    
    return len(errors) == 0, errors


def check_project_structure():
    """Check if project structure is correct"""
    print("\nChecking project structure...")
    
    required_files = [
        "models/__init__.py",
        "models/cogvideox_transformer.py",
        "models/hunyuan_video_transformer.py",
        "data/__init__.py",
        "data/video_dataset.py",
        "utils/__init__.py",
        "utils/config.py",
        "utils/training.py",
        "configs/video/train/cogvideox.yaml",
        "configs/video/train/hunyuan.yaml",
        "train_video.py",
        "train_video_demo.py",
        "README.md",
        "requirements.txt",
    ]
    
    missing = []
    
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} - Missing!")
            missing.append(file_path)
    
    return len(missing) == 0, missing


def check_data_directory():
    """Check if data directory exists"""
    print("\nChecking data directory...")
    
    data_dir = Path("data/videos")
    videos_dir = data_dir / "videos"
    metadata_file = data_dir / "metadata.json"
    
    if data_dir.exists():
        print(f"  ✅ data/videos directory exists")
    else:
        print(f"  ⚠️  data/videos directory not found (will be created)")
    
    if videos_dir.exists():
        video_files = list(videos_dir.glob("*.mp4"))
        if video_files:
            print(f"  ✅ Found {len(video_files)} video files")
        else:
            print(f"  ℹ️  No video files yet (use create_sample_videos.py)")
    else:
        print(f"  ℹ️  videos subdirectory not found (will be created)")
    
    if metadata_file.exists():
        print(f"  ✅ metadata.json exists")
    else:
        print(f"  ℹ️  metadata.json not found (optional)")
    
    return True, []


def main():
    print("=" * 60)
    print("Blueberry Video - Setup Check")
    print("=" * 60)
    print()
    
    all_good = True
    
    # Check imports
    imports_ok, missing_deps = check_imports()
    if not imports_ok:
        all_good = False
        print(f"\n⚠️  Missing dependencies: {', '.join(missing_deps)}")
        print("   Install with: pip install -r requirements.txt")
    
    # Check structure
    structure_ok, missing_files = check_project_structure()
    if not structure_ok:
        all_good = False
        print(f"\n⚠️  Missing files: {', '.join(missing_files)}")
    
    # Check data
    data_ok, _ = check_data_directory()
    
    print("\n" + "=" * 60)
    
    if all_good:
        print("✅ Setup looks good!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Test with demo: python train_video_demo.py")
        print("2. Create samples: python create_sample_videos.py")
        print("3. Start training: python train_video.py")
        print("\nHappy researching! 🚀")
    else:
        print("⚠️  Some issues found. Please fix them before continuing.")
        print("=" * 60)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

