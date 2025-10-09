# Blueberry Video - Project Summary

## What Was Done

Successfully transformed the blueberry-llm repository into a video generation research repository, maintaining the experimental structure while replacing LLM logic with video diffusion.

## Repository Structure

```
blueberry-video/
├── models/                      # Video Generation Architectures
│   ├── cogvideox_transformer.py      # CogVideoX 3D Transformer
│   ├── hunyuan_video_transformer.py  # Hunyuan Video Transformer
│   ├── attention.py                  # Attention mechanisms
│   ├── embeddings.py                 # Position & time embeddings
│   └── ...                           # Supporting components
│
├── configs/                     # Configuration Files
│   └── video/
│       └── train/
│           ├── cogvideox.yaml       # CogVideoX training config
│           └── hunyuan.yaml         # Hunyuan training config
│
├── data/                        # Data Loading
│   ├── video_dataset.py            # Video dataset loader
│   └── videos/                     # Dataset location
│       ├── videos/                 # .mp4 files go here
│       └── metadata.json           # Video descriptions
│
├── experiments/                 # Research Experiments
│   └── exp_template/               # Template for new experiments
│       ├── README.md               # Experiment documentation
│       └── config.yaml             # Experiment config
│
├── utils/                       # Utilities
│   ├── config.py                   # Config loading
│   └── training.py                 # Training utilities
│
├── train_video.py               # Main Training Script
├── train_video_demo.py          # Demo with Dummy Data
├── create_sample_videos.py      # Generate Test Videos
│
├── README.md                    # Main Documentation
├── QUICKSTART.md                # Quick Start Guide
├── CONTRIBUTING.md              # Contribution Guidelines
└── requirements.txt             # Dependencies

```

## Key Features

### ✅ Two State-of-the-Art Architectures

1. **CogVideoX** - 3D Transformer with temporal attention
2. **Hunyuan Video** - Advanced video generation model

Both architectures copied directly from diffusers library.

### ✅ Complete Training Pipeline

- `train_video.py` - Production training script
- `train_video_demo.py` - Test with dummy data (no videos needed)
- Supports both architectures via config selection
- Flexible optimizer and scheduler configuration

### ✅ Data Pipeline

- `VideoDataset` - Loads .mp4 files with text prompts
- `DummyVideoDataset` - For testing without real data
- Automatic frame sampling and resizing
- Simple metadata format (JSON)

### ✅ Experiment Structure

Following the same philosophy as blueberry-llm:

- Template-based experiment creation
- Clear documentation requirements
- Hypothesis-driven research
- Results tracking

### ✅ Documentation

- **README.md** - Complete project overview
- **QUICKSTART.md** - Get started in 5 minutes
- **CONTRIBUTING.md** - How to contribute experiments
- **PROJECT_SUMMARY.md** - This file

## How to Use

### Quick Test (No Data Required)

```bash
pip install -r requirements.txt
python train_video_demo.py
```

This runs a demo training loop with dummy data to verify everything works.

### Train with Real Data

1. **Prepare dataset:**
   ```
   data/videos/
     videos/
       video_001.mp4
       video_002.mp4
     metadata.json
   ```

2. **Choose architecture:**
   ```bash
   # CogVideoX
   python train_video.py --config configs/video/train/cogvideox.yaml
   
   # Or Hunyuan Video
   python train_video.py --config configs/video/train/hunyuan.yaml
   ```

### Create an Experiment

1. **Copy template:**
   ```bash
   cp -r experiments/exp_template experiments/exp1_my_research
   ```

2. **Edit config:**
   ```bash
   vim experiments/exp1_my_research/config.yaml
   ```

3. **Document research question:**
   ```bash
   vim experiments/exp1_my_research/README.md
   ```

4. **Run experiment:**
   ```bash
   python train_video.py --config experiments/exp1_my_research/config.yaml
   ```

5. **Document findings and submit PR**

## What Was Removed

From the original blueberry-llm repository:

- ❌ LLM models (DeepSeek, Qwen3)
- ❌ MoE architecture code
- ❌ LLM training scripts
- ❌ LLM-specific experiments
- ❌ LLM optimizers and utilities

## What Was Added

- ✅ CogVideoX transformer architecture
- ✅ Hunyuan Video transformer architecture  
- ✅ Video dataset loader
- ✅ Video-specific training script
- ✅ Video experiment configs
- ✅ Demo training script
- ✅ Sample video generator
- ✅ Complete documentation

## Maintained Philosophy

The core philosophy from blueberry-llm was preserved:

- **Freedom to explore** - No prescribed research directions
- **Scientific rigor** - Hypothesis-driven experiments
- **Open research** - All findings are public
- **Community-driven** - Collaborative learning environment

## Next Steps for Users

1. **Test the setup:**
   ```bash
   python train_video_demo.py
   ```

2. **Create sample videos (optional):**
   ```bash
   python create_sample_videos.py
   ```

3. **Prepare your dataset or use dummy data**

4. **Start your first experiment!**

## Technical Details

### Model Architecture

Both architectures are 3D transformers that:
- Process video as sequences of frames
- Use temporal attention for coherence
- Support text conditioning
- Generate via diffusion process

### Training Process

1. Load video frames
2. Encode to latent space (implicit)
3. Add noise (diffusion forward process)
4. Train model to predict noise
5. MSE loss between predicted and actual noise

### Configuration

All experiments are configured via YAML:
- Model hyperparameters
- Training settings
- Data parameters
- Optimizer/scheduler settings

## Dependencies

Main requirements:
- PyTorch 2.0+
- OpenCV (video processing)
- diffusers (model architectures)
- transformers (text encoding)

See `requirements.txt` for complete list.

## Research Ideas

Some directions to explore:

1. **Architecture variants** - Modify attention mechanisms
2. **Temporal modeling** - Different frame sampling strategies
3. **Resolution scaling** - Multi-resolution training
4. **Loss functions** - Custom loss terms
5. **Efficiency** - Faster training/inference
6. **Quality metrics** - Better evaluation methods

## Summary

This repository is now a fully functional video generation research platform that:

✅ Supports two SOTA architectures  
✅ Has complete training pipeline  
✅ Includes data loading utilities  
✅ Provides experiment framework  
✅ Contains comprehensive documentation  
✅ Enables quick testing with dummy data  
✅ Follows research-first philosophy  

**Status: Ready for research! 🚀**

## Contact & Support

- Open issues for questions
- Submit PRs for experiments
- Share findings with community
- Help others learn

---

**Happy researching!**

*Open Superintelligence Lab*

