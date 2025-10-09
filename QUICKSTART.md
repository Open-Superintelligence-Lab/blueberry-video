# Quick Start Guide

Get started with video generation research in 5 minutes!

## Installation

```bash
# Clone the repository
git clone https://github.com/Open-Superintelligence-Lab/blueberry-video.git
cd blueberry-video

# Install dependencies
pip install -r requirements.txt
```

## Test the Pipeline (No Data Required!)

Run the demo script to verify everything works:

```bash
python train_video_demo.py
```

This will:
- Initialize a small CogVideoX model
- Create dummy data
- Run 3 epochs of training
- Complete in ~1-2 minutes

Expected output:
```
============================================================
Blueberry Video - Demo Training
============================================================

Using device: cuda
Initializing CogVideoX model (small version for demo)...
Model parameters: 1,234,567

Starting demo training for 3 epochs...

Epoch 0: Average Loss = 2.1234
Epoch 1: Average Loss = 1.8765
Epoch 2: Average Loss = 1.5432

Demo completed successfully!
```

## Train with Real Data

### Step 1: Prepare Your Dataset

Create this structure:

```
data/videos/
  videos/
    video_001.mp4
    video_002.mp4
    ...
  metadata.json
```

Example `metadata.json`:

```json
{
  "video_001": "A cat playing with a ball",
  "video_002": "Sunset over the ocean"
}
```

### Step 2: Choose Architecture

**Option A: CogVideoX (Recommended for beginners)**
```bash
python train_video.py --config configs/video/train/cogvideox.yaml
```

**Option B: Hunyuan Video (More advanced)**
```bash
python train_video.py --config configs/video/train/hunyuan.yaml
```

### Step 3: Monitor Training

Watch the loss decrease:
```
Epoch 0: Average Loss = 4.5678
Epoch 1: Average Loss = 3.2345
Epoch 2: Average Loss = 2.1234
...
```

Checkpoints are saved in `checkpoints/` by default.

## Start Your First Experiment

### 1. Copy the Template

```bash
cp -r experiments/exp_template experiments/exp1_my_first_experiment
```

### 2. Define Your Research Question

Edit `experiments/exp1_my_first_experiment/README.md`:

```markdown
## Research Question

Can reducing the number of attention heads improve training speed 
without sacrificing quality?

## Hypothesis

Using 8 heads instead of 16 should be 2x faster with minimal quality loss.
```

### 3. Configure

Edit `experiments/exp1_my_first_experiment/config.yaml`:

```yaml
# Try different attention heads
num_attention_heads: 8  # Changed from 16

# Rest of config...
```

### 4. Run

```bash
python train_video.py --config experiments/exp1_my_first_experiment/config.yaml
```

### 5. Document Findings

After training, update your experiment README with:
- Training time comparison
- Quality metrics
- Sample videos
- Conclusions

### 6. Share

```bash
git add experiments/exp1_my_first_experiment
git commit -m "Experiment: Attention head ablation study"
git push origin main
```

Then create a Pull Request!

## Common Issues

### Out of Memory

Reduce batch size or resolution in your config:

```yaml
batch_size: 1
resolution: 128  # Instead of 256
num_frames: 8    # Instead of 16
```

### No GPU

The code works on CPU too (just slower):

```bash
# It will automatically detect and use CPU
python train_video_demo.py
```

### Dependencies Issues

Try installing with specific versions:

```bash
pip install torch==2.0.0 torchvision==0.15.0
pip install -r requirements.txt
```

## Next Steps

- 📖 Read [README.md](README.md) for architecture details
- 🤝 Read [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines
- 🔬 Check `experiments/` for example research
- 💬 Open an issue to discuss ideas

## Research Ideas to Get Started

- **Attention Mechanisms**: Try different attention patterns
- **Frame Rates**: Train with different temporal resolutions
- **Architectures**: Compare CogVideoX vs Hunyuan
- **Loss Functions**: Experiment with different loss weightings
- **Data Augmentation**: Add temporal or spatial augmentations

---

**Happy researching! 🚀**

Questions? Open an issue or check the docs!

