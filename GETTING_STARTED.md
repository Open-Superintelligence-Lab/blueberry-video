# Getting Started with Video Training & Generation

## ✅ Quick Start (You've already done this!)

Your environment is now set up and working! Here's what we've accomplished:

1. ✅ Installed dependencies
2. ✅ Created sample training videos
3. ✅ Successfully ran demo training
4. ✅ Fixed model configuration issues

## 📚 What You Have Now

### Sample Videos Created
```
data/videos/videos/
├── sample_001_circle.mp4    - Green circle moving left to right
├── sample_002_square.mp4    - Blue square expanding from center  
└── sample_003_line.mp4      - Red line rotating around center
```

### Working Scripts
- `train_video_demo.py` - ✅ Working demo with dummy data
- `train_video.py` - Main training script (needs minor fixes for real data)
- `generate_video.py` - NEW! Video generation script
- `create_sample_videos.py` - Creates synthetic training videos

## 🚀 How to Train & Generate Videos

### Option 1: Quick Demo Training (Already Working!)

```bash
# This runs a quick 3-epoch demo with dummy data
python train_video_demo.py
```

**Output:**
```
Epoch 0: Average Loss = 1.3956
Epoch 1: Average Loss = 1.3075
Epoch 2: Average Loss = 1.2174
Demo completed successfully!
```

### Option 2: Train with Real Sample Videos

```bash
# Train on the synthetic videos we created
python train_video.py --config configs/video/train/cogvideox.yaml
```

**Note:** The main training script needs the same fixes as the demo. For now, use the demo for testing.

### Option 3: Generate a Video

```bash
# Generate a video using the model (random initialization without checkpoint)
python generate_video.py --prompt "A beautiful sunset" --output my_video.mp4

# Or with a trained checkpoint:
python generate_video.py \
  --checkpoint checkpoints/checkpoint_epoch_10.pt \
  --prompt "A cat playing" \
  --output generated.mp4 \
  --num_frames 16 \
  --resolution 128 \
  --steps 50
```

## 📊 Training Your Own Model - Step by Step

### Step 1: Prepare Your Videos

Create your dataset:
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
  "video_002": "Sunset over the ocean",
  "video_003": "Person walking through forest"
}
```

### Step 2: Configure Training

Edit `configs/video/train/cogvideox.yaml`:
```yaml
# Model settings
model_type: cogvideox
num_attention_heads: 16
attention_head_dim: 64
in_channels: 16
num_layers: 24

# Data settings  
data_path: ./data/videos
resolution: 256
num_frames: 16

# Training settings
batch_size: 1
num_epochs: 100
learning_rate: 1e-4

# Checkpointing
save_dir: ./checkpoints/cogvideox
save_every: 10
```

### Step 3: Start Training

```bash
# For a quick test with sample videos (3 synthetic videos)
python train_video_demo.py

# For full training (when train_video.py is fixed)
python train_video.py --config configs/video/train/cogvideox.yaml
```

### Step 4: Monitor Progress

Watch for decreasing loss:
```
Epoch 0: Average Loss = 1.40
Epoch 1: Average Loss = 1.31
Epoch 2: Average Loss = 1.22
...
```

Checkpoints are saved to `checkpoints/` every 10 epochs.

### Step 5: Generate Videos

```bash
python generate_video.py \
  --checkpoint checkpoints/checkpoint_epoch_50.pt \
  --prompt "Your creative prompt here" \
  --output my_generated_video.mp4 \
  --num_frames 16 \
  --resolution 128
```

## 🎯 Current Status

### ✅ What's Working
- Demo training with dummy data
- Sample video creation
- Model initialization (CogVideoX)
- Loss calculation and backpropagation
- Checkpoint saving
- Video generation script (basic)

### ⚠️  What Needs Work
- `train_video.py` needs the same fixes as demo (tensor reshaping, text encoding)
- Real text encoding (currently using random embeddings)
- VAE encoding/decoding for real videos (currently using placeholder)
- Full inference pipeline with proper denoising scheduler

## 🔧 Key Technical Details

### Model Input Format
```python
# Latents shape: [batch, frames, channels, height, width]
latents = torch.randn(2, 8, 16, 64, 64)  # Example

# Text embeddings: [batch, seq_length, embed_dim]
text_embeds = torch.randn(2, 10, 4096)

# Model forward pass
output = model(
    hidden_states=latents,
    timestep=timesteps,
    encoder_hidden_states=text_embeds,
    return_dict=True
)
```

### Important Dimension Calculations
```python
# For positional embeddings to work correctly:
sample_width = resolution // patch_size
sample_height = resolution // patch_size
sample_frames = (num_frames - 1) * temporal_compression_ratio + 1

# Example: 64x64 resolution, 8 frames
sample_width = 64 // 2 = 32
sample_height = 64 // 2 = 32
sample_frames = (8 - 1) * 4 + 1 = 29
```

### Text Sequence Length
```python
# Must match in both model init and forward pass
max_text_seq_length = 10  # or 226 for full model
```

## 🐛 Troubleshooting

### Error: "dimension mismatch"
- Check that input latents match model's expected dimensions
- Verify `sample_width`, `sample_height`, `sample_frames` are correct
- Ensure `max_text_seq_length` matches text embeddings

### Error: "requires grad" 
- Make sure model forward pass is used (not just tensor operations)
- Check that loss is computed from model output

### Out of Memory
```yaml
# Reduce in config:
batch_size: 1
resolution: 64  # Instead of 256
num_frames: 8   # Instead of 16
```

## 📁 Project Structure

```
blueberry-video/
├── models/                   # Model architectures
│   ├── cogvideox_transformer.py
│   └── hunyuan_video_transformer.py
├── data/
│   ├── video_dataset.py     # Dataset loader
│   └── videos/              # Your videos go here
├── configs/                 # Training configs
│   └── video/train/
│       ├── cogvideox.yaml
│       └── hunyuan.yaml
├── train_video_demo.py      # ✅ Working demo
├── train_video.py           # Main training (needs fixes)
├── generate_video.py        # Video generation
└── create_sample_videos.py  # Create synthetic videos
```

## 🎓 Next Steps

1. **Quick Experiment:**
   ```bash
   # Run demo a few times to see it work
   python train_video_demo.py
   ```

2. **Generate a Test Video:**
   ```bash
   python generate_video.py --output test.mp4
   ```

3. **Train on Sample Videos:**
   - The 3 sample videos are ready in `data/videos/videos/`
   - Adapt `train_video.py` with fixes from `train_video_demo.py`
   - Run full training

4. **Create Your Own Experiment:**
   ```bash
   cp -r experiments/exp_template experiments/exp1_my_research
   # Edit experiment config and document your research question
   ```

## 🤝 Research Ideas

Since this is a research platform, here are some ideas:

1. **Architecture**: Try different numbers of attention heads/layers
2. **Training**: Experiment with different learning rates
3. **Data**: Create videos with different patterns/motions
4. **Loss**: Try different loss functions or weightings
5. **Efficiency**: Optimize for faster training/inference

## 📝 Summary

**You can now:**
- ✅ Run demo training that works
- ✅ See loss decreasing over epochs
- ✅ Generate videos (with random model)
- ✅ Understand the model's input/output format

**To do real video generation:**
1. Train on more data for more epochs
2. Implement proper text encoding (T5/CLIP)
3. Add VAE encoding/decoding
4. Use proper diffusion scheduler

---

Happy researching! 🚀

*If you have questions, check the code in `train_video_demo.py` - it has all the working examples.*

