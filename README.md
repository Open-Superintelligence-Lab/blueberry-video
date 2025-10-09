# Blueberry Video

**Open Superintelligence Lab** - Open research for everyone. We publish all of our research for the sake of accelerating science. Learn real AI research from a real research lab.

## Quick Start

```bash
pip install -r requirements.txt

# Train with CogVideoX architecture
python train_video.py

# Or specify a different config
python train_video.py --config configs/video/train/hunyuan.yaml
```

## About

Purpose of this repository is to research better, faster, smarter **video generation models**.

This repository contains cutting-edge video generation experiments and architectures. We believe scientists do their best work when given freedom to explore, so this is a space for your independent research and discovery.

Fork this repository, create a new experiment in `experiments/` folder, then create a pull request to merge it back.

## Architectures

We currently support two state-of-the-art video generation architectures:

### 1. **CogVideoX** (3D Transformer)
- 3D attention mechanism for temporal coherence
- Efficient for medium-length videos
- Located in: `models/cogvideox_transformer.py`

### 2. **Hunyuan Video** (Advanced Transformer)
- Advanced temporal modeling
- High-quality video generation
- Located in: `models/hunyuan_video_transformer.py`

## Experiments

> Each experiment below is validated on a specific git tag. Later commits may introduce breaking changes. To run an experiment with correct version of the repo, checkout its validated tag using: `git checkout <tag-name>`

| Experiment | Researcher | Validated Tag | Research Question | Key Findings |
|------------|-----------|---------------|-------------------|--------------|
| _Your experiments will be added here_ | | | | |

## Getting Started

### 1. Fork and Clone

```bash
git clone https://github.com/YOUR-USERNAME/blueberry-video.git
cd blueberry-video
pip install -r requirements.txt
```

### 2. Prepare Your Dataset

Create a dataset with this structure:

```
data/videos/
  videos/
    video_001.mp4
    video_002.mp4
    ...
  metadata.json
```

The `metadata.json` should contain prompts for each video:

```json
{
  "video_001": "A cat playing with a ball",
  "video_002": "Sunset over the ocean",
  ...
}
```

### 3. Choose Your Architecture

Edit `configs/video/train/cogvideox.yaml` or `configs/video/train/hunyuan.yaml` to configure your experiment.

### 4. Train

```bash
python train_video.py
```

### 5. Create an Experiment

```bash
# Copy the template
cp -r experiments/exp_template experiments/exp1_my_research

# Edit the config
vim experiments/exp1_my_research/config.yaml

# Document your research question
vim experiments/exp1_my_research/README.md

# Run your experiment
python train_video.py --config experiments/exp1_my_research/config.yaml
```

### 6. Share Your Findings

Once you finish with your research, create a pull request to merge it back to this repo!

## Philosophy

We don't prescribe what to research. Instead, we provide:

* **Freedom to explore** interesting ideas
* **Infrastructure to test** hypotheses
* **A collaborative environment** for learning

Research questions you might explore:

- How does attention mechanism design affect temporal consistency?
- Can we generate longer videos with limited compute?
- What's the optimal trade-off between quality and speed?
- How do different frame rates affect training dynamics?
- Can we improve motion modeling with custom architectures?

## Structure

```
blueberry-video/
├── models/              # Video generation architectures
│   ├── cogvideox_transformer.py
│   ├── hunyuan_video_transformer.py
│   └── ...
├── configs/             # Configuration files
│   └── video/
│       ├── train/       # Training configs
│       └── inference/   # Inference configs
├── data/                # Dataset utilities
│   └── video_dataset.py
├── experiments/         # Research experiments
│   └── exp_template/    # Template for new experiments
├── utils/               # Training utilities
├── train_video.py       # Main training script
└── README.md            # This file
```

## Key Features

✅ **Simple & Clean** - Minimal boilerplate, maximum research  
✅ **Two SOTA Architectures** - CogVideoX and Hunyuan Video  
✅ **Easy Experimentation** - Template-based experiment system  
✅ **Well Documented** - Every experiment tells a story  
✅ **Open Science** - All research is public  

## Research Areas

Some areas you might want to explore:

### Architecture
- Novel attention mechanisms
- Temporal modeling improvements
- Multi-scale processing
- Efficient transformers

### Training
- Loss function variants
- Curriculum learning strategies
- Data augmentation techniques
- Transfer learning approaches

### Quality
- Motion consistency
- Temporal coherence
- Resolution scaling
- Frame interpolation

### Efficiency
- Model compression
- Faster inference
- Lower memory usage
- Quantization strategies

## Contributing

See `CONTRIBUTING.md` for guidelines on how to contribute to this project.

## Citation

If you use this codebase for your research, please cite:

```bibtex
@software{blueberry_video,
  author = {Open Superintelligence Lab},
  title = {Blueberry Video: Open Video Generation Research},
  year = {2025},
  url = {https://github.com/Open-Superintelligence-Lab/blueberry-video}
}
```

## License

MIT License - See LICENSE file for details

## Community

Join our research community:
- Share your experiments
- Discuss findings
- Collaborate on ideas
- Learn from others

---

**Made with ❤️ by Open Superintelligence Lab**

*Accelerating video generation research, one experiment at a time.*
