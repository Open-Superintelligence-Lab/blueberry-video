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

## Contributing

See `CONTRIBUTING.md` for guidelines on how to contribute to this project.

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
