# Experiment 1: Baseline CogVideoX Training

**Status**: 🔵 Planned  
**Researcher**: TBD  
**Date**: October 2025

## Research Question

Can a small CogVideoX model learn to generate simple synthetic motion patterns (moving circles, expanding squares, rotating lines)?

## Hypothesis

A reduced CogVideoX model with 12 layers and 8 attention heads should be capable of learning basic motion patterns from synthetic training data, demonstrating that the architecture can capture temporal coherence even at a small scale.

## Methodology

1. **Model**: CogVideoX with reduced parameters (12 layers, 8 heads)
2. **Data**: 3 synthetic videos with simple geometric motions
3. **Training**: 10 epochs with AdamW optimizer, cosine learning rate schedule
4. **Evaluation**: Visual inspection of generated videos for motion consistency

### Changes from Default
- Reduced model size (12 layers instead of 24, 8 heads instead of 16)
- Small dataset (3 synthetic videos)
- Short training (10 epochs)

## Results

### Quantitative Results

_To be filled after running experiment_

- Final loss: 
- Training time: 
- GPU memory usage: 

### Qualitative Observations

_To be filled after running experiment_

### Generated Samples

_Add generated video samples here_

## Conclusions

_To be filled after running experiment_

## Future Work

Potential follow-ups:
- Increase model size and compare quality
- Add more complex motion patterns
- Test different architectures (Hunyuan)
- Experiment with temporal compression ratios

## How to Run

```bash
cd /path/to/blueberry-video
python train_video.py --config experiments/exp1_baseline/config.yaml
```

