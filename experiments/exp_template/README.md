# Experiment Template

This is a template for creating new video generation experiments.

## Structure

- `config.yaml` - Experiment-specific configuration
- `train.py` - Custom training script (optional, if you need modifications)
- `results/` - Store experiment results here
- `README.md` - This file - document your research question and findings

## How to Use This Template

1. Copy this folder and rename it to your experiment name (e.g., `exp1_attention_variants`)
2. Update the config.yaml with your experiment parameters
3. Document your research question below
4. Run the experiment
5. Document your findings

## Research Question

**What are you trying to discover?**

Write your research question here. For example:
- Can we improve video generation by modifying the attention mechanism?
- Does training with different frame rates affect quality?
- How does temporal consistency change with different architectures?

## Hypothesis

What do you expect to happen?

## Methodology

1. Describe your approach
2. What are you changing from the baseline?
3. How will you measure success?

## Results

### Quantitative Results

- Metric 1: X
- Metric 2: Y
- Training time: Z

### Qualitative Observations

What did you notice during training?

### Generated Samples

Include links to generated videos or images here.

## Conclusions

What did you learn?

## Future Work

What would you try next based on these results?

