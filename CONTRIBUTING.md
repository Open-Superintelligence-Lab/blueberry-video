# Contributing to Blueberry Video

Welcome to the Blueberry Video research community! We're excited to have you contribute to advancing video generation research.

## Philosophy

This is a **research-first** repository. We value:

- 🔬 **Rigorous experimentation** over quick hacks
- 📊 **Documented findings** over undocumented code
- 🤝 **Open collaboration** over competition
- 🎯 **Focused research questions** over broad goals

## How to Contribute

### 1. Research Experiments (Primary)

The main way to contribute is by conducting experiments and sharing your findings.

**Steps:**

1. **Fork the repository**
   ```bash
   git clone https://github.com/YOUR-USERNAME/blueberry-video.git
   cd blueberry-video
   ```

2. **Set up your environment**
   ```bash
   pip install -r requirements.txt
   ```

3. **Create your experiment**
   ```bash
   cp -r experiments/exp_template experiments/exp1_your_research
   ```

4. **Define your research question**
   
   Edit `experiments/exp1_your_research/README.md` and clearly state:
   - Research question
   - Hypothesis
   - Methodology
   - Expected outcomes

5. **Configure your experiment**
   
   Edit `experiments/exp1_your_research/config.yaml` with your parameters.

6. **Run your experiment**
   ```bash
   python train_video.py --config experiments/exp1_your_research/config.yaml
   ```

7. **Document your findings**
   
   Update your experiment's README with:
   - Quantitative results
   - Qualitative observations
   - Sample outputs
   - Conclusions
   - Future work suggestions

8. **Submit a Pull Request**
   
   Once you've completed your experiment and documented the results, submit a PR to merge it back into the main repository.

### 2. Code Improvements

You can also contribute by:

- **Adding new architectures** - Implement and document new video generation models
- **Improving training utilities** - Better optimizers, schedulers, data loaders
- **Bug fixes** - Fix issues you encounter
- **Documentation** - Improve explanations and examples

## Experiment Guidelines

### What Makes a Good Experiment?

✅ **Clear Research Question**
- "How does attention sparsity affect temporal consistency?"
- NOT: "Try some stuff with attention"

✅ **Controlled Variables**
- Change ONE thing at a time
- Document what you changed and why

✅ **Reproducible**
- Include all configs
- Document random seeds
- List dependencies

✅ **Well Documented**
- Write clear explanations
- Include visual results when possible
- Share negative results too!

### Experiment Naming

Use descriptive names:
- ✅ `exp1_sparse_attention_temporal`
- ✅ `exp2_frame_rate_ablation`
- ❌ `exp1_test`
- ❌ `my_experiment`

## Code Style

We value **readable code** over clever code.

### Python Style
- Follow PEP 8
- Use meaningful variable names
- Add docstrings to functions
- Keep functions focused and small

### Example

```python
def train_epoch(model, dataloader, optimizer, device, epoch):
    """
    Train model for one epoch
    
    Args:
        model: Video generation model
        dataloader: Training data loader
        optimizer: Optimizer instance
        device: torch device (cuda/cpu)
        epoch: Current epoch number
        
    Returns:
        Average loss for the epoch
    """
    # Clear implementation here
    pass
```

## Pull Request Process

1. **Create a descriptive PR title**
   - ✅ "Experiment: Sparse Attention for Temporal Consistency"
   - ✅ "Fix: Training script batch size handling"
   - ❌ "Update"

2. **Describe your changes**
   - What did you change?
   - Why did you change it?
   - What did you learn?

3. **Include results**
   - For experiments: Include key findings
   - For code changes: Show before/after behavior

4. **Wait for review**
   - Address feedback constructively
   - Be open to suggestions

## Research Ethics

- **Give credit** - Cite papers and acknowledge ideas
- **Share failures** - Negative results are valuable
- **Be honest** - Don't cherry-pick results
- **Help others** - Answer questions, review PRs

## Questions?

- Open an issue for discussions
- Tag experiments with your GitHub handle
- Help others with their experiments

## Recognition

All contributors will be:
- Listed in the experiments table
- Credited in the README
- Part of the research community

---

**Thank you for contributing to open science! 🚀**

Together we're accelerating video generation research.
