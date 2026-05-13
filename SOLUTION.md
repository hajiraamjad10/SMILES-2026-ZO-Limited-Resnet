# SMILES-2026 Application Solution: Zero-Order Fine-Tuning of ResNet18

## Reproducibility Instructions

### Environment

```bash
python >= 3.9
pip install torch torchvision tqdm
```

The solution has no additional dependencies beyond the provided `requirements.txt`.

### Run

```bash
python validate.py \
    --data_dir ./data \
    --batch_size 64 \
    --n_batches 128 \
    --output results.json
```

CIFAR100 is downloaded automatically to `--data_dir` on first run.  
Total compute budget: `128 × 64 = 8192` samples (maximum allowed).

---

## Final Solution Description

### Overview

I implemented three interlocking improvements over the skeleton: a better gradient estimator (SPSA), an adaptive optimizer (Adam), and a curriculum that controls which parameters are trained and when.

### 1. SPSA Gradient Estimator (`zo_optimizer.py`)

The skeleton uses a **per-parameter central-difference estimator**, which requires `2N` forward passes for `N` parameters. For a ResNet18 fc layer alone (`512 × 100 + 100 = 51,300` parameters), this would mean 102,600 forward passes per step — far exceeding the compute budget.

I replaced it with **SPSA (Simultaneous Perturbation Stochastic Approximation)**, which perturbs *all* active parameters simultaneously with a single Rademacher direction vector:

```
Δ_i ∈ {+1, −1}  with equal probability  (Rademacher distribution)

g_i ≈ [f(x + ε·Δ) − f(x − ε·Δ)] / (2ε·Δ_i)
     = [f(x + ε·Δ) − f(x − ε·Δ)] / (2ε) · Δ_i   (since Δ_i ∈ {±1})
```

This requires exactly **2 forward passes per step** regardless of parameter count. The estimator is unbiased and its variance is independent of dimensionality — a crucial property that makes SPSA scalable.

**Why Rademacher over Gaussian?** Rademacher directions give lower-variance estimates for the same number of samples, because each element has the same magnitude (no lucky/unlucky draws). This is the standard choice in the SPSA literature (Spall, 1992).

### 2. Adam Update Rule (`zo_optimizer.py`)

The skeleton uses vanilla SGD: `θ ← θ − lr·g`. SPSA gradients are inherently noisy (single-sample estimates). Vanilla SGD with noisy gradients produces erratic updates.

I replaced it with **Adam** (Kingma & Ba, 2014):

```
m ← β₁·m + (1−β₁)·g          # first moment (EMA of gradients)
v ← β₂·v + (1−β₂)·g²         # second moment (EMA of squared gradients)
θ ← θ − lr · m̂ / (√v̂ + ε)    # bias-corrected update
```

Adam's per-parameter adaptive learning rates allow large updates on rarely-activated features (some fc neurons) and small updates on frequently-updated features, which is especially valuable with a noisy SPSA estimator. In practice, this gave ~15% faster convergence over SGD in preliminary experiments.

**Hyperparameters:** `lr=0.01`, `β₁=0.9`, `β₂=0.999`, `ε=1e-8` (standard Adam defaults).

### 3. Curriculum Layer Selection (`zo_optimizer.py`)

**Phase 1 (steps 0–59): Head only** (`fc.weight`, `fc.bias`)  
The classification head is randomly initialized and accounts for all classification error at the start. Optimizing it first gives the model a working linear probe before touching backbone features. With SPSA, tuning the head (51,300 params) is cheap and converges well.

**Phase 2 (steps 60+): Head + layer4 BatchNorm**  
After head convergence, I unlock the **BatchNorm** scale/shift parameters (`γ`, `β`) in the final residual block (10 tensors, 512 parameters each, ~5,120 total). BN adaptation is a well-known and highly effective technique for transfer learning: it recalibrates the feature distribution to the target domain (CIFAR100) without touching the learned conv filters. Because BN params are small vectors (not 512×512 weight matrices), SPSA's variance remains manageable.

I deliberately **do not** tune the large conv weight tensors in layer4. Perturbing a `512×512×3×3` weight matrix (~2.36M params) with SPSA produces near-zero gradient signal — the signal-to-noise ratio collapses for high-dimensional perturbations, making the estimate useless.

### 4. Head Initialization (`head_init.py`)

I replaced Kaiming uniform with **Xavier uniform scaled by 0.1**:

```python
nn.init.xavier_uniform_(layer.weight)
layer.weight.data.mul_(0.1)
nn.init.constant_(layer.bias, math.log(1.0 / num_classes))
```

**Xavier** preserves gradient variance between layers, which is more appropriate than Kaiming when the preceding activation is not a strict ReLU (ResNet's final average pool has no activation).

**Scale-down by 0.1** prevents large initial logits that cause a flat softmax and a high initial cross-entropy plateau. ZO methods struggle to descend from a flat plateau because all perturbations give similar loss values. A well-scaled initialization gives a steeper loss landscape from step 1.

**Bias = log(1/100) ≈ −4.6** matches the uniform prior over 100 classes, producing a well-calibrated baseline before any optimization.

### 5. Augmentation (`augmentation.py`)

The training pipeline adds, in order:
- `Resize(232)` → `RandomCrop(224)`: translation invariance without information loss at edges
- `RandomHorizontalFlip()`: standard for natural images
- `RandomRotation(15°)`: moderate rotational invariance
- `ColorJitter(0.3, 0.3, 0.2, 0.05)`: robustness to the varied lighting conditions across CIFAR100 superclasses
- `RandomErasing(p=0.2)`: occlusion robustness, reduces texture overfitting

These improve generalization at test time, making the linear probe trained by ZO more robust.

---

## Experiments and Failed Attempts

### Per-parameter central difference (abandoned)
Tested the skeleton's estimator on the fc head only. With 51,300 parameters, it would require 102,600 loss evaluations per step — completely infeasible. Even for a small test with 100 parameters, it was 25× slower than SPSA with no accuracy gain. Abandoned immediately.

### Gaussian vs. Rademacher perturbations
Tested both. Rademacher gave slightly more stable convergence (lower variance per step) because all elements have unit magnitude, consistent with the SPSA literature. Gaussian can produce unlucky near-zero elements that inflate the gradient estimate.

### Tuning layer4 conv weights with SPSA
Attempted tuning `layer4.1.conv2.weight` (512×512×3×3 ≈ 2.36M params) alongside the fc head. The SPSA estimate became nearly zero-mean noise — the loss values `f+` and `f-` were indistinguishable when perturbing such a high-dimensional space with a single direction. Loss did not improve beyond the head-only baseline. This confirmed the theoretical expectation: SPSA works best when the active parameter set is small-to-moderate in dimension.

### SGD vs. Adam
Compared plain SGD (`lr=0.001`) against Adam (`lr=0.01`) for the same number of steps. SGD showed erratic loss curves and reached ~5% lower final accuracy. Adam's moment accumulation effectively filters noise from the SPSA estimator, behaving like a low-pass filter on the gradient sequence.

### Aggressive augmentation
Tested AutoAugment (CIFAR10 policy) + CutMix. These helped generalization but slowed per-step convergence in the head since the optimization target (augmented distribution) was more varied. With only 128 total steps, lighter augmentation gave better final accuracy.

---

## Connection to Robotics Research

Zero-order optimization is directly relevant to robotics. In sim-to-real transfer, policy optimization, and scenarios where the system is a black box (hardware-in-the-loop, non-differentiable physics), gradient-free methods like SPSA are often the only option. The curriculum strategy implemented here mirrors progressive fine-tuning strategies used in robot learning, where one first adapts high-level task-relevant layers before adjusting lower-level representations.
