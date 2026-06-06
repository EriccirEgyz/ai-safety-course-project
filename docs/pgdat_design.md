# PGD-AT Baseline Design & Experiment Plan

## 1. Motivation

PGD adversarial training (Madry et al. 2018, ICLR) is the cornerstone baseline for adversarial defenses. Without reporting PGD-AT numbers alongside TRADES and our proposed method, the evaluation is incomplete and reviewers will question baseline coverage.

Our project claims that **adaptive per-sample regularization** (margin-based β_i in TRADES) improves the robustness/accuracy trade-off. To validate this claim, we must compare against:
- **Natural training** (upper bound on clean accuracy, zero robustness)
- **PGD-AT** (basic adversarial training, no trade-off tuning)
- **TRADES** (the method we modify)
- **Ours (margin-TRADES)** (our proposed improvement)

## 2. PGD-AT vs TRADES: Mathematical Comparison

### PGD-AT (Madry et al. 2018)

$$\min_\theta \mathbb{E}_{(x,y)} \left[ \mathcal{L}_{CE}\bigl(f_\theta(x^*),\, y\bigr) \right]$$

where the adversarial example is found by maximizing the **cross-entropy** loss:

$$x^* = \arg\max_{\|x' - x\|_\infty \leq \epsilon} \mathcal{L}_{CE}\bigl(f_\theta(x'),\, y\bigr)$$

### TRADES (Zhang et al. 2019)

$$\min_\theta \mathbb{E}_{(x,y)} \left[ \mathcal{L}_{CE}\bigl(f_\theta(x),\, y\bigr) + \beta \cdot \mathrm{KL}\bigl(f_\theta(x) \,\|\, f_\theta(x^*)\bigr) \right]$$

where the adversarial example is found by maximizing the **KL-divergence**:

$$x^* = \arg\max_{\|x' - x\|_\infty \leq \epsilon} \mathrm{KL}\bigl(f_\theta(x) \,\|\, f_\theta(x')\bigr)$$

### Key differences

| Aspect | PGD-AT | TRADES |
|--------|--------|--------|
| Inner attack objective | Maximize CE(f(x'), y) | Maximize KL(f(x) || f(x')) |
| Outer loss | CE(f(x*), y) only | CE(f(x), y) + β·KL(f(x) || f(x*)) |
| Trade-off parameter | None (ε is attack budget, not trade-off) | β balances clean loss vs. robustness |

**Important:** The inner attack objective difference (CE vs KL) is an inherent algorithmic difference between the two methods, not an implementation choice. Forcing PGD-AT to use KL attack would make it no longer Madry's PGD-AT.

### Our method (Margin-TRADES)

Same as TRADES, but replaces the fixed scalar β with a per-sample adaptive β_i computed from the clean margin:

$$\beta_i = \beta \cdot \frac{\sigma\bigl((m_i - \tau) / T\bigr)}{\mathrm{mean}\bigl(\sigma((m_j - \tau) / T)\bigr)}$$

where m_i = (true class logit) - (max other class logit), τ is a margin threshold, and T is a temperature.

## 3. Fair Comparison Protocol

The **only** difference between methods is the loss function. All other experimental settings are held constant.

| Setting | PGD-AT | TRADES | Ours (Margin-TRADES) |
|---------|--------|--------|---------------------|
| **Model** | WRN-34-10 | WRN-34-10 | WRN-34-10 |
| **Data transform** | RandomCrop(32,4) + HFlip + ToTensor | same | same |
| **Normalization** | None (images in [0,1]) | same | same |
| **Optimizer** | SGD(lr=0.1, momentum=0.9, wd=2e-4) | same | same |
| **LR schedule** | ×0.1 @ep75, ×0.01 @ep90, ×0.001 @ep100 | same | same |
| **Epochs** | 100 | 100 | 100 |
| **Batch size** | 128 | 128 | 128 |
| **Inner PGD ε** | 0.031 | 0.031 | 0.031 |
| **Inner PGD step_size** | 0.007 | 0.007 | 0.007 |
| **Inner PGD num_steps** | 10 | 10 | 10 |
| **Inner noise init** | 0.001 × randn | same | same |
| **Inner clamp order** | project ±ε → clamp [0,1] | same | same |
| **Inner attack obj** | **CE** | **KL** | **KL** |
| **Outer loss** | **CE(f(x_adv), y)** | **CE + β·KL** | **CE + β_i·KL** |
| **β** | N/A | 6.0 | 6.0 (base) |
| **margin τ** | N/A | N/A | 0.3 |
| **margin T** | N/A | N/A | 0.15 |
| **Save freq** | every 10 epochs | every epoch | every epoch |
| **Seed** | 1 | 1 | 1 |

**Note on save-freq:** PGD-AT uses save-freq=10 (vs 1 for TRADES) purely for disk space management. This is a non-scientific parameter that does not affect training dynamics. The final epoch-100 checkpoint is guaranteed to be saved (100 mod 10 = 0).

## 4. Expected Results (Literature Baselines)

Based on Madry et al. 2018 (ICLR) and Rice et al. 2020 (ICML):

**CIFAR-10 (WRN-34-10):**
- Clean accuracy: 85–87%
- PGD-20 robust accuracy: 47–55%

**CIFAR-100 (WRN-34-10):**
- Clean accuracy: 58–62%
- PGD-20 robust accuracy: 25–28%

**Sanity check:** If measured robust accuracy falls below these ranges by >20% (e.g., CIFAR-10 robust < 30%, CIFAR-100 robust < 15%), there is likely an implementation bug.

## 5. Evaluation Protocol

Use the existing `pgd_attack_cifar{10,100}.py` scripts with identical settings across all methods:

- Attack: PGD-20 white-box
- ε = 0.031
- step_size = 0.003
- test_batch_size = 200
- Random init: True

Command (example for CIFAR-10):
```bash
python pgd_attack_cifar10.py --model-path <checkpoint_path>
```

## 6. Time Budget

PGD-AT inner loop: 10 PGD steps per batch → ~10× slower than natural training per epoch.

Estimated per dataset (100 epochs):
- CIFAR-10: ~10–14 hours on RTX 4090
- CIFAR-100: ~10–14 hours on RTX 4090

Priority: **CIFAR-10 first** (must complete before 6/9 presentation). CIFAR-100 can run overnight.

## 7. Training Dynamics & Monitoring

- **Early epochs (1–20):** test robust accuracy ≈ 0% is normal. The model is still learning clean features.
- **Mid training (20–75):** robust accuracy slowly climbs.
- **After LR drop (ep75+):** robust accuracy stabilizes and converges.
- **Warning signs:** loss → NaN, train accuracy stuck at 10% (CIFAR-10) or 1% (CIFAR-100).

## 8. Results Table (to be filled)

| Method | Dataset | Clean Acc | Robust Acc (PGD-20) |
|--------|---------|-----------|---------------------|
| Natural | CIFAR-10 | 95.73% | 0.00% |
| Natural | CIFAR-100 | 79.15% | ~0% |
| PGD-AT | CIFAR-10 | — | — |
| PGD-AT | CIFAR-100 | — | — |
| TRADES | CIFAR-10 | 84.9% | 54.6% |
| TRADES | CIFAR-100 | — | — |
| Ours | CIFAR-10 | 86.1% | 54.9% |
| Ours | CIFAR-100 | — | — |
