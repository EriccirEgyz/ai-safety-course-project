import os
import sys

import torch
import pytest

# Allow `from trades import compute_beta_i` when run from repo root via
# `python -m pytest tests/test_beta_i.py -v`.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TRADES_OURS = os.path.join(_REPO_ROOT, 'TRADES_ours')
if _TRADES_OURS not in sys.path:
    sys.path.insert(0, _TRADES_OURS)

from trades import compute_beta_i


# Defaults mirror TRADES_ours/train_trades_cifar10.py (tau=0.3, T=0.15, beta=6.0).
TAU = 0.3
T = 0.15
BETA = 6.0


def test_beta_i_strictly_monotonic_in_margin():
    # Design A: d/dm sigmoid((m - tau)/T) = sigmoid'(z) / T > 0 for finite m and T > 0,
    # and the post-normalization divisor (mean of positive weights) does not change
    # the ordering. So beta_i must be strictly increasing in margin.
    margins = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    beta_i = compute_beta_i(margins, beta=BETA, margin_tau=TAU, margin_temperature=T)
    diffs = beta_i[1:] - beta_i[:-1]
    assert torch.all(diffs > 0), f"beta_i not strictly increasing in margin: {beta_i.tolist()}"


def test_beta_i_batch_mean_matches_baseline_beta():
    # Normalization guarantees mean(beta_i) == beta exactly in the population limit;
    # for any finite batch it equals beta up to the 1e-12 numerical slack.
    torch.manual_seed(0)
    margins = torch.randn(1024)
    beta_i = compute_beta_i(margins, beta=BETA, margin_tau=TAU, margin_temperature=T)
    assert abs(beta_i.mean().item() - BETA) < 1e-3, \
        f"mean(beta_i)={beta_i.mean().item():.6f}, expected {BETA} within 1e-3"


def test_beta_i_carries_no_grad_through_margins():
    # The original inline mapping lived inside `with torch.no_grad():`; the helper
    # preserves that. After backward through beta_i, margins.grad must remain None,
    # i.e. the model cannot lower the loss by directly nudging the margin weights.
    margins = torch.randn(100, requires_grad=True)
    beta_i = compute_beta_i(margins, beta=BETA, margin_tau=TAU, margin_temperature=T)
    assert not beta_i.requires_grad, "beta_i must be detached from the autograd graph"
    if beta_i.requires_grad:
        beta_i.sum().backward()
    assert margins.grad is None, "gradient leaked from beta_i back into margins"


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
