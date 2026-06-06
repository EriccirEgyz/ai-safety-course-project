"""
PGD-AT loss (Madry et al. 2018, ICLR).

Inner attack: maximize CE(f(x + delta), y) under l_inf constraint.
Outer loss:   CE(f(x_adv), y).

Inner PGD hyperparameters (noise init, epsilon, step_size, perturb_steps,
clamp order) are kept identical to TRADES for fair comparison.
The only algorithmic difference vs TRADES is:
  - Inner attack objective: CE (PGD-AT) vs KL-divergence (TRADES)
  - Outer loss: CE on x_adv only (PGD-AT) vs CE on x + beta*KL(x, x_adv) (TRADES)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def pgdat_loss(model,
               x_natural,
               y,
               optimizer,
               step_size=0.007,
               epsilon=0.031,
               perturb_steps=10,
               distance='l_inf'):
    model.eval()
    # generate adversarial example — noise init identical to TRADES
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                # PGD-AT inner attack: maximize CE loss (not KL as in TRADES)
                loss_attack = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss_attack, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        raise NotImplementedError("Only l_inf distance is supported for PGD-AT baseline.")

    model.train()
    x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()

    # zero gradient
    optimizer.zero_grad()
    # PGD-AT outer loss: CE on adversarial example
    logits_adv = model(x_adv)
    loss = F.cross_entropy(logits_adv, y)
    return loss
