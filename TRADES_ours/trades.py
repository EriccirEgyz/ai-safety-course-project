import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def squared_l2_norm(x):
    # MODIFIED: x.unsqueeze(0).shape[0] -> x.shape[0], Variable已废弃，直接用tensor
    flattened = x.view(x.shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


def compute_beta_i(margins, beta, margin_tau=0.0, margin_temperature=1.0):
    # Per-sample robustness weight from clean margins (Design A):
    #   beta_i = beta * sigmoid((m_i - tau) / T) / mean_batch sigmoid((m_j - tau) / T)
    # Larger clean margin -> stronger robustness regularization.
    # Wrapped in no_grad so the model cannot lower the loss by directly manipulating beta_i.
    with torch.no_grad():
        margin_weights = torch.sigmoid((margins - margin_tau) / margin_temperature)
        return float(beta) * margin_weights / (margin_weights.mean() + 1e-12)


def trades_loss(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                margin_tau=0.0,
                margin_temperature=1.0,
                distance='l_inf'):
    # define KL-loss
    # MODIFIED: size_average=False 已废弃，改为 reduction='sum'
    criterion_kl = nn.KLDivLoss(reduction='sum')
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        # MODIFIED: Variable已废弃，直接用requires_grad=True
        delta.requires_grad = True

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
                                           F.softmax(model(x_natural), dim=1))
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        # MODIFIED: Variable已废弃，用.detach()代替
        x_adv = (x_natural + delta).detach()
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    # MODIFIED: Variable已废弃，用.detach()代替
    x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    logits_adv = model(x_adv)

    # MODIFIED: Use a margin-based per-sample beta_i instead of a fixed batch-level beta.
    # The clean margin is defined as the true-class logit minus the largest non-true-class logit.
    # Samples with larger clean margins are better separated, so they receive stronger robustness regularization.
    true_logits = logits.gather(1, y.view(-1, 1)).squeeze(1)
    other_logits = logits.masked_fill(F.one_hot(y, num_classes=logits.size(1)).bool(), float('-inf'))
    max_other_logits = other_logits.max(dim=1)[0]
    margins = true_logits - max_other_logits

    beta_i = compute_beta_i(margins, beta,
                            margin_tau=margin_tau,
                            margin_temperature=margin_temperature)

    robust_kl = F.kl_div(F.log_softmax(logits_adv, dim=1),
                         F.softmax(logits, dim=1),
                         reduction='none').sum(dim=1)
    loss_robust = torch.mean(beta_i * robust_kl)
    loss = loss_natural + loss_robust
    return loss
