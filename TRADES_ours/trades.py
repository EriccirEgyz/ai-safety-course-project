import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def tensor_stats(x):
    values = x.detach().float().view(-1).cpu()
    sorted_values = torch.sort(values)[0]
    last_idx = values.numel() - 1
    return {
        'mean': values.mean().item(),
        'std': values.std(unbiased=False).item(),
        'min': values.min().item(),
        'max': values.max().item(),
        'p10': sorted_values[int(0.10 * last_idx)].item(),
        'p50': sorted_values[int(0.50 * last_idx)].item(),
        'p90': sorted_values[int(0.90 * last_idx)].item(),
    }


def squared_l2_norm(x):
    # MODIFIED: x.unsqueeze(0).shape[0] -> x.shape[0], Variable已废弃，直接用tensor
    flattened = x.view(x.shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


def trades_loss(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                beta_schedule='margin_easy',
                margin_tau=0.0,
                margin_temperature=1.0,
                beta_i_min=1.0,
                beta_i_max=10.0,
                distance='l_inf',
                return_stats=False):
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

    # MODIFIED: Support three beta scheduling strategies:
    # - 'fixed': Original TRADES with constant beta (no per-sample adaptation)
    # - 'margin_easy': Higher beta for easy samples (large margin) - protects already-correct samples
    # - 'margin_hard': Higher beta for hard samples (small margin) - hard example mining

    if beta_schedule == 'fixed':
        # Original TRADES: uniform beta across all samples
        beta_i = torch.full((batch_size,), float(beta), device=x_natural.device)

    elif beta_schedule in ['margin_easy', 'margin_hard']:
        # Compute clean margin: true-class logit minus largest non-true-class logit
        true_logits = logits.gather(1, y.view(-1, 1)).squeeze(1)
        other_logits = logits.masked_fill(F.one_hot(y, num_classes=logits.size(1)).bool(), float('-inf'))
        max_other_logits = other_logits.max(dim=1)[0]
        margins = true_logits - max_other_logits

        # Detach beta_i so the model cannot reduce its loss by directly manipulating beta weights.
        # tau shifts the margin threshold, and temperature controls how sharply margins are mapped to weights.
        # Normalize beta_i to keep the batch-average regularization strength equal to the original beta.
        # Clip beta_i to the common TRADES beta range to avoid unstable extreme per-sample weights.
        with torch.no_grad():
            if beta_schedule == 'margin_easy':
                # Easy samples (large margin) get higher beta
                margin_weights = torch.sigmoid((margins - margin_tau) / margin_temperature)
            else:  # 'margin_hard'
                # Hard samples (small margin) get higher beta - REVERSED strategy
                margin_weights = torch.sigmoid((margin_tau - margins) / margin_temperature)

            beta_i = float(beta) * margin_weights / (margin_weights.mean() + 1e-12)
            beta_i = torch.clamp(beta_i, min=beta_i_min, max=beta_i_max)

    else:
        raise ValueError(f"Unknown beta_schedule: '{beta_schedule}'. "
                        f"Must be one of: 'fixed', 'margin_easy', 'margin_hard'.")

    robust_kl = F.kl_div(F.log_softmax(logits_adv, dim=1),
                         F.softmax(logits, dim=1),
                         reduction='none').sum(dim=1)
    loss_robust = torch.mean(beta_i * robust_kl)
    loss = loss_natural + loss_robust
    if return_stats:
        # Compute margin stats (only if margin-based schedule is used)
        if beta_schedule in ['margin_easy', 'margin_hard']:
            true_logits = logits.gather(1, y.view(-1, 1)).squeeze(1)
            other_logits = logits.masked_fill(F.one_hot(y, num_classes=logits.size(1)).bool(), float('-inf'))
            max_other_logits = other_logits.max(dim=1)[0]
            margins = true_logits - max_other_logits
            margin_stats = tensor_stats(margins)
        else:
            margin_stats = {'mean': 0.0, 'std': 0.0, 'p10': 0.0, 'p50': 0.0, 'p90': 0.0}

        beta_stats = tensor_stats(beta_i)
        stats = {
            'num_samples': batch_size,
            'loss_total': loss.detach().item(),
            'loss_natural': loss_natural.detach().item(),
            'loss_robust': loss_robust.detach().item(),
            'margin_mean': margin_stats['mean'],
            'margin_std': margin_stats['std'],
            'margin_p10': margin_stats['p10'],
            'margin_p50': margin_stats['p50'],
            'margin_p90': margin_stats['p90'],
            'beta_i_mean': beta_stats['mean'],
            'beta_i_std': beta_stats['std'],
            'beta_i_min': beta_stats['min'],
            'beta_i_max': beta_stats['max'],
            'beta_i_p10': beta_stats['p10'],
            'beta_i_p50': beta_stats['p50'],
            'beta_i_p90': beta_stats['p90'],
            'margin_values': margins.detach().float().cpu() if beta_schedule in ['margin_easy', 'margin_hard'] else None,
            'beta_i_values': beta_i.detach().float().cpu(),
        }
        return loss, stats
    return loss
