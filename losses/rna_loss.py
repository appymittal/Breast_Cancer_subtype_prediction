import torch
from torch import nn
from .focal import FocalLoss


class RNASeqLoss(nn.Module):
    def __init__(self, beta=0.1, class_weights=None, annealing_steps=10000,
                 use_focal=False, focal_gamma=2.0, label_smoothing=0.0, kl_epsilon=1e-8):
        super().__init__()
        self.register_buffer('current_step', torch.tensor(0))
        self.target_beta = beta
        self.annealing_steps = annealing_steps
        self.kl_epsilon = kl_epsilon

        if class_weights is not None:
            self.register_buffer('class_weights', class_weights.clone().detach())

        else:
            self.class_weights = None

        self.use_focal = use_focal
        self.label_smoothing = label_smoothing
        self.focal_gamma = focal_gamma

        if use_focal:
            self.ce_loss = FocalLoss(
                weight=self.class_weights,
                gamma=focal_gamma,
                label_smoothing=label_smoothing
            )
        else:
            self.ce_loss = nn.CrossEntropyLoss(
                weight=self.class_weights,
                label_smoothing=label_smoothing
            )

    @property
    def beta(self):
        # Sigmoid warm-up curve
        center = self.annealing_steps // 2
        steepness = 5 / max(self.annealing_steps, 1)
        ratio = 1 / (1 + torch.exp(-steepness * (self.current_step.float() - center)))
        return self.target_beta * ratio

    def kl_divergence(self, mu, logvar):
        kl_terms = -0.5 * (1 + logvar - mu.pow(2) - (logvar.exp() + self.kl_epsilon))
        return torch.sum(kl_terms, dim=1).mean()

    def reset_annealing(self):
        self.current_step.zero_()

    def forward(self, outputs, targets):
        if outputs['logits'].shape[0] != targets.shape[0]:
            raise ValueError(
                f"Batch size mismatch: logits {outputs['logits'].shape[0]} vs "
                f"targets {targets.shape[0]}"
            )

        ce = self.ce_loss(outputs['logits'], targets)
        if 'mu_rna' in outputs:
            kl_rna = self.kl_divergence(outputs['mu_rna'], outputs['logvar_rna']).detach()
            total_loss = ce + self.beta * kl_rna
        else:
            kl_rna = torch.tensor(0.0, device=self.current_step.device)
            total_loss = ce

        loss_dict = {
            'total': total_loss,
            'ce': ce.detach(),
            'kl_total': kl_rna,
            'beta': self.beta.clone().detach().to(self.current_step.device)
        }

        if self.training:
            self.current_step += 1

        return loss_dict