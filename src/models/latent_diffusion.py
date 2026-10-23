"""Conditional latent DDPM with scenario event/missingness conditioning."""
import math
import torch
from torch import nn


class LatentDiffusion(nn.Module):
    def __init__(self, latent_dim, condition_dim, hidden_dim=128, steps=100):
        super().__init__()
        self.latent_dim, self.steps = latent_dim, steps
        self.network = nn.Sequential(nn.Linear(latent_dim + condition_dim + 32, hidden_dim), nn.SiLU(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, latent_dim))
        t = torch.linspace(0, 1, steps + 1, dtype=torch.float64)
        ab = torch.cos((t + .008) / 1.008 * math.pi / 2).square()
        beta = (1 - ab[1:] / ab[:-1]).clamp(max=.999).float()
        self.register_buffer('beta', beta)
        self.register_buffer('alpha_bar', (1 - beta).cumprod(0))

    def forward(self, z, t, condition):
        f = torch.exp(torch.arange(16, device=z.device) * (-math.log(10000) / 15))
        phase = t[:, None] * f
        return self.network(torch.cat([z, phase.sin(), phase.cos(), condition], -1))

    def losses(self, clean, condition):
        t = torch.randint(self.steps, (len(clean),), device=clean.device)
        noise = torch.randn_like(clean)
        a = self.alpha_bar[t, None]
        noisy = a.sqrt() * clean + (1 - a).sqrt() * noise
        return (self(noisy, t, condition) - noise).square().mean(-1)

    @torch.no_grad()
    def sample(self, condition, initial=None, start_step=None):
        if initial is None:
            if start_step is not None:
                raise ValueError('start_step requires initial latents')
            x = torch.randn(len(condition), self.latent_dim, device=condition.device)
            start_step = self.steps - 1
        else:
            if start_step is None or not 0 <= start_step < self.steps or initial.shape != (len(condition), self.latent_dim):
                raise ValueError('Invalid diffusion anchor')
            a = self.alpha_bar[start_step]
            x = a.sqrt() * initial + (1-a).sqrt() * torch.randn_like(initial)
        for step in reversed(range(start_step + 1)):
            t = torch.full((len(x),), step, device=x.device)
            a, b = self.alpha_bar[step], self.beta[step]
            prev = self.alpha_bar[step-1] if step else torch.ones_like(a)
            clean = ((x - (1-a).sqrt() * self(x, t, condition)) / a.sqrt()).clamp(-5, 5)
            mean = prev.sqrt()*b/(1-a)*clean + (1-b).sqrt()*(1-prev)/(1-a)*x
            x = mean + (b*(1-prev)/(1-a)).sqrt()*torch.randn_like(x) if step else mean
        return x
