import torch
from torch import nn


class EpisodeEncoder(nn.Module):
    """Encodes a past observation episode into a fixed-size context vector."""

    def __init__(self, input_dim, hidden_dim, latent_dim, num_heads, num_layers, dropout):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by transformer_heads.")
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.projection = nn.Sequential(nn.Linear(hidden_dim, latent_dim), nn.SiLU())

    def forward(self, history):
        """Return a context vector from an ordered history of shape [B, T, C]."""
        sequence_length = history.shape[1]
        positions = torch.arange(sequence_length, device=history.device).unsqueeze(1)
        dimensions = torch.arange(0, self.input_projection.out_features, 2, device=history.device)
        angles = positions / (10000 ** (dimensions / self.input_projection.out_features))
        positional_encoding = torch.zeros(sequence_length, self.input_projection.out_features, device=history.device)
        positional_encoding[:, 0::2] = torch.sin(angles)
        positional_encoding[:, 1::2] = torch.cos(angles)

        tokens = self.input_projection(history) + positional_encoding.unsqueeze(0)
        encoded_history = self.transformer(tokens)# The final token summarizes the past up to the forecast start.
        # The final token summarizes the past up to the forecast start.
        return self.projection(encoded_history[:, -1, :])


class NoisePredictor(nn.Module):
    """Predict diffusion velocity v (legacy class name retained)."""
    def __init__(self, sequence_length, latent_dim, hidden_dim):
        super().__init__()
        self.sequence_length = sequence_length
        self.time_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.network = nn.Sequential(
            nn.Linear(sequence_length + latent_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, sequence_length),
        )

    def forward(self, noisy_future, timestep, context):
        # A normalized scalar timestep is enough for this compact baseline.
        timestep = timestep.float().view(-1, 1)
        time_features = self.time_embedding(timestep)
        flattened = noisy_future.squeeze(-1)
        return self.network(torch.cat([flattened, context, time_features], dim=-1)).unsqueeze(-1)


class EpisodicDiffusion(nn.Module):
    # Old epsilon/absolute-RSRP checkpoints have incompatible semantics.
    FORMAT = "zero_snr_v_residual_v1"

    def __init__(
        self, input_dim, future_length, hidden_dim, latent_dim, timesteps,
        transformer_heads=4, transformer_layers=2, dropout=0.1,
    ):
        super().__init__()
        if timesteps < 2 or input_dim != 1:
            raise ValueError("Residual diffusion requires timesteps >= 2 and input_dim = 1.")
        self.encoder = EpisodeEncoder(
            input_dim, hidden_dim, latent_dim, transformer_heads, transformer_layers, dropout
        )
        self.noise_predictor = NoisePredictor(future_length, latent_dim, hidden_dim)
        self.timesteps = timesteps
        # Rescale sqrt(alpha_bar) so the terminal distribution is exactly N(0,I).
        original = torch.cumprod(1 - torch.linspace(1e-4, 0.02, timesteps, dtype=torch.float64), 0).sqrt()
        signal = (original - original[-1]) * original[0] / (original[0] - original[-1])
        alpha_bars = signal.square()
        previous = torch.cat([torch.ones(1, dtype=torch.float64), alpha_bars[:-1]])
        alphas = alpha_bars / previous
        betas = 1 - alphas
        self.register_buffer("posterior_variance", (betas * (1 - previous) / (1 - alpha_bars)).float())
        self.register_buffer("posterior_x0_coef", (betas * previous.sqrt() / (1 - alpha_bars)).float())
        self.register_buffer("posterior_xt_coef", ((1 - previous) * alphas.sqrt() / (1 - alpha_bars)).float())
        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas", alphas.float())
        self.register_buffer("alpha_bars", alpha_bars.float())

    def loss(self, history, future):
        # Both inputs are standardized using the same training statistics.
        # This is (future_dBm - last_history_dBm) / training_std.
        future = future - history[:, -1:, :]
        batch_size = future.shape[0]
        steps = torch.randint(0, self.timesteps, (batch_size,), device=future.device)
        alpha_bar = self.alpha_bars[steps].view(batch_size, 1, 1)
        noise = torch.randn_like(future)
        noisy_future = alpha_bar.sqrt() * future + (1.0 - alpha_bar).sqrt() * noise
        context = self.encoder(history)
        predicted_velocity = self.noise_predictor(
            noisy_future, steps / max(self.timesteps - 1, 1), context
        )
        target_velocity = alpha_bar.sqrt() * noise - (1 - alpha_bar).sqrt() * future
        return nn.functional.mse_loss(predicted_velocity, target_velocity)

    @torch.no_grad()
    def sample(self, history, num_samples=1):
        """Generate residuals, then return absolute standardized RSRP [B,S,F,1].

        Use eval() when sampling to disable dropout. v prediction and the x0
        posterior form avoid division by alpha=0 at the terminal step.
        """
        if num_samples < 1:
            raise ValueError("num_samples must be positive.")
        batch_size = history.shape[0]
        context = self.encoder(history).repeat_interleave(num_samples, dim=0)
        x = torch.randn(
            batch_size * num_samples,
            self.noise_predictor.sequence_length,
            1,
            device=history.device,
        )
        for step in reversed(range(self.timesteps)):
            t = torch.full((x.shape[0],), step / max(self.timesteps - 1, 1), device=x.device)
            velocity = self.noise_predictor(x, t, context)
            alpha_bar = self.alpha_bars[step]
            residual = alpha_bar.sqrt() * x - (1 - alpha_bar).sqrt() * velocity
            x = self.posterior_x0_coef[step] * residual + self.posterior_xt_coef[step] * x
            if step > 0:
                x = x + self.posterior_variance[step].sqrt() * torch.randn_like(x)
        return x.view(batch_size, num_samples, -1, 1) + history[:, None, -1:, :]
