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
        encoded_history = self.transformer(tokens)
        # The final token summarizes the past up to the forecast start.
        return self.projection(encoded_history[:, -1, :])


class NoisePredictor(nn.Module):
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
    def __init__(
        self, input_dim, future_length, hidden_dim, latent_dim, timesteps,
        transformer_heads=4, transformer_layers=2, dropout=0.1,
    ):
        super().__init__()
        self.encoder = EpisodeEncoder(
            input_dim, hidden_dim, latent_dim, transformer_heads, transformer_layers, dropout
        )
        self.noise_predictor = NoisePredictor(future_length, latent_dim, hidden_dim)
        self.timesteps = timesteps
        betas = torch.linspace(1e-4, 0.02, timesteps)
        alphas = 1.0 - betas
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", torch.cumprod(alphas, dim=0))

    def loss(self, history, future):
        batch_size = future.shape[0]
        steps = torch.randint(0, self.timesteps, (batch_size,), device=future.device)
        alpha_bar = self.alpha_bars[steps].view(batch_size, 1, 1)
        noise = torch.randn_like(future)
        noisy_future = alpha_bar.sqrt() * future + (1.0 - alpha_bar).sqrt() * noise
        context = self.encoder(history)
        predicted_noise = self.noise_predictor(
            noisy_future, steps / max(self.timesteps - 1, 1), context
        )
        return nn.functional.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def sample(self, history, num_samples=1):
        """Generate normalized future trajectories using DDPM reverse diffusion."""
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
            predicted_noise = self.noise_predictor(x, t, context)
            alpha = self.alphas[step]
            alpha_bar = self.alpha_bars[step]
            x = (x - (1 - alpha) / (1 - alpha_bar).sqrt() * predicted_noise) / alpha.sqrt()
            if step > 0:
                x = x + self.betas[step].sqrt() * torch.randn_like(x)
        return x.view(batch_size, num_samples, -1, 1)
