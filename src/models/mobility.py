"""Episode encoder and stochastic waypoint policy, without event classifiers."""
import math

import torch
from torch import nn
from torch.distributions import Normal


class MobilityModel(nn.Module):
    FORMAT = 'ns3_waypoint_policy_v4'

    def __init__(self, feature_dim, points, slots, profiles, sites, duration_s,
                 waypoint_interval_s=.5, hidden_dim=32, latent_dim=96,
                 num_heads=4, num_layers=1, action_std=.35,
                 bounds=((-100., 1100.), (-200., 200.))):
        super().__init__()
        self.slots, self.profiles = slots, profiles
        self.duration_s, self.latent_dim = duration_s, latent_dim
        self.segments = math.ceil(duration_s / waypoint_interval_s)
        self.register_buffer('times', torch.linspace(0, duration_s, self.segments + 1))
        self.register_buffer('bounds', torch.tensor(bounds, dtype=torch.float32))
        self.register_buffer('sites', torch.tensor(sites, dtype=torch.float32))
        self.input = nn.Linear(feature_dim * 2, hidden_dim)
        self.position = nn.Parameter(torch.randn(1, points, hidden_dim) * .01)
        layer = nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 4,
                                          dropout=0., batch_first=True, activation='gelu')
        self.temporal = nn.TransformerEncoder(layer, num_layers)
        self.encoder_head = nn.Sequential(nn.Linear(len(slots) * points * hidden_dim, 128), nn.SiLU(),
                                          nn.Linear(128, latent_dim))
        self.action_dim = len(slots) * (3 + 3 * self.segments)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 128), nn.SiLU(), nn.Linear(128, self.action_dim))
        nn.init.normal_(self.decoder[-1].weight, std=.005)
        nn.init.zeros_(self.decoder[-1].bias)
        # Initial points near the deployment; motion starts without gratuitous turns/stops.
        with torch.no_grad():
            bias = self.decoder[-1].bias.view(len(slots), -1)
            center = self.sites[:, :2].mean(0)
            p = ((center - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])).clamp(.01, .99)
            bias[:, :2] = torch.logit(p)
            bias[:, 3:].view(len(slots), -1, 3)[..., 2] = -1.
        self.action_std = action_std
        used = torch.ones(len(slots), 3 + 3 * self.segments)
        for i, profile in enumerate(profiles):
            if profile['max_speed_mps'] == 0:
                used[i, 2:] = 0
            elif not profile['allow_stop']:
                used[i, 5::3] = 0
        self.register_buffer('action_mask', used.flatten())

    def encode(self, batch):
        x, mask = batch['values'], batch['mask']
        b, n, t, _ = x.shape
        x = torch.cat([x.masked_fill(~mask, 0), mask.float()], -1)
        tokens = self.input(x).reshape(b*n, t, -1) + self.position
        return self.encoder_head(self.temporal(tokens).reshape(b, -1))

    def distribution(self, z):
        return Normal(self.decoder(z), self.action_std)

    def sample(self, z, deterministic=False):
        dist = self.distribution(z)
        action = dist.mean.detach() if deterministic else dist.sample()
        log_prob = (dist.log_prob(action) * self.action_mask).sum(-1)
        return action, log_prob

    def decode(self, action):
        """Return [batch, UE, waypoint, (t,x,y)]; equal positions encode stops."""
        raw = action.reshape(-1, len(self.slots), 3 + 3*self.segments)
        initial = self.bounds[:, 0] + raw[..., :2].sigmoid() * (self.bounds[:, 1] - self.bounds[:, 0])
        params = raw[..., 3:].reshape(-1, len(self.slots), self.segments, 3)
        dt = float(self.times[1])
        velocity = []
        for i, profile in enumerate(self.profiles):
            turns = params[:, i, :, 0]
            heading = raw[:, i, 2:3] + turns.cumsum(-1)
            speed = profile['min_moving_speed_mps'] + params[:, i, :, 1].sigmoid() * (
                profile['max_speed_mps'] - profile['min_moving_speed_mps'])
            if profile['allow_stop']:
                speed = speed * (params[:, i, :, 2] < 0)
            velocity.append(torch.stack([heading.cos(), heading.sin()], -1) * speed[..., None])
        velocity = torch.stack(velocity, 1)
        xy = torch.cat([initial[:, :, None], initial[:, :, None] + (velocity * dt).cumsum(2)], 2)
        times = self.times[None, None, :, None].expand(*xy.shape[:-1], 1)
        return torch.cat([times, xy], -1)

