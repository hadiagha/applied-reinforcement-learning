#!/usr/bin/env python3
"""
Chapter 9: Data Center Cooling Control with REINFORCE and Transformers
=======================================================================
This module demonstrates solving a data-center cooling control problem using:
  - A Transformer-based policy network (attention over multiple cooling zones)
  - REINFORCE (policy gradient) with a learned value baseline for variance reduction
  - Batch training with vectorized environments for GPU efficiency
"""

import os
import math
import random
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class Config:
    """Central configuration for the data center cooling RL experiment."""
    # Environment Settings
    n_zones: int = 8
    horizon: int = 96
    dt: float = 0.25
    temp_min: float = 18.0
    temp_max: float = 27.0
    temp_target: float = 22.0
    temp_init_mean: float = 22.0
    temp_init_std: float = 1.0
    dp_min: float = 5.0
    dp_max: float = 25.0
    dp_target: float = 15.0
    dp_init_mean: float = 15.0
    dp_init_std: float = 2.0
    load_base: float = 50.0
    load_amplitude: float = 20.0
    load_noise_std: float = 5.0
    tamb_base: float = 30.0
    tamb_amplitude: float = 8.0
    tamb_noise_std: float = 1.0
    ewt_base: float = 12.0
    ewt_noise_std: float = 0.5
    thermal_mass: float = 0.7
    fan_cooling_coeff: float = 3.0
    valve_cooling_coeff: float = 4.0
    load_heating_coeff: float = 0.08
    ambient_leakage: float = 0.02
    zone_coupling: float = 0.1
    fan_pressure_coeff: float = 0.8
    pressure_decay: float = 0.1
    pressure_noise_std: float = 0.5
    action_rate_limit: float = 0.15
    # Reward Weights
    w_energy_fan: float = 0.3
    w_energy_valve: float = 0.2
    w_temp_violation: float = 5.0
    w_pressure_violation: float = 2.0
    softplus_beta: float = 2.0
    # Model Architecture
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 128
    dropout: float = 0.1
    # Training Hyperparameters
    batch_size: int = 32
    n_updates: int = 1000
    gamma: float = 0.99
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    lr_warmup_updates: int = 20
    lr_decay: bool = True
    # Evaluation
    eval_episodes: int = 20
    # Misc
    seed: int = 1042
    output_dir: str = "./outputs"
    device: str = ""

    def __post_init__(self):
        if not self.device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        os.makedirs(self.output_dir, exist_ok=True)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def soft_hinge_penalty(x: torch.Tensor, lower: float, upper: float, beta: float) -> torch.Tensor:
    """Smooth penalty for values outside [lower, upper] using softplus."""
    lower_violation = F.softplus(beta * (lower - x)) / beta
    upper_violation = F.softplus(beta * (x - upper)) / beta
    return lower_violation + upper_violation


# ==============================================================================
# VECTORIZED DATA CENTER ENVIRONMENT
# ==============================================================================

class DataCenterEnv:
    """
    Vectorized Data Center Cooling Environment running B parallel environments.
    State: temperature and differential pressure per zone
    Actions: fan speed and valve opening per zone (continuous [0,1])
    """

    def __init__(self, config: Config, batch_size: Optional[int] = None):
        self.cfg = config
        self.batch_size = batch_size if batch_size is not None else config.batch_size
        self.device = torch.device(config.device)
        self.adjacency = self._create_adjacency_matrix()
        self.temp = None
        self.dp = None
        self.prev_fan = None
        self.prev_valve = None
        self.step_count = None
        self.ewt = None

    def _create_adjacency_matrix(self) -> torch.Tensor:
        n = self.cfg.n_zones
        adj = torch.zeros(n, n, device=self.device)
        for i in range(n):
            adj[i, (i - 1) % n] = 1.0
            adj[i, (i + 1) % n] = 1.0
        adj = adj / adj.sum(dim=1, keepdim=True).clamp(min=1.0)
        return adj

    def reset(self) -> torch.Tensor:
        B, N = self.batch_size, self.cfg.n_zones
        self.temp = torch.normal(self.cfg.temp_init_mean, self.cfg.temp_init_std, (B, N), device=self.device)
        self.dp = torch.normal(self.cfg.dp_init_mean, self.cfg.dp_init_std, (B, N), device=self.device)
        self.prev_fan = torch.full((B, N), 0.5, device=self.device)
        self.prev_valve = torch.full((B, N), 0.5, device=self.device)
        self.ewt = torch.full((B,), self.cfg.ewt_base, device=self.device)
        self.step_count = torch.zeros(B, dtype=torch.long, device=self.device)
        return self._get_observation()

    def _get_disturbances(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N = self.batch_size, self.cfg.n_zones
        t = 2 * math.pi * self.step_count.float() / self.cfg.horizon
        load_pattern = self.cfg.load_base + self.cfg.load_amplitude * torch.cos(t - math.pi).unsqueeze(1)
        zone_variation = torch.linspace(0.8, 1.2, N, device=self.device).unsqueeze(0)
        load = load_pattern * zone_variation + torch.randn(B, N, device=self.device) * self.cfg.load_noise_std
        load = load.clamp(min=10.0)
        tamb = self.cfg.tamb_base + self.cfg.tamb_amplitude * torch.cos(t - math.pi * 0.7)
        tamb = tamb + torch.randn(B, device=self.device) * self.cfg.tamb_noise_std
        self.ewt = self.ewt + torch.randn(B, device=self.device) * self.cfg.ewt_noise_std * 0.1
        self.ewt = self.ewt.clamp(self.cfg.ewt_base - 3, self.cfg.ewt_base + 3)
        return load, tamb, self.ewt

    def _get_observation(self) -> torch.Tensor:
        B, N = self.batch_size, self.cfg.n_zones
        load, tamb, ewt = self._get_disturbances()
        neighbor_temp = torch.matmul(self.temp, self.adjacency.T)
        temp_norm = (self.temp - self.cfg.temp_target) / 10.0
        neighbor_temp_norm = (neighbor_temp - self.cfg.temp_target) / 10.0
        dp_norm = (self.dp - self.cfg.dp_target) / 10.0
        load_norm = (load - self.cfg.load_base) / (self.cfg.load_amplitude * 2)
        tamb_norm = ((tamb - self.cfg.tamb_base) / self.cfg.tamb_amplitude).unsqueeze(1).expand(B, N)
        ewt_norm = ((ewt - self.cfg.ewt_base) / 3.0).unsqueeze(1).expand(B, N)
        prev_fan_norm = self.prev_fan - 0.5
        prev_valve_norm = self.prev_valve - 0.5
        obs = torch.stack([temp_norm, dp_norm, load_norm, neighbor_temp_norm,
                           tamb_norm, ewt_norm, prev_fan_norm, prev_valve_norm], dim=-1)
        return obs

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        B, N = self.batch_size, self.cfg.n_zones
        fan = actions[:, :, 0]
        valve = actions[:, :, 1]
        fan_delta = (fan - self.prev_fan).clamp(-self.cfg.action_rate_limit, self.cfg.action_rate_limit)
        valve_delta = (valve - self.prev_valve).clamp(-self.cfg.action_rate_limit, self.cfg.action_rate_limit)
        fan = (self.prev_fan + fan_delta).clamp(0, 1)
        valve = (self.prev_valve + valve_delta).clamp(0, 1)
        load, tamb, ewt = self._get_disturbances()
        cooling = self.cfg.fan_cooling_coeff * fan + self.cfg.valve_cooling_coeff * valve * (self.temp - ewt.unsqueeze(1)) / 20.0
        heating = self.cfg.load_heating_coeff * load + self.cfg.ambient_leakage * (tamb.unsqueeze(1) - self.temp)
        neighbor_temp = torch.matmul(self.temp, self.adjacency.T)
        coupling = self.cfg.zone_coupling * (neighbor_temp - self.temp)
        temp_change = heating - cooling + coupling
        self.temp = self.cfg.thermal_mass * self.temp + (1 - self.cfg.thermal_mass) * (self.temp + temp_change)
        dp_change = self.cfg.fan_pressure_coeff * fan - self.cfg.pressure_decay * (self.dp - self.cfg.dp_target)
        self.dp = self.dp + dp_change + torch.randn(B, N, device=self.device) * self.cfg.pressure_noise_std
        energy_cost = self.cfg.w_energy_fan * (fan ** 2).mean(dim=1) + self.cfg.w_energy_valve * (valve ** 2).mean(dim=1)
        temp_penalty = soft_hinge_penalty(self.temp, self.cfg.temp_min, self.cfg.temp_max, self.cfg.softplus_beta).mean(dim=1) * self.cfg.w_temp_violation
        pressure_penalty = soft_hinge_penalty(self.dp, self.cfg.dp_min, self.cfg.dp_max, self.cfg.softplus_beta).mean(dim=1) * self.cfg.w_pressure_violation
        rewards = -(energy_cost + temp_penalty + pressure_penalty)
        temp_violations = ((self.temp < self.cfg.temp_min) | (self.temp > self.cfg.temp_max)).float().mean(dim=1)
        dp_violations = ((self.dp < self.cfg.dp_min) | (self.dp > self.cfg.dp_max)).float().mean(dim=1)
        info = {'energy_cost': energy_cost.detach(), 'temp_penalty': temp_penalty.detach(),
                'pressure_penalty': pressure_penalty.detach(), 'temp_violations': temp_violations.detach(),
                'dp_violations': dp_violations.detach(), 'mean_temp': self.temp.mean(dim=1).detach(),
                'mean_dp': self.dp.mean(dim=1).detach(), 'mean_fan': fan.mean(dim=1).detach(),
                'mean_valve': valve.mean(dim=1).detach()}
        self.prev_fan = fan.detach()
        self.prev_valve = valve.detach()
        self.step_count = self.step_count + 1
        dones = self.step_count >= self.cfg.horizon
        obs = self._get_observation()
        return obs, rewards, dones, info


# ==============================================================================
# TRANSFORMER POLICY AND VALUE NETWORKS
# ==============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TransformerPolicyValue(nn.Module):
    """Transformer-based Actor-Critic for Data Center Control."""

    def __init__(self, config: Config):
        super().__init__()
        self.cfg = config
        obs_dim = 8
        action_dim = 2
        self.input_embed = nn.Sequential(nn.Linear(obs_dim, config.d_model), nn.LayerNorm(config.d_model), nn.ReLU())
        self.pos_encoding = PositionalEncoding(config.d_model, max_len=config.n_zones + 1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=config.d_model, nhead=config.n_heads,
                                                    dim_feedforward=config.d_ff, dropout=config.dropout,
                                                    activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        self.policy_head = nn.Sequential(nn.Linear(config.d_model, config.d_model), nn.ReLU(),
                                          nn.Linear(config.d_model, action_dim * 2))
        self.value_head = nn.Sequential(nn.Linear(config.d_model, config.d_model), nn.ReLU(), nn.Linear(config.d_model, 1))
        self.log_std_min = -5.0
        self.log_std_max = 2.0

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.input_embed(obs)
        x = self.pos_encoding(x)
        features = self.transformer(x)
        policy_out = self.policy_head(features)
        action_mean = policy_out[:, :, :2]
        action_log_std = policy_out[:, :, 2:].clamp(self.log_std_min, self.log_std_max)
        action_std = action_log_std.exp()
        pooled = features.mean(dim=1)
        value = self.value_head(pooled).squeeze(-1)
        return action_mean, action_std, value, features

    def get_action_and_value(self, obs: torch.Tensor, deterministic: bool = False):
        action_mean, action_std, values, _ = self.forward(obs)
        dist = Normal(action_mean, action_std)
        raw_actions = action_mean if deterministic else dist.rsample()
        actions = torch.sigmoid(raw_actions)
        eps = 1e-6
        log_prob_raw = dist.log_prob(raw_actions)
        squash_correction = torch.log(actions.clamp(eps, 1-eps)) + torch.log((1 - actions).clamp(eps, 1-eps))
        log_probs = (log_prob_raw + squash_correction).sum(dim=[1, 2])
        entropy = dist.entropy().sum(dim=[1, 2])
        return actions, log_probs, entropy, values

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        action_mean, action_std, values, _ = self.forward(obs)
        eps = 1e-6
        actions_clamped = actions.clamp(eps, 1 - eps)
        raw_actions = torch.log(actions_clamped / (1 - actions_clamped))
        dist = Normal(action_mean, action_std)
        log_prob_raw = dist.log_prob(raw_actions)
        squash_correction = torch.log(actions_clamped) + torch.log(1 - actions_clamped)
        log_probs = (log_prob_raw + squash_correction).sum(dim=[1, 2])
        entropy = dist.entropy().sum(dim=[1, 2])
        return log_probs, entropy, values


# ==============================================================================
# HEURISTIC BASELINE CONTROLLER
# ==============================================================================

class HeuristicController:
    """Simple PI-like controller for comparison."""

    def __init__(self, config: Config):
        self.cfg = config
        self.device = torch.device(config.device)
        self.kp_temp = 0.15
        self.ki_temp = 0.02
        self.kp_dp = 0.1
        self.temp_deadband = 1.0
        self.dp_deadband = 3.0
        self.integral_error = None

    def reset(self, batch_size: int):
        self.integral_error = torch.zeros(batch_size, self.cfg.n_zones, device=self.device)

    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        temp = obs[:, :, 0] * 10.0 + self.cfg.temp_target
        dp = obs[:, :, 1] * 10.0 + self.cfg.dp_target
        temp_error = temp - self.cfg.temp_target
        temp_error_db = torch.where(temp_error.abs() < self.temp_deadband, torch.zeros_like(temp_error), temp_error)
        if self.integral_error is None:
            self.integral_error = torch.zeros_like(temp_error)
        self.integral_error = (self.integral_error + temp_error_db * 0.1).clamp(-10, 10)
        cooling_demand = self.kp_temp * temp_error_db + self.ki_temp * self.integral_error
        dp_error = dp - self.cfg.dp_target
        dp_error_db = torch.where(dp_error.abs() < self.dp_deadband, torch.zeros_like(dp_error), dp_error)
        fan = (0.5 + cooling_demand * 0.5 - self.kp_dp * dp_error_db).clamp(0.1, 0.9)
        valve = (0.5 + cooling_demand * 0.6).clamp(0.1, 0.9)
        return torch.stack([fan, valve], dim=-1)


# ==============================================================================
# ROLLOUT BUFFER
# ==============================================================================

class RolloutBuffer:
    def __init__(self, config: Config):
        self.cfg = config
        self.device = torch.device(config.device)
        self.clear()

    def clear(self):
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.entropies = []

    def add(self, obs, action, log_prob, reward, value, done, entropy):
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.entropies.append(entropy)

    def compute_returns_and_advantages(self, last_value: torch.Tensor):
        T = len(self.rewards)
        B = self.rewards[0].shape[0]
        rewards = torch.stack(self.rewards)
        values = torch.stack(self.values)
        dones = torch.stack(self.dones)
        returns = torch.zeros(T, B, device=self.device)
        next_return = last_value * (1 - dones[-1].float())
        for t in reversed(range(T)):
            returns[t] = rewards[t] + self.cfg.gamma * next_return * (1 - dones[t].float())
            next_return = returns[t]
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    def get_tensors(self):
        return {'observations': torch.stack(self.observations), 'actions': torch.stack(self.actions),
                'log_probs': torch.stack(self.log_probs), 'entropies': torch.stack(self.entropies)}


# ==============================================================================
# TRAINING
# ==============================================================================

def train(config: Config):
    print(f"Training on device: {config.device}")
    print(f"Batch size: {config.batch_size}, Horizon: {config.horizon}, Zones: {config.n_zones}")
    print("-" * 60)
    env = DataCenterEnv(config)
    model = TransformerPolicyValue(config).to(config.device)
    optimizer = torch.optim.Adam([
        {'params': model.input_embed.parameters(), 'lr': config.lr_actor},
        {'params': model.pos_encoding.parameters(), 'lr': config.lr_actor},
        {'params': model.transformer.parameters(), 'lr': config.lr_actor},
        {'params': model.policy_head.parameters(), 'lr': config.lr_actor},
        {'params': model.value_head.parameters(), 'lr': config.lr_critic},
    ])

    def lr_lambda(update):
        if update < config.lr_warmup_updates:
            return (update + 1) / config.lr_warmup_updates
        elif config.lr_decay:
            progress = (update - config.lr_warmup_updates) / max(1, config.n_updates - config.lr_warmup_updates)
            return 0.5 * (1 + math.cos(math.pi * progress))
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    history = {'episode_return': [], 'policy_loss': [], 'value_loss': [], 'entropy': [],
               'temp_violations': [], 'energy_cost': [], 'lr': []}
    buffer = RolloutBuffer(config)

    for update in range(config.n_updates):
        buffer.clear()
        obs = env.reset()
        episode_rewards, episode_violations, episode_energy = [], [], []
        for step in range(config.horizon):
            with torch.no_grad():
                actions, log_probs, entropy, values = model.get_action_and_value(obs)
            next_obs, rewards, dones, info = env.step(actions)
            buffer.add(obs, actions, log_probs, rewards, values, dones, entropy)
            episode_rewards.append(rewards.mean().item())
            episode_violations.append(info['temp_violations'].mean().item())
            episode_energy.append(info['energy_cost'].mean().item())
            obs = next_obs
        with torch.no_grad():
            _, _, _, last_value = model.get_action_and_value(obs)
        returns, advantages = buffer.compute_returns_and_advantages(last_value)
        data = buffer.get_tensors()
        T, B = returns.shape
        obs_flat = data['observations'].view(T * B, config.n_zones, -1)
        actions_flat = data['actions'].view(T * B, config.n_zones, 2)
        returns_flat = returns.view(T * B)
        advantages_flat = advantages.view(T * B)
        # Normalize returns for value loss to keep loss scale reasonable
        returns_mean = returns_flat.mean()
        returns_std = returns_flat.std() + 1e-8
        returns_normalized = (returns_flat - returns_mean) / returns_std
        log_probs, entropy, values = model.evaluate_actions(obs_flat, actions_flat)
        policy_loss = -(log_probs * advantages_flat.detach()).mean()
        # Value loss on normalized returns (critic predicts normalized values)
        value_loss = F.mse_loss(values, returns_normalized.detach())
        entropy_loss = -entropy.mean()
        total_loss = policy_loss + config.value_coef * value_loss + config.entropy_coef * entropy_loss
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        scheduler.step()
        episode_return = sum(episode_rewards)
        history['episode_return'].append(episode_return)
        history['policy_loss'].append(policy_loss.item())
        history['value_loss'].append(value_loss.item())
        history['entropy'].append(-entropy_loss.item())
        history['temp_violations'].append(np.mean(episode_violations))
        history['energy_cost'].append(np.mean(episode_energy))
        history['lr'].append(scheduler.get_last_lr()[0])
        if (update + 1) % 10 == 0 or update == 0:
            print(f"Update {update+1:4d}/{config.n_updates} | Return: {episode_return:7.2f} | "
                  f"P_Loss: {policy_loss.item():7.4f} | V_Loss: {value_loss.item():7.4f} | "
                  f"Entropy: {-entropy_loss.item():6.2f} | Violations: {np.mean(episode_violations):.3f}")
    print("-" * 60)
    print("Training complete!")
    return model, history


# ==============================================================================
# EVALUATION
# ==============================================================================

def evaluate_policy(model, config: Config, n_episodes: int, deterministic: bool = True):
    env = DataCenterEnv(config, batch_size=n_episodes)
    model.eval()
    obs = env.reset()
    episode_return = torch.zeros(n_episodes, device=config.device)
    episode_energy = torch.zeros(n_episodes, device=config.device)
    episode_temp_viol = torch.zeros(n_episodes, device=config.device)
    all_temps, all_fans, all_valves = [], [], []
    with torch.no_grad():
        for _ in range(config.horizon):
            actions, _, _, _ = model.get_action_and_value(obs, deterministic=deterministic)
            obs, rewards, dones, info = env.step(actions)
            episode_return += rewards
            episode_energy += info['energy_cost']
            episode_temp_viol += info['temp_violations']
            all_temps.append(info['mean_temp'].cpu().numpy())
            all_fans.append(info['mean_fan'].cpu().numpy())
            all_valves.append(info['mean_valve'].cpu().numpy())
    model.train()
    return {'returns': episode_return.cpu().numpy(), 'energy_costs': episode_energy.cpu().numpy() / config.horizon,
            'temp_violations': episode_temp_viol.cpu().numpy() / config.horizon,
            'mean_temps': np.array(all_temps), 'mean_fans': np.array(all_fans), 'mean_valves': np.array(all_valves)}


def evaluate_heuristic(config: Config, n_episodes: int):
    env = DataCenterEnv(config, batch_size=n_episodes)
    controller = HeuristicController(config)
    controller.reset(n_episodes)
    obs = env.reset()
    episode_return = torch.zeros(n_episodes, device=config.device)
    episode_energy = torch.zeros(n_episodes, device=config.device)
    episode_temp_viol = torch.zeros(n_episodes, device=config.device)
    all_temps, all_fans, all_valves = [], [], []
    for _ in range(config.horizon):
        actions = controller.get_action(obs)
        obs, rewards, dones, info = env.step(actions)
        episode_return += rewards
        episode_energy += info['energy_cost']
        episode_temp_viol += info['temp_violations']
        all_temps.append(info['mean_temp'].cpu().numpy())
        all_fans.append(info['mean_fan'].cpu().numpy())
        all_valves.append(info['mean_valve'].cpu().numpy())
    return {'returns': episode_return.cpu().numpy(), 'energy_costs': episode_energy.cpu().numpy() / config.horizon,
            'temp_violations': episode_temp_viol.cpu().numpy() / config.horizon,
            'mean_temps': np.array(all_temps), 'mean_fans': np.array(all_fans), 'mean_valves': np.array(all_valves)}


def run_single_episode_rollout(model, config: Config, deterministic: bool = True):
    env = DataCenterEnv(config, batch_size=1)
    model.eval()
    trajectory = {'temps': [], 'dps': [], 'fans': [], 'valves': [], 'rewards': [], 'energy_costs': [], 'temp_penalties': []}
    obs = env.reset()
    with torch.no_grad():
        for _ in range(config.horizon):
            trajectory['temps'].append(env.temp[0].cpu().numpy())
            trajectory['dps'].append(env.dp[0].cpu().numpy())
            actions, _, _, _ = model.get_action_and_value(obs, deterministic=deterministic)
            obs, rewards, dones, info = env.step(actions)
            trajectory['fans'].append(actions[0, :, 0].cpu().numpy())
            trajectory['valves'].append(actions[0, :, 1].cpu().numpy())
            trajectory['rewards'].append(rewards[0].item())
            trajectory['energy_costs'].append(info['energy_cost'][0].item())
            trajectory['temp_penalties'].append(info['temp_penalty'][0].item())
    for key in trajectory:
        trajectory[key] = np.array(trajectory[key])
    model.train()
    return trajectory


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def plot_training_curves(history, config: Config):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle('REINFORCE Training Progress - Data Center Cooling', fontsize=14)
    updates = range(1, len(history['episode_return']) + 1)
    cmap = plt.cm.tab10
    ax = axes[0, 0]
    ax.plot(updates, history['episode_return'], color=cmap(0), alpha=0.3, linewidth=0.5)
    window = min(20, len(history['episode_return']) // 5) if len(history['episode_return']) > 5 else 1
    if window > 1:
        smoothed = np.convolve(history['episode_return'], np.ones(window)/window, mode='valid')
        ax.plot(range(window, len(history['episode_return']) + 1), smoothed, color=cmap(0), linewidth=2)
    ax.set_xlabel('Update'); ax.set_ylabel('Episode Return'); ax.set_title('Episode Return'); ax.grid(True, alpha=0.3)
    ax = axes[0, 1]
    ax.plot(updates, history['policy_loss'], color=cmap(1), alpha=0.5)
    ax.set_xlabel('Update'); ax.set_ylabel('Policy Loss'); ax.set_title('Policy Loss'); ax.grid(True, alpha=0.3)
    ax = axes[0, 2]
    ax.plot(updates, history['value_loss'], color=cmap(2), alpha=0.5)
    ax.set_xlabel('Update'); ax.set_ylabel('Value Loss'); ax.set_title('Value Loss'); ax.grid(True, alpha=0.3)
    ax = axes[1, 0]
    ax.plot(updates, history['entropy'], color=cmap(3), alpha=0.5)
    ax.set_xlabel('Update'); ax.set_ylabel('Entropy'); ax.set_title('Policy Entropy'); ax.grid(True, alpha=0.3)
    ax = axes[1, 1]
    ax.plot(updates, history['temp_violations'], color=cmap(4), alpha=0.5)
    ax.set_xlabel('Update'); ax.set_ylabel('Violation Rate'); ax.set_title('Temperature Violations'); ax.grid(True, alpha=0.3)
    ax = axes[1, 2]
    ax.plot(updates, history['energy_cost'], color=cmap(5), alpha=0.5)
    ax.set_xlabel('Update'); ax.set_ylabel('Energy Cost'); ax.set_title('Energy Cost'); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.output_dir, 'training_curves.png'), dpi=150)
    plt.close()
    print(f"Saved training curves to {config.output_dir}/training_curves.png")


def plot_episode_rollout(trajectory, config: Config):
    import matplotlib.pyplot as plt
    T = len(trajectory['rewards'])
    time_steps = np.arange(T)
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('Single Episode Rollout - RL Policy', fontsize=14)
    cmap = plt.cm.viridis
    n_zones = trajectory['temps'].shape[1]
    colors = [cmap(i / n_zones) for i in range(n_zones)]
    ax = axes[0, 0]
    for z in range(n_zones):
        ax.plot(time_steps, trajectory['temps'][:, z], color=colors[z], alpha=0.7, label=f'Zone {z}')
    ax.axhline(config.temp_min, color='red', linestyle='--', alpha=0.5, label='Bounds')
    ax.axhline(config.temp_max, color='red', linestyle='--', alpha=0.5)
    ax.axhline(config.temp_target, color='green', linestyle=':', alpha=0.5, label='Target')
    ax.set_xlabel('Time Step'); ax.set_ylabel('Temperature (°C)'); ax.set_title('Zone Temperatures'); ax.grid(True, alpha=0.3)
    ax = axes[0, 1]
    for z in range(n_zones):
        ax.plot(time_steps, trajectory['dps'][:, z], color=colors[z], alpha=0.7)
    ax.axhline(config.dp_min, color='red', linestyle='--', alpha=0.5)
    ax.axhline(config.dp_max, color='red', linestyle='--', alpha=0.5)
    ax.axhline(config.dp_target, color='green', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time Step'); ax.set_ylabel('Pressure (Pa)'); ax.set_title('Differential Pressure'); ax.grid(True, alpha=0.3)
    ax = axes[1, 0]
    for z in range(n_zones):
        ax.plot(time_steps, trajectory['fans'][:, z], color=colors[z], alpha=0.7)
    ax.set_xlabel('Time Step'); ax.set_ylabel('Fan Speed'); ax.set_title('Fan Actions'); ax.set_ylim(-0.05, 1.05); ax.grid(True, alpha=0.3)
    ax = axes[1, 1]
    for z in range(n_zones):
        ax.plot(time_steps, trajectory['valves'][:, z], color=colors[z], alpha=0.7)
    ax.set_xlabel('Time Step'); ax.set_ylabel('Valve Opening'); ax.set_title('Valve Actions'); ax.set_ylim(-0.05, 1.05); ax.grid(True, alpha=0.3)
    ax = axes[2, 0]
    ax.plot(time_steps, trajectory['rewards'], color=plt.cm.tab10(0))
    ax.set_xlabel('Time Step'); ax.set_ylabel('Reward'); ax.set_title('Step Rewards'); ax.grid(True, alpha=0.3)
    ax = axes[2, 1]
    ax.plot(time_steps, trajectory['energy_costs'], color=plt.cm.tab10(1), label='Energy')
    ax.plot(time_steps, trajectory['temp_penalties'], color=plt.cm.tab10(3), label='Temp Penalty')
    ax.set_xlabel('Time Step'); ax.set_ylabel('Cost'); ax.set_title('Cost Components'); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.output_dir, 'episode_rollout.png'), dpi=150)
    plt.close()
    print(f"Saved episode rollout to {config.output_dir}/episode_rollout.png")


def plot_comparison(rl_results, heuristic_results, config: Config):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle('RL Policy vs Heuristic Controller', fontsize=14)
    cmap = plt.cm.tab10
    metrics = [('returns', 'Episode Return'), ('energy_costs', 'Avg Energy Cost'), ('temp_violations', 'Violation Rate')]
    for idx, (key, title) in enumerate(metrics):
        ax = axes[idx]
        rl_data = rl_results[key]
        heur_data = heuristic_results[key]
        positions = [0, 1]
        bp = ax.boxplot([rl_data, heur_data], positions=positions, widths=0.6, patch_artist=True)
        bp['boxes'][0].set_facecolor(cmap(0))
        bp['boxes'][1].set_facecolor(cmap(1))
        ax.set_xticks(positions)
        ax.set_xticklabels(['RL Policy', 'Heuristic'])
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(config.output_dir, 'comparison.png'), dpi=150)
    plt.close()
    print(f"Saved comparison plot to {config.output_dir}/comparison.png")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    config = Config()
    set_seed(config.seed)
    print("=" * 60)
    print("Data Center Cooling Control with REINFORCE + Transformer")
    print("=" * 60)
    print(f"Device: {config.device}")
    print(f"Zones: {config.n_zones}, Horizon: {config.horizon}, Batch: {config.batch_size}")
    print(f"Updates: {config.n_updates}, Gamma: {config.gamma}")
    print("=" * 60)

    # Train
    model, history = train(config)

    # Plot training curves
    plot_training_curves(history, config)

    # Evaluate RL policy
    print("\nEvaluating RL policy...")
    rl_results = evaluate_policy(model, config, config.eval_episodes)
    print(f"  RL Avg Return: {rl_results['returns'].mean():.2f} ± {rl_results['returns'].std():.2f}")
    print(f"  RL Avg Energy: {rl_results['energy_costs'].mean():.4f}")
    print(f"  RL Violation Rate: {rl_results['temp_violations'].mean():.4f}")

    # Evaluate heuristic
    print("\nEvaluating heuristic controller...")
    heuristic_results = evaluate_heuristic(config, config.eval_episodes)
    print(f"  Heuristic Avg Return: {heuristic_results['returns'].mean():.2f} ± {heuristic_results['returns'].std():.2f}")
    print(f"  Heuristic Avg Energy: {heuristic_results['energy_costs'].mean():.4f}")
    print(f"  Heuristic Violation Rate: {heuristic_results['temp_violations'].mean():.4f}")

    # Plot comparison
    plot_comparison(rl_results, heuristic_results, config)

    # Single episode rollout visualization
    print("\nGenerating episode rollout visualization...")
    trajectory = run_single_episode_rollout(model, config)
    plot_episode_rollout(trajectory, config)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    rl_return = rl_results['returns'].mean()
    heur_return = heuristic_results['returns'].mean()
    improvement = ((rl_return - heur_return) / abs(heur_return)) * 100 if heur_return != 0 else 0
    print(f"RL Policy Return:        {rl_return:.2f}")
    print(f"Heuristic Return:        {heur_return:.2f}")
    print(f"Improvement:             {improvement:.1f}%")
    print(f"RL Energy Cost:          {rl_results['energy_costs'].mean():.4f}")
    print(f"Heuristic Energy Cost:   {heuristic_results['energy_costs'].mean():.4f}")
    print(f"RL Violation Rate:       {rl_results['temp_violations'].mean():.4f}")
    print(f"Heuristic Violation Rate:{heuristic_results['temp_violations'].mean():.4f}")
    print("=" * 60)
    print(f"Plots saved to: {config.output_dir}/")


if __name__ == "__main__":
    main()
