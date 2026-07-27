"""
Chapter 12 -- Step 2: The RL Portfolio Agent (PPO), trained WITHOUT LLM signals
===============================================================================
Manning Publications -- Applied Reinforcement Learning (Capstone Chapter)

This script trains the reinforcement-learning portfolio manager on the
environment from Step 1 (`ch12_trading_env.py`). It is the FIRST rung of the
chapter's five-mode comparison ladder:

    (this file)  RL, price/portfolio features only  -- news block is all zeros
    later        RL + naive sentiment signals
    later        RL + base (off-the-shelf) LLM signals
    later        RL + fine-tuned LLM signals
    baselines    equal-weight  and  buy-and-hold SPY   (no learning)

We deliberately keep the LLM out of the picture here. The goal of Step 2 is a
stable, understandable RL-only agent that clearly beats (or at least matches)
the non-learning baselines, so that when we later switch the news block ON we
can attribute any change to the *signal quality* -- nothing else.

WHY PPO (see the chapter's Step-2 brainstorm)
---------------------------------------------
Portfolio management is a sequential decision problem with a dense daily reward
and a small discrete action set (8 allocation templates). PPO -- introduced in
Chapter 10 -- is the natural fit: a policy-gradient method whose clipped
surrogate objective and value baseline keep updates stable on noisy financial
rewards, while a categorical policy handles the discrete templates directly.
This reuses the exact machinery readers met in Chapter 10 (clipped ratio, GAE,
value head, entropy bonus), so nothing here is new except the application.

METHODOLOGY (honest, leakage-free)
----------------------------------
- CHRONOLOGICAL train/test split: the agent trains only on the EARLIER portion
  of the price history and is evaluated only on the LATER, unseen portion. It
  never samples an episode that starts in the test window.
- NO LOOK-AHEAD: inherited from the environment -- the state at day t uses only
  data up to t; the reward uses the t -> t+1 return, realized after the action.
- Same evaluation start days are used for the RL agent AND both baselines, so
  the comparison is apples-to-apples.
- This is a TEACHING system, not a profit claim.

Runs on CPU or GPU unchanged (device auto-detected). The network is tiny, so
CPU is perfectly fine for Step 2; the GPU matters only for the LLM steps later.
"""

import argparse
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from ch12_trading_env import (
    ACTION_NAMES,
    ALL_TICKERS,
    ASSET_TICKERS,
    CASH_TICKER,
    EnvConfig,
    MarketSignal,
    PortfolioTradingEnv,
    generate_synthetic_market_data,
    set_seed,
)


# ============================================================================
# CONFIGURATION
# ============================================================================
# One dataclass for the agent + training knobs, in the Chapter 9-11 style. The
# environment keeps its own EnvConfig; here we only configure the RL side.

@dataclass
class PPOConfig:
    """Configuration for the PPO portfolio agent and its training loop."""

    # --- data / split ---
    n_days: int = 1000                 # length of the synthetic history (~4 trading years)
    train_frac: float = 0.70           # first 70% of days = train, last 30% = test (chronological)

    # --- model architecture ---
    hidden_dim: int = 128              # width of the actor-critic MLP
    n_hidden_layers: int = 2

    # --- rollout collection ---
    rollout_episodes: int = 16         # episodes gathered per policy update
    n_updates: int = 300               # total PPO updates (1 update = rollout + optimize)

    # --- PPO objective (same names/roles as Chapter 10) ---
    gamma: float = 0.99                # discount factor
    gae_lambda: float = 0.95           # GAE bias-variance tradeoff
    clip_epsilon: float = 0.2          # clipped-surrogate trust region
    value_coef: float = 0.5            # weight on the value (critic) loss
    entropy_coef: float = 0.01         # weight on the entropy bonus (exploration)
    ppo_epochs: int = 4                # optimization passes over each rollout batch
    minibatch_size: int = 256          # SGD minibatch size within a PPO epoch

    # --- optimization ---
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5

    # --- evaluation ---
    eval_stride: int = 2               # evaluate on every k-th valid test start day

    # --- logging / output ---
    log_every: int = 10                # print a summary line every k updates
    output_dir: str = "./outputs_rl_no_llm"

    # --- reproducibility & device ---
    seed: int = 42
    device: str = ""                   # "" => auto (cuda if available else cpu)

    def __post_init__(self):
        if not self.device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        os.makedirs(self.output_dir, exist_ok=True)


# ============================================================================
# ACTOR-CRITIC NETWORK
# ============================================================================
# A small shared-trunk MLP with two heads, mirroring the actor-critic design of
# Chapter 9's TransformerPolicyValue (policy head + value head on shared
# features) but with a plain MLP trunk suited to our 62-dim tabular state.


class ActorCritic(nn.Module):
    """Shared-trunk actor-critic for a discrete (categorical) policy."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128,
                 n_hidden_layers: int = 2):
        super().__init__()
        # Shared trunk: state -> hidden features.
        layers: List[nn.Module] = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        self.trunk = nn.Sequential(*layers)

        # Two heads: policy logits (actor) and a scalar state value (critic).
        self.policy_head = nn.Linear(hidden_dim, action_dim)   #A actor: one logit per template
        self.value_head = nn.Linear(hidden_dim, 1)             #B critic: expected return baseline

        self._init_weights()

    def _init_weights(self):
        # Orthogonal init is a small but real PPO stability trick: a large gain in
        # the trunk preserves signal, a tiny gain (0.01) on the policy head keeps
        # the initial policy near-uniform so early exploration is broad.
        for m in self.trunk:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)  #C near-uniform initial policy
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (action logits, state value) for a batch of states."""
        features = self.trunk(state)
        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return logits, value

    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        """Sample (or argmax) an action and return (action, log_prob, value)."""
        logits, value = self.forward(state)
        dist = Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(logits, dim=-1)   #D greedy: used at evaluation
        else:
            action = dist.sample()                  #E stochastic: used while training
        return action, dist.log_prob(action), value

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor):
        """Re-score stored (state, action) pairs under the CURRENT policy.

        Returns (log_probs, entropy, values) -- exactly the quantities PPO's
        clipped objective, entropy bonus, and value loss need.
        """
        logits, values = self.forward(states)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), values

#A One output per allocation template; softmax over these is the policy
#B Single scalar estimating the state's value, the PPO advantage baseline
#C Small gain => logits near zero => nearly uniform action probabilities at start
#D Deterministic decoding makes evaluation reproducible
#E Sampling drives on-policy exploration during rollouts


# ============================================================================
# GENERALIZED ADVANTAGE ESTIMATION (GAE) -- same formulation as Chapter 10
# ============================================================================

def compute_gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray,
                last_value: float, gamma: float, gae_lambda: float
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Compute GAE advantages and returns for ONE episode.

    delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
    A_t     = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
    return_t = A_t + V(s_t)

    lambda trades bias vs variance: 0 => low-variance TD error, 1 => Monte Carlo.
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float64)
    gae = 0.0
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * (1.0 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1.0 - dones[t]) * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


# ============================================================================
# PPO TRAINER
# ============================================================================


class PPOTrainer:
    """Collects rollouts on the trading env and updates the policy with PPO."""

    def __init__(self, train_env: PortfolioTradingEnv, config: PPOConfig):
        self.env = train_env
        self.cfg = config
        self.device = torch.device(config.device)

        self.model = ActorCritic(
            state_dim=train_env.state_dim,
            action_dim=train_env.action_dim,
            hidden_dim=config.hidden_dim,
            n_hidden_layers=config.n_hidden_layers,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)

        # Valid range of episode start days WITHIN THE TRAIN ENV only (no leakage).
        self.train_starts = self._valid_start_days(train_env)

        # Training history (metric lists), Chapter 9-11 logging style.
        self.history: Dict[str, List[float]] = {
            "episode_reward": [], "total_return": [], "max_drawdown": [],
            "turnover": [], "policy_loss": [], "value_loss": [],
            "entropy": [], "approx_kl": [],
        }

    @staticmethod
    def _valid_start_days(env: PortfolioTradingEnv) -> List[int]:
        """Start days that leave room for the lookback behind and the episode ahead."""
        lo = env.cfg.lookback
        hi = env.n_days - env.cfg.horizon - 1
        return list(range(lo, hi + 1))

    # ------------------------------------------------------------------
    # ROLLOUT COLLECTION
    # ------------------------------------------------------------------
    @torch.no_grad()
    def collect_rollouts(self) -> Dict[str, torch.Tensor]:
        """Run `rollout_episodes` on-policy episodes and package them for PPO.

        Advantages are computed per episode with GAE, then all transitions are
        flattened into one batch and the advantages normalized -- the standard,
        stable PPO recipe from Chapter 10.
        """
        self.model.eval()
        b_states, b_actions, b_logp, b_returns, b_adv = [], [], [], [], []

        for _ in range(self.cfg.rollout_episodes):
            # Each episode starts on a random TRAIN-window day -> diverse markets.
            start = self.train_starts[np.random.randint(len(self.train_starts))]
            state = self.env.reset(start_day=start)

            ep_states, ep_actions, ep_logp = [], [], []
            ep_rewards, ep_values, ep_dones = [], [], []
            done = False
            while not done:
                s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                action, logp, value = self.model.get_action(s)
                next_state, reward, done, info = self.env.step(int(action.item()))

                ep_states.append(state)
                ep_actions.append(int(action.item()))
                ep_logp.append(float(logp.item()))
                ep_rewards.append(reward)
                ep_values.append(float(value.item()))
                ep_dones.append(1.0 if done else 0.0)
                state = next_state

            # Episodes are fixed-horizon and always terminal at the end, so the
            # bootstrap value beyond the last step is 0.
            adv, ret = compute_gae(
                np.array(ep_rewards), np.array(ep_values), np.array(ep_dones),
                last_value=0.0, gamma=self.cfg.gamma, gae_lambda=self.cfg.gae_lambda,
            )
            b_states.extend(ep_states)
            b_actions.extend(ep_actions)
            b_logp.extend(ep_logp)
            b_returns.extend(ret.tolist())
            b_adv.extend(adv.tolist())

            # Episode-level metrics for logging.
            self.history["episode_reward"].append(float(np.sum(ep_rewards)))
            self.history["total_return"].append(self.env.value / self.env.cfg.initial_capital - 1.0)
            self.history["max_drawdown"].append(_episode_max_drawdown(self.env))
            self.history["turnover"].append(float(self.env.last_turnover))

        # Flatten into tensors and normalize advantages over the whole batch.
        adv_t = torch.as_tensor(b_adv, dtype=torch.float32, device=self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)   #A batch-normalized advantages
        return {
            "states": torch.as_tensor(np.array(b_states), dtype=torch.float32, device=self.device),
            "actions": torch.as_tensor(b_actions, dtype=torch.long, device=self.device),
            "old_log_probs": torch.as_tensor(b_logp, dtype=torch.float32, device=self.device),
            "returns": torch.as_tensor(b_returns, dtype=torch.float32, device=self.device),
            "advantages": adv_t,
        }

    #A Normalizing advantages across the full rollout keeps gradient scale stable

    # ------------------------------------------------------------------
    # PPO UPDATE
    # ------------------------------------------------------------------
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Run several PPO epochs of minibatch updates over one rollout batch."""
        self.model.train()
        n = batch["states"].shape[0]
        idx = np.arange(n)

        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}
        n_minibatches = 0

        for _ in range(self.cfg.ppo_epochs):
            np.random.shuffle(idx)
            for start in range(0, n, self.cfg.minibatch_size):
                mb = idx[start:start + self.cfg.minibatch_size]
                mb_states = batch["states"][mb]
                mb_actions = batch["actions"][mb]
                mb_old_logp = batch["old_log_probs"][mb]
                mb_returns = batch["returns"][mb]
                mb_adv = batch["advantages"][mb]

                new_logp, entropy, values = self.model.evaluate_actions(mb_states, mb_actions)

                # Clipped surrogate objective (identical logic to Chapter 10).
                ratio = torch.exp(new_logp - mb_old_logp)               #B pi_new / pi_old
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - self.cfg.clip_epsilon,
                                    1 + self.cfg.clip_epsilon) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()           #C pessimistic min

                # Value (critic) regression toward the GAE returns.
                value_loss = F.mse_loss(values, mb_returns)

                # Entropy bonus encourages exploration / prevents premature collapse.
                entropy_loss = -entropy.mean()

                loss = (policy_loss
                        + self.cfg.value_coef * value_loss
                        + self.cfg.entropy_coef * entropy_loss)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (mb_old_logp - new_logp).mean().item()  #D drift diagnostic
                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += entropy.mean().item()
                stats["approx_kl"] += approx_kl
                n_minibatches += 1

        for k in stats:
            stats[k] /= max(1, n_minibatches)
        return stats

    #B Probability ratio between the updated and behavior policies
    #C Taking the min of clipped/unclipped is what bounds each update's size
    #D approx_kl ~ how far the policy moved this update; watch it stay small/stable

    # ------------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------------
    def train(self) -> Dict[str, List[float]]:
        print("=" * 70)
        print("PPO TRAINING (RL portfolio agent, NO LLM signals)")
        print("=" * 70)
        print(f"Device        : {self.cfg.device}")
        print(f"State dim     : {self.env.state_dim}   Actions: {self.env.action_dim}")
        print(f"Train starts  : {len(self.train_starts)} valid days "
              f"(episode horizon {self.env.cfg.horizon})")
        print(f"Updates       : {self.cfg.n_updates}  x  {self.cfg.rollout_episodes} episodes/update")
        print("-" * 70)

        running_reward = None
        best_return = -float("inf")
        for update in range(self.cfg.n_updates):
            batch = self.collect_rollouts()
            stats = self.update(batch)
            for k in ("policy_loss", "value_loss", "entropy", "approx_kl"):
                self.history[k].append(stats[k])

            # Exponentially smoothed reward for a readable trend (Chapter 8 style).
            recent_reward = float(np.mean(self.history["episode_reward"][-self.cfg.rollout_episodes:]))
            running_reward = recent_reward if running_reward is None \
                else 0.9 * running_reward + 0.1 * recent_reward

            recent_return = float(np.mean(self.history["total_return"][-self.cfg.rollout_episodes:]))
            if recent_return > best_return:
                best_return = recent_return

            if (update + 1) % self.cfg.log_every == 0 or update == 0:
                recent_dd = float(np.mean(self.history["max_drawdown"][-self.cfg.rollout_episodes:]))
                recent_to = float(np.mean(self.history["turnover"][-self.cfg.rollout_episodes:]))
                print(f"Update {update+1:4d}/{self.cfg.n_updates} | "
                      f"Reward: {recent_reward:7.3f} | Running: {running_reward:7.3f} | "
                      f"Return: {recent_return:+.3%} | MaxDD: {recent_dd:+.3%} | "
                      f"Turn: {recent_to:.3f} | "
                      f"P_Loss: {stats['policy_loss']:+.4f} | V_Loss: {stats['value_loss']:.4f} | "
                      f"Ent: {stats['entropy']:.3f} | KL: {stats['approx_kl']:+.4f}")

        print("-" * 70)
        print(f"Training complete. Best rollout-avg return: {best_return:+.3%}")
        return self.history


def _episode_max_drawdown(env: PortfolioTradingEnv) -> float:
    """Max drawdown reached during the episode the env just finished.

    We reconstruct it from the env's peak vs. final value; for a per-step curve
    the evaluation path below tracks values explicitly.
    """
    return env.value / env.peak_value - 1.0


# ============================================================================
# BASELINE POLICIES (no learning) -- evaluated with the SAME env mechanics
# ============================================================================
# Both baselines submit explicit target weights through env.step_target_weights,
# so they pay the same transaction costs and are scored by the same reward as
# the RL agent -- a fair comparison.


def _equal_weight_target(env: PortfolioTradingEnv) -> np.ndarray:
    """Equal weight across the risky assets, no cash (rebalanced daily)."""
    w = np.zeros(env.n_slots, dtype=np.float64)
    w[: env.n_assets] = 1.0 / env.n_assets
    return w


def run_equal_weight_episode(env: PortfolioTradingEnv, start_day: int) -> Dict:
    """Buy an equal-weight basket and rebalance to it every day."""
    env.reset(start_day=start_day)
    target = _equal_weight_target(env)
    return _run_weight_episode(env, lambda e: target)


def run_buy_hold_spy_episode(env: PortfolioTradingEnv, start_day: int) -> Dict:
    """Put 100% in SPY on day 0, then HOLD (never rebalance -> zero later turnover)."""
    env.reset(start_day=start_day)
    spy_idx = ASSET_TICKERS.index("SPY")

    def policy(e: PortfolioTradingEnv) -> np.ndarray:
        if e.step_count == 0:
            w = np.zeros(e.n_slots, dtype=np.float64)
            w[spy_idx] = 1.0
            return w
        return e.weights.copy()   # hold: target == current drifted weights => no trade
    return _run_weight_episode(env, policy)


def _run_weight_episode(env: PortfolioTradingEnv, policy_fn) -> Dict:
    """Run one episode where `policy_fn(env)` returns target weights each step."""
    values = [env.value]
    rewards, turnovers = [], []
    done = False
    while not done:
        target = policy_fn(env)
        _, reward, done, info = env.step_target_weights(target)
        values.append(info["portfolio_value"])
        rewards.append(reward)
        turnovers.append(info["turnover"])
    return _episode_stats(values, rewards, turnovers, env.cfg.initial_capital)


# ============================================================================
# EVALUATION
# ============================================================================


def _episode_stats(values, rewards, turnovers, initial_capital) -> Dict:
    """Package one episode's trajectory into the metrics we compare on."""
    values = np.asarray(values, dtype=np.float64)
    running_peak = np.maximum.accumulate(values)
    max_dd = float((values / running_peak - 1.0).min())
    return {
        "values": values,                                     # equity curve (len horizon+1)
        "episode_reward": float(np.sum(rewards)),
        "total_return": float(values[-1] / initial_capital - 1.0),
        "max_drawdown": max_dd,
        "turnover": float(np.sum(turnovers)),
    }


@torch.no_grad()
def run_agent_episode(model: ActorCritic, env: PortfolioTradingEnv, start_day: int,
                      device: torch.device, deterministic: bool = True) -> Dict:
    """Run one greedy (deterministic) episode of the trained agent."""
    model.eval()
    state = env.reset(start_day=start_day)
    values = [env.value]
    rewards, turnovers = [], []
    done = False
    while not done:
        s = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        action, _, _ = model.get_action(s, deterministic=deterministic)
        state, reward, done, info = env.step(int(action.item()))
        values.append(info["portfolio_value"])
        rewards.append(reward)
        turnovers.append(info["turnover"])
    return _episode_stats(values, rewards, turnovers, env.cfg.initial_capital)


def evaluate_all(model: ActorCritic, test_env: PortfolioTradingEnv, config: PPOConfig
                 ) -> Dict[str, Dict]:
    """Evaluate the RL agent and both baselines on the SAME test-window starts."""
    lo = test_env.cfg.lookback
    hi = test_env.n_days - test_env.cfg.horizon - 1
    start_days = list(range(lo, hi + 1, config.eval_stride))
    device = torch.device(config.device)

    results: Dict[str, Dict] = {"rl_agent": [], "equal_weight": [], "buy_hold_spy": []}
    for sd in start_days:
        results["rl_agent"].append(run_agent_episode(model, test_env, sd, device))
        results["equal_weight"].append(run_equal_weight_episode(test_env, sd))
        results["buy_hold_spy"].append(run_buy_hold_spy_episode(test_env, sd))

    # Aggregate each policy across all start days.
    summary: Dict[str, Dict] = {}
    for name, eps in results.items():
        summary[name] = {
            "n_episodes": len(eps),
            "mean_total_return": float(np.mean([e["total_return"] for e in eps])),
            "std_total_return": float(np.std([e["total_return"] for e in eps])),
            "mean_max_drawdown": float(np.mean([e["max_drawdown"] for e in eps])),
            "mean_turnover": float(np.mean([e["turnover"] for e in eps])),
            "mean_episode_reward": float(np.mean([e["episode_reward"] for e in eps])),
            "mean_equity_curve": np.mean(np.stack([e["values"] for e in eps]), axis=0),
            "std_equity_curve": np.std(np.stack([e["values"] for e in eps]), axis=0),
        }
    return summary


def print_comparison(summary: Dict[str, Dict]):
    """Print the head-to-head comparison table (RL vs. the two baselines)."""
    print("\n" + "=" * 70)
    print("EVALUATION ON HELD-OUT TEST WINDOW (chronologically later, unseen)")
    print("=" * 70)
    header = f"{'policy':<16}{'mean ret':>12}{'std ret':>10}{'mean MDD':>12}{'turnover':>11}{'ep reward':>12}"
    print(header)
    print("-" * len(header))
    label = {"rl_agent": "RL agent (PPO)", "equal_weight": "equal-weight",
             "buy_hold_spy": "buy & hold SPY"}
    for name in ("rl_agent", "equal_weight", "buy_hold_spy"):
        s = summary[name]
        print(f"{label[name]:<16}{s['mean_total_return']:>+11.2%}{s['std_total_return']:>10.2%}"
              f"{s['mean_max_drawdown']:>+11.2%}{s['mean_turnover']:>11.2f}"
              f"{s['mean_episode_reward']:>12.3f}")
    print("-" * len(header))
    rl = summary["rl_agent"]["mean_total_return"]
    ew = summary["equal_weight"]["mean_total_return"]
    bh = summary["buy_hold_spy"]["mean_total_return"]
    print(f"RL vs equal-weight : {(rl - ew):+.2%} mean-return difference")
    print(f"RL vs buy&hold SPY : {(rl - bh):+.2%} mean-return difference")
    print("=" * 70)


# ============================================================================
# PLOTTING (matplotlib is optional; guarded import like Chapter 9)
# ============================================================================


def plot_training_curves(history: Dict[str, List[float]], config: PPOConfig):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[plot] matplotlib unavailable ({exc}); skipping training-curve plot.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("PPO Training Progress -- RL Portfolio Agent (no LLM signals)", fontsize=14)

    def smooth(x, w=20):
        x = np.asarray(x, dtype=np.float64)
        if len(x) < w:
            return x
        return np.convolve(x, np.ones(w) / w, mode="valid")

    ep = history["episode_reward"]
    ax = axes[0, 0]
    ax.plot(ep, alpha=0.25, color="tab:blue")
    ax.plot(range(len(ep) - len(smooth(ep)), len(ep)), smooth(ep), color="tab:blue", lw=2)
    ax.set_title("Episode Reward"); ax.set_xlabel("episode"); ax.grid(True, alpha=0.3)

    tr = np.array(history["total_return"]) * 100
    ax = axes[0, 1]
    ax.plot(tr, alpha=0.25, color="tab:green")
    ax.plot(range(len(tr) - len(smooth(tr)), len(tr)), smooth(tr), color="tab:green", lw=2)
    ax.axhline(0, color="k", ls="--", alpha=0.4)
    ax.set_title("Episode Total Return (%)"); ax.set_xlabel("episode"); ax.grid(True, alpha=0.3)

    dd = np.array(history["max_drawdown"]) * 100
    ax = axes[0, 2]
    ax.plot(dd, alpha=0.3, color="tab:red")
    ax.set_title("Episode Max Drawdown (%)"); ax.set_xlabel("episode"); ax.grid(True, alpha=0.3)

    for (r, c), key, title in [((1, 0), "policy_loss", "Policy Loss"),
                               ((1, 1), "value_loss", "Value Loss"),
                               ((1, 2), "entropy", "Policy Entropy")]:
        ax = axes[r, c]
        ax.plot(history[key], color="tab:purple", alpha=0.8)
        ax.set_title(title); ax.set_xlabel("update"); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(config.output_dir, "training_curves.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"[plot] saved training curves -> {path}")


def plot_evaluation(summary: Dict[str, Dict], config: PPOConfig):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[plot] matplotlib unavailable ({exc}); skipping evaluation plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("RL Agent vs. Baselines on Held-Out Test Window", fontsize=14)
    colors = {"rl_agent": "tab:blue", "equal_weight": "tab:orange", "buy_hold_spy": "tab:green"}
    label = {"rl_agent": "RL agent (PPO)", "equal_weight": "equal-weight",
             "buy_hold_spy": "buy & hold SPY"}

    # (1) Mean equity curve with +/- std band (cumulative return, base 1.0).
    ax = axes[0]
    for name in ("rl_agent", "equal_weight", "buy_hold_spy"):
        mean_c = summary[name]["mean_equity_curve"]
        std_c = summary[name]["std_equity_curve"]
        steps = np.arange(len(mean_c))
        ax.plot(steps, mean_c, color=colors[name], lw=2, label=label[name])
        ax.fill_between(steps, mean_c - std_c, mean_c + std_c, color=colors[name], alpha=0.15)
    ax.axhline(1.0, color="k", ls="--", alpha=0.4)
    ax.set_title("Mean Portfolio Value over Episode"); ax.set_xlabel("day"); ax.set_ylabel("value")
    ax.legend(); ax.grid(True, alpha=0.3)

    names = ["rl_agent", "equal_weight", "buy_hold_spy"]
    x = np.arange(len(names))
    # (2) Mean total return.
    ax = axes[1]
    ax.bar(x, [summary[n]["mean_total_return"] * 100 for n in names],
           color=[colors[n] for n in names])
    ax.set_xticks(x); ax.set_xticklabels([label[n] for n in names], rotation=15)
    ax.axhline(0, color="k", ls="--", alpha=0.4)
    ax.set_title("Mean Total Return (%)"); ax.grid(True, alpha=0.3, axis="y")
    # (3) Mean max drawdown & turnover (twin axis).
    ax = axes[2]
    ax.bar(x - 0.2, [summary[n]["mean_max_drawdown"] * 100 for n in names], width=0.4,
           color="tab:red", alpha=0.7, label="max drawdown (%)")
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, [summary[n]["mean_turnover"] for n in names], width=0.4,
            color="tab:gray", alpha=0.7, label="turnover")
    ax.set_xticks(x); ax.set_xticklabels([label[n] for n in names], rotation=15)
    ax.set_title("Risk / Cost: Drawdown & Turnover")
    ax.set_ylabel("max drawdown (%)"); ax2.set_ylabel("total turnover")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(config.output_dir, "evaluation_comparison.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"[plot] saved evaluation comparison -> {path}")


# ============================================================================
# TRAIN/TEST ENVIRONMENTS (chronological split -- no leakage)
# ============================================================================


def make_train_test_envs(env_config: EnvConfig, ppo_config: PPOConfig
                          ) -> Tuple[PortfolioTradingEnv, PortfolioTradingEnv]:
    """Generate one price history and split it CHRONOLOGICALLY into train/test.

    The agent only ever samples episodes from the train env; evaluation only ever
    uses the test env. Because the split is by time, the agent cannot have seen
    any test-period price movement during training.
    """
    prices, volumes = generate_synthetic_market_data(
        env_config.tickers, ppo_config.n_days, env_config.seed
    )
    split = int(ppo_config.train_frac * ppo_config.n_days)

    train_cfg = EnvConfig(**{**env_config.__dict__})
    train_cfg.n_days = split
    test_cfg = EnvConfig(**{**env_config.__dict__})
    test_cfg.n_days = ppo_config.n_days - split

    train_env = PortfolioTradingEnv(train_cfg, prices=prices[:split], volumes=volumes[:split])
    test_env = PortfolioTradingEnv(test_cfg, prices=prices[split:], volumes=volumes[split:])
    return train_env, test_env


# ============================================================================
# ENTRY POINT
# ============================================================================


def run_training(env_config: EnvConfig, ppo_config: PPOConfig):
    """Full Step-2 pipeline: split data, train PPO, evaluate vs. baselines, plot."""
    set_seed(ppo_config.seed)

    print("=" * 70)
    print("CHAPTER 12 -- STEP 2: RL PORTFOLIO AGENT (PPO), NO LLM SIGNALS")
    print("=" * 70)
    print(f"Universe : {', '.join(env_config.tickers)} + {CASH_TICKER}")
    print(f"Data     : {ppo_config.n_days} days, "
          f"chronological split {int(ppo_config.train_frac*100)}/"
          f"{100-int(ppo_config.train_frac*100)} (train/test)")
    print(f"News/LLM : {'ON' if env_config.use_news else 'OFF (zeros) -- RL-only baseline mode'}")
    print(f"Reward   : return - cost - {env_config.turnover_penalty_coef:g}*turnover "
          f"- {env_config.drawdown_penalty_coef:g}*drawdown - {env_config.vol_penalty_coef:g}*vol "
          f"(scale {env_config.reward_scale:g})")

    train_env, test_env = make_train_test_envs(env_config, ppo_config)
    print(f"Train env: {train_env.n_days} days | Test env: {test_env.n_days} days")
    print(train_env.describe_state_layout())

    trainer = PPOTrainer(train_env, ppo_config)
    history = trainer.train()

    summary = evaluate_all(trainer.model, test_env, ppo_config)
    print_comparison(summary)

    plot_training_curves(history, ppo_config)
    plot_evaluation(summary, ppo_config)

    # Save the trained policy so later steps / eval can reload it.
    ckpt = os.path.join(ppo_config.output_dir, "ppo_agent.pt")
    torch.save({"model_state": trainer.model.state_dict(),
                "state_dim": train_env.state_dim,
                "action_dim": train_env.action_dim,
                "ppo_config": ppo_config.__dict__}, ckpt)
    print(f"\nSaved trained agent -> {ckpt}")
    print(f"All outputs in       -> {ppo_config.output_dir}/")
    return trainer, summary


def main():
    parser = argparse.ArgumentParser(
        description="Chapter 12 -- Step 2: train the PPO portfolio agent (no LLM signals)."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_days", type=int, default=1000, help="Length of synthetic history.")
    parser.add_argument("--horizon", type=int, default=30, help="Trading days per episode.")
    parser.add_argument("--n_updates", type=int, default=300, help="PPO updates.")
    parser.add_argument("--rollout_episodes", type=int, default=16, help="Episodes per update.")
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--output_dir", type=str, default="./outputs_rl_no_llm")
    # Reward-shaping overrides (default None => keep EnvConfig defaults). These let
    # us tune how much the agent cares about drawdown / volatility / turnover
    # without editing the environment -- the Step-2 refinement knobs.
    parser.add_argument("--drawdown_penalty_coef", type=float, default=None)
    parser.add_argument("--vol_penalty_coef", type=float, default=None)
    parser.add_argument("--turnover_penalty_coef", type=float, default=None)
    args = parser.parse_args()

    env_config = EnvConfig(seed=args.seed, horizon=args.horizon, use_news=False)
    if args.drawdown_penalty_coef is not None:
        env_config.drawdown_penalty_coef = args.drawdown_penalty_coef
    if args.vol_penalty_coef is not None:
        env_config.vol_penalty_coef = args.vol_penalty_coef
    if args.turnover_penalty_coef is not None:
        env_config.turnover_penalty_coef = args.turnover_penalty_coef
    ppo_config = PPOConfig(
        seed=args.seed,
        n_days=args.n_days,
        n_updates=args.n_updates,
        rollout_episodes=args.rollout_episodes,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
    )
    run_training(env_config, ppo_config)


if __name__ == "__main__":
    main()
