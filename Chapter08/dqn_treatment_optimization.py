import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt


# ============================================================
#  Treatment Planning Environment
# ============================================================

class TreatmentEnv:
    """
    Fractionated radiotherapy planning environment.

    Features:
    - Multiple fractions (sessions)
    - Discrete actions: region x intensity
    - Stochastic dose delivery
    - Organ-specific risk weights
    - Tumor regression / progression dynamics
    - Reward shaping friendly
    """

    def __init__(
        self,
        n_organs: int = 5,
        n_fractions: int = 30,
        n_intensity_levels: int = 3,
        seed: int = 0,
    ):
        """
        n_organs: includes tumor as index 0, OARs as 1..n_organs-1
        """
        self.rng = np.random.RandomState(seed)
        self.n_organs = n_organs
        self.n_fractions = n_fractions
        self.n_intensity_levels = n_intensity_levels

        # Tumor dose target for whole course
        self.target_dose = 70.0  # [Gy]

        # Organ dose limits (course total). index 0 = tumor -> inf
        self.organ_limits = np.array(
            [np.inf, 20.0, 15.0, 25.0, 18.0], dtype=np.float32
        )

        # Relative risk weights for organs (OARs only)
        self.organ_risk_weights = np.array(
            [0.0, 3.0, 1.5, 4.0, 2.0], dtype=np.float32
        )

        # Discrete beam intensities
        self.intensity_levels = np.array([3.0, 5.0, 7.0], dtype=np.float32)

        # Dose delivery noise (curriculum will adjust this)
        self.dose_noise_std = 0.1

        # Tumor dynamics parameters (curriculum-adjustable)
        self.initial_tumor_size = 1.0
        self.min_tumor_size = 0.2
        self.max_tumor_size = 2.0
        self.tumor_shrink_rate = 0.02
        self.tumor_growth_rate = 0.005
        self.min_effective_daily_dose = 2.0

        # Cost per step to encourage efficient plans
        self.step_cost = 0.02

        # Clinical success / failure thresholds (curriculum-adjustable)
        self.max_overdose_factor = 1.4    # fail if OAR > factor * limit
        self.success_tumor_window = 5.0   # |dose-target| < window
        self.success_margin_factor = 1.1  # success if OAR <= factor * limit

        # State:
        #   cum_doses (n_organs)
        #   tumor_remaining_fraction (1)
        #   oar_margins (n_organs-1)
        #   tumor_size_norm (1)
        #   fraction_progress (1)
        self.state_dim = (
            self.n_organs + 1 + (self.n_organs - 1) + 1 + 1
        )
        self.action_dim = self.n_organs * self.n_intensity_levels

        self.cum_doses = None
        self.tumor_size = None
        self.current_fraction = None
        self.done = False

        self.reset()

    # ------------ Curriculum support ------------

    def set_difficulty(self, difficulty: float):
        """
        difficulty in [0,1]:
          0 = very easy, 1 = final harder setting
        """
        difficulty = float(np.clip(difficulty, 0.0, 1.0))

        # Increase noise and tumor growth with difficulty
        self.dose_noise_std = 0.05 + 0.25 * difficulty
        self.tumor_growth_rate = 0.0 + 0.01 * difficulty
        self.tumor_shrink_rate = 0.015 + 0.02 * difficulty

        # Make clinical constraints stricter with difficulty
        self.max_overdose_factor = 1.5 - 0.3 * difficulty   # from 1.5 → 1.2
        self.success_margin_factor = 1.25 - 0.25 * difficulty  # from 1.25 → 1.0

    # ------------ Core environment logic ------------

    def reset(self):
        self.cum_doses = np.zeros(self.n_organs, dtype=np.float32)
        self.tumor_size = self.initial_tumor_size
        self.current_fraction = 0
        self.done = False
        return self._get_state()

    def _action_to_params(self, action: int):
        organ_idx = action // self.n_intensity_levels
        intensity_idx = action % self.n_intensity_levels
        intensity = self.intensity_levels[intensity_idx]
        return organ_idx, intensity

    def _apply_dose(self, organ_idx: int, intensity: float):
        """
        Compute and apply stochastic dose (including spillover).
        Returns dose_change vector for this fraction.
        """
        dose_change = np.zeros(self.n_organs, dtype=np.float32)

        # Primary dose
        dose_change[organ_idx] += intensity

        # Spillover to neighbors
        base_spill_factor = 0.3
        # Slightly modulate spills related to tumor size if tumor involved
        tumor_spill_factor = base_spill_factor * (1.0 + 0.3 * (self.tumor_size - 1.0))

        for neighbor in (organ_idx - 1, organ_idx + 1):
            if 0 <= neighbor < self.n_organs:
                if organ_idx == 0 or neighbor == 0:
                    spill = intensity * tumor_spill_factor
                else:
                    spill = intensity * base_spill_factor
                dose_change[neighbor] += spill

        # Stochastic noise on non-zero doses
        noise = self.rng.normal(0.0, self.dose_noise_std, size=self.n_organs)
        dose_change = np.maximum(dose_change + noise, 0.0)

        self.cum_doses += dose_change
        return dose_change

    def _update_tumor_dynamics(self, tumor_dose_this_step: float):
        if tumor_dose_this_step >= self.min_effective_daily_dose:
            self.tumor_size *= (1.0 - self.tumor_shrink_rate)
        else:
            self.tumor_size *= (1.0 + self.tumor_growth_rate)

        self.tumor_size = np.clip(
            self.tumor_size, self.min_tumor_size, self.max_tumor_size
        )

    def _get_state(self):
        cum_norm = self.cum_doses / 100.0  # rough normalization

        tumor_dose = self.cum_doses[0]
        tumor_remaining = (self.target_dose - tumor_dose) / self.target_dose
        tumor_remaining = np.clip(tumor_remaining, -1.0, 1.0)

        oar_doses = self.cum_doses[1:]
        oar_limits = self.organ_limits[1:]
        oar_margins = (oar_limits - oar_doses) / oar_limits
        oar_margins = np.clip(oar_margins, -1.0, 1.0)

        tumor_size_norm = (
            (self.tumor_size - self.min_tumor_size)
            / (self.max_tumor_size - self.min_tumor_size)
        )
        tumor_size_norm = np.clip(tumor_size_norm, 0.0, 1.0)

        frac_progress = self.current_fraction / float(self.n_fractions)

        state = np.concatenate(
            [
                cum_norm,
                np.array([tumor_remaining], dtype=np.float32),
                oar_margins,
                np.array([tumor_size_norm], dtype=np.float32),
                np.array([frac_progress], dtype=np.float32),
            ]
        )
        return state.astype(np.float32)

    def _check_success_failure(self):
        tumor_dose = self.cum_doses[0]
        oar_doses = self.cum_doses[1:]
        oar_limits = self.organ_limits[1:]

        # Failure: severe overdose of any OAR
        if np.any(oar_doses > self.max_overdose_factor * oar_limits):
            return False, True

        tumor_ok = np.abs(tumor_dose - self.target_dose) < self.success_tumor_window
        oars_ok = np.all(oar_doses <= self.success_margin_factor * oar_limits)

        if tumor_ok and oars_ok:
            return True, False

        return False, False

    def _compute_reward(
        self,
        dose_change: np.ndarray,
        done: bool,
        success: bool,
        failure: bool,
        prev_tumor_dose: float,
    ):
        tumor_dose = self.cum_doses[0]
        oar_doses = self.cum_doses[1:]
        oar_limits = self.organ_limits[1:]
        oar_risk = self.organ_risk_weights[1:]

        # Tumor coverage (symmetric around target)
        prev_error = np.abs(prev_tumor_dose - self.target_dose)
        curr_error = np.abs(tumor_dose - self.target_dose)
        tumor_error = curr_error / self.target_dose
        tumor_reward = -1.5 * tumor_error

        # Directional shaping: reward moving closer to target
        if curr_error < prev_error:
            direction_bonus = 0.3
        else:
            direction_bonus = -0.15

        # OAR violation penalty (weighted)
        violations = np.maximum(0.0, oar_doses - oar_limits)
        weighted_violations = violations * oar_risk
        organ_penalty = -0.3 * np.sum(weighted_violations)

        # Tumor size penalty
        tumor_size_penalty = -0.4 * self.tumor_size

        # Per-step cost
        step_penalty = -self.step_cost

        reward = (
            tumor_reward
            + direction_bonus
            + organ_penalty
            + tumor_size_penalty
            + step_penalty
        )

        if done:
            if success:
                reward += 200.0
            if failure:
                reward -= 200.0

        return float(reward)

    def step(self, action: int):
        if self.done:
            raise RuntimeError("Step called on terminated episode. Call reset().")

        prev_tumor_dose = float(self.cum_doses[0])

        organ_idx, intensity = self._action_to_params(action)
        dose_change = self._apply_dose(organ_idx, intensity)

        tumor_dose_this_step = dose_change[0]
        self._update_tumor_dynamics(tumor_dose_this_step)

        self.current_fraction += 1
        episode_over = self.current_fraction >= self.n_fractions

        success, failure = self._check_success_failure()
        done = episode_over or success or failure
        self.done = done

        reward = self._compute_reward(
            dose_change, done, success, failure, prev_tumor_dose
        )

        next_state = self._get_state()
        info = {
            "success": success,
            "failure": failure,
            "tumor_dose": float(self.cum_doses[0]),
            "oar_doses": self.cum_doses[1:].copy(),
            "tumor_size": float(self.tumor_size),
        }
        return next_state, reward, done, info


# ============================================================
#  Replay Buffer
# ============================================================

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(
            (
                np.array(state, dtype=np.float32),
                int(action),
                float(reward),
                np.array(next_state, dtype=np.float32),
                bool(done),
            )
        )

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.bool_),
        )

    def __len__(self):
        return len(self.buffer)


# ============================================================
#  Simple NumPy MLP for Q-function
# ============================================================

class MLPQNetwork:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, rng=None):
        if rng is None:
            rng = np.random.RandomState(42)
        self.rng = rng

        limit1 = np.sqrt(6 / (input_dim + hidden_dim))
        self.W1 = rng.uniform(-limit1, limit1, size=(input_dim, hidden_dim)).astype(
            np.float32
        )
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)

        limit2 = np.sqrt(6 / (hidden_dim + output_dim))
        self.W2 = rng.uniform(-limit2, limit2, size=(hidden_dim, output_dim)).astype(
            np.float32
        )
        self.b2 = np.zeros(output_dim, dtype=np.float32)

    def forward(self, x):
        z1 = x @ self.W1 + self.b1
        h1 = np.maximum(z1, 0.0)
        q_values = h1 @ self.W2 + self.b2
        cache = (x, z1, h1)
        return q_values, cache

    def predict(self, x):
        z1 = x @ self.W1 + self.b1
        h1 = np.maximum(z1, 0.0)
        q_values = h1 @ self.W2 + self.b2
        return q_values

    def update(self, x, grad_q, cache, lr: float):
        batch_size = x.shape[0]
        x, z1, h1 = cache

        dW2 = h1.T @ grad_q / batch_size
        db2 = np.sum(grad_q, axis=0) / batch_size

        dh1 = grad_q @ self.W2.T
        dz1 = dh1 * (z1 > 0)

        dW1 = x.T @ dz1 / batch_size
        db1 = np.sum(dz1, axis=0) / batch_size

        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

    def copy_from(self, other):
        self.W1 = other.W1.copy()
        self.b1 = other.b1.copy()
        self.W2 = other.W2.copy()
        self.b2 = other.b2.copy()


# ============================================================
#  Double DQN Agent (NumPy)
# ============================================================

class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        gamma: float = 0.99,
        lr: float = 1e-3,
        buffer_capacity: int = 50000,
        batch_size: int = 64,
        min_buffer_size: int = 1000,
        target_update_freq: int = 1000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_frames: int = 30000,
        seed: int = 123,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.min_buffer_size = min_buffer_size
        self.target_update_freq = target_update_freq

        self.rng = np.random.RandomState(seed)

        self.online_net = MLPQNetwork(state_dim, hidden_dim, action_dim, rng=self.rng)
        self.target_net = MLPQNetwork(state_dim, hidden_dim, action_dim, rng=self.rng)
        self.target_net.copy_from(self.online_net)

        self.replay_buffer = ReplayBuffer(buffer_capacity)

        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_frames = epsilon_decay_frames
        self.total_steps = 0

    def select_action(self, state: np.ndarray, greedy: bool = False):
        """
        If greedy=True, ignore epsilon and act greedily (for evaluation).
        """
        if greedy:
            state_batch = state.reshape(1, -1)
            q_values = self.online_net.predict(state_batch)[0]
            return int(np.argmax(q_values))

        self.total_steps += 1
        frac = min(1.0, self.total_steps / float(self.epsilon_decay_frames))
        self.epsilon = (
            self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)
        )

        if self.rng.rand() < self.epsilon:
            return self.rng.randint(self.action_dim)
        else:
            state_batch = state.reshape(1, -1)
            q_values = self.online_net.predict(state_batch)[0]
            return int(np.argmax(q_values))

    def push_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self):
        if len(self.replay_buffer) < self.min_buffer_size:
            return None

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
        ) = self.replay_buffer.sample(self.batch_size)

        q_values, cache = self.online_net.forward(states)

        # Double DQN target
        next_q_online = self.online_net.predict(next_states)
        best_next_actions = np.argmax(next_q_online, axis=1)

        next_q_target = self.target_net.predict(next_states)
        best_next_q = next_q_target[
            np.arange(self.batch_size), best_next_actions
        ]

        targets = rewards + self.gamma * best_next_q * (~dones)

        q_selected = q_values[np.arange(self.batch_size), actions]
        td_errors = q_selected - targets

        grad_q = np.zeros_like(q_values)
        grad_q[np.arange(self.batch_size), actions] = td_errors

        self.online_net.update(states, grad_q, cache, self.lr)

        if self.total_steps % self.target_update_freq == 0:
            self.target_net.copy_from(self.online_net)

        loss = 0.5 * np.mean(td_errors ** 2)
        return loss


# ============================================================
#  Visualization Helpers (for teaching)
# ============================================================

def plot_training_curves(rewards, success_flags, window: int = 50):
    episodes = np.arange(1, len(rewards) + 1)

    plt.figure()
    plt.plot(episodes, rewards)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title("Episode Reward Over Time")
    plt.grid()

    plt.figure()
    if len(success_flags) >= window:
        kernel = np.ones(window) / window
        smoothed = np.convolve(success_flags, kernel, mode="valid")
        plt.plot(
            np.arange(window, len(success_flags) + 1),
            smoothed,
        )
    else:
        plt.plot(episodes, success_flags)
    plt.xlabel("Episode")
    plt.ylabel("Success Rate (moving average)")
    plt.title(f"Success Rate (window={window})")
    plt.ylim(-0.05, 1.05)
    plt.grid()
    plt.show()


def visualize_greedy_episode(env: TreatmentEnv, agent: DQNAgent):
    """
    Run a single greedy episode and visualize:
      - tumor dose vs OAR doses over time
      - tumor size over time
    """
    state = env.reset()
    tumor_doses = []
    oar_doses = []
    tumor_sizes = []

    for _ in range(env.n_fractions):
        action = agent.select_action(state, greedy=True)
        next_state, reward, done, info = env.step(action)

        tumor_doses.append(info["tumor_dose"])
        oar_doses.append(info["oar_doses"])
        tumor_sizes.append(info["tumor_size"])

        state = next_state
        if done:
            break

    oar_doses = np.array(oar_doses)

    plt.figure()
    plt.plot(tumor_doses, label="Tumor")
    for i in range(oar_doses.shape[1]):
        plt.plot(oar_doses[:, i], label=f"OAR {i+1}")
    plt.xlabel("Fraction")
    plt.ylabel("Cumulative Dose [Gy]")
    plt.title("Dose Accumulation During Greedy Episode")
    plt.legend()
    plt.grid()

    plt.figure()
    plt.plot(tumor_sizes)
    plt.xlabel("Fraction")
    plt.ylabel("Tumor Size (arb. units)")
    plt.title("Tumor Size Dynamics During Greedy Episode")
    plt.grid()
    plt.show()


def evaluate_policy(env: TreatmentEnv, agent: DQNAgent, n_episodes: int = 50):
    """
    Run greedy policy on multiple episodes and collect outcomes.
    """
    tumor_final = []
    max_oar_final = []
    success_flags = []

    for _ in range(n_episodes):
        state = env.reset()
        done = False
        info = {}
        while not done:
            action = agent.select_action(state, greedy=True)
            state, reward, done, info = env.step(action)
        tumor_final.append(info["tumor_dose"])
        max_oar_final.append(float(np.max(info["oar_doses"])))
        success_flags.append(1.0 if info["success"] else 0.0)

    return np.array(tumor_final), np.array(max_oar_final), np.array(success_flags)


def plot_outcome_scatter(
    tumor_final, max_oar_final, success_flags, env: TreatmentEnv
):
    """
    Scatter: final tumor dose vs max OAR dose, colored by success.
    """
    plt.figure()
    successes = success_flags > 0.5
    failures = ~successes

    plt.scatter(
        tumor_final[successes],
        max_oar_final[successes],
        label="Success",
        marker="o",
    )
    plt.scatter(
        tumor_final[failures],
        max_oar_final[failures],
        label="Failure",
        marker="x",
    )

    plt.axvline(env.target_dose, linestyle="--")
    # Reasonable rough limit line for reference (max of OAR limits)
    plt.axhline(
        np.max(env.organ_limits[1:]), linestyle="--"
    )

    plt.xlabel("Final Tumor Dose [Gy]")
    plt.ylabel("Max Final OAR Dose [Gy]")
    plt.title("Outcome Scatter: Tumor vs Max OAR Dose")
    plt.legend()
    plt.grid()
    plt.show()


# ============================================================
#  Training Loop with Curriculum + Visualizations
# ============================================================

def main():
    # Base environment
    env = TreatmentEnv()
    agent = DQNAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_dim=128,
        gamma=0.99,
        lr=1e-3,
        buffer_capacity=50000,
        batch_size=64,
        min_buffer_size=1000,
        target_update_freq=1000,
        epsilon_start=1.0,
        epsilon_end=0.03,
        epsilon_decay_frames=50000,
        seed=42,
    )

    num_episodes = 10000
    curriculum_episodes = 1000  # ramp difficulty from 0→1 over this many episodes

    rewards_history = []
    success_history = []
    losses_history = []

    for episode in range(num_episodes):
        # Curriculum: gradually increase difficulty
        difficulty = min(1.0, episode / float(curriculum_episodes))
        env.set_difficulty(difficulty)

        state = env.reset()
        episode_reward = 0.0
        success_flag = False

        for t in range(env.n_fractions):
            action = agent.select_action(state, greedy=False)
            next_state, reward, done, info = env.step(action)

            agent.push_transition(state, action, reward, next_state, done)
            loss = agent.train_step()
            if loss is not None:
                losses_history.append(loss)

            episode_reward += reward
            state = next_state

            if done:
                success_flag = info["success"]
                break

        rewards_history.append(episode_reward)
        success_history.append(1.0 if success_flag else 0.0)

        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(rewards_history[-20:])
            avg_success = np.mean(success_history[-20:])
            print(
                f"Episode {episode+1:4d} | "
                f"AvgReward(20): {avg_reward:8.2f} | "
                f"Success(20): {avg_success*100:5.1f}% | "
                f"Epsilon: {agent.epsilon:5.3f} | "
                f"Difficulty: {difficulty:4.2f}"
            )

    print("Training finished.")

    # ---------------- Visualizations ----------------

    plot_training_curves(rewards_history, success_history, window=50)

    # Evaluate greedy policy on fixed difficulty (final difficulty)
    env.set_difficulty(1.0)
    tumor_final, max_oar_final, success_eval = evaluate_policy(
        env, agent, n_episodes=50
    )
    print(
        f"Greedy evaluation success rate over 50 episodes: "
        f"{np.mean(success_eval)*100:.1f}%"
    )

    visualize_greedy_episode(env, agent)
    plot_outcome_scatter(tumor_final, max_oar_final, success_eval, env)


if __name__ == "__main__":
    main()
