"""
Chapter 12 -- Steps 5 & 6: Connect LLM Signals to RL + Final 5-Mode Comparison
==============================================================================
Manning Publications -- Applied Reinforcement Learning (Capstone Chapter)

This is the capstone integration. It wires the LLM signal extractor (Steps 3-4)
into the RL trading environment (Steps 1-2) and produces the final comparison
across five modes:

    1. heuristic      -- buy-and-hold SPY / equal-weight (no learning)
    2. rl_no_news     -- PPO, news block fixed to zeros
    3. rl_naive       -- PPO + rule-based sentiment signals
    4. rl_base_llm    -- PPO + base (off-the-shelf) LLM signals
    5. rl_finetuned   -- PPO + fine-tuned LLM signals

THE DISTILLATION FRAMING (why this is an honest teaching design)
----------------------------------------------------------------
For LLM signals to *help* an RL agent, the signals must genuinely predict price
moves. Real markets are near-efficient (weak signal) and our env prices are
synthetic, so we make the pedagogy explicit and clean:

  * The DeepSeek label is the ORACLE signal -- expensive, not deployable in real
    time. We CONSTRUCT the market so that this oracle signal truly drives next-
    day returns (plus noise): see generate_signal_driven_market().
  * We then DISTILL the oracle into a cheap fine-tuned small model (Step 4).
  * The agent never sees the oracle -- only each mode's extractor output. The
    question the chapter answers: how much oracle signal survives distillation,
    and how much trading value does that buy?

Because fine-tuned signals approximate the oracle best (Step-4 MAE 0.11 vs. the
base model's 0.34), the rl_finetuned agent should observe the cleanest state and
trade best -- while rl_no_news is blind to the driver and rl_naive/base see a
noisier version. This isolates exactly the chapter's claim: signal quality ->
state quality -> policy quality.

FAIRNESS / LEAKAGE CONTROLS (identical across all modes)
--------------------------------------------------------
- Same market prices, same env settings, same transaction costs & risk penalties.
- Chronological train/val/test split; RL only samples episodes from the TRAIN
  window; all modes are evaluated on the same held-out TEST window.
- No look-ahead: the signal observed on day t is built from news up to t, and it
  drives the return realized from t -> t+1 (after the action is committed).

STAGES (cached so you can iterate without recomputing the expensive parts)
--------------------------------------------------------------------------
  signals : build the per-mode daily signal matrices (base/fine-tuned need the
            GPU) + the signal-driven market; cache to disk.
  train   : train a PPO agent per RL mode, evaluate every mode on the test
            window, and write the final metrics/report/figures.
  all     : signals -> train.
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ch12_trading_env import (
    ASSET_TICKERS, CASH_TICKER, ACTION_NAMES, EnvConfig, MarketSignal,
    PortfolioTradingEnv, generate_synthetic_market_data, set_seed,
)
from rl_w_o_llm_training import ActorCritic, PPOConfig, PPOTrainer
from build_signal_dataset import SIGNAL_FIELDS, naive_signal, read_jsonl

TRADING_DAYS = 252  # for annualization


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class IntegrationConfig:
    """Everything Steps 5-6 need, in one dataclass (Chapters 9-11 style)."""
    # --- paths ---
    data_dir: str = "./signal_dataset"          # clean.jsonl + naive_labels.jsonl
    signal_llm_dir: str = "./signal_llm_lora"    # fine-tuned adapter from Step 4
    output_dir: str = "./outputs_integration"
    cache_file: str = "market_and_signals.npz"

    # --- base model (for base/fine-tuned signal generation) ---
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"

    # --- chronological split (fractions of the news timeline) ---
    train_frac: float = 0.70
    val_frac: float = 0.15                       # reserved (held out of training)
    # test = remainder

    # --- signal-driven market data-generating process ---
    # The signal is the DOMINANT alpha: we zero the free market drift and shrink the
    # shared factor so being constantly invested is NOT automatically rewarded --
    # the only way to earn return is to rotate toward the ETF the signal favors.
    # That is what forces the agent to actually read the news block. signal_coef
    # is tuned so signal QUALITY matters (a noisy naive/base signal mistimes and is
    # punished; the fine-tuned signal tracks the oracle and profits).
    signal_coef: float = 0.04                    # how strongly the ORACLE signal moves returns
    market_vol: float = 0.004                    # shared market-factor daily vol (kept small)
    risk_coef: float = 0.004                     # extra downside when risk signals are high
    base_drift: float = 0.0                       # NO free drift -> selection/timing is the alpha

    # --- reward shaping (the Step-2 v2 tuning; defaults are too risk-averse) ---
    drawdown_penalty_coef: float = 0.03
    vol_penalty_coef: float = 0.01

    # --- which modes to run ---
    rl_modes: Tuple[str, ...] = ("rl_no_news", "rl_naive", "rl_base_llm", "rl_finetuned")
    heuristic_modes: Tuple[str, ...] = ("buy_hold_spy", "equal_weight")

    # --- RL training (shared across all RL modes for fairness) ---
    n_updates: int = 600
    rollout_episodes: int = 16
    entropy_coef: float = 0.02                   # keep exploration up -> avoid action collapse
    eval_stride: int = 2

    seed: int = 42
    device: str = ""

    def __post_init__(self):
        if not self.device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        os.makedirs(self.output_dir, exist_ok=True)


# Nice labels/colors for the report.
MODE_LABELS = {
    "buy_hold_spy": "buy & hold SPY", "equal_weight": "equal-weight",
    "rl_no_news": "RL (no news)", "rl_naive": "RL + naive signals",
    "rl_base_llm": "RL + base LLM", "rl_finetuned": "RL + fine-tuned LLM",
}
MODE_COLORS = {
    "buy_hold_spy": "tab:gray", "equal_weight": "tab:orange",
    "rl_no_news": "tab:blue", "rl_naive": "tab:green",
    "rl_base_llm": "tab:purple", "rl_finetuned": "tab:red",
}


# ============================================================================
# SIGNAL <-> ASSET ADAPTER  (the 13 theme signals -> per-ETF effects)
# ============================================================================
# This single mapping does double duty:
#   (a) it is the DATA-GENERATING PROCESS: applied to the ORACLE signal it decides
#       how each ETF's return responds to the news (used only to build the market);
#   (b) conceptually it is also what the agent must implicitly learn from the news
#       block of its state (tech_signal -> QQQ, energy_signal -> XLE, ...).
# Indices follow MarketSignal.FIELDS / SIGNAL_FIELDS order.

_IDX = {f: i for i, f in enumerate(SIGNAL_FIELDS)}


def signal_to_asset_effect(vec: np.ndarray) -> np.ndarray:
    """Map a 13-field signal vector to a per-ETF return-effect vector (SPY..XLV)."""
    g = lambda name: float(vec[_IDX[name]])
    mkt, ro, rates, gr = g("market_sentiment"), g("risk_on_signal"), g("rates_pressure"), g("growth_signal")
    tech, en, fin, dfn = g("tech_signal"), g("energy_signal"), g("financials_signal"), g("defensive_signal")
    eff = np.zeros(len(ASSET_TICKERS), dtype=np.float64)
    eff[0] = 0.6 * mkt + 0.4 * ro                     # SPY  (broad market)
    eff[1] = 0.5 * tech + 0.3 * gr + 0.3 * mkt        # QQQ  (tech / growth)
    eff[2] = 0.5 * ro + 0.4 * gr + 0.2 * mkt          # IWM  (small caps)
    eff[3] = 0.8 * en + 0.1 * mkt                     # XLE  (energy)
    eff[4] = 0.6 * fin + 0.2 * rates + 0.2 * mkt      # XLF  (financials)
    eff[5] = 0.7 * dfn - 0.3 * mkt                    # XLV  (defensive, inverse to market)
    return eff


def sig_dict_to_vec(d: Dict[str, float]) -> np.ndarray:
    """Turn a signal dict into a vector in the canonical field order."""
    return np.asarray([float(d.get(f, 0.0)) for f in SIGNAL_FIELDS], dtype=np.float32)


# ============================================================================
# TIMELINE + PER-MODE SIGNAL MATRICES
# ============================================================================


def load_timeline(cfg: IntegrationConfig) -> List[Dict[str, Any]]:
    """Ordered list of DATED examples with their oracle signal + naive signal.

    Uses clean.jsonl (input text + oracle DeepSeek signal) joined with
    naive_labels.jsonl (rule-based signal), keeping only dated examples and
    sorting chronologically so the sequence is a valid trading timeline.
    """
    clean = read_jsonl(os.path.join(cfg.data_dir, "clean.jsonl"))
    naive = {r["id"]: r["signals"] for r in read_jsonl(os.path.join(cfg.data_dir, "naive_labels.jsonl"))}
    dated = [e for e in clean if e.get("date")]
    dated.sort(key=lambda e: e["date"])
    timeline = []
    for e in dated:
        timeline.append({
            "id": e["id"], "date": e["date"], "input": e["input"],
            "oracle": sig_dict_to_vec(e["signals"]),
            "naive": sig_dict_to_vec(naive.get(e["id"], {})),
        })
    print(f"[timeline] {len(timeline)} dated trading days "
          f"({timeline[0]['date']} .. {timeline[-1]['date']})")
    return timeline


@torch.no_grad()
def generate_llm_signal_matrices(cfg: IntegrationConfig, timeline: List[Dict[str, Any]]
                                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Generate BASE and FINE-TUNED signal vectors for every timeline day (GPU).

    We load the base model once, inject + load the Step-4 adapter, and run
    inference twice over the same inputs: adapters OFF (base) and ON (fine-tuned).
    Unparseable generations fall back to a zero signal.
    """
    from finetune_signal_extractor import (
        ModelConfig, LoRAConfig, EvalConfig, build_model_and_tokenizer,
        generate_signals_batched, parse_signal, load_adapter, set_lora_enabled,
    )
    model_cfg = ModelConfig(model_name=cfg.model_name)
    # Infer the LoRA rank from the saved adapter so the injected scaffold matches
    # (Iter-1 used r=16, v2 uses r=32); mismatched rank -> tensor-size error.
    adapter_state = torch.load(os.path.join(cfg.signal_llm_dir, "lora_adapter.pt"),
                               map_location="cpu")
    lora_r = next(v.shape[0] for k, v in adapter_state.items() if "lora_A" in k)
    print(f"[signals] detected LoRA rank r={lora_r} in {cfg.signal_llm_dir}")
    model, tokenizer, _ = build_model_and_tokenizer(model_cfg, LoRAConfig(r=lora_r))
    load_adapter(model, cfg.signal_llm_dir, model_cfg.device)
    eval_cfg = EvalConfig()

    inputs = [ex["input"] for ex in timeline]

    def gen(tag: str) -> np.ndarray:
        print(f"[signals] generating {tag} signals for {len(inputs)} days ...")
        raw = generate_signals_batched(model, tokenizer, inputs, model_cfg, eval_cfg)
        mat = np.zeros((len(inputs), len(SIGNAL_FIELDS)), dtype=np.float32)
        parsed_ok = 0
        for i, txt in enumerate(raw):
            sig, flags = parse_signal(txt)
            if sig is not None:
                mat[i] = sig_dict_to_vec(sig); parsed_ok += 1
        print(f"[signals] {tag}: parsed {parsed_ok}/{len(inputs)} "
              f"({100*parsed_ok/len(inputs):.1f}%)")
        return mat

    set_lora_enabled(model, False)
    base_mat = gen("base")
    set_lora_enabled(model, True)
    tuned_mat = gen("fine-tuned")
    return base_mat, tuned_mat


# ============================================================================
# SIGNAL-DRIVEN MARKET  (the data-generating process)
# ============================================================================


def generate_signal_driven_market(oracle_mat: np.ndarray, cfg: IntegrationConfig
                                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Build prices/volumes whose returns are driven by the ORACLE signal + noise.

    returns[t] responds to the oracle signal from day t-1 (so the signal observed
    at t-1 predicts the t-1 -> t move -- matching how the env realizes rewards and
    keeping the setup free of look-ahead). High volatility/geopolitical/liquidity
    risk adds a downside drag and inflates noise, so those signals matter too.
    """
    from ch12_trading_env import _ASSET_DYNAMICS
    n = oracle_mat.shape[0]
    A = len(ASSET_TICKERS)
    rng = np.random.default_rng(cfg.seed)

    drift = np.array([_ASSET_DYNAMICS[t]["drift"] for t in ASSET_TICKERS])
    beta = np.array([_ASSET_DYNAMICS[t]["beta"] for t in ASSET_TICKERS])
    idio = np.array([_ASSET_DYNAMICS[t]["idio"] for t in ASSET_TICKERS])
    p0 = np.array([_ASSET_DYNAMICS[t]["p0"] for t in ASSET_TICKERS])
    vol0 = np.array([_ASSET_DYNAMICS[t]["vol0"] for t in ASSET_TICKERS])

    market_factor = rng.normal(cfg.base_drift, cfg.market_vol, n)
    returns = np.zeros((n, A), dtype=np.float64)
    for t in range(1, n):
        s = oracle_mat[t - 1]
        eff = signal_to_asset_effect(s)
        vol_r = max(0.0, float(s[_IDX["volatility_risk"]]))
        geo_r = max(0.0, float(s[_IDX["geopolitical_risk"]]))
        liq_r = max(0.0, float(s[_IDX["liquidity_risk"]]))
        risk_drag = cfg.risk_coef * (vol_r + geo_r + liq_r)          # broad downside in stress
        noise = rng.normal(0.0, idio * (1.0 + 0.5 * vol_r))
        returns[t] = drift + beta * market_factor[t] + cfg.signal_coef * eff - risk_drag + noise

    prices = p0 * np.cumprod(1.0 + returns, axis=0)
    volumes = vol0 * np.exp(rng.normal(0.0, 0.30, (n, A)) + 3.0 * np.abs(returns))
    return prices, volumes


def build_signal_matrices(cfg: IntegrationConfig, timeline: List[Dict[str, Any]],
                          base_mat: np.ndarray, tuned_mat: np.ndarray
                          ) -> Dict[str, np.ndarray]:
    """Assemble the observed signal matrix the agent sees, per RL mode."""
    n = len(timeline)
    zeros = np.zeros((n, len(SIGNAL_FIELDS)), dtype=np.float32)
    naive = np.stack([ex["naive"] for ex in timeline])
    return {
        "rl_no_news": zeros,        # env.use_news=False also zeros this; kept explicit
        "rl_naive": naive,
        "rl_base_llm": base_mat,
        "rl_finetuned": tuned_mat,
    }


# ============================================================================
# CACHING (so `train` reruns don't redo the GPU signal generation)
# ============================================================================


def save_cache(cfg: IntegrationConfig, prices, volumes, matrices, timeline):
    path = os.path.join(cfg.output_dir, cfg.cache_file)
    np.savez_compressed(
        path, prices=prices, volumes=volumes, oracle=np.stack([e["oracle"] for e in timeline]),
        dates=np.array([e["date"] for e in timeline]),
        **{f"sig_{k}": v for k, v in matrices.items()})
    # inputs/ids are text -> separate json
    meta = [{"id": e["id"], "date": e["date"], "input": e["input"]} for e in timeline]
    with open(os.path.join(cfg.output_dir, "timeline_meta.json"), "w") as f:
        json.dump(meta, f)
    print(f"[cache] saved market + signals -> {path}")


def load_cache(cfg: IntegrationConfig):
    path = os.path.join(cfg.output_dir, cfg.cache_file)
    d = np.load(path, allow_pickle=True)
    matrices = {k[4:]: d[k] for k in d.files if k.startswith("sig_")}
    meta = json.load(open(os.path.join(cfg.output_dir, "timeline_meta.json")))
    return d["prices"], d["volumes"], matrices, d["oracle"], meta


# ============================================================================
# ENV CONSTRUCTION PER MODE (chronological split, shared market)
# ============================================================================


def make_mode_envs(cfg: IntegrationConfig, prices, volumes, signal_matrix,
                   use_news: bool, env_config: EnvConfig
                   ) -> Tuple[PortfolioTradingEnv, PortfolioTradingEnv]:
    """Build train/test envs sharing the SAME market; only the news block differs.

    Chronological split: train = first train_frac; test = last (1-train-val). The
    validation slice is held out of both (reserved for tuning), never trained on.
    """
    n = prices.shape[0]
    train_end = int(cfg.train_frac * n)
    test_start = int((cfg.train_frac + cfg.val_frac) * n)

    def make(lo, hi):
        c = EnvConfig(**{**env_config.__dict__})
        c.n_days = hi - lo
        c.use_news = use_news
        sm = signal_matrix[lo:hi] if signal_matrix is not None else None
        return PortfolioTradingEnv(c, prices=prices[lo:hi], volumes=volumes[lo:hi],
                                   signal_matrix=sm)
    return make(0, train_end), make(test_start, n)


# ============================================================================
# EVALUATION (rich: daily returns + weights, for financial metrics + exposure)
# ============================================================================


@torch.no_grad()
def _run_agent_steps(model, env, start_day, device):
    """Greedy agent episode; return per-step (values, net_returns, turnovers, weights, rewards)."""
    model.eval()
    state = env.reset(start_day=start_day)
    vals, rets, turns, weights, rewards = [env.value], [], [], [], []
    done = False
    while not done:
        s = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        action, _, _ = model.get_action(s, deterministic=True)
        state, reward, done, info = env.step(int(action.item()))
        vals.append(info["portfolio_value"]); rets.append(info["net_return"])
        turns.append(info["turnover"]); weights.append(info["weights"]); rewards.append(reward)
    return vals, rets, turns, weights, rewards


def _run_weights_steps(env, start_day, policy_fn):
    """Heuristic episode driven by explicit target weights; same per-step outputs."""
    env.reset(start_day=start_day)
    vals, rets, turns, weights, rewards = [env.value], [], [], [], []
    done = False
    while not done:
        _, reward, done, info = env.step_target_weights(policy_fn(env))
        vals.append(info["portfolio_value"]); rets.append(info["net_return"])
        turns.append(info["turnover"]); weights.append(info["weights"]); rewards.append(reward)
    return vals, rets, turns, weights, rewards


def _financial_metrics(episodes: List[Dict], initial_capital: float) -> Dict[str, Any]:
    """Aggregate per-episode trajectories into the report's financial metrics."""
    daily = np.concatenate([np.asarray(e["rets"]) for e in episodes])   # pooled daily net returns
    mean_d, std_d = float(daily.mean()), float(daily.std())
    total_returns = [e["vals"][-1] / initial_capital - 1.0 for e in episodes]
    mdds = []
    for e in episodes:
        v = np.asarray(e["vals"]); peak = np.maximum.accumulate(v)
        mdds.append(float((v / peak - 1.0).min()))
    equity = np.mean(np.stack([e["vals"] for e in episodes]), axis=0)     # mean equity curve
    equity_std = np.std(np.stack([e["vals"] for e in episodes]), axis=0)
    weights = np.mean(np.stack([np.asarray(e["weights"]) for e in episodes]), axis=0)  # (T, n_slots)
    return {
        "sharpe": (mean_d / std_d * np.sqrt(TRADING_DAYS)) if std_d > 1e-9 else 0.0,
        "ann_return": mean_d * TRADING_DAYS,
        "ann_vol": std_d * np.sqrt(TRADING_DAYS),
        "mean_total_return": float(np.mean(total_returns)),
        "mean_max_drawdown": float(np.mean(mdds)),
        "mean_turnover": float(np.mean([np.sum(e["turns"]) for e in episodes])),
        "mean_episode_reward": float(np.mean([np.sum(e["rewards"]) for e in episodes])),
        "final_value": float(np.mean([e["vals"][-1] for e in episodes])),
        "mean_equity_curve": equity, "std_equity_curve": equity_std,
        "mean_weights": weights, "n_episodes": len(episodes),
    }


def evaluate_mode(cfg: IntegrationConfig, mode: str, model, test_env) -> Dict[str, Any]:
    """Run a mode's policy over all test-window start days and compute metrics."""
    lo, hi = test_env.cfg.lookback, test_env.n_days - test_env.cfg.horizon - 1
    starts = list(range(lo, hi + 1, cfg.eval_stride))
    device = torch.device(cfg.device)
    spy_idx = ASSET_TICKERS.index("SPY")

    def equal_weight(env):
        w = np.zeros(env.n_slots); w[:env.n_assets] = 1.0 / env.n_assets; return w

    def buy_hold(env):
        if env.step_count == 0:
            w = np.zeros(env.n_slots); w[spy_idx] = 1.0; return w
        return env.weights.copy()

    episodes = []
    for sd in starts:
        if mode == "buy_hold_spy":
            v, r, t, w, rw = _run_weights_steps(test_env, sd, buy_hold)
        elif mode == "equal_weight":
            v, r, t, w, rw = _run_weights_steps(test_env, sd, equal_weight)
        else:
            v, r, t, w, rw = _run_agent_steps(model, test_env, sd, device)
        episodes.append({"vals": v, "rets": r, "turns": t, "weights": w, "rewards": rw})
    m = _financial_metrics(episodes, test_env.cfg.initial_capital)
    m["mode"] = mode
    return m


# ============================================================================
# TRAINING (one PPO agent per RL mode; heuristics need no training)
# ============================================================================


def train_rl_mode(cfg: IntegrationConfig, mode: str, prices, volumes,
                  signal_matrix, env_config: EnvConfig):
    """Train a PPO agent for one RL mode on the shared signal-driven market."""
    use_news = (mode != "rl_no_news")
    # Reset RNG identically before each mode so any performance DIFFERENCE is due
    # to the news signal, not to different random rollouts (fair comparison).
    set_seed(cfg.seed)
    train_env, test_env = make_mode_envs(cfg, prices, volumes, signal_matrix,
                                          use_news, env_config)
    ppo_cfg = PPOConfig(seed=cfg.seed, n_updates=cfg.n_updates,
                        rollout_episodes=cfg.rollout_episodes,
                        entropy_coef=cfg.entropy_coef,
                        eval_stride=cfg.eval_stride, device=cfg.device,
                        output_dir=os.path.join(cfg.output_dir, mode))
    print("\n" + "#" * 70)
    print(f"# TRAINING MODE: {mode}  (news={'ON' if use_news else 'OFF'})  "
          f"state_dim={train_env.state_dim}")
    print("#" * 70)
    trainer = PPOTrainer(train_env, ppo_cfg)
    trainer.train()
    return trainer.model, test_env


# ============================================================================
# REPORTING
# ============================================================================


def print_comparison_table(results: Dict[str, Dict]):
    print("\n" + "=" * 96)
    print("FINAL COMPARISON -- 5 MODES ON THE HELD-OUT TEST WINDOW")
    print("=" * 96)
    hdr = (f"{'mode':<22}{'Sharpe':>8}{'annRet':>9}{'annVol':>9}{'meanRet':>9}"
           f"{'maxDD':>9}{'turnov':>8}{'epRew':>9}{'finVal':>9}")
    print(hdr); print("-" * len(hdr))
    order = ["buy_hold_spy", "equal_weight", "rl_no_news", "rl_naive",
             "rl_base_llm", "rl_finetuned"]
    for mode in order:
        if mode not in results:
            continue
        m = results[mode]
        print(f"{MODE_LABELS[mode]:<22}{m['sharpe']:>8.2f}{m['ann_return']:>8.1%}"
              f"{m['ann_vol']:>8.1%}{m['mean_total_return']:>8.2%}"
              f"{m['mean_max_drawdown']:>8.2%}{m['mean_turnover']:>8.2f}"
              f"{m['mean_episode_reward']:>9.2f}{m['final_value']:>9.4f}")
    print("=" * 96)


def plot_final_comparison(cfg: IntegrationConfig, results: Dict[str, Dict]):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot] matplotlib unavailable ({exc}); skipping figures.")
        return
    order = [m for m in ["buy_hold_spy", "equal_weight", "rl_no_news", "rl_naive",
                         "rl_base_llm", "rl_finetuned"] if m in results]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Chapter 12 Capstone -- 5-Mode Comparison (held-out test)", fontsize=15)

    # (1) Mean cumulative-return / equity curve.
    ax = axes[0, 0]
    for mode in order:
        m = results[mode]; c = m["mean_equity_curve"]; steps = np.arange(len(c))
        ax.plot(steps, c, color=MODE_COLORS[mode], lw=2, label=MODE_LABELS[mode])
        ax.fill_between(steps, c - m["std_equity_curve"], c + m["std_equity_curve"],
                        color=MODE_COLORS[mode], alpha=0.08)
    ax.axhline(1.0, color="k", ls="--", alpha=0.4)
    ax.set_title("Mean Portfolio Value over Episode"); ax.set_xlabel("day"); ax.set_ylabel("value")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (2) Mean drawdown curve.
    ax = axes[0, 1]
    for mode in order:
        c = results[mode]["mean_equity_curve"]; peak = np.maximum.accumulate(c)
        ax.plot(np.arange(len(c)), (c / peak - 1.0) * 100, color=MODE_COLORS[mode],
                lw=2, label=MODE_LABELS[mode])
    ax.set_title("Mean Drawdown (%)"); ax.set_xlabel("day"); ax.grid(True, alpha=0.3)

    # (3) Risk/return scatter (annualized).
    ax = axes[1, 0]
    for mode in order:
        m = results[mode]
        ax.scatter(m["ann_vol"] * 100, m["ann_return"] * 100, color=MODE_COLORS[mode],
                   s=90, label=MODE_LABELS[mode])
        ax.annotate(f"Sharpe {m['sharpe']:.2f}", (m["ann_vol"] * 100, m["ann_return"] * 100),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_title("Annualized Risk vs. Return"); ax.set_xlabel("ann. volatility (%)")
    ax.set_ylabel("ann. return (%)"); ax.grid(True, alpha=0.3)

    # (4) Sharpe bar chart.
    ax = axes[1, 1]
    xs = np.arange(len(order))
    ax.bar(xs, [results[m]["sharpe"] for m in order], color=[MODE_COLORS[m] for m in order])
    ax.set_xticks(xs); ax.set_xticklabels([MODE_LABELS[m] for m in order], rotation=20, ha="right")
    ax.set_title("Sharpe Ratio by Mode"); ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    p1 = os.path.join(cfg.output_dir, "final_comparison.png")
    plt.savefig(p1, dpi=150); plt.close()
    print(f"[plot] saved {p1}")

    # Exposure plot: mean weight per asset over the episode, for the RL modes.
    rl_order = [m for m in order if m.startswith("rl_")]
    fig, axes = plt.subplots(1, len(rl_order), figsize=(5 * len(rl_order), 4), squeeze=False)
    fig.suptitle("Average Asset Exposure over Episode (RL modes)", fontsize=14)
    labels = list(ASSET_TICKERS) + [CASH_TICKER]
    for k, mode in enumerate(rl_order):
        ax = axes[0, k]; W = results[mode]["mean_weights"]  # (T, n_slots)
        ax.stackplot(np.arange(W.shape[0]), *[W[:, j] for j in range(W.shape[1])],
                     labels=labels, alpha=0.85)
        ax.set_title(MODE_LABELS[mode]); ax.set_xlabel("day"); ax.set_ylim(0, 1)
        if k == 0:
            ax.set_ylabel("weight"); ax.legend(fontsize=6, loc="upper right")
    plt.tight_layout()
    p2 = os.path.join(cfg.output_dir, "exposure_by_mode.png")
    plt.savefig(p2, dpi=150); plt.close()
    print(f"[plot] saved {p2}")


def print_qualitative(cfg: IntegrationConfig, timeline_meta, matrices, oracle, k: int = 3):
    """Show news -> naive/base/fine-tuned signals for a few high-signal test days."""
    print("\n" + "=" * 70)
    print("QUALITATIVE: news snippet -> signals (naive vs base vs fine-tuned vs oracle)")
    print("=" * 70)
    n = len(timeline_meta)
    test_start = int((cfg.train_frac + cfg.val_frac) * n)
    # pick the test-window days where the oracle signal is strongest
    strengths = [(i, float(np.abs(oracle[i]).sum())) for i in range(test_start, n)]
    strengths.sort(key=lambda x: x[1], reverse=True)
    show = [i for i, _ in strengths[:k]]
    for i in show:
        meta = timeline_meta[i]
        print(f"\n--- {meta['date']} ---")
        print(meta["input"][:220])
        for mode in ("rl_naive", "rl_base_llm", "rl_finetuned"):
            v = matrices[mode][i]
            top = sorted(zip(SIGNAL_FIELDS, v), key=lambda x: -abs(x[1]))[:4]
            print(f"  {MODE_LABELS[mode]:<20} " + ", ".join(f"{n}={val:+.2f}" for n, val in top))
        top = sorted(zip(SIGNAL_FIELDS, oracle[i]), key=lambda x: -abs(x[1]))[:4]
        print(f"  {'ORACLE (DeepSeek)':<20} " + ", ".join(f"{n}={val:+.2f}" for n, val in top))


def save_report(cfg: IntegrationConfig, results: Dict[str, Dict]):
    slim = {}
    for mode, m in results.items():
        slim[mode] = {k: v for k, v in m.items()
                      if k not in ("mean_equity_curve", "std_equity_curve", "mean_weights")}
    with open(os.path.join(cfg.output_dir, "final_report.json"), "w") as f:
        json.dump(slim, f, indent=2)
    print(f"[report] wrote {cfg.output_dir}/final_report.json")

    # Also persist the RAW curve/weight arrays so ANY new visualization can be made
    # later from data in hand (no need to re-run or reload the trained agents).
    arrays = {}
    for mode, m in results.items():
        arrays[f"{mode}__equity"] = m["mean_equity_curve"]
        arrays[f"{mode}__equity_std"] = m["std_equity_curve"]
        arrays[f"{mode}__weights"] = m["mean_weights"]
    np.savez_compressed(os.path.join(cfg.output_dir, "eval_curves.npz"), **arrays)
    print(f"[report] wrote {cfg.output_dir}/eval_curves.npz (equity + weight arrays)")


# ============================================================================
# ORCHESTRATION
# ============================================================================


def stage_signals(cfg: IntegrationConfig):
    """Build per-mode signal matrices + the signal-driven market; cache to disk."""
    set_seed(cfg.seed)
    timeline = load_timeline(cfg)
    oracle_mat = np.stack([e["oracle"] for e in timeline])
    base_mat, tuned_mat = generate_llm_signal_matrices(cfg, timeline)
    matrices = build_signal_matrices(cfg, timeline, base_mat, tuned_mat)
    prices, volumes = generate_signal_driven_market(oracle_mat, cfg)
    save_cache(cfg, prices, volumes, matrices, timeline)
    return prices, volumes, matrices, timeline


def stage_train(cfg: IntegrationConfig):
    """Train each RL mode, evaluate every mode on the test window, and report."""
    set_seed(cfg.seed)
    _, _, matrices, oracle, meta = load_cache(cfg)
    # Regenerate the market from the cached ORACLE signals using the CURRENT DGP
    # parameters. This lets us iterate on signal_coef / drift / noise by re-running
    # only `train` (cheap) without re-running the GPU signal generation.
    prices, volumes = generate_signal_driven_market(oracle, cfg)
    env_config = EnvConfig(seed=cfg.seed,
                           drawdown_penalty_coef=cfg.drawdown_penalty_coef,
                           vol_penalty_coef=cfg.vol_penalty_coef)

    results: Dict[str, Dict] = {}
    # RL modes: train then evaluate on their own test env (same market).
    for mode in cfg.rl_modes:
        model, test_env = train_rl_mode(cfg, mode, prices, volumes,
                                        matrices.get(mode), env_config)
        results[mode] = evaluate_mode(cfg, mode, model, test_env)

    # Heuristics: evaluate on a plain test env (no news needed).
    _, heur_test = make_mode_envs(cfg, prices, volumes, None, False, env_config)
    for mode in cfg.heuristic_modes:
        results[mode] = evaluate_mode(cfg, mode, None, heur_test)

    print_comparison_table(results)
    plot_final_comparison(cfg, results)
    print_qualitative(cfg, meta, matrices, oracle)
    save_report(cfg, results)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Chapter 12 -- Steps 5&6: connect LLM signals to RL + final comparison.")
    parser.add_argument("stage", choices=["signals", "train", "all"])
    parser.add_argument("--data_dir", type=str, default="./signal_dataset")
    parser.add_argument("--signal_llm_dir", type=str, default="./signal_llm_lora")
    parser.add_argument("--output_dir", type=str, default="./outputs_integration")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--n_updates", type=int, default=None)
    parser.add_argument("--signal_coef", type=float, default=None)
    args = parser.parse_args()

    cfg = IntegrationConfig(data_dir=args.data_dir, signal_llm_dir=args.signal_llm_dir,
                            output_dir=args.output_dir)
    if args.model_name:
        cfg.model_name = args.model_name
    if args.n_updates:
        cfg.n_updates = args.n_updates
    if args.signal_coef is not None:
        cfg.signal_coef = args.signal_coef

    print("=" * 70)
    print("CHAPTER 12 -- STEPS 5 & 6: LLM-SIGNAL RL INTEGRATION + FINAL COMPARISON")
    print("=" * 70)
    if args.stage in ("signals", "all"):
        stage_signals(cfg)
    if args.stage in ("train", "all"):
        stage_train(cfg)


if __name__ == "__main__":
    main()
