"""
Chapter 12: LLM-Guided Reinforcement Learning for Portfolio Management
======================================================================
Manning Publications -- Applied Reinforcement Learning (Capstone Chapter)

STEP 1 of 6: THE MODULAR PORTFOLIO TRADING ENVIRONMENT
------------------------------------------------------
This is the capstone chapter. It ties together the ideas of Chapters 8-11:
  - Ch.8  gave us value-based control (DQN) on a discrete action space.
  - Ch.9  gave us policy gradients (REINFORCE) on a simulated environment.
  - Ch.10 taught PPO and how a *learned* signal can shape an agent.
  - Ch.11 taught GRPO + LoRA and how to fine-tune an LLM cheaply.

Chapter 12 combines two of these worlds. An RL agent manages a small ETF
portfolio, and -- this is the whole point of the chapter -- an LLM reads
financial news and turns it into *structured numeric market signals* that
become part of the agent's state. The LLM never trades. It is a STATE ENCODER.
The RL agent is the only thing that makes portfolio decisions.

    news + market context
        -> LLM signal extractor        (Steps 3-4: fine-tuned later)
        -> structured JSON signals      (the MarketSignal schema below)
        -> RL trading environment state (THIS FILE)
        -> RL portfolio decision        (Step 2)
        -> portfolio reward + evaluation

Because the chapter's lesson is about *representation quality*, the environment
must let us switch the news/LLM block of the state ON or OFF without touching
anything else. That modularity is the single most important design goal of this
file, and it is why the state is built from clearly separated blocks and why the
news features arrive as a plain numeric vector (the MarketSignal) that defaults
to zeros when the LLM is disabled.

==============================================================================
WHAT THIS FILE CONTAINS
==============================================================================
- EnvConfig        : one dataclass holding every knob (Ch.9/10/11 style)
- MarketSignal     : the LLM <-> environment contract (structured news signal)
- ACTION_TEMPLATES : 8 discrete allocation templates (the action space)
- generate_synthetic_market_data : seeded, offline, realistic-ish ETF prices
- PortfolioTradingEnv : reset / step / state construction / reward / costs
- a smoke test in main() that runs one random-action episode and reports stats

We deliberately keep dependencies tiny (numpy only for the environment;
matplotlib is optional and only used for an illustrative plot) so this code
stays runnable for years -- the same discipline we followed in Chapters 10-11.

==============================================================================
MODELING CHOICES (and why they are honest)
==============================================================================
- DAILY rebalancing over a short episode (default 30 trading days). Short
  episodes make the sequential decision problem easy to visualize and fast to
  train, and let us start each episode from a different slice of history.
- NO LOOK-AHEAD BIAS. At decision day t the agent sees only features computed
  from data up to and including day t. It then picks target weights, and the
  reward is the return realized from day t to day t+1. Information the agent
  could not have known at t never enters its state.
- This is a TEACHING system, not a profit claim. We use synthetic data by
  default, ignore many market frictions, and keep the reward simple. The goal
  is to show how better state representations help an RL agent -- nothing more.
"""

import argparse
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ============================================================================
# CONFIGURATION
# ============================================================================
# As in Chapters 9-11, every tunable lives in a single dataclass. Grouped
# headers keep the knobs organized, and each default is chosen for a readable,
# fast-to-run book example rather than for maximum realism.

# The tradable universe. CASH is always the last "asset" and always available.
ASSET_TICKERS: Tuple[str, ...] = ("SPY", "QQQ", "IWM", "XLE", "XLF", "XLV")
CASH_TICKER: str = "CASH"
ALL_TICKERS: Tuple[str, ...] = ASSET_TICKERS + (CASH_TICKER,)

# Human-readable descriptions, used only for printing/plots.
ASSET_DESCRIPTIONS: Dict[str, str] = {
    "SPY": "broad U.S. market",
    "QQQ": "technology / growth",
    "IWM": "small caps",
    "XLE": "energy",
    "XLF": "financials",
    "XLV": "healthcare / defensive",
    "CASH": "cash position",
}


@dataclass
class EnvConfig:
    """Central configuration for the portfolio trading environment."""

    # --- universe & episode ---
    tickers: Tuple[str, ...] = ASSET_TICKERS      # risky assets (cash added internally)
    horizon: int = 30                             # trading days per episode (decisions)
    lookback: int = 20                            # days of history used for features
    initial_capital: float = 1.0                  # start each episode at $1 (returns are unit-free)

    # --- market data (synthetic by default; see generate_synthetic_market_data) ---
    n_days: int = 400                             # length of the generated price history
    data_source: str = "synthetic"               # "synthetic" now; "csv" seam left for later

    # --- execution & risk ---
    tx_cost_rate: float = 0.001                   # 10 bps per unit of one-way turnover
    max_position: float = 0.50                    # cap any single asset weight (agent sees this)

    # --- reward weights (reward = return - cost - turnover - drawdown - vol) ---
    # The headline reward mirrors the chapter text:
    #   reward = portfolio_return - transaction_cost - turnover_penalty
    #            - drawdown_penalty - volatility_penalty
    # Each penalty coefficient below scales one of those terms. They are small so
    # the daily return stays the dominant signal; raise them to make the agent
    # more conservative.
    turnover_penalty_coef: float = 0.001
    drawdown_penalty_coef: float = 0.10
    vol_penalty_coef: float = 0.05
    reward_scale: float = 100.0                   # scale decimal returns to ~O(1) for stable RL

    # --- news / LLM signal block (the modular switch) ---
    use_news: bool = False                        # False => news features are all zeros
    # When use_news is True the env reads a per-day MarketSignal from a matrix
    # supplied at construction (Steps 3-5 produce that matrix from an LLM). The
    # dimension is fixed by the MarketSignal schema below.

    # --- feature toggles (let the book turn blocks on/off for ablations) ---
    include_market_features: bool = True
    include_portfolio_features: bool = True
    include_execution_features: bool = True

    # --- reproducibility ---
    seed: int = 42

    def __post_init__(self):
        self.tickers = tuple(self.tickers)
        assert self.lookback >= 5, "lookback must be >= 5 (needed for 5-day momentum)"
        assert self.horizon >= 1


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility (identical helper to Chapters 9-11)."""
    random.seed(seed)
    np.random.seed(seed)
    # torch is not needed by the environment itself, but we seed it if present so
    # the whole capstone pipeline is reproducible from this one call.
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# ============================================================================
# THE LLM <-> ENVIRONMENT CONTRACT: MarketSignal
# ============================================================================
# This is the structured JSON signal the LLM will emit in Steps 3-5. Defining it
# now -- before we have any LLM -- is deliberate: it fixes the interface so every
# later "signal producer" (naive sentiment, base LLM, fine-tuned LLM) plugs into
# the SAME slot in the state. The environment only ever sees the numeric vector
# returned by to_vector(); it does not know or care who produced it.
#
# All fields are bounded so the numbers are comparable and safe to feed a neural
# net. Directional fields live in [-1, 1] (negative = bearish/easing, positive =
# bullish/tightening). Risk fields live in [0, 1] (0 = calm, 1 = extreme).


@dataclass
class MarketSignal:
    """Structured market signal extracted from news/macro context by the LLM.

    This is the canonical 13-field schema produced by the LLM signal extractor
    (Steps 3-4). The field ORDER here defines the layout of the news block in the
    state and must match `SIGNAL_FIELDS` in build_signal_dataset.py. Twelve are
    directional/risk signals in [-1, 1]; `confidence` is in [0, 1].
    """

    market_sentiment: float = 0.0     # overall mood: -1 bearish .. +1 bullish
    risk_on_signal: float = 0.0       # -1 risk-off (flight to safety) .. +1 risk-on
    rates_pressure: float = 0.0       # -1 easing expected .. +1 tightening expected
    inflation_pressure: float = 0.0   # -1 disinflation .. +1 rising inflation
    growth_signal: float = 0.0        # -1 slowdown .. +1 strong growth
    tech_signal: float = 0.0          # technology / growth-stock outlook
    energy_signal: float = 0.0        # energy sector outlook
    financials_signal: float = 0.0    # financials / banks outlook
    defensive_signal: float = 0.0     # defensive sectors (healthcare/staples) outlook
    volatility_risk: float = 0.0      # -1 unusually calm .. +1 high expected vol
    liquidity_risk: float = 0.0       # -1 ample liquidity .. +1 tightening/stress
    geopolitical_risk: float = 0.0    # -1 de-escalation .. +1 high stress
    confidence: float = 0.0           # 0 (guessing) .. 1 (clear signal)

    # The 13 field names, in vector order (kept in sync with SIGNAL_FIELDS).
    FIELDS: Tuple[str, ...] = (
        "market_sentiment", "risk_on_signal", "rates_pressure", "inflation_pressure",
        "growth_signal", "tech_signal", "energy_signal", "financials_signal",
        "defensive_signal", "volatility_risk", "liquidity_risk", "geopolitical_risk",
        "confidence",
    )

    def to_vector(self) -> np.ndarray:
        """Flatten the 13 signal fields into the news-block vector, in FIELDS order."""
        return np.asarray([getattr(self, f) for f in self.FIELDS], dtype=np.float32)

    @staticmethod
    def dim() -> int:
        """Length of to_vector(): the 13 canonical signal fields."""
        return len(MarketSignal.FIELDS)

    @staticmethod
    def feature_names() -> List[str]:
        """Human-readable names for each entry of to_vector() (for logging/plots)."""
        return list(MarketSignal.FIELDS)


# ============================================================================
# ACTION SPACE: DISCRETE ALLOCATION TEMPLATES (Option A)
# ============================================================================
# Instead of letting the agent output raw continuous weights, it picks one of a
# few hand-designed portfolios ("templates"). This keeps the action space small
# and every decision human-readable ("the agent went defensive on day 12"),
# which is ideal for a teaching chapter -- and it is directly compatible with a
# DQN (Ch.8) or a categorical-policy PPO/GRPO (Ch.10/11) in Step 2.
#
# Each template is a target weight vector over [SPY, QQQ, IWM, XLE, XLF, XLV,
# CASH] that sums to 1. All templates respect the max_position cap by design.

_RAW_TEMPLATES: Dict[str, List[float]] = {
    #                  SPY   QQQ   IWM   XLE   XLF   XLV   CASH
    "cash_heavy":     [0.10, 0.05, 0.00, 0.00, 0.00, 0.05, 0.80],
    "defensive":      [0.20, 0.05, 0.00, 0.00, 0.05, 0.30, 0.40],
    "equal_weight":   [1/6,  1/6,  1/6,  1/6,  1/6,  1/6,  0.00],
    "growth_tech":    [0.25, 0.45, 0.10, 0.00, 0.05, 0.05, 0.10],
    "energy_tilt":    [0.20, 0.05, 0.05, 0.45, 0.05, 0.05, 0.15],
    "financials_tilt":[0.20, 0.05, 0.05, 0.05, 0.45, 0.05, 0.15],
    "broad_risk_on":  [0.40, 0.30, 0.20, 0.05, 0.05, 0.00, 0.00],
    "risk_off":       [0.15, 0.00, 0.00, 0.00, 0.00, 0.25, 0.60],
}

# Freeze the ordered list of action names and a normalized weight matrix.
ACTION_NAMES: Tuple[str, ...] = tuple(_RAW_TEMPLATES.keys())
# (n_actions, n_assets+1) matrix; each row re-normalized to sum to exactly 1.
ACTION_TEMPLATES: np.ndarray = np.array(
    [np.array(w, dtype=np.float64) / float(np.sum(w)) for w in _RAW_TEMPLATES.values()],
    dtype=np.float64,
)


# ============================================================================
# SYNTHETIC MARKET DATA (offline, seeded, no network)
# ============================================================================
# We generate daily prices with a one-factor model: every asset shares a common
# "market" shock plus its own idiosyncratic noise, scaled by an asset-specific
# beta and volatility. This produces realistic-looking correlations (SPY/QQQ/IWM
# move together; XLE is more independent) without any external data, so the smoke
# test runs anywhere. Volume is a lognormal series that spikes on big moves.
#
# A real-data loader can replace this later (data_source="csv") without changing
# the environment: all the env needs is a (n_days, n_assets) price array and a
# matching volume array.

# Per-asset (drift, beta, idiosyncratic vol, starting price, base volume).
_ASSET_DYNAMICS: Dict[str, Dict[str, float]] = {
    "SPY": {"drift": 0.0003, "beta": 1.00, "idio": 0.003, "p0": 100.0, "vol0": 8e7},
    "QQQ": {"drift": 0.0004, "beta": 1.20, "idio": 0.006, "p0": 300.0, "vol0": 5e7},
    "IWM": {"drift": 0.0001, "beta": 1.10, "idio": 0.007, "p0": 180.0, "vol0": 3e7},
    "XLE": {"drift": 0.0000, "beta": 0.80, "idio": 0.010, "p0": 80.0,  "vol0": 2e7},
    "XLF": {"drift": 0.0001, "beta": 1.05, "idio": 0.006, "p0": 35.0,  "vol0": 4e7},
    "XLV": {"drift": 0.0002, "beta": 0.70, "idio": 0.004, "p0": 120.0, "vol0": 1e7},
}


def generate_synthetic_market_data(
    tickers: Tuple[str, ...],
    n_days: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate seeded daily prices and volumes for the given tickers.

    Returns:
        prices:  (n_days, n_assets) float array of daily closing prices.
        volumes: (n_days, n_assets) float array of daily traded volume.
    """
    rng = np.random.default_rng(seed)  #A independent generator keeps data reproducible
    n_assets = len(tickers)

    # Shared market factor: a mild upward drift with daily Gaussian shocks.
    market_shock = rng.normal(0.0004, 0.010, size=n_days)  #B common component

    prices = np.zeros((n_days, n_assets), dtype=np.float64)
    volumes = np.zeros((n_days, n_assets), dtype=np.float64)

    for j, ticker in enumerate(tickers):
        dyn = _ASSET_DYNAMICS[ticker]
        idio_shock = rng.normal(0.0, dyn["idio"], size=n_days)  #C asset-specific noise
        # Daily return = drift + beta * market + idiosyncratic noise.
        asset_returns = dyn["drift"] + dyn["beta"] * market_shock + idio_shock  #D one-factor model
        asset_returns[0] = 0.0
        # Compound returns into a price path starting at p0.
        prices[:, j] = dyn["p0"] * np.cumprod(1.0 + asset_returns)  #E price = product of gross returns
        # Volume rises with the size of the move (lognormal, so always positive).
        log_vol = rng.normal(0.0, 0.30, size=n_days) + 3.0 * np.abs(asset_returns)  #F spikes on big days
        volumes[:, j] = dyn["vol0"] * np.exp(log_vol)

    return prices, volumes

#A A dedicated Generator (not global np.random) so data is stable regardless of other draws
#B Every asset is exposed to this same daily market move -> realistic correlation
#C Layered on top of the market factor, this is what makes assets differ
#D Classic single-factor return model: systematic (beta*market) + idiosyncratic
#E Cumulative product turns daily returns into a price level series
#F Trading volume tends to surge on large price moves; we mimic that loosely


# ============================================================================
# THE ENVIRONMENT
# ============================================================================


class PortfolioTradingEnv:
    """A modular daily portfolio-management environment.

    The agent starts each episode fully in cash and, once per day, chooses one
    allocation template. The environment charges transaction costs for trading,
    realizes the next day's return, and returns a shaped daily reward. The state
    is assembled from clearly separated BLOCKS so the news/LLM block can be
    switched on or off (the central experiment of this chapter).

    State blocks (concatenated in this fixed order):
      1. MARKET     : per-asset price features (returns, MA ratio, vol, drawdown, volume)
      2. NEWS/LLM   : the MarketSignal vector (all zeros when use_news is False)
      3. PORTFOLIO  : current weights, value, drawdown, time remaining
      4. EXECUTION  : last turnover, realized portfolio vol, cost & position limits
    """

    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        prices: Optional[np.ndarray] = None,
        volumes: Optional[np.ndarray] = None,
        signal_matrix: Optional[np.ndarray] = None,
    ):
        self.cfg = config if config is not None else EnvConfig()
        self.tickers = self.cfg.tickers
        self.n_assets = len(self.tickers)          # risky assets only
        self.n_slots = self.n_assets + 1           # + cash
        self.cash_idx = self.n_assets              # cash is the last slot
        self.n_actions = len(ACTION_NAMES)

        # ---- market data ----
        if prices is None or volumes is None:
            prices, volumes = generate_synthetic_market_data(
                self.tickers, self.cfg.n_days, self.cfg.seed
            )
        self.prices = np.asarray(prices, dtype=np.float64)     #A (n_days, n_assets)
        self.volumes = np.asarray(volumes, dtype=np.float64)
        self.n_days = self.prices.shape[0]
        assert self.prices.shape[1] == self.n_assets, "price columns must match tickers"

        # Precompute daily simple returns once; returns[t] is the move from t-1 to t.
        self.returns = np.zeros_like(self.prices)              #B returns[0] = 0 by construction
        self.returns[1:] = self.prices[1:] / self.prices[:-1] - 1.0

        # ---- news / LLM signals ----
        # signal_matrix, if given, is (n_days, MarketSignal.dim()): one signal row
        # per market day. When use_news is False we never read it (block = zeros).
        self.signal_matrix = None
        if signal_matrix is not None:
            signal_matrix = np.asarray(signal_matrix, dtype=np.float32)
            assert signal_matrix.shape == (self.n_days, MarketSignal.dim()), (
                f"signal_matrix must be (n_days={self.n_days}, "
                f"signal_dim={MarketSignal.dim()}), got {signal_matrix.shape}"
            )
            self.signal_matrix = signal_matrix

        # ---- state dimensionality (computed, never hardcoded) ----
        self.market_dim = 6 * self.n_assets if self.cfg.include_market_features else 0
        self.news_dim = MarketSignal.dim()          # always present; zeros when disabled
        self.portfolio_dim = (self.n_slots + 3) if self.cfg.include_portfolio_features else 0
        self.execution_dim = 4 if self.cfg.include_execution_features else 0
        self.state_dim = (
            self.market_dim + self.news_dim + self.portfolio_dim + self.execution_dim
        )
        self.action_dim = self.n_actions

        # ---- episode state (set in reset) ----
        self.day = 0                # absolute index into the price arrays
        self.start_day = 0
        self.step_count = 0
        self.weights = None         # current (drifted) weights, shape (n_slots,)
        self.value = 0.0            # current portfolio value
        self.peak_value = 0.0       # running max value (for drawdown)
        self.last_turnover = 0.0
        self.returns_history = None # recent realized net returns (for vol estimate)

        set_seed(self.cfg.seed)

    #A Prices/volumes are the only market data the env needs; a real loader can supply them
    #B Simple daily returns computed up front so step() and features never recompute them

    # ------------------------------------------------------------------
    # RESET
    # ------------------------------------------------------------------
    def reset(self, start_day: Optional[int] = None) -> np.ndarray:
        """Begin a new episode and return the initial state.

        A random (or fixed) start day is chosen so that there is enough history
        BEHIND it for the feature lookback, and enough days AHEAD of it for the
        full episode plus the one extra day whose return closes the last step.
        """
        lo = self.cfg.lookback                                  #A need lookback days of history
        hi = self.n_days - self.cfg.horizon - 1                 #B need horizon+1 days ahead
        assert hi > lo, "price history too short for this horizon/lookback"

        if start_day is None:
            self.start_day = random.randint(lo, hi)             #C random slice -> episode variety
        else:
            self.start_day = int(np.clip(start_day, lo, hi))    #D fixed start for deterministic eval

        self.day = self.start_day
        self.step_count = 0

        # Start fully in cash: an unambiguous, cost-free starting portfolio.
        self.weights = np.zeros(self.n_slots, dtype=np.float64)
        self.weights[self.cash_idx] = 1.0                       #E 100% cash at t=0
        self.value = float(self.cfg.initial_capital)
        self.peak_value = self.value
        self.last_turnover = 0.0
        self.returns_history = deque(maxlen=self.cfg.lookback)  #F for the rolling vol feature

        return self._get_state()

    #A The lower bound guarantees market features have a full lookback window
    #B The upper bound guarantees returns[day+1] exists for every step of the episode
    #C Different start days act like different "episodes" from one price history
    #D Evaluation passes a fixed start so base/tuned modes see identical markets
    #E Starting in cash means the first rebalance's turnover is well-defined
    #F Only the most recent `lookback` net returns feed the volatility estimate

    # ------------------------------------------------------------------
    # STATE CONSTRUCTION (no look-ahead: uses data up to and including self.day)
    # ------------------------------------------------------------------
    def _market_features(self, day: int) -> np.ndarray:
        """Per-asset price features computed ONLY from data up to `day`.

        For each risky asset we compute six classic, interpretable features:
          1. 1-day return
          2. 5-day momentum
          3. price / 20-day moving average - 1   (trend)
          4. 20-day volatility of returns
          5. drawdown from the 20-day high
          6. volume z-score over the lookback window
        """
        lb = self.cfg.lookback
        feats = np.zeros((self.n_assets, 6), dtype=np.float64)
        # Windows END at `day` (inclusive) so nothing after the decision leaks in.
        price_win = self.prices[day - lb + 1: day + 1]          #A shape (lb, n_assets)
        ret_win = self.returns[day - lb + 1: day + 1]
        vol_win = self.volumes[day - lb + 1: day + 1]

        for j in range(self.n_assets):
            p = self.prices[day, j]
            feats[j, 0] = self.returns[day, j]                              # 1-day return
            feats[j, 1] = self.prices[day, j] / self.prices[day - 5, j] - 1 # 5-day momentum
            ma = price_win[:, j].mean()
            feats[j, 2] = p / ma - 1.0                                      # trend vs MA
            feats[j, 3] = ret_win[:, j].std()                              # volatility
            roll_max = price_win[:, j].max()
            feats[j, 4] = p / roll_max - 1.0                               # drawdown (<= 0)
            v = vol_win[:, j]
            feats[j, 5] = (self.volumes[day, j] - v.mean()) / (v.std() + 1e-8)  # volume z-score

        return feats.reshape(-1)  #B flatten to (6 * n_assets,)

    #A Every window is sliced to end at `day`; this is where look-ahead bias is prevented
    #B One flat vector per state block keeps concatenation simple and dims predictable

    def _news_features(self, day: int) -> np.ndarray:
        """The MarketSignal block. Zeros when the LLM/news is disabled.

        This is the modular switch at the heart of the chapter. When use_news is
        False (RL with no news), or no signal matrix was supplied, the entire
        block is zeros -- the agent literally cannot see any news. When enabled,
        we read the precomputed signal for this day (produced by a naive
        extractor or an LLM in later steps).
        """
        if not self.cfg.use_news or self.signal_matrix is None:
            return np.zeros(self.news_dim, dtype=np.float64)     #A LLM OFF -> neutral zeros
        return self.signal_matrix[day].astype(np.float64)        #B LLM ON  -> that day's signal

    #A Disabling news must not change the state layout, only zero this block
    #B The signal for day t is information available at t (built from news up to t)

    def _portfolio_features(self) -> np.ndarray:
        """Internal account state: where the portfolio stands right now."""
        value_ratio = self.value / self.cfg.initial_capital - 1.0   # total return so far
        drawdown = self.value / self.peak_value - 1.0               # current drawdown (<= 0)
        time_remaining = (self.cfg.horizon - self.step_count) / self.cfg.horizon
        # Current weights (includes cash) capture "what we hold" and "last action".
        return np.concatenate([
            self.weights,                                   # (n_slots,)
            [value_ratio, drawdown, time_remaining],        # 3 scalars
        ]).astype(np.float64)

    def _execution_features(self) -> np.ndarray:
        """Execution / risk context: costs, limits, and recent portfolio risk."""
        if len(self.returns_history) >= 2:
            port_vol = float(np.std(self.returns_history))  # realized vol of the strategy
        else:
            port_vol = 0.0
        return np.array([
            self.last_turnover,          # how much we traded last step
            port_vol,                    # recent realized volatility of the portfolio
            self.cfg.tx_cost_rate,       # the cost regime the agent operates in
            self.cfg.max_position,       # the position cap it must respect
        ], dtype=np.float64)

    def _get_state(self) -> np.ndarray:
        """Assemble the full state by concatenating the enabled blocks in order."""
        blocks = []
        if self.cfg.include_market_features:
            blocks.append(self._market_features(self.day))
        blocks.append(self._news_features(self.day))            # always present (zeros if off)
        if self.cfg.include_portfolio_features:
            blocks.append(self._portfolio_features())
        if self.cfg.include_execution_features:
            blocks.append(self._execution_features())
        return np.concatenate(blocks).astype(np.float32)

    # ------------------------------------------------------------------
    # STEP
    # ------------------------------------------------------------------
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Apply one allocation decision (a discrete template) and advance a day.

        This is the action interface the PPO agent (Step 2) uses: it hands us an
        integer, we decode it into the template's target weights, and delegate to
        step_target_weights() for the actual mechanics. Keeping the mechanics in
        one place means baselines and a future continuous-weight agent share the
        exact same cost/return/reward math -- only the action DECODING differs.
        """
        assert 0 <= action < self.n_actions, f"invalid action {action}"
        target_w = ACTION_TEMPLATES[action].copy()             #A decode discrete action
        return self.step_target_weights(target_w, action_name=ACTION_NAMES[action])

    def step_target_weights(
        self, target_w: np.ndarray, action_name: str = "custom"
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        """Apply an explicit target-weight vector and advance one trading day.

        Used directly by the equal-weight and buy-and-hold baselines (Step 2) and
        available to any future continuous-action agent. `target_w` must sum to 1
        over the (n_assets + cash) slots.

        Sequence (this ORDER is what avoids look-ahead bias):
          1. Take the target weights (decided using info up to today).
          2. Charge transaction cost on the turnover from current -> target.
          3. Realize the return from today (t) to tomorrow (t+1).
          4. Update portfolio value, drawdown, and the drifted weights.
          5. Compute the shaped daily reward.
          6. Advance the clock and build the next state.
        """
        target_w = np.asarray(target_w, dtype=np.float64).copy()
        assert target_w.shape == (self.n_slots,), f"target_w must be ({self.n_slots},)"
        assert abs(target_w.sum() - 1.0) < 1e-6, "target weights must sum to 1"

        # 2) Turnover = one-way fraction of the book we must trade to reach target.
        #    (0.5 * L1 distance so a full switch out-and-in counts once, not twice.)
        turnover = 0.5 * np.sum(np.abs(target_w - self.weights))   #B in [0, 1]
        transaction_cost = self.cfg.tx_cost_rate * turnover        #C paid now, before returns

        # 3) Realize tomorrow's return. Cash earns nothing. This is the ONLY place
        #    future data is touched, and only AFTER the action is fixed.
        next_ret = np.zeros(self.n_slots, dtype=np.float64)
        next_ret[: self.n_assets] = self.returns[self.day + 1]     #D returns[t+1] = t -> t+1 move
        gross_return = float(np.dot(target_w, next_ret))           #E weighted portfolio return

        # 4) Update value (net of cost), running peak, and drawdown.
        net_return = gross_return - transaction_cost
        self.value *= (1.0 + net_return)
        self.peak_value = max(self.peak_value, self.value)
        drawdown = self.value / self.peak_value - 1.0              #F <= 0
        self.returns_history.append(net_return)

        # Weights drift with the realized returns to become tomorrow's holdings.
        grown = target_w * (1.0 + next_ret)
        denom = grown.sum()
        self.weights = grown / denom if denom > 1e-12 else target_w  #G renormalize to sum 1

        # 5) Shaped daily reward (see EnvConfig for the coefficients).
        port_vol = float(np.std(self.returns_history)) if len(self.returns_history) >= 2 else 0.0
        turnover_penalty = self.cfg.turnover_penalty_coef * turnover
        drawdown_penalty = self.cfg.drawdown_penalty_coef * (-drawdown)   # penalize magnitude
        vol_penalty = self.cfg.vol_penalty_coef * port_vol
        reward = (
            gross_return
            - transaction_cost
            - turnover_penalty
            - drawdown_penalty
            - vol_penalty
        ) * self.cfg.reward_scale                                   #H scale to O(1) for the agent

        # 6) Advance and build the next observation.
        self.last_turnover = turnover
        self.day += 1
        self.step_count += 1
        done = self.step_count >= self.cfg.horizon

        info = {
            "action_name": action_name,
            "gross_return": gross_return,
            "net_return": net_return,
            "transaction_cost": transaction_cost,
            "turnover": turnover,
            "drawdown": drawdown,
            "portfolio_value": self.value,
            "portfolio_vol": port_vol,
            "weights": self.weights.copy(),
        }
        return self._get_state(), float(reward), done, info

    #A Decoding the discrete action into fixed target weights (Option A action space)
    #B Turnover drives both the real cost and the shaping penalty
    #C Cost is charged on today's rebalance, independent of how tomorrow turns out
    #D The single controlled use of "future" data -- after the action is committed
    #E Portfolio return is the target weights dotted with next-day asset returns
    #F Drawdown is measured from the running peak of portfolio value
    #G After earning returns, weights drift; renormalizing gives next-day holdings
    #H Reward scaling keeps daily rewards near O(1), which stabilizes RL later

    # ------------------------------------------------------------------
    # Convenience helpers used by baselines / evaluation (Steps 2 & 6)
    # ------------------------------------------------------------------
    def action_index(self, name: str) -> int:
        """Look up the integer action for a template name (e.g. 'equal_weight')."""
        return ACTION_NAMES.index(name)

    def describe_state_layout(self) -> str:
        """Return a human-readable summary of how the state vector is laid out."""
        lines = ["State layout (concatenated blocks):"]
        if self.cfg.include_market_features:
            lines.append(f"  [market]     {self.market_dim:3d}  (6 features x {self.n_assets} assets)")
        lines.append(f"  [news/LLM]   {self.news_dim:3d}  ({'ON' if self.cfg.use_news else 'OFF -> zeros'})")
        if self.cfg.include_portfolio_features:
            lines.append(f"  [portfolio]  {self.portfolio_dim:3d}  (weights + value + drawdown + time)")
        if self.cfg.include_execution_features:
            lines.append(f"  [execution]  {self.execution_dim:3d}  (turnover, vol, cost, max_pos)")
        lines.append(f"  = state_dim  {self.state_dim:3d}")
        return "\n".join(lines)


# ============================================================================
# SMOKE TEST
# ============================================================================
# A short, self-contained sanity check: build the environment, run ONE episode
# with random actions, and print the diagnostics the chapter asks for. This is
# not training -- it only proves the environment resets, steps, avoids
# look-ahead, and produces sensible reward/value/turnover numbers.


def _max_drawdown(values: List[float]) -> float:
    """Largest peak-to-trough drop of a value series, as a negative fraction."""
    values = np.asarray(values, dtype=np.float64)
    running_peak = np.maximum.accumulate(values)
    drawdowns = values / running_peak - 1.0
    return float(drawdowns.min())


def run_smoke_test(config: EnvConfig, demo_news: bool = True):
    """Run one random-action episode and report environment diagnostics."""
    print("=" * 70)
    print("CHAPTER 12 - STEP 1: PORTFOLIO TRADING ENVIRONMENT SMOKE TEST")
    print("=" * 70)
    print(f"Universe : {', '.join(config.tickers)} + {CASH_TICKER}")
    print(f"Horizon  : {config.horizon} trading days   Lookback: {config.lookback} days")
    print(f"Actions  : {len(ACTION_NAMES)} allocation templates -> {list(ACTION_NAMES)}")
    print(f"Seed     : {config.seed}")
    print("-" * 70)

    set_seed(config.seed)
    env = PortfolioTradingEnv(config)

    # --- reset & state shape ---
    state = env.reset(start_day=config.lookback)   # fixed start so the test is deterministic
    print(env.describe_state_layout())
    print(f"\nInitial state shape : {state.shape}   (dtype={state.dtype})")
    print(f"Initial state[:8]   : {np.round(state[:8], 4)}")
    print(f"Initial weights     : {dict(zip(ALL_TICKERS, np.round(env.weights, 3)))}")

    # --- run one random-action episode ---
    values = [env.value]
    rewards, turnovers, actions_taken = [], [], []
    done = False
    first_action, first_reward = None, None

    while not done:
        action = random.randint(0, env.n_actions - 1)    #A random policy -> pure env test
        state, reward, done, info = env.step(action)
        if first_action is None:
            first_action, first_reward = action, reward
        values.append(info["portfolio_value"])
        rewards.append(reward)
        turnovers.append(info["turnover"])
        actions_taken.append(info["action_name"])

    # --- report ---
    total_return = env.value / config.initial_capital - 1.0
    mdd = _max_drawdown(values)
    print("\n" + "-" * 70)
    print("SAMPLE TRANSITION (first step)")
    print(f"  action taken   : {first_action} ({ACTION_NAMES[first_action]})")
    print(f"  reward         : {first_reward:.4f}")
    print("\nEPISODE TRAJECTORY (portfolio value, starting at "
          f"{config.initial_capital:.2f})")
    traj = "  " + "  ".join(f"{v:.4f}" for v in values)
    # Wrap the trajectory so it prints tidily.
    for i in range(0, len(values), 10):
        chunk = values[i:i + 10]
        print("  " + " ".join(f"{v:6.4f}" for v in chunk))

    print("\nEPISODE SUMMARY")
    print(f"  steps taken         : {len(rewards)}")
    print(f"  mean daily reward   : {np.mean(rewards):+.4f}")
    print(f"  final value         : {env.value:.4f}")
    print(f"  total return        : {total_return:+.2%}")
    print(f"  max drawdown        : {mdd:+.2%}")
    print(f"  total turnover      : {np.sum(turnovers):.3f}")
    print(f"  mean turnover/step  : {np.mean(turnovers):.3f}")

    # --- demonstrate the modular news switch ---
    if demo_news:
        print("\n" + "-" * 70)
        print("NEWS/LLM BLOCK MODULARITY CHECK")
        news_off = env._news_features(env.start_day)
        print(f"  use_news=False -> news block all zeros? {np.allclose(news_off, 0.0)}"
              f"  (dim={news_off.shape[0]})")

        # Build a tiny random signal matrix (a STAND-IN for the LLM in Steps 3-5)
        # and rebuild the env with news enabled to show the block turns on.
        rng = np.random.default_rng(config.seed)
        fake_signals = rng.uniform(-1.0, 1.0, size=(env.n_days, MarketSignal.dim()))
        cfg_on = EnvConfig(**{**config.__dict__})
        cfg_on.use_news = True
        env_on = PortfolioTradingEnv(cfg_on, prices=env.prices, volumes=env.volumes,
                                     signal_matrix=fake_signals.astype(np.float32))
        env_on.reset(start_day=config.lookback)
        news_on = env_on._news_features(env_on.start_day)
        named = dict(zip(MarketSignal.feature_names(), np.round(news_on, 3)))
        print(f"  use_news=True  -> news block non-zero? {not np.allclose(news_on, 0.0)}")
        print(f"  example signal : {named}")
        print(f"  state_dim (off vs on): {env.state_dim} vs {env_on.state_dim} "
              f"(identical layout, only the values change)")

    print("=" * 70)
    print("Smoke test complete. Environment is ready for Step 2 (the RL agent).")
    print("=" * 70)
    return env

#A Random actions exercise every code path in step() without needing a trained agent


def main():
    parser = argparse.ArgumentParser(
        description="Chapter 12 - Step 1: portfolio trading environment smoke test."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=int, default=30, help="Trading days per episode.")
    parser.add_argument("--n_days", type=int, default=400, help="Length of synthetic history.")
    parser.add_argument("--use_news", action="store_true",
                        help="Enable the news/LLM block (uses random placeholder signals).")
    args = parser.parse_args()

    config = EnvConfig(
        seed=args.seed,
        horizon=args.horizon,
        n_days=args.n_days,
        use_news=args.use_news,
    )
    run_smoke_test(config)


if __name__ == "__main__":
    main()
