"""
Chapter 12 -- Step 3: Build the News-to-Signal Fine-Tuning Dataset
==================================================================
Manning Publications -- Applied Reinforcement Learning (Capstone Chapter)

This script builds (and validates) the dataset used to fine-tune the LLM
"signal extractor" in Step 4. The extractor's one job is to read financial
news (plus a little market context) and emit a STRUCTURED JSON market signal.
It never recommends trades -- it only describes market context. The RL agent
(Steps 1-2) is the only component that makes portfolio decisions.

    news headlines + (optional) market snapshot
        -> LLM signal extractor
        -> structured JSON signal  (the 13-field schema below)
        -> RL trading environment state (Step 5)

WHY A SEPARATE SCRIPT
---------------------
Dataset construction is deliberately decoupled from the RL code: it has its own
dependencies (`datasets`, `openai`), its own outputs (JSONL files), and its own
lifecycle. The only contract it shares with the environment is the SIGNAL
SCHEMA -- the set of signal names. (Those names are the general, open-sourceable
13-field schema; a thin adapter in Step 5 maps them onto the environment's news
vector, so the dataset stays reusable and not tied to our specific ETFs.)

PIPELINE (staged, resumable, cost-gated)
----------------------------------------
    collect  ->  snapshot  ->  naive  ->  label  ->  clean  ->  build  ->  stats
    (news)      (market)     (rules)    (DeepSeek)  (validate) (JSONL)   (report)

Each stage writes a file and the next stage reads it, so you can run them one at
a time, inspect the output, and resume without recomputing (crucial because the
DeepSeek labeling stage costs real money -- it is resumable and can be piloted
on a small sample first).

SOURCES (easiest + legally safe first; store headlines/metadata, not full text)
-------------------------------------------------------------------------------
- FNSPID (HuggingFace `Zihan1004/FNSPID`): dated financial headlines across many
  tickers. We stream it and keep only headline + date + ticker + url + publisher.
- GDELT DOC 2.0 API: broad macro / geopolitical news headlines with dates. Free,
  no key. Feeds the macro and geopolitical_risk signals.
- FiQA (`pauri32/fiqa-2018`): financial sentiment sentences (undated). Adds
  language diversity for the sentiment mapping.
- Market snapshot (optional): compact daily features (return / vol / trend) for a
  small ETF set, from a bulk historical download (Stooq), cached once.

THREE-WAY COMPARISON (designed in)
----------------------------------
The same 13-field schema is emitted by all three "signal producers" we compare
later: (1) a naive rule-based extractor (in this file), (2) a base off-the-shelf
LLM, and (3) the fine-tuned LLM (Steps 4-5). Building the naive labels here means
mode (1) needs no extra code later.

Dependencies: `datasets` (news), `openai` (DeepSeek client). Both are imported
lazily inside the stages that need them, so the offline stages (naive signals,
validation, JSONL building, and the `selftest`) run with just the standard
library + numpy.
"""

import argparse
import hashlib
import json
import os
import random
import re
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class DataConfig:
    """Every knob for dataset construction (Chapters 9-11 dataclass style)."""

    # --- target sizes ---
    n_train: int = 6500
    n_test: int = 500

    # --- which sources to use ---
    use_fnspid: bool = True
    use_gdelt: bool = True
    use_fiqa: bool = True

    # --- FNSPID streaming ---
    fnspid_dataset: str = "Zihan1004/FNSPID"
    fnspid_max_rows: int = 300_000        # cap how many rows we stream (memory/time)
    fnspid_shuffle_buffer: int = 20_000   # reservoir buffer (FNSPID is ticker-sorted; mixes dates)

    # --- GDELT windows ---
    gdelt_start_year: int = 2017
    gdelt_end_year: int = 2023
    gdelt_max_records: int = 250          # per monthly window (GDELT hard cap is 250)
    gdelt_sleep_s: float = 5.0            # GDELT rate-limits hard; 5s between windows
    gdelt_max_retries: int = 3            # exponential backoff on HTTP 429 (5s, 10s)

    # --- day bucketing (dated news -> one example per market day) ---
    min_headlines_per_day: int = 3
    max_headlines_per_example: int = 12
    fiqa_fraction: float = 0.10           # fraction of examples drawn from FiQA sentences

    # --- curation (quality filter run before labeling) ---
    curate_min_relevant: int = 1          # daily example must keep >= this many market-relevant headlines
    curate_min_richness: float = 0.10     # min |naive-signal| magnitude to keep (drops noise-only days)

    # --- market snapshot (optional context) ---
    use_market_snapshot: bool = True
    snapshot_tickers: Tuple[str, ...] = ("SPY", "QQQ", "IWM", "XLE", "XLF", "XLV")

    # --- DeepSeek labeling ---
    model: str = "deepseek-v4-pro"
    reasoning_effort: str = "medium"      # "low"/"medium"/"high" -- cost grows with effort
    thinking: bool = True                 # DeepSeek "thinking" mode (extra_body)
    # QUALITY-FIRST default: one example per call (full model attention, no
    # cross-example interference or truncation). Raise to amortize the prompt if
    # you want to trade some quality for lower cost.
    label_batch_size: int = 1
    # Concurrency is what makes labeling fast: many API calls run in parallel, so
    # thousands of reasoning-model calls finish in tens of minutes, not hours.
    label_concurrency: int = 32
    max_completion_tokens: int = 4096
    api_timeout_s: float = 120.0
    api_max_retries: int = 4

    # --- paths & misc ---
    data_dir: str = "./signal_dataset"
    seed: int = 42

    # derived file paths
    @property
    def f_raw(self) -> str: return os.path.join(self.data_dir, "raw_news.jsonl")
    @property
    def f_raw_uncurated(self) -> str: return os.path.join(self.data_dir, "raw_news_uncurated.jsonl")
    @property
    def f_snapshot(self) -> str: return os.path.join(self.data_dir, "market_history.json")
    @property
    def f_naive(self) -> str: return os.path.join(self.data_dir, "naive_labels.jsonl")
    @property
    def f_labeled(self) -> str: return os.path.join(self.data_dir, "labeled_raw.jsonl")
    @property
    def f_clean(self) -> str: return os.path.join(self.data_dir, "clean.jsonl")
    @property
    def f_train(self) -> str: return os.path.join(self.data_dir, "train.jsonl")
    @property
    def f_test(self) -> str: return os.path.join(self.data_dir, "test.jsonl")
    @property
    def f_report(self) -> str: return os.path.join(self.data_dir, "validation_report.json")


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility (same helper as the RL code)."""
    random.seed(seed)
    np.random.seed(seed)


# ============================================================================
# THE SIGNAL SCHEMA (canonical, general, open-sourceable)
# ============================================================================
# This is the contract every signal producer (naive / base LLM / fine-tuned LLM)
# must satisfy. It is intentionally NOT tied to our specific ETF universe -- it
# describes market *context* with sector THEMES, so the dataset could be reused
# for any equity strategy. Step 5 maps these themes onto the environment's news
# vector (e.g. tech_signal -> QQQ, energy_signal -> XLE).

SIGNAL_FIELDS: Tuple[str, ...] = (
    "market_sentiment",     # overall mood: -1 very bearish .. +1 very bullish
    "risk_on_signal",       # -1 risk-off (flight to safety) .. +1 risk-on
    "rates_pressure",       # -1 easing expected .. +1 hikes/tightening expected
    "inflation_pressure",   # -1 disinflation .. +1 rising inflation
    "growth_signal",        # -1 slowdown/recession .. +1 strong growth
    "tech_signal",          # theme: technology / growth stocks outlook
    "energy_signal",        # theme: energy sector outlook
    "financials_signal",    # theme: financials/banks outlook
    "defensive_signal",     # theme: defensive sectors (healthcare/staples) outlook
    "volatility_risk",      # -1 unusually calm .. +1 high expected volatility
    "liquidity_risk",       # -1 ample liquidity .. +1 tightening/stress
    "geopolitical_risk",    # -1 de-escalation .. +1 high geopolitical stress
    "confidence",           # 0 (guessing) .. 1 (clear signal in the news)
)

# Validation ranges: per the chapter rules, every signal is in [-1, 1] EXCEPT
# `confidence`, which is in [0, 1].
SIGNAL_RANGES: Dict[str, Tuple[float, float]] = {
    f: (0.0, 1.0) if f == "confidence" else (-1.0, 1.0) for f in SIGNAL_FIELDS
}

# The system prompt is shared by (a) DeepSeek label generation and (b) the
# fine-tuning target format, so the fine-tuned model sees the same instruction.
# Kept compact: cost during labeling scales with prompt tokens on every call.
SIGNAL_SYSTEM_PROMPT = (
    "You are a financial market analyst. You read financial news headlines and an "
    "optional market snapshot, and you output a single JSON object of market "
    "signals. You DESCRIBE market context; you do NOT give trading advice or "
    "recommend any action.\n"
    "Output ONLY a JSON object with EXACTLY these keys (floats):\n"
    "  market_sentiment, risk_on_signal, rates_pressure, inflation_pressure,\n"
    "  growth_signal, tech_signal, energy_signal, financials_signal,\n"
    "  defensive_signal, volatility_risk, liquidity_risk, geopolitical_risk,\n"
    "  confidence\n"
    "Every value is in [-1, 1] except confidence, which is in [0, 1]. "
    "Negative = bearish/easing/lower-risk; positive = bullish/tightening/"
    "higher-risk. Use 0.0 when the news says nothing about a dimension. "
    "confidence reflects how clearly the news supports your signals. "
    "Return valid JSON only -- no prose, no markdown fences."
)


def zero_signal() -> Dict[str, float]:
    """A neutral signal: all zeros (used as a safe default / fallback)."""
    return {f: 0.0 for f in SIGNAL_FIELDS}


# ============================================================================
# SMALL IO HELPERS
# ============================================================================

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def stable_id(*parts: str) -> str:
    """Deterministic short id from content (so re-runs align and dedupe works)."""
    h = hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()
    return h[:16]


# ============================================================================
# STAGE 1: COLLECT RAW NEWS
# ============================================================================
# Each source is wrapped so a failure (network hiccup, dataset moved, rate limit)
# is non-fatal: we log what we got and continue with the other sources. Every
# collected article is normalized to:
#   {"date": "YYYY-MM-DD" | None, "title": str, "ticker": str|None,
#    "url": str, "source": str}


def _clean_title(title: str) -> str:
    title = re.sub(r"\s+", " ", (title or "").strip())
    return title


def collect_fnspid(cfg: DataConfig) -> List[Dict[str, Any]]:
    """Stream FNSPID and keep compact headline metadata (no full article text)."""
    print(f"[collect] FNSPID: streaming up to {cfg.fnspid_max_rows:,} rows ...")
    try:
        from datasets import load_dataset
    except Exception as exc:
        print(f"[collect] FNSPID skipped -- `datasets` unavailable ({exc})")
        return []

    out: List[Dict[str, Any]] = []
    try:
        ds = load_dataset(cfg.fnspid_dataset, split="train", streaming=True)
        # FNSPID rows are sorted by ticker, so a reservoir shuffle mixes tickers
        # and dates before we cap -- otherwise we'd over-sample early-alphabet names.
        if cfg.fnspid_shuffle_buffer > 0:
            ds = ds.shuffle(seed=cfg.seed, buffer_size=cfg.fnspid_shuffle_buffer)
        for i, row in enumerate(ds):
            if i >= cfg.fnspid_max_rows:
                break
            title = _clean_title(row.get("Article_title", ""))
            raw_date = row.get("Date", "") or ""
            if not title or len(title) < 12:
                continue
            # "2020-06-05 06:30:54 UTC" -> "2020-06-05"
            date = raw_date[:10] if len(raw_date) >= 10 and raw_date[4] == "-" else None
            out.append({
                "date": date,
                "title": title,
                "ticker": (row.get("Stock_symbol") or None),
                "url": (row.get("Url") or ""),
                "source": "fnspid",
            })
            if (i + 1) % 200_000 == 0:
                print(f"[collect] FNSPID: streamed {i+1:,} rows, kept {len(out):,} ...")
    except Exception as exc:
        print(f"[collect] FNSPID error after {len(out):,} rows: {exc}")
    print(f"[collect] FNSPID: kept {len(out):,} headlines")
    return out


def collect_gdelt(cfg: DataConfig) -> List[Dict[str, Any]]:
    """Query GDELT DOC 2.0 for macro/geopolitical headlines, month by month."""
    print(f"[collect] GDELT: {cfg.gdelt_start_year}-{cfg.gdelt_end_year} monthly windows ...")
    query = ('(inflation OR "interest rates" OR "federal reserve" OR recession OR '
             '"stock market" OR "oil prices" OR earnings OR unemployment OR '
             'sanctions OR war OR crisis) sourcelang:english')
    out: List[Dict[str, Any]] = []
    consecutive_fail = 0
    abort_threshold = 8          # if GDELT blocks this IP repeatedly, give up gracefully
    aborted = False
    for year in range(cfg.gdelt_start_year, cfg.gdelt_end_year + 1):
        if aborted:
            break
        for month in range(1, 13):
            if consecutive_fail >= abort_threshold:
                print(f"[collect] GDELT: {consecutive_fail} consecutive failures "
                      f"(likely IP rate-limited); skipping remaining windows.")
                aborted = True
                break
            start = f"{year}{month:02d}01000000"
            end_month = month % 12 + 1
            end_year = year + (1 if month == 12 else 0)
            end = f"{end_year}{end_month:02d}01000000"
            params = {
                "query": query, "mode": "artlist", "format": "json",
                "maxrecords": cfg.gdelt_max_records,
                "startdatetime": start, "enddatetime": end, "sort": "hybridrel",
            }
            url = "https://api.gdeltproject.org/api/v2/doc/doc?" + urllib.parse.urlencode(params)
            # GDELT rate-limits aggressively; retry 429s with exponential backoff.
            got = None
            for attempt in range(cfg.gdelt_max_retries):
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(req, timeout=30) as r:
                        got = json.loads(r.read().decode("utf-8", "replace"))
                    break
                except urllib.error.HTTPError as exc:
                    if exc.code == 429 and attempt < cfg.gdelt_max_retries - 1:
                        time.sleep(cfg.gdelt_sleep_s * (2 ** attempt))   # 5s,10s,20s,...
                        continue
                    print(f"[collect] GDELT {year}-{month:02d} skipped ({exc})")
                    break
                except Exception as exc:
                    print(f"[collect] GDELT {year}-{month:02d} skipped ({exc})")
                    break
            if got is not None:
                consecutive_fail = 0                 # a 200 (even with 0 articles) resets
                for a in got.get("articles", []):
                    title = _clean_title(a.get("title", ""))
                    seen = a.get("seendate", "")  # "20230115T120000Z"
                    date = (f"{seen[0:4]}-{seen[4:6]}-{seen[6:8]}"
                            if len(seen) >= 8 and seen[:8].isdigit() else None)
                    if title and len(title) >= 12:
                        out.append({"date": date, "title": title, "ticker": None,
                                    "url": a.get("url", ""), "source": "gdelt"})
            else:
                consecutive_fail += 1
            time.sleep(cfg.gdelt_sleep_s)
    print(f"[collect] GDELT: kept {len(out):,} headlines")
    return out


def collect_fiqa(cfg: DataConfig) -> List[Dict[str, Any]]:
    """Load FiQA sentiment sentences (undated) as standalone news snippets."""
    print("[collect] FiQA: loading sentiment sentences ...")
    try:
        from datasets import load_dataset
    except Exception as exc:
        print(f"[collect] FiQA skipped -- `datasets` unavailable ({exc})")
        return []
    out: List[Dict[str, Any]] = []
    try:
        ds = load_dataset("pauri32/fiqa-2018", split="train")
        for row in ds:
            title = _clean_title(row.get("sentence", ""))
            if title and len(title) >= 12:
                out.append({"date": None, "title": title,
                            "ticker": (row.get("target") or None),
                            "url": "", "source": "fiqa"})
    except Exception as exc:
        print(f"[collect] FiQA error: {exc}")
    print(f"[collect] FiQA: kept {len(out):,} sentences")
    return out


def bucket_into_examples(cfg: DataConfig, articles: List[Dict[str, Any]]
                         ) -> List[Dict[str, Any]]:
    """Turn raw articles into examples.

    Dated articles (FNSPID/GDELT) are grouped BY DAY into one market-level example
    (signals are day-level context). Undated sentences (FiQA) each become a small
    single-snippet example. Each example gets a stable id and records which
    headlines and tickers it came from.
    """
    by_day: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    undated: List[Dict[str, Any]] = []
    for a in articles:
        (by_day[a["date"]].append(a) if a["date"] else undated.append(a))

    examples: List[Dict[str, Any]] = []

    # --- dated day-examples ---
    for date, arts in by_day.items():
        # Dedupe headlines within the day, keep a diverse, capped subset.
        seen_titles, headlines, tickers, urls = set(), [], [], []
        for a in arts:
            t = a["title"]
            if t.lower() in seen_titles:
                continue
            seen_titles.add(t.lower())
            headlines.append(t)
            if a["ticker"]:
                tickers.append(a["ticker"])
            if a["url"]:
                urls.append(a["url"])
            if len(headlines) >= cfg.max_headlines_per_example:
                break
        if len(headlines) < cfg.min_headlines_per_day:
            continue
        examples.append({
            "id": stable_id(date, *headlines[:3]),
            "date": date,
            "headlines": headlines,
            "tickers": sorted(set(tickers))[:12],
            "urls": urls[:cfg.max_headlines_per_example],
            "sources": sorted(set(a["source"] for a in arts)),
            "kind": "daily",
        })

    # --- undated single-snippet examples (FiQA) ---
    for a in undated:
        examples.append({
            "id": stable_id("fiqa", a["title"]),
            "date": None,
            "headlines": [a["title"]],
            "tickers": [a["ticker"]] if a["ticker"] else [],
            "urls": [],
            "sources": [a["source"]],
            "kind": "sentence",
        })

    # Deduplicate by id.
    dedup = {e["id"]: e for e in examples}
    print(f"[collect] bucketed {len(dedup):,} unique examples "
          f"({sum(e['kind']=='daily' for e in dedup.values()):,} daily, "
          f"{sum(e['kind']=='sentence' for e in dedup.values()):,} sentence)")
    return list(dedup.values())


def stage_collect(cfg: DataConfig):
    """Run all collectors, bucket, cap to target size, and write raw_news.jsonl."""
    set_seed(cfg.seed)
    articles: List[Dict[str, Any]] = []
    if cfg.use_fnspid:
        articles += collect_fnspid(cfg)
    if cfg.use_gdelt:
        articles += collect_gdelt(cfg)
    if cfg.use_fiqa:
        articles += collect_fiqa(cfg)

    examples = bucket_into_examples(cfg, articles)

    # Split by kind so we can control the FiQA (sentence) fraction.
    daily = [e for e in examples if e["kind"] == "daily"]
    sentence = [e for e in examples if e["kind"] == "sentence"]
    daily.sort(key=lambda e: e["date"])            # chronological order (for a clean split later)
    random.shuffle(sentence)

    target = cfg.n_train + cfg.n_test
    n_sentence = min(len(sentence), int(cfg.fiqa_fraction * target))
    n_daily = min(len(daily), target - n_sentence)
    chosen = daily[-n_daily:] + sentence[:n_sentence]   # keep the most RECENT daily examples
    random.shuffle(chosen)

    write_jsonl(cfg.f_raw, chosen)
    print(f"[collect] wrote {len(chosen):,} raw examples -> {cfg.f_raw} "
          f"(target {target:,}: {n_daily:,} daily + {n_sentence:,} sentence)")
    if len(chosen) < target:
        print(f"[collect] NOTE: got {len(chosen):,} < target {target:,}. "
              f"Increase fnspid_max_rows / GDELT range, or lower min_headlines_per_day.")


# ============================================================================
# STAGE 2: MARKET SNAPSHOT (optional context, cached once)
# ============================================================================
# We download bulk daily history for a small ETF set from Stooq (free, no key,
# not rate-limited like Yahoo), cache it, and expose a compact per-date snapshot.
# Entirely optional: if the download fails, examples simply carry no snapshot.


def _download_stooq(symbol: str) -> Dict[str, float]:
    """Return {date: close} for one Stooq symbol, or {} on failure."""
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            txt = r.read().decode("utf-8", "replace")
        closes = {}
        for line in txt.strip().splitlines()[1:]:      # skip CSV header
            cols = line.split(",")
            if len(cols) >= 5:
                try:
                    closes[cols[0]] = float(cols[4])   # Date, O, H, L, Close, Vol
                except ValueError:
                    pass
        return closes
    except Exception as exc:
        print(f"[snapshot] {symbol} download failed ({exc})")
        return {}


def stage_snapshot(cfg: DataConfig):
    """Download & cache daily closes for the snapshot tickers + a VIX proxy."""
    if not cfg.use_market_snapshot:
        print("[snapshot] disabled by config")
        return
    history: Dict[str, Dict[str, float]] = {}
    for t in cfg.snapshot_tickers:
        history[t] = _download_stooq(f"{t.lower()}.us")
        print(f"[snapshot] {t}: {len(history[t]):,} daily closes")
    history["VIX"] = _download_stooq("^vix")
    print(f"[snapshot] VIX: {len(history['VIX']):,} daily closes")
    os.makedirs(cfg.data_dir, exist_ok=True)
    with open(cfg.f_snapshot, "w") as f:
        json.dump(history, f)
    print(f"[snapshot] wrote market history -> {cfg.f_snapshot}")


def _sorted_dates(closes: Dict[str, float]) -> List[str]:
    return sorted(closes.keys())


def compute_snapshot(history: Dict[str, Dict[str, float]], date: str
                     ) -> Optional[Dict[str, Any]]:
    """Compact market snapshot for `date`: per-ticker 1d return + 20d trend, VIX.

    Uses only data up to `date` (no look-ahead). Returns None if we have no
    history at or before that date.
    """
    if not history:
        return None
    snap: Dict[str, Any] = {}
    for t, closes in history.items():
        if not closes:
            continue
        dates = _sorted_dates(closes)
        # Most recent trading day at or before `date`.
        idx = None
        lo, hi = 0, len(dates) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if dates[mid] <= date:
                idx, lo = mid, mid + 1
            else:
                hi = mid - 1
        if idx is None or idx < 20:
            continue
        c = closes[dates[idx]]
        c_prev = closes[dates[idx - 1]]
        c_ma = np.mean([closes[dates[idx - k]] for k in range(20)])
        ret_1d = c / c_prev - 1.0 if c_prev else 0.0
        if t == "VIX":
            snap["vix"] = round(c, 2)
        else:
            snap[t] = {"ret_1d": round(ret_1d, 4), "trend_20d": round(c / c_ma - 1.0, 4)}
    return snap or None


# ============================================================================
# STAGE 3: NAIVE RULE-BASED SIGNALS (comparison mode 1)
# ============================================================================
# A transparent, dependency-free extractor: count keyword hits per dimension and
# map them to [-1, 1]. It is intentionally crude -- its whole purpose is to be
# the "cheap baseline" the LLM must beat. It emits the SAME 13-field schema.

_POS_WORDS = ["surge", "rally", "gain", "jump", "beat", "record high", "soar",
              "optimism", "upgrade", "profit", "growth", "recover", "boost", "strong"]
_NEG_WORDS = ["plunge", "slump", "fall", "drop", "miss", "record low", "crash",
              "fear", "downgrade", "loss", "recession", "weak", "cut", "slump", "tumble"]
_RISK_OFF = ["safe haven", "flight to safety", "sell-off", "selloff", "risk-off",
             "panic", "volatility", "defensive", "treasuries", "gold"]
_RISK_ON = ["risk-on", "risk appetite", "rally", "record high", "bullish", "inflows"]
_RATES_UP = ["rate hike", "hike", "tightening", "hawkish", "raise rates", "yields rise"]
_RATES_DOWN = ["rate cut", "cut rates", "dovish", "easing", "yields fall"]
_INFLATION_UP = ["inflation", "cpi rises", "price pressure", "hot cpi", "prices rise"]
_INFLATION_DOWN = ["disinflation", "cooling inflation", "cpi falls", "prices fall"]
_GROWTH_UP = ["growth", "expansion", "gdp rises", "hiring", "jobs added", "strong demand"]
_GROWTH_DOWN = ["recession", "slowdown", "contraction", "layoffs", "weak demand"]
_TECH = ["tech", "semiconductor", "chip", "software", "ai ", "nasdaq", "apple",
         "nvidia", "microsoft"]
_ENERGY = ["oil", "energy", "crude", "opec", "gas prices", "petroleum"]
_FIN = ["bank", "financial", "lending", "credit", "interest rate", "yields"]
_DEFENSIVE = ["healthcare", "pharma", "utilities", "staples", "defensive"]
_VOL = ["volatility", "vix", "turbulent", "swing", "uncertainty", "plunge", "crash"]
_LIQUIDITY = ["liquidity", "credit crunch", "funding", "tightening", "default", "bond stress"]
_GEO = ["war", "sanction", "conflict", "geopolitical", "military", "invasion", "tension"]


def _score(text: str, pos: List[str], neg: List[str]) -> Tuple[float, int]:
    """Signed score in [-1, 1] from positive vs negative keyword counts."""
    p = sum(text.count(w) for w in pos)
    n = sum(text.count(w) for w in neg)
    total = p + n
    if total == 0:
        return 0.0, 0
    return float(np.clip((p - n) / total, -1.0, 1.0)), total


def _presence(text: str, words: List[str]) -> Tuple[float, int]:
    """Unsigned presence score in [0, 1] scaled by how many keywords appear."""
    hits = sum(text.count(w) for w in words)
    return float(np.clip(hits / 3.0, 0.0, 1.0)), hits


def naive_signal(text: str) -> Dict[str, float]:
    """Rule-based mapping from news text to the 13-field signal schema."""
    t = " " + text.lower() + " "
    sig = zero_signal()
    total_hits = 0

    sig["market_sentiment"], h = _score(t, _POS_WORDS, _NEG_WORDS); total_hits += h
    risk_on, h1 = _presence(t, _RISK_ON); risk_off, h2 = _presence(t, _RISK_OFF)
    sig["risk_on_signal"] = float(np.clip(risk_on - risk_off, -1.0, 1.0)); total_hits += h1 + h2
    sig["rates_pressure"], h = _score(t, _RATES_UP, _RATES_DOWN); total_hits += h
    sig["inflation_pressure"], h = _score(t, _INFLATION_UP, _INFLATION_DOWN); total_hits += h
    sig["growth_signal"], h = _score(t, _GROWTH_UP, _GROWTH_DOWN); total_hits += h

    # Sector themes: presence * overall sentiment sign (a mentioned sector inherits
    # the headline's mood). Crude on purpose.
    mood = sig["market_sentiment"]
    for key, words in (("tech_signal", _TECH), ("energy_signal", _ENERGY),
                       ("financials_signal", _FIN), ("defensive_signal", _DEFENSIVE)):
        pres, h = _presence(t, words); total_hits += h
        # Defensive sectors tend to be favored when mood is negative -> invert.
        sign = -mood if key == "defensive_signal" else mood
        sig[key] = float(np.clip(pres * (sign if sign != 0 else 0.1), -1.0, 1.0))

    sig["volatility_risk"], h = _presence(t, _VOL); total_hits += h
    sig["liquidity_risk"], h = _presence(t, _LIQUIDITY); total_hits += h
    sig["geopolitical_risk"], h = _presence(t, _GEO); total_hits += h

    # Confidence grows with the number of keyword hits (saturating).
    sig["confidence"] = float(np.clip(total_hits / 15.0, 0.0, 1.0))
    return sig


def build_input_text(example: Dict[str, Any],
                     history: Optional[Dict[str, Dict[str, float]]] = None) -> str:
    """Format one example into the text the LLM (and fine-tuned model) reads.

    Kept general (not tied to the RL env). Reused verbatim at inference time in
    Step 5 so training and deployment see identically-formatted inputs.
    """
    lines: List[str] = []
    if example.get("date"):
        lines.append(f"Date: {example['date']}")
    if example.get("tickers"):
        lines.append(f"Tickers in the news: {', '.join(example['tickers'])}")
    if history and example.get("date"):
        snap = compute_snapshot(history, example["date"])
        if snap:
            bits = []
            if "vix" in snap:
                bits.append(f"VIX={snap['vix']}")
            for t in ("SPY", "QQQ", "XLE", "XLF", "XLV"):
                if t in snap:
                    bits.append(f"{t} {snap[t]['ret_1d']:+.2%}")
            if bits:
                lines.append("Market snapshot: " + ", ".join(bits))
    lines.append("News headlines:")
    for h in example["headlines"]:
        lines.append(f"- {h}")
    return "\n".join(lines)


def stage_naive(cfg: DataConfig):
    """Compute naive rule-based signals for every raw example (comparison mode 1)."""
    raw = read_jsonl(cfg.f_raw)
    if not raw:
        print("[naive] no raw examples; run `collect` first."); return
    rows = []
    for ex in raw:
        text = " ".join(ex["headlines"])
        rows.append({"id": ex["id"], "signals": naive_signal(text)})
    write_jsonl(cfg.f_naive, rows)
    print(f"[naive] wrote {len(rows):,} naive-signal rows -> {cfg.f_naive}")


# ============================================================================
# CURATION: quality filter run BEFORE labeling
# ============================================================================
# Real financial news feeds are mostly noise (analyst blurbs, "week ahead"
# roundups, 52-week-high lists) that carry no market signal. Labeling those
# wastes API budget and dilutes the dataset with near-zero targets. Curation
# keeps only market/macro-relevant headlines per day and drops days whose news
# has no discernible signal, concentrating spend on examples worth learning from.

# Substrings that mark a headline as boilerplate/noise (checked lowercased).
_NOISE_SUBSTRINGS = [
    "week ahead", "top performing", "top-performing", "movers", "zacks",
    "52-week", "52 week", "watchlist", "watch list", "what to watch",
    "stocks to watch", "premarket", "pre-market", "after hours", "after-hours",
    "analyst blog", "highlights:", "hit 52", "hits 52", "biggest movers",
    "midday", "market update", "earnings scheduled", "on the move",
    "trending", "unusual options", "insider", "price target", "initiates coverage",
]

# Market/macro relevance vocabulary: a headline is "relevant" if it mentions any
# of these (built from the naive lexicons plus broad macro terms).
_MARKET_KEYWORDS = sorted(set(
    _POS_WORDS + _NEG_WORDS + _RISK_OFF + _RISK_ON + _RATES_UP + _RATES_DOWN
    + _INFLATION_UP + _INFLATION_DOWN + _GROWTH_UP + _GROWTH_DOWN + _TECH
    + _ENERGY + _FIN + _DEFENSIVE + _VOL + _LIQUIDITY + _GEO
    + ["fed", "cpi", "gdp", "jobs", "unemployment", "yield", "bond", "dollar",
       "guidance", "beats", "misses", "outlook", "forecast", "stimulus",
       "tariff", "trade war", "downgrade", "upgrade", "merger", "acquisition"]
))


def _is_noise(headline: str) -> bool:
    h = headline.lower()
    return any(sub in h for sub in _NOISE_SUBSTRINGS)


def _is_relevant(headline: str) -> bool:
    h = headline.lower()
    return any(kw in h for kw in _MARKET_KEYWORDS)


def stage_curate(cfg: DataConfig):
    """Filter raw examples to signal-rich ones before labeling (quality gate).

    For each daily example: drop noise headlines, keep only market-relevant ones,
    and require both a minimum count of relevant headlines and a minimum naive
    signal magnitude. FiQA sentences (already clean sentiment) are kept as-is.
    The original raw file is backed up to raw_news_uncurated.jsonl.
    """
    raw = read_jsonl(cfg.f_raw)
    if not raw:
        print("[curate] no raw examples; run `collect` first."); return

    # Back up the uncurated raw once (so curation is repeatable / reversible).
    if not os.path.exists(cfg.f_raw_uncurated):
        write_jsonl(cfg.f_raw_uncurated, raw)
        print(f"[curate] backed up uncurated raw -> {cfg.f_raw_uncurated}")

    kept: List[Dict[str, Any]] = []
    dropped_noise = dropped_lowsig = 0
    for ex in raw:
        if ex.get("kind") == "sentence":
            kept.append(ex); continue        # FiQA: already a clean signal sentence

        # Drop boilerplate, then keep only market-relevant headlines.
        non_noise = [h for h in ex["headlines"] if not _is_noise(h)]
        relevant = [h for h in non_noise if _is_relevant(h)]
        if len(relevant) < cfg.curate_min_relevant:
            dropped_noise += 1; continue

        curated = {**ex, "headlines": relevant[:cfg.max_headlines_per_example]}
        sig = naive_signal(" ".join(curated["headlines"]))
        richness = sum(abs(sig[f]) for f in SIGNAL_FIELDS if f != "confidence")
        if richness < cfg.curate_min_richness:
            dropped_lowsig += 1; continue
        curated["richness"] = round(float(richness), 3)
        kept.append(curated)

    # Rank daily examples by richness (strongest signals first); keep FiQA too.
    daily = sorted([e for e in kept if e.get("kind") == "daily"],
                   key=lambda e: e.get("richness", 0.0), reverse=True)
    sentence = [e for e in kept if e.get("kind") == "sentence"]
    curated_all = daily + sentence
    random.seed(cfg.seed); random.shuffle(curated_all)

    write_jsonl(cfg.f_raw, curated_all)
    print(f"[curate] kept {len(curated_all):,} / {len(raw):,} examples "
          f"({len(daily):,} daily + {len(sentence):,} sentence)")
    print(f"[curate] dropped: {dropped_noise:,} noise-only days, "
          f"{dropped_lowsig:,} low-signal days")
    if daily:
        rich_vals = [e["richness"] for e in daily]
        print(f"[curate] daily richness: mean {np.mean(rich_vals):.2f}, "
              f"median {np.median(rich_vals):.2f}, max {np.max(rich_vals):.2f}")


# ============================================================================
# STAGE 4: DEEPSEEK LABELING (resumable, batched, pilot-able)
# ============================================================================


def _extract_json_objects(text: str) -> List[Dict[str, Any]]:
    """Robustly pull JSON object(s) from a model response.

    Handles a bare object, a JSON array, or objects wrapped in ```json fences.
    Returns a list of dicts (possibly empty).
    """
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    # Try a direct parse first (array or object).
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, list) else [obj]
    except Exception:
        pass
    # Fallback: find each {...} block.
    out = []
    for m in re.finditer(r"\{[^{}]*\}", text, flags=re.DOTALL):
        try:
            out.append(json.loads(m.group(0)))
        except Exception:
            continue
    return out


# Two curated few-shot exemplars. These are the single biggest lever on label
# quality: they calibrate the SCALE of every signal (what a +0.7 vs +0.3 looks
# like) and pin the exact output format. Prepended to every labeling call.
_FEW_SHOT = [
    ("Date: 2023-01-13\nNews headlines:\n"
     "- Stocks rally as inflation cools for a sixth straight month\n"
     "- Fed officials signal smaller rate hikes ahead\n"
     "- Tech shares lead gains; Nasdaq jumps 2%",
     {"market_sentiment": 0.7, "risk_on_signal": 0.6, "rates_pressure": -0.5,
      "inflation_pressure": -0.6, "growth_signal": 0.3, "tech_signal": 0.7,
      "energy_signal": 0.0, "financials_signal": 0.1, "defensive_signal": -0.2,
      "volatility_risk": -0.2, "liquidity_risk": -0.1, "geopolitical_risk": 0.0,
      "confidence": 0.8}),
    ("Date: 2022-02-24\nNews headlines:\n"
     "- Markets plunge as conflict escalates and new sanctions are announced\n"
     "- Oil surges past $100 on supply fears\n"
     "- Investors flee to safe-haven bonds and gold; VIX spikes",
     {"market_sentiment": -0.8, "risk_on_signal": -0.9, "rates_pressure": 0.1,
      "inflation_pressure": 0.5, "growth_signal": -0.4, "tech_signal": -0.5,
      "energy_signal": 0.8, "financials_signal": -0.4, "defensive_signal": 0.4,
      "volatility_risk": 0.9, "liquidity_risk": 0.5, "geopolitical_risk": 0.9,
      "confidence": 0.85}),
]


def _few_shot_messages() -> List[Dict[str, str]]:
    """Few-shot turns (user -> ideal assistant JSON) inserted after the system prompt."""
    msgs: List[Dict[str, str]] = []
    for inp, sig in _FEW_SHOT:
        msgs.append({"role": "user", "content": inp})
        msgs.append({"role": "assistant", "content": json.dumps(sig)})
    return msgs


def _deepseek_client(cfg: DataConfig):
    from openai import OpenAI
    import httpx
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set in environment (see deepseek_api.txt).")
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com",
                  timeout=httpx.Timeout(cfg.api_timeout_s, connect=30.0),
                  max_retries=cfg.api_max_retries)


def _label_call(client, cfg: DataConfig, inputs: List[str]) -> Tuple[List[Dict[str, Any]], Dict]:
    """One DeepSeek call labeling `inputs` (usually one); returns (signals, usage).

    With the quality-first default (batch_size=1) `inputs` has a single element and
    we ask for one JSON object; with batch_size>1 we ask for a JSON array in order.
    """
    messages = [{"role": "system", "content": SIGNAL_SYSTEM_PROMPT}]
    messages += _few_shot_messages()
    if len(inputs) == 1:
        messages.append({"role": "user", "content": inputs[0]})
    else:
        numbered = "\n\n".join(f"[Example {i+1}]\n{txt}" for i, txt in enumerate(inputs))
        messages.append({"role": "user", "content":
            f"Label the following {len(inputs)} examples. Return a JSON ARRAY of "
            f"{len(inputs)} objects, one per example IN ORDER.\n\n{numbered}"})
    resp = client.chat.completions.create(
        model=cfg.model,
        messages=messages,
        stream=False,
        reasoning_effort=cfg.reasoning_effort,
        extra_body={"thinking": {"type": "enabled" if cfg.thinking else "disabled"}},
        max_tokens=cfg.max_completion_tokens,
    )
    content = resp.choices[0].message.content or ""
    objs = _extract_json_objects(content)
    usage = getattr(resp, "usage", None)
    usage_d = {"prompt_tokens": getattr(usage, "prompt_tokens", 0),
               "completion_tokens": getattr(usage, "completion_tokens", 0)} if usage else {}
    return objs, usage_d


def _label_chunk(client, cfg: DataConfig, chunk: List[Dict[str, Any]],
                 history: Dict) -> Tuple[List[Dict[str, Any]], int, int]:
    """Label one chunk (thread worker). Returns (rows, prompt_tokens, completion_tokens).

    Robust: if a batch response doesn't yield one object per input, or the call
    fails, we fall back to per-item calls so one bad row never wastes the rest.
    """
    inputs = [build_input_text(ex, history) for ex in chunk]
    p_tok = c_tok = 0
    try:
        objs, usage = _label_call(client, cfg, inputs)
        p_tok += usage.get("prompt_tokens", 0); c_tok += usage.get("completion_tokens", 0)
    except Exception:
        objs = []
    if len(objs) != len(chunk):
        objs = []
        for txt in inputs:
            try:
                o, u = _label_call(client, cfg, [txt])
                objs.append(o[0] if o else zero_signal())
                p_tok += u.get("prompt_tokens", 0); c_tok += u.get("completion_tokens", 0)
            except Exception:
                objs.append(zero_signal())
    rows = [{"id": ex["id"], "input": txt, "raw_signals": obj}
            for ex, txt, obj in zip(chunk, inputs, objs)]
    return rows, p_tok, c_tok


def stage_label(cfg: DataConfig, pilot: int = 0):
    """Label raw examples with DeepSeek, CONCURRENTLY. Resumable; `pilot>0` = N only.

    Concurrency (`label_concurrency` parallel calls) is what makes this fast; each
    worker labels a chunk of `label_batch_size` examples (default 1 for quality).
    Results are appended as soon as each chunk completes, so a crash loses at most
    the in-flight chunks and a restart skips everything already written.
    """
    raw = read_jsonl(cfg.f_raw)
    if not raw:
        print("[label] no raw examples; run `collect` first."); return

    history = {}
    if os.path.exists(cfg.f_snapshot):
        with open(cfg.f_snapshot) as f:
            history = json.load(f)

    done_ids = {r["id"] for r in read_jsonl(cfg.f_labeled)}
    todo = [ex for ex in raw if ex["id"] not in done_ids]
    if pilot > 0:
        todo = todo[:pilot]
    print(f"[label] {len(done_ids):,} already labeled; {len(todo):,} to do "
          f"({'PILOT' if pilot else 'full'} run, batch={cfg.label_batch_size}, "
          f"concurrency={cfg.label_concurrency}, effort={cfg.reasoning_effort})")
    if not todo:
        print("[label] nothing to do."); return

    client = _deepseek_client(cfg)
    chunks = [todo[i:i + cfg.label_batch_size] for i in range(0, len(todo), cfg.label_batch_size)]
    tot_prompt = tot_completion = done = 0
    lock = threading.Lock()
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=cfg.label_concurrency) as pool:
        futures = {pool.submit(_label_chunk, client, cfg, ch, history): ch for ch in chunks}
        for fut in as_completed(futures):
            rows, p_tok, c_tok = fut.result()
            with lock:                                   # serialize file writes + counters
                append_jsonl(cfg.f_labeled, rows)
                tot_prompt += p_tok; tot_completion += c_tok
                done += len(rows)
                if done % 50 == 0 or done >= len(todo):
                    rate = done / max(1e-9, time.time() - t0)
                    print(f"[label] {done:,}/{len(todo):,} | {rate:.1f} ex/s | "
                          f"tokens: {tot_prompt:,} in / {tot_completion:,} out", flush=True)

    dt = time.time() - t0
    print(f"[label] DONE in {dt:.0f}s. tokens: {tot_prompt:,} prompt + {tot_completion:,} completion")
    # Project the full-run cost from the pilot's average tokens per example.
    if pilot and done:
        avg_p, avg_c = tot_prompt / done, tot_completion / done
        full = len(raw)
        print(f"\n[label] PILOT PROJECTION for the full {full:,} examples:")
        print(f"   avg tokens/example: {avg_p:.0f} prompt + {avg_c:.0f} completion")
        print(f"   projected total   : {avg_p*full/1e6:.2f}M prompt + {avg_c*full/1e6:.2f}M completion tokens")
        print(f"   projected wall time @ {cfg.label_concurrency} workers: "
              f"~{(dt/done)*full/cfg.label_concurrency*cfg.label_concurrency/60:.1f} min "
              f"(observed {done/dt:.1f} ex/s)")
        print("\n[label] PILOT samples (inspect quality before the full run):")
        for r in read_jsonl(cfg.f_labeled)[:4]:
            print("-" * 60)
            print(r["input"][:280])
            print("signals:", json.dumps(r["raw_signals"]))


# ============================================================================
# STAGE 5: CLEAN + VALIDATE
# ============================================================================


def validate_signal(sig: Any) -> Tuple[bool, Optional[Dict[str, float]], List[str]]:
    """Check one signal dict. Returns (ok, cleaned_or_None, problems).

    A value out of range is CLAMPED (not rejected) and noted; a missing/non-numeric
    key is a hard failure (the example is dropped).
    """
    problems: List[str] = []
    if not isinstance(sig, dict):
        return False, None, ["not a dict"]
    cleaned: Dict[str, float] = {}
    for f in SIGNAL_FIELDS:
        if f not in sig:
            problems.append(f"missing:{f}")
            return False, None, problems
        try:
            v = float(sig[f])
        except (TypeError, ValueError):
            problems.append(f"nonnumeric:{f}")
            return False, None, problems
        lo, hi = SIGNAL_RANGES[f]
        if v < lo or v > hi:
            problems.append(f"clamped:{f}")
            v = float(np.clip(v, lo, hi))
        cleaned[f] = v
    extra = [k for k in sig if k not in SIGNAL_FIELDS]
    if extra:
        problems.append(f"extra_keys:{len(extra)}")
    return True, cleaned, problems


def stage_clean(cfg: DataConfig):
    """Validate labeled examples, drop bad ones, dedupe, and write clean.jsonl."""
    labeled = read_jsonl(cfg.f_labeled)
    raw_by_id = {ex["id"]: ex for ex in read_jsonl(cfg.f_raw)}
    if not labeled:
        print("[clean] no labeled examples; run `label` first."); return

    kept, drop_reasons, clamp_counter = [], Counter(), Counter()
    seen_ids, seen_inputs = set(), set()
    for r in labeled:
        rid = r["id"]
        if rid in seen_ids:
            drop_reasons["duplicate_id"] += 1; continue
        inp = r.get("input", "")
        ihash = stable_id(inp)
        if ihash in seen_inputs:
            drop_reasons["duplicate_input"] += 1; continue
        ok, cleaned, problems = validate_signal(r.get("raw_signals"))
        for p in problems:
            if p.startswith("clamped:"):
                clamp_counter[p.split(":")[1]] += 1
        if not ok:
            drop_reasons[problems[0].split(":")[0] if problems else "invalid"] += 1
            continue
        seen_ids.add(rid); seen_inputs.add(ihash)
        ex = raw_by_id.get(rid, {})
        kept.append({"id": rid, "date": ex.get("date"), "kind": ex.get("kind"),
                     "input": inp, "signals": cleaned})

    write_jsonl(cfg.f_clean, kept)
    print(f"[clean] kept {len(kept):,} / {len(labeled):,} labeled examples")
    if drop_reasons:
        print(f"[clean] drops: {dict(drop_reasons)}")
    if clamp_counter:
        print(f"[clean] clamped out-of-range values: {dict(clamp_counter)}")


# ============================================================================
# STAGE 6: BUILD TRAIN/TEST JSONL (fine-tuning format)
# ============================================================================


def stage_build(cfg: DataConfig):
    """Split clean examples chronologically and write train/test JSONL (chat format)."""
    clean = read_jsonl(cfg.f_clean)
    if not clean:
        print("[build] no clean examples; run `clean` first."); return

    # Chronological split: undated (FiQA) go to train; dated go by date, the most
    # recent dates forming the test set (mirrors the RL chronological convention).
    dated = sorted([e for e in clean if e.get("date")], key=lambda e: e["date"])
    undated = [e for e in clean if not e.get("date")]
    n_test = min(cfg.n_test, len(dated))
    test = dated[-n_test:]
    train = dated[:-n_test] + undated if n_test else dated + undated
    random.seed(cfg.seed); random.shuffle(train)

    def to_chat(e: Dict[str, Any]) -> Dict[str, Any]:
        return {"id": e["id"], "messages": [
            {"role": "system", "content": SIGNAL_SYSTEM_PROMPT},
            {"role": "user", "content": e["input"]},
            {"role": "assistant", "content": json.dumps(e["signals"])},
        ]}

    write_jsonl(cfg.f_train, [to_chat(e) for e in train])
    write_jsonl(cfg.f_test, [to_chat(e) for e in test])
    print(f"[build] wrote {len(train):,} train -> {cfg.f_train}")
    print(f"[build] wrote {len(test):,} test  -> {cfg.f_test}")


# ============================================================================
# STAGE 7: SUMMARY STATISTICS + REPORT
# ============================================================================


def stage_stats(cfg: DataConfig):
    """Per-signal summary statistics, checks, and a few printed examples."""
    clean = read_jsonl(cfg.f_clean)
    if not clean:
        print("[stats] no clean examples; run `clean` first."); return

    per_field: Dict[str, List[float]] = {f: [] for f in SIGNAL_FIELDS}
    missing = Counter()
    for e in clean:
        for f in SIGNAL_FIELDS:
            if f in e["signals"]:
                per_field[f].append(e["signals"][f])
            else:
                missing[f] += 1

    report: Dict[str, Any] = {
        "n_clean": len(clean),
        "n_dated": sum(1 for e in clean if e.get("date")),
        "n_sentence": sum(1 for e in clean if e.get("kind") == "sentence"),
        "missing_values": dict(missing),
        "signal_stats": {},
    }
    print("\n" + "=" * 70)
    print("SIGNAL SUMMARY STATISTICS")
    print("=" * 70)
    print(f"{'field':<20}{'mean':>9}{'std':>9}{'min':>8}{'max':>8}{'%nonzero':>10}")
    print("-" * 64)
    for f in SIGNAL_FIELDS:
        vals = np.array(per_field[f], dtype=np.float64) if per_field[f] else np.zeros(1)
        nonzero = float(np.mean(np.abs(vals) > 1e-6)) * 100
        stats = {"mean": float(vals.mean()), "std": float(vals.std()),
                 "min": float(vals.min()), "max": float(vals.max()),
                 "pct_nonzero": nonzero}
        report["signal_stats"][f] = stats
        print(f"{f:<20}{stats['mean']:>9.3f}{stats['std']:>9.3f}"
              f"{stats['min']:>8.2f}{stats['max']:>8.2f}{nonzero:>9.1f}%")

    os.makedirs(cfg.data_dir, exist_ok=True)
    with open(cfg.f_report, "w") as f:
        json.dump(report, f, indent=2)
    print("-" * 64)
    print(f"[stats] wrote report -> {cfg.f_report}")

    print("\n[stats] sample examples for manual inspection:")
    for e in clean[:3]:
        print("-" * 60)
        print(e["input"][:280])
        print("signals:", json.dumps(e["signals"]))


# ============================================================================
# OFFLINE SELF-TEST (no network / no API) -- run this locally before the server
# ============================================================================


def stage_selftest(cfg: DataConfig):
    """Exercise every OFFLINE component on tiny synthetic data.

    Verifies: naive signal schema+ranges, input formatting, JSON extraction,
    validation/clamping, cleaning, chronological build, and stats -- all without
    touching the network or the DeepSeek API. This is the local gate before we
    copy the file to the GPU box for the networked collect/label stages.
    """
    print("=" * 70)
    print("OFFLINE SELF-TEST (no network, no API)")
    print("=" * 70)
    tmp = os.path.join(cfg.data_dir, "_selftest")
    scfg = DataConfig(data_dir=tmp, n_train=4, n_test=2)
    os.makedirs(tmp, exist_ok=True)

    # 1) Synthetic raw examples (a few dated day-examples + a sentence example).
    raw = [
        {"id": "d1", "date": "2023-01-03", "kind": "daily",
         "headlines": ["Stocks surge as inflation cools and Fed signals rate cut",
                       "Tech rally lifts Nasdaq to record high",
                       "Oil prices plunge on weak demand"], "tickers": ["AAPL"], "urls": []},
        {"id": "d2", "date": "2023-02-10", "kind": "daily",
         "headlines": ["War fears and sanctions rattle markets, volatility spikes",
                       "Banks tumble on credit crunch worries",
                       "Investors flee to safe haven treasuries"], "tickers": [], "urls": []},
        {"id": "d3", "date": "2023-03-15", "kind": "daily",
         "headlines": ["Healthcare stocks gain as defensive rotation continues",
                       "GDP growth beats expectations",
                       "Energy sector recovers as crude rallies"], "tickers": [], "urls": []},
        {"id": "s1", "date": None, "kind": "sentence",
         "headlines": ["The company reported a record quarterly profit and raised guidance."],
         "tickers": ["MSFT"], "urls": []},
    ]
    write_jsonl(scfg.f_raw, raw)

    # 2) Naive signals -- check schema + ranges.
    stage_naive(scfg)
    naive = read_jsonl(scfg.f_naive)
    assert len(naive) == len(raw), "naive count mismatch"
    for r in naive:
        ok, cleaned, problems = validate_signal(r["signals"])
        assert ok and not any(p.startswith("clamped") for p in problems), \
            f"naive signal out of range: {problems}"
    print("[selftest] naive signals: schema + ranges OK")
    print("  example:", json.dumps(naive[0]["signals"]))

    # 2b) Curation: noise-only day dropped, signal-rich day kept & filtered.
    curate_raw = [
        {"id": "rich", "date": "2023-01-03", "kind": "daily",
         "headlines": ["Stocks surge as inflation cools and Fed signals rate cut",
                       "Top Performing Industries For Jan 3",   # noise -> dropped
                       "Oil prices plunge on weak demand"], "tickers": [], "urls": []},
        {"id": "noise", "date": "2023-01-04", "kind": "daily",
         "headlines": ["The Week Ahead: earnings and Comic-Con",   # noise
                       "Zacks Analyst Blog Highlights: MSFT",       # noise
                       "Biggest movers at midday"], "tickers": [], "urls": []},
        {"id": "s1", "date": None, "kind": "sentence",
         "headlines": ["The company reported a record quarterly profit."],
         "tickers": [], "urls": []},
    ]
    ccfg = DataConfig(data_dir=os.path.join(tmp, "curate"))
    write_jsonl(ccfg.f_raw, curate_raw)
    stage_curate(ccfg)
    curated = read_jsonl(ccfg.f_raw)
    kept_ids = {e["id"] for e in curated}
    assert "rich" in kept_ids and "s1" in kept_ids, "curate dropped good examples"
    assert "noise" not in kept_ids, "curate kept a noise-only day"
    rich_ex = next(e for e in curated if e["id"] == "rich")
    assert all("Top Performing" not in h for h in rich_ex["headlines"]), "noise headline survived"
    print("[selftest] curation: noise dropped, signal-rich kept & headline-filtered OK")

    # 3) JSON extraction robustness (bare, fenced, array).
    assert len(_extract_json_objects('{"a": 1}')) == 1
    assert len(_extract_json_objects('```json\n{"a":1}\n```')) == 1
    assert len(_extract_json_objects('[{"a":1},{"b":2}]')) == 2
    print("[selftest] JSON extraction: bare/fenced/array OK")

    # 4) Validation: clamp out-of-range, reject missing.
    bad = zero_signal(); bad["market_sentiment"] = 5.0; bad["confidence"] = -0.3
    ok, cleaned, problems = validate_signal(bad)
    assert ok and cleaned["market_sentiment"] == 1.0 and cleaned["confidence"] == 0.0
    assert any(p.startswith("clamped") for p in problems)
    missing = zero_signal(); del missing["tech_signal"]
    ok2, _, _ = validate_signal(missing)
    assert not ok2, "missing key should fail validation"
    print("[selftest] validation: clamp + missing-key rejection OK")

    # 5) Simulate a labeled file (naive signals stand in for the LLM here) and run
    #    clean -> build -> stats end to end.
    labeled = [{"id": ex["id"], "input": build_input_text(ex),
                "raw_signals": naive_signal(" ".join(ex["headlines"]))} for ex in raw]
    write_jsonl(scfg.f_labeled, labeled)
    stage_clean(scfg)
    clean = read_jsonl(scfg.f_clean)
    assert len(clean) == len(raw), "clean dropped valid rows"
    stage_build(scfg)
    train = read_jsonl(scfg.f_train); test = read_jsonl(scfg.f_test)
    assert len(test) == 2 and len(train) == len(raw) - 2, "split sizes wrong"
    # Test set must be the two most-RECENT dated examples.
    test_ids = {t["id"] for t in test}
    assert test_ids == {"d2", "d3"}, f"chronological split wrong: {test_ids}"
    # Chat format sanity.
    msgs = train[0]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    json.loads(msgs[2]["content"])  # assistant content must be valid JSON
    print("[selftest] clean/build/chronological-split/chat-format OK")

    # 6) Concurrent labeling with a MOCKED DeepSeek API (no network) -- verify the
    #    thread pool, incremental writes, and resume-skip logic all work.
    import sys
    mod = sys.modules[__name__]
    orig_client, orig_call = mod._deepseek_client, mod._label_call
    mod._deepseek_client = lambda cfg: object()  # dummy client (unused by the mock)

    def _fake_call(client, cfg, inputs):
        objs = [naive_signal(txt) for txt in inputs]   # deterministic stand-in
        return objs, {"prompt_tokens": 10 * len(inputs), "completion_tokens": 20 * len(inputs)}
    mod._label_call = _fake_call
    try:
        if os.path.exists(scfg.f_labeled):
            os.remove(scfg.f_labeled)
        stage_label(scfg, pilot=0)
        lab1 = read_jsonl(scfg.f_labeled)
        assert len(lab1) == len(raw), f"labeled {len(lab1)} != raw {len(raw)}"
        assert all(set(r["raw_signals"]) == set(SIGNAL_FIELDS) for r in lab1)
        stage_label(scfg, pilot=0)                     # resume: should add nothing
        assert len(read_jsonl(scfg.f_labeled)) == len(raw), "resume re-labeled existing ids"
    finally:
        mod._deepseek_client, mod._label_call = orig_client, orig_call
    print("[selftest] concurrent labeling (mocked) + resume-skip OK")

    stage_stats(scfg)
    print("\n[selftest] ALL OFFLINE CHECKS PASSED")


# ============================================================================
# ENTRY POINT
# ============================================================================


def build_config(args) -> DataConfig:
    cfg = DataConfig(data_dir=args.data_dir, seed=args.seed,
                     n_train=args.n_train, n_test=args.n_test)
    if args.model:
        cfg.model = args.model
    if args.reasoning_effort:
        cfg.reasoning_effort = args.reasoning_effort
    if args.batch_size:
        cfg.label_batch_size = args.batch_size
    if args.fnspid_max_rows:
        cfg.fnspid_max_rows = args.fnspid_max_rows
    if args.min_headlines:
        cfg.min_headlines_per_day = args.min_headlines
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="Chapter 12 -- Step 3: build the news->signal fine-tuning dataset.")
    parser.add_argument("stage", choices=[
        "selftest", "collect", "snapshot", "naive", "curate", "label", "clean",
        "build", "stats", "all"], help="which pipeline stage to run")
    parser.add_argument("--data_dir", type=str, default="./signal_dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_train", type=int, default=6500)
    parser.add_argument("--n_test", type=int, default=500)
    parser.add_argument("--pilot", type=int, default=0,
                        help="label only N examples (cost-safe pilot before full run)")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--reasoning_effort", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--fnspid_max_rows", type=int, default=None)
    parser.add_argument("--min_headlines", type=int, default=None,
                        help="min market-relevant headlines to keep a trading day")
    args = parser.parse_args()

    cfg = build_config(args)
    os.makedirs(cfg.data_dir, exist_ok=True)

    if args.stage == "selftest":
        stage_selftest(cfg)
    elif args.stage == "collect":
        stage_collect(cfg)
    elif args.stage == "snapshot":
        stage_snapshot(cfg)
    elif args.stage == "naive":
        stage_naive(cfg)
    elif args.stage == "curate":
        stage_curate(cfg)
    elif args.stage == "label":
        stage_label(cfg, pilot=args.pilot)
    elif args.stage == "clean":
        stage_clean(cfg)
    elif args.stage == "build":
        stage_build(cfg)
    elif args.stage == "stats":
        stage_stats(cfg)
    elif args.stage == "all":
        stage_collect(cfg)
        stage_snapshot(cfg)
        stage_curate(cfg)
        stage_naive(cfg)
        stage_label(cfg, pilot=args.pilot)
        stage_clean(cfg)
        stage_build(cfg)
        stage_stats(cfg)


if __name__ == "__main__":
    main()
