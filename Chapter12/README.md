# Chapter 12 — LLM-Guided Reinforcement Learning for Portfolio Management

Code for the capstone chapter of *Applied Reinforcement Learning* (Manning).

This is a **teaching system**, not an investment product. It shows how to *engineer*
a reinforcement-learning system that works end to end — the environment, the state
representation, the reward, the agent, and an honest evaluation — and how to wire a
fine-tuned language model into that system as a **state encoder**. Nothing here is a
profit claim, and none of it is investment advice.

---

## The idea in one picture

An RL agent manages a small ETF portfolio. A language model reads financial news and
turns it into a **structured numeric market signal** that becomes part of the agent's
state. The LLM never trades — it only describes the market. The RL agent is the only
component that makes portfolio decisions.

```
   financial news + market context
        -> LLM signal extractor            (fine-tuned in Step 4)
        -> structured JSON signal           (the 13-field MarketSignal schema)
        -> RL trading environment state     (Step 1)
        -> RL portfolio decision (PPO)      (Step 2)
        -> portfolio reward + evaluation    (Steps 5-6)
```

The chapter's whole argument is about **representation quality**: keep the algorithm
fixed, change only the quality of the news signal, and watch what happens to
performance. That is why the news block of the state can be switched on or off (and
swapped between naive / base-LLM / fine-tuned-LLM signals) without touching anything
else.

---

## The files (each is one "Step" from the chapter)

| File | Step | What it does |
|---|---|---|
| `ch12_trading_env.py`         | 1   | The modular portfolio trading environment: action templates, the `MarketSignal` schema, the state builder, the shaped reward, and a seeded synthetic market. No external data needed. |
| `rl_w_o_llm_training.py`      | 2   | Trains the PPO portfolio agent with the news block **off** (price/portfolio features only). The first rung of the comparison ladder and a clean RL-only baseline. |
| `build_signal_dataset.py`     | 3   | Builds and validates the news→signal fine-tuning dataset. Staged, resumable pipeline: `collect → snapshot → curate → naive → label → clean → build → stats`. The `label` stage calls the DeepSeek "oracle." |
| `finetune_signal_extractor.py`| 4   | Fine-tunes a small instruct LLM (LoRA + SFT, loss masked to the JSON tokens) to emit valid signal JSON, then evaluates base-vs-fine-tuned on the held-out test set. |
| `rl_llm_integration.py`       | 5-6 | Wires the signal extractor into the environment and runs the final 5-mode comparison (heuristic / rl_no_news / rl_naive / rl_base_llm / rl_finetuned). Also contains the signal-driven market generator. |

Import graph (so you know what depends on what):

```
ch12_trading_env.py
      ^                         build_signal_dataset.py
      |                                 ^
rl_w_o_llm_training.py                  |
      ^                         finetune_signal_extractor.py
      |                                 ^
      +------------ rl_llm_integration.py ------------+
```

Keep all files in the same directory so the cross-imports resolve.

---

## Requirements

```bash
python -m venv .venv && source .venv/bin/activate      # optional but recommended
pip install -r requirements.txt
```

`requirements.txt` covers Steps 1, 2, 4, 5, 6 (numpy / torch / matplotlib / tqdm /
transformers). **Step 3 (dataset build) needs two extra packages** that are
deliberately kept out of the core file, because dataset construction has its own
lifecycle:

```bash
pip install datasets openai
```

**GPU.** Steps 1–2 run fine on a laptop CPU in a few minutes. Steps 4–6 fine-tune and
run a small LLM (`Qwen/Qwen2.5-1.5B-Instruct` by default) and want an NVIDIA GPU
(24–48 GB is plenty; the code auto-detects the GPU and uses bf16 when available, and
falls back to CPU if you shrink the model). See `RUNNING_ON_CLOUD_GPU.md` in the book
repo for a step-by-step cloud-GPU walkthrough.

**DeepSeek API key** (only for Step 3's `label` stage). The oracle labels are produced
by the DeepSeek API:

```bash
export DEEPSEEK_API_KEY=sk-...        # your key
```

The labeling stage costs real money, so it is **resumable** and can be **piloted** on a
handful of examples first (`--pilot 20`).

---

## How to run

Every script takes `--help`. The commands below use each script's defaults; the exact
hyperparameters used in the chapter are in the book text.

### Step 1 — sanity-check the environment
```bash
python ch12_trading_env.py --use_news        # builds a synthetic market and prints a state
```

### Step 2 — train the RL-only agent (no LLM)
```bash
python rl_w_o_llm_training.py --output_dir ./outputs_rl_no_llm
```
Trains PPO on the earlier part of the price history and evaluates on the later, unseen
part (chronological split, no look-ahead). Writes metrics and plots to `--output_dir`.

### Step 3 — build the fine-tuning dataset
Run the whole pipeline (needs `DEEPSEEK_API_KEY`):
```bash
python build_signal_dataset.py all --data_dir ./signal_dataset --pilot 20   # cheap pilot first
python build_signal_dataset.py all --data_dir ./signal_dataset              # full run
```
Or run one stage at a time (each writes a file the next stage reads, so you can inspect
and resume):
```bash
python build_signal_dataset.py collect  --data_dir ./signal_dataset
python build_signal_dataset.py snapshot  --data_dir ./signal_dataset
python build_signal_dataset.py curate    --data_dir ./signal_dataset
python build_signal_dataset.py naive     --data_dir ./signal_dataset
python build_signal_dataset.py label     --data_dir ./signal_dataset --pilot 20
python build_signal_dataset.py clean     --data_dir ./signal_dataset
python build_signal_dataset.py build     --data_dir ./signal_dataset
python build_signal_dataset.py stats     --data_dir ./signal_dataset
```
(`selftest` runs offline logic checks and needs no network or key.)

### Step 4 — fine-tune the signal extractor (LoRA + SFT)
```bash
python finetune_signal_extractor.py \
  --data_dir ./signal_dataset \
  --output_dir ./signal_llm_lora
```
Trains the LoRA adapters and prints a base-vs-fine-tuned report (JSON validity, per-signal
MAE, directional accuracy). Use `--skip_train` to reload a saved adapter and only
evaluate, or `--selftest` for offline checks.

### Steps 5–6 — integrate and run the 5-mode comparison
```bash
python rl_llm_integration.py all \
  --data_dir ./signal_dataset \
  --signal_llm_dir ./signal_llm_lora \
  --output_dir ./outputs_integration
```
`signals` generates each mode's signals over the price history; `train` trains and
evaluates the agents; `all` does both. The final report compares the five modes on
Sharpe, annualized return/volatility, max drawdown, and turnover.

---

## Outputs

Each script writes to its `--output_dir` / `--data_dir`: JSON metrics, per-mode results,
and matplotlib figures. The chapter's result figures (the comparison ladder, the
fidelity→performance link, and the risk/return scatter) are generated from the final
report of Steps 5–6.

## Reproducibility & honesty notes

- **Seeded** synthetic market and training, so runs are repeatable.
- **Chronological** train/test split and **no look-ahead** by construction — the state at
  day *t* uses only data up to *t*.
- The market in Steps 5–6 is **signal-driven by design**: the oracle (DeepSeek) signal
  genuinely drives next-day returns, so the experiment cleanly measures *how much oracle
  signal survives distillation into the small model, and how much trading value that
  buys*. On real, near-efficient markets you should not expect these magnitudes — the
  point is the engineering method, not the numbers.

## License / attribution

Companion code for *Applied Reinforcement Learning* (Manning), Chapter 12. Educational
use. Not investment advice.
