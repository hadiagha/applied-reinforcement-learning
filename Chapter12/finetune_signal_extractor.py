"""
Chapter 12 -- Step 4: Fine-Tune the LLM Signal Extractor (LoRA + SFT)
=====================================================================
Manning Publications -- Applied Reinforcement Learning (Capstone Chapter)

We now fine-tune a small instruct LLM to turn financial news into the STRUCTURED
JSON market signal defined in Step 3. The model is NOT learning to trade -- it is
learning to be a reliable *state encoder*: news in, valid 13-field signal JSON
out. The RL agent (Steps 1-2) remains the only thing that makes trades.

    news headlines  ->  [fine-tuned LLM]  ->  {market_sentiment, risk_on_signal, ...}

METHOD (deliberately practical, reusing Chapters 10 & 11)
---------------------------------------------------------
- Supervised fine-tuning (SFT): the target is the DeepSeek label from Step 3.
  We compute the loss ONLY on the assistant's JSON tokens (the prompt is masked
  with -100), exactly as in Chapter 10's SFT stage.
- LoRA: we freeze the base model and train tiny low-rank adapters, the same
  hand-written LoRA from Chapter 11 (LoRALinear / inject_lora / set_lora_enabled).
  This keeps the fine-tune cheap and lets us recover the BASE model for free by
  disabling the adapters -- which is exactly how we compare base vs. fine-tuned.
- Why not RL here? The signal has a known-good target (the DeepSeek label), so
  supervised learning is the right, simple tool. (One COULD wrap a JSON/range
  "verifier" reward and run GRPO as in Chapter 11 -- we note where that hooks in
  -- but the main pipeline stays SFT to keep the chapter focused.)

The signal SCHEMA, system prompt, JSON parsing, and validation are imported from
`build_signal_dataset.py` so the fine-tuner and the dataset can never disagree
about what a valid signal is.

EVALUATION (base LLM vs. fine-tuned LLM on the 500 held-out test examples)
--------------------------------------------------------------------------
- JSON parse success rate, schema match rate, numeric range validity
- Mean absolute error (MAE) per signal vs. the reference labels
- Directional accuracy per signal (does the sign match when the reference is
  non-neutral?)
- Confidence calibration summary (does higher predicted confidence track higher
  accuracy?)
- A few qualitative target-vs-base-vs-fine-tuned examples

Runs on GPU (bf16) or CPU (fp32), auto-detected, like Chapters 10-11.
"""

import argparse
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# transformers is imported lazily inside build_model_and_tokenizer() so the
# offline selftest (LoRA math + metrics) runs without transformers installed.

# Reuse the Step-3 schema / prompt / parsing so definitions never drift.
from build_signal_dataset import (
    SIGNAL_FIELDS,
    SIGNAL_RANGES,
    SIGNAL_SYSTEM_PROMPT,
    zero_signal,
    _extract_json_objects,
    validate_signal,
)


# ============================================================================
# CONFIGURATION (dataclasses, exactly the Chapter 10/11 style)
# ============================================================================

@dataclass
class ModelConfig:
    """Base model + runtime configuration (device/dtype derived at import)."""
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    max_length: int = 768                      # prompt + JSON fits comfortably
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float32
    )


@dataclass
class LoRAConfig:
    """Hand-written LoRA adapters (identical scheme to Chapter 11)."""
    r: int = 16
    alpha: int = 32                            # effective scale = alpha / r
    dropout: float = 0.05
    target_modules: Tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )


@dataclass
class SFTConfig:
    """Supervised fine-tuning hyperparameters (Chapter 10 SFT lineage)."""
    learning_rate: float = 2e-4                # LoRA tolerates a higher LR than full FT
    batch_size: int = 8
    gradient_accumulation_steps: int = 2       # effective batch = 16
    num_epochs: int = 3
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    logging_steps: int = 20


@dataclass
class EvalConfig:
    """Evaluation configuration (base vs. fine-tuned on the test split)."""
    num_eval_samples: int = 500
    max_new_tokens: int = 200                  # the JSON answer is short
    gen_batch_size: int = 16                   # batched greedy decoding for speed
    num_examples_to_print: int = 4


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility (same helper as the rest of Ch.12)."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_parameter_count(model: nn.Module) -> Dict[str, int]:
    """Total vs. trainable parameter counts (LoRA => trainable is a tiny fraction)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


# ============================================================================
# LORA FROM SCRATCH (same implementation as Chapter 11)
# ============================================================================
# W  ->  W + (alpha/r) * B @ A, with A random and B zero-initialized so the
# adapted model starts identical to the base model. Disabling the adapters
# (set_lora_enabled(False)) recovers the exact base model -- that is how we get
# the "base LLM" for the comparison without loading a second copy of the weights.


class LoRALinear(nn.Module):
    """Drop-in wrapper around a frozen nn.Linear that adds a low-rank update."""

    def __init__(self, base_linear: nn.Linear, r: int, alpha: int, dropout: float):
        super().__init__()
        self.base = base_linear
        self.r = r
        self.scaling = alpha / r
        self.enabled = True

        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        self.lora_A = nn.Parameter(torch.empty(r, base_linear.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base_linear.out_features, r))
        self.dropout = nn.Dropout(p=dropout)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))   # A ~ kaiming, B = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base(x)
        if not self.enabled:
            return result                       # adapters off => exact base layer
        lora_A = self.lora_A.to(x.dtype)
        lora_B = self.lora_B.to(x.dtype)
        update = self.dropout(x) @ lora_A.t() @ lora_B.t()
        return result + self.scaling * update


def inject_lora(model: nn.Module, cfg: LoRAConfig) -> List[nn.Parameter]:
    """Replace targeted nn.Linear layers with LoRALinear; return trainable params."""
    for p in model.parameters():
        p.requires_grad = False
    to_replace = []
    for _, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear) and child_name in cfg.target_modules:
                to_replace.append((module, child_name, child))
    for parent, child_name, child in to_replace:
        setattr(parent, child_name, LoRALinear(child, cfg.r, cfg.alpha, cfg.dropout))
    return [p for p in model.parameters() if p.requires_grad]


def set_lora_enabled(model: nn.Module, enabled: bool):
    """Toggle every adapter. Disabled => the model behaves as the frozen base."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.enabled = enabled


def save_adapter(model: nn.Module, tokenizer, output_dir: str):
    """Save just the tiny LoRA weights + tokenizer (Chapter 11 style)."""
    os.makedirs(output_dir, exist_ok=True)
    state = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    torch.save(state, os.path.join(output_dir, "lora_adapter.pt"))
    tokenizer.save_pretrained(output_dir)
    print(f"[save] adapter ({len(state)} tensors) -> {output_dir}/lora_adapter.pt")


def load_adapter(model: nn.Module, output_dir: str, device: str):
    """Load LoRA weights into an already-injected model (for inference/eval)."""
    path = os.path.join(output_dir, "lora_adapter.pt")
    state = torch.load(path, map_location=device)
    own = dict(model.named_parameters())
    for name, tensor in state.items():
        if name in own:
            own[name].data.copy_(tensor.to(device))
    print(f"[load] adapter <- {path}")


# ============================================================================
# DATA: chat formatting + prompt-masked SFT targets
# ============================================================================


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class SignalSFTDataset(Dataset):
    """Tokenize chat examples and mask everything but the assistant JSON.

    Each row is {"messages": [system, user, assistant]}. We tokenize the full
    conversation for the input, and a SECOND time up to the assistant header to
    find how many leading tokens to mask (-100) so the loss is computed ONLY on
    the assistant's JSON answer -- the standard SFT recipe from Chapter 10.
    """

    def __init__(self, rows: List[Dict[str, Any]], tokenizer, max_length: int):
        self.rows = rows
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        msgs = self.rows[idx]["messages"]
        # Render to TEXT first, then tokenize to plain id lists. (apply_chat_template
        # with tokenize=True returns a tokenizers.Encoding in transformers 5.x, so we
        # avoid it and re-tokenize the rendered string with add_special_tokens=False.)
        full_text = self.tok.apply_chat_template(msgs, tokenize=False,
                                                 add_generation_prompt=False)
        prompt_text = self.tok.apply_chat_template(msgs[:-1], tokenize=False,
                                                   add_generation_prompt=True)
        full = self.tok(full_text, add_special_tokens=False)["input_ids"]
        prompt = self.tok(prompt_text, add_special_tokens=False)["input_ids"]
        full = full[: self.max_length]
        prompt_len = min(len(prompt), len(full))
        labels = list(full)
        for i in range(prompt_len):
            labels[i] = -100                      # mask the prompt; train on the JSON only
        return {"input_ids": torch.tensor(full, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long)}


def make_collate_fn(pad_id: int):
    """Dynamic right-padding to the longest sequence in the batch (memory-friendly)."""
    def collate(batch):
        maxlen = max(b["input_ids"].shape[0] for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            ids, lab = b["input_ids"], b["labels"]
            pad = maxlen - ids.shape[0]
            input_ids.append(torch.cat([ids, torch.full((pad,), pad_id, dtype=torch.long)]))
            labels.append(torch.cat([lab, torch.full((pad,), -100, dtype=torch.long)]))
            attn.append(torch.cat([torch.ones(ids.shape[0], dtype=torch.long),
                                    torch.zeros(pad, dtype=torch.long)]))
        return {"input_ids": torch.stack(input_ids),
                "attention_mask": torch.stack(attn),
                "labels": torch.stack(labels)}
    return collate


# ============================================================================
# SFT TRAINER (LoRA params only)
# ============================================================================


class SignalSFTTrainer:
    """Fine-tune the LoRA adapters with masked-language SFT (Chapter 10 loop)."""

    def __init__(self, model, tokenizer, lora_params, model_cfg: ModelConfig,
                 sft_cfg: SFTConfig):
        self.model = model
        self.tok = tokenizer
        self.model_cfg = model_cfg
        self.cfg = sft_cfg
        self.device = model_cfg.device
        self.optimizer = AdamW(lora_params, lr=sft_cfg.learning_rate,
                               weight_decay=sft_cfg.weight_decay)
        self.use_amp = self.device == "cuda" and model_cfg.dtype != torch.float32
        self.train_losses: List[float] = []

    def train(self, train_rows: List[Dict[str, Any]], output_dir: str):
        ds = SignalSFTDataset(train_rows, self.tok, self.model_cfg.max_length)
        loader = DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=True,
                            collate_fn=make_collate_fn(self.tok.pad_token_id))
        steps_per_epoch = math.ceil(len(loader) / self.cfg.gradient_accumulation_steps)
        total_steps = steps_per_epoch * self.cfg.num_epochs
        warmup_steps = max(1, int(total_steps * self.cfg.warmup_ratio))
        scheduler = CosineAnnealingLR(self.optimizer, T_max=max(1, total_steps - warmup_steps),
                                      eta_min=self.cfg.learning_rate * 0.1)

        print("=" * 70)
        print("SFT TRAINING (LoRA adapters, signal extractor)")
        print("=" * 70)
        print(f"Device        : {self.device}  dtype: {self.model_cfg.dtype}")
        print(f"Train examples: {len(train_rows)}   Epochs: {self.cfg.num_epochs}")
        print(f"Optimizer steps: {total_steps}  (warmup {warmup_steps})")
        print("-" * 70)

        self.model.train()
        global_step = 0
        accum_loss = 0.0
        for epoch in range(self.cfg.num_epochs):
            for i, batch in enumerate(loader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                with torch.autocast(device_type="cuda", dtype=self.model_cfg.dtype,
                                    enabled=self.use_amp):
                    out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                     labels=labels)
                loss = out.loss / self.cfg.gradient_accumulation_steps
                loss.backward()
                accum_loss += loss.item()

                if (i + 1) % self.cfg.gradient_accumulation_steps == 0:
                    nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.cfg.max_grad_norm)
                    # Linear warmup, then cosine decay (Chapter 10 schedule).
                    if global_step < warmup_steps:
                        for g in self.optimizer.param_groups:
                            g["lr"] = self.cfg.learning_rate * (global_step + 1) / warmup_steps
                    self.optimizer.step()
                    if global_step >= warmup_steps:
                        scheduler.step()
                    self.optimizer.zero_grad()

                    self.train_losses.append(accum_loss)
                    if global_step % self.cfg.logging_steps == 0:
                        lr = self.optimizer.param_groups[0]["lr"]
                        print(f"epoch {epoch+1} | step {global_step:4d}/{total_steps} | "
                              f"loss {accum_loss:.4f} | lr {lr:.2e}", flush=True)
                    accum_loss = 0.0
                    global_step += 1
            print(f"[epoch {epoch+1}] done.")

        print("-" * 70)
        print("SFT training complete.")
        save_adapter(self.model, self.tok, output_dir)
        return {"train_losses": self.train_losses}


# ============================================================================
# INFERENCE
# ============================================================================


def build_prompt_text(tokenizer, user_content: str) -> str:
    """Render the (system + user) chat prompt with the assistant turn open."""
    messages = [{"role": "system", "content": SIGNAL_SYSTEM_PROMPT},
                {"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)


@torch.no_grad()
def generate_signals_batched(model, tokenizer, user_contents: List[str],
                             model_cfg: ModelConfig, eval_cfg: EvalConfig) -> List[str]:
    """Greedy-decode signal JSON for many prompts, in batches, and return raw text."""
    model.eval()
    tokenizer.padding_side = "left"             # left-pad so generation aligns
    use_amp = model_cfg.device == "cuda" and model_cfg.dtype != torch.float32
    outputs: List[str] = []
    for start in range(0, len(user_contents), eval_cfg.gen_batch_size):
        chunk = user_contents[start:start + eval_cfg.gen_batch_size]
        prompts = [build_prompt_text(tokenizer, u) for u in chunk]
        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=model_cfg.max_length,
                        add_special_tokens=False).to(model_cfg.device)
        with torch.autocast(device_type="cuda", dtype=model_cfg.dtype, enabled=use_amp):
            gen = model.generate(**enc, max_new_tokens=eval_cfg.max_new_tokens,
                                 do_sample=False, pad_token_id=tokenizer.pad_token_id)
        for j in range(gen.shape[0]):
            new_tokens = gen[j, enc["input_ids"].shape[1]:]
            outputs.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
    return outputs


def parse_signal(text: str) -> Tuple[Optional[Dict[str, float]], Dict[str, bool]]:
    """Parse generated text into a validated signal dict + flag diagnostics.

    Returns (signal_or_None, flags) where flags has json_ok / schema_ok / range_ok.
    """
    flags = {"json_ok": False, "schema_ok": False, "range_ok": False}
    objs = _extract_json_objects(text)
    if not objs:
        return None, flags
    flags["json_ok"] = True
    raw = objs[0]
    # schema: exactly the 13 keys present (extra keys tolerated but noted as not-clean)
    flags["schema_ok"] = all(f in raw for f in SIGNAL_FIELDS) and \
        all(k in SIGNAL_FIELDS for k in raw)
    ok, cleaned, problems = validate_signal(raw)
    if not ok:
        return None, flags
    flags["range_ok"] = not any(p.startswith("clamped") for p in problems)
    return cleaned, flags


# ============================================================================
# EVALUATION METRICS
# ============================================================================


def evaluate_model(model, tokenizer, test_rows: List[Dict[str, Any]],
                   model_cfg: ModelConfig, eval_cfg: EvalConfig, label: str
                   ) -> Dict[str, Any]:
    """Generate on the test set and compute the full metric suite for one model."""
    user_contents = [r["messages"][1]["content"] for r in test_rows]
    references = [json.loads(r["messages"][2]["content"]) for r in test_rows]

    print(f"\n[eval:{label}] generating {len(user_contents)} completions ...")
    raw_outputs = generate_signals_batched(model, tokenizer, user_contents,
                                            model_cfg, eval_cfg)

    n = len(test_rows)
    json_ok = schema_ok = range_ok = 0
    # Per-signal accumulators (only over successfully-parsed examples).
    abs_err = {f: [] for f in SIGNAL_FIELDS}
    dir_correct = {f: 0 for f in SIGNAL_FIELDS}
    dir_total = {f: 0 for f in SIGNAL_FIELDS}
    conf_bins = {"low": [], "med": [], "high": []}      # directional-hit rate by confidence
    parsed_signals: List[Optional[Dict[str, float]]] = []

    for out, ref in zip(raw_outputs, references):
        sig, flags = parse_signal(out)
        json_ok += flags["json_ok"]; schema_ok += flags["schema_ok"]; range_ok += flags["range_ok"]
        parsed_signals.append(sig)
        if sig is None:
            continue
        # per-signal MAE + directional accuracy vs the reference label
        example_hits, example_dirs = 0, 0
        for f in SIGNAL_FIELDS:
            abs_err[f].append(abs(sig[f] - ref.get(f, 0.0)))
            if f == "confidence":
                continue
            if abs(ref.get(f, 0.0)) >= 0.1:                 # reference is non-neutral
                dir_total[f] += 1
                hit = (np.sign(sig[f]) == np.sign(ref[f]))
                dir_correct[f] += int(hit)
                example_hits += int(hit); example_dirs += 1
        # confidence calibration: does higher predicted confidence => higher hit rate?
        if example_dirs > 0:
            acc = example_hits / example_dirs
            c = sig["confidence"]
            bucket = "low" if c < 0.34 else ("med" if c < 0.67 else "high")
            conf_bins[bucket].append(acc)

    metrics = {
        "label": label,
        "n": n,
        "json_success_rate": json_ok / n,
        "schema_match_rate": schema_ok / n,
        "range_validity_rate": range_ok / n,
        "mae_per_signal": {f: (float(np.mean(abs_err[f])) if abs_err[f] else None)
                           for f in SIGNAL_FIELDS},
        "mae_overall": (float(np.mean([e for f in SIGNAL_FIELDS for e in abs_err[f]]))
                        if any(abs_err.values()) else None),
        "directional_acc_per_signal": {
            f: (dir_correct[f] / dir_total[f] if dir_total[f] else None) for f in SIGNAL_FIELDS},
        "directional_support": {f: dir_total[f] for f in SIGNAL_FIELDS},
        "confidence_calibration": {
            b: (float(np.mean(v)) if v else None) for b, v in conf_bins.items()},
        "confidence_bin_counts": {b: len(v) for b, v in conf_bins.items()},
        "raw_outputs": raw_outputs,
        "parsed_signals": parsed_signals,
    }
    return metrics


def print_eval_report(base: Dict[str, Any], tuned: Dict[str, Any]):
    """Print the base-vs-fine-tuned comparison across every metric."""
    print("\n" + "=" * 70)
    print("EVALUATION: BASE LLM vs. FINE-TUNED LLM (500 held-out test examples)")
    print("=" * 70)
    row = lambda name, b, t: print(f"{name:<26}{b:>18}{t:>18}")
    row("metric", "base", "fine-tuned")
    print("-" * 62)
    for key, name in [("json_success_rate", "JSON parse success"),
                      ("schema_match_rate", "schema match"),
                      ("range_validity_rate", "numeric range valid")]:
        row(name, f"{base[key]:.1%}", f"{tuned[key]:.1%}")
    row("MAE overall (lower=better)",
        f"{base['mae_overall']:.3f}" if base['mae_overall'] is not None else "n/a",
        f"{tuned['mae_overall']:.3f}" if tuned['mae_overall'] is not None else "n/a")

    print("\nDirectional accuracy per signal (base -> fine-tuned; support = #non-neutral refs):")
    for f in SIGNAL_FIELDS:
        if f == "confidence":
            continue
        b = base["directional_acc_per_signal"][f]
        t = tuned["directional_acc_per_signal"][f]
        sup = tuned["directional_support"][f]
        bs = f"{b:.2f}" if b is not None else " n/a"
        ts = f"{t:.2f}" if t is not None else " n/a"
        print(f"  {f:<20} {bs} -> {ts}   (support {sup})")

    print("\nMAE per signal (base -> fine-tuned):")
    for f in SIGNAL_FIELDS:
        b = base["mae_per_signal"][f]; t = tuned["mae_per_signal"][f]
        bs = f"{b:.3f}" if b is not None else "n/a"
        ts = f"{t:.3f}" if t is not None else "n/a"
        print(f"  {f:<20} {bs} -> {ts}")

    print("\nConfidence calibration (mean directional hit-rate by predicted confidence):")
    for b in ("low", "med", "high"):
        bc = base["confidence_calibration"][b]; tc = tuned["confidence_calibration"][b]
        print(f"  {b:<5} conf: base {bc if bc is None else round(bc,2)} "
              f"(n={base['confidence_bin_counts'][b]})  ->  "
              f"fine-tuned {tc if tc is None else round(tc,2)} "
              f"(n={tuned['confidence_bin_counts'][b]})")
    print("=" * 70)


def print_qualitative_examples(test_rows, base, tuned, k: int = 4):
    """Show target JSON vs base output vs fine-tuned output for a few examples."""
    print("\n" + "=" * 70)
    print("QUALITATIVE EXAMPLES (target vs base vs fine-tuned)")
    print("=" * 70)
    for i in range(min(k, len(test_rows))):
        user = test_rows[i]["messages"][1]["content"]
        target = test_rows[i]["messages"][2]["content"]
        print(f"\n--- Example {i+1} ---")
        print(user[:240])
        print(f"[TARGET]     {target}")
        print(f"[BASE]       {base['raw_outputs'][i][:200]}")
        print(f"[FINE-TUNED] {tuned['raw_outputs'][i][:200]}")


def save_eval_report(base: Dict[str, Any], tuned: Dict[str, Any], output_dir: str):
    """Persist a JSON report (dropping the bulky raw outputs)."""
    def slim(m):
        return {k: v for k, v in m.items() if k not in ("raw_outputs", "parsed_signals")}
    report = {"base": slim(base), "fine_tuned": slim(tuned)}
    path = os.path.join(output_dir, "signal_eval_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[eval] wrote report -> {path}")


# ============================================================================
# PIPELINE
# ============================================================================


def build_model_and_tokenizer(model_cfg: ModelConfig, lora_cfg: LoRAConfig):
    """Load base model + tokenizer and inject LoRA adapters."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_cfg.model_name, dtype=model_cfg.dtype)
    lora_params = inject_lora(model, lora_cfg)
    model.to(model_cfg.device)
    counts = get_parameter_count(model)
    print(f"Model: {model_cfg.model_name}")
    print(f"Parameters: {counts['trainable']:,} trainable / {counts['total']:,} total "
          f"({100*counts['trainable']/counts['total']:.3f}% trainable)")
    return model, tokenizer, lora_params


def run_pipeline(data_dir: str, output_dir: str, model_cfg: ModelConfig,
                 lora_cfg: LoRAConfig, sft_cfg: SFTConfig, eval_cfg: EvalConfig,
                 skip_train: bool = False, max_train_samples: Optional[int] = None):
    """Fine-tune (or load) the signal extractor, then compare base vs. fine-tuned."""
    set_seed(42)
    os.makedirs(output_dir, exist_ok=True)

    train_rows = load_jsonl(os.path.join(data_dir, "train.jsonl"))
    if max_train_samples:                        # smoke-test knob: train on a small slice
        train_rows = train_rows[:max_train_samples]
    test_rows = load_jsonl(os.path.join(data_dir, "test.jsonl"))[: eval_cfg.num_eval_samples]
    print(f"Loaded {len(train_rows)} train / {len(test_rows)} test examples")

    model, tokenizer, lora_params = build_model_and_tokenizer(model_cfg, lora_cfg)

    if not skip_train:
        trainer = SignalSFTTrainer(model, tokenizer, lora_params, model_cfg, sft_cfg)
        trainer.train(train_rows, output_dir)
    else:
        load_adapter(model, output_dir, model_cfg.device)

    # BASE = adapters off; FINE-TUNED = adapters on. Same weights in memory.
    set_lora_enabled(model, False)
    base = evaluate_model(model, tokenizer, test_rows, model_cfg, eval_cfg, "base")
    set_lora_enabled(model, True)
    tuned = evaluate_model(model, tokenizer, test_rows, model_cfg, eval_cfg, "fine-tuned")

    print_eval_report(base, tuned)
    print_qualitative_examples(test_rows, base, tuned, eval_cfg.num_examples_to_print)
    save_eval_report(base, tuned, output_dir)
    return base, tuned


# ============================================================================
# OFFLINE SELF-TEST (no model download / no GPU)
# ============================================================================


def stage_selftest():
    """Exercise the GPU-free logic: LoRA math, parsing, and all eval metrics."""
    print("=" * 70)
    print("OFFLINE SELF-TEST (no model, no GPU)")
    print("=" * 70)

    # 1) LoRA: zero-init B => adapted layer == base layer; toggling works.
    torch.manual_seed(0)
    base = nn.Linear(8, 4)
    lin = LoRALinear(base, r=2, alpha=4, dropout=0.0)
    x = torch.randn(3, 8)
    assert torch.allclose(lin(x), base(x), atol=1e-6), "LoRA must start as identity (B=0)"
    nn.init.normal_(lin.lora_B, std=0.1)                       # now adapters do something
    assert not torch.allclose(lin(x), base(x), atol=1e-6)
    set_lora_enabled_module = lin.enabled
    lin.enabled = False
    assert torch.allclose(lin(x), base(x), atol=1e-6), "disabled adapter must equal base"
    print("[selftest] LoRA identity-init + enable/disable OK")

    # 2) parse_signal: valid JSON, out-of-range, garbage.
    good = json.dumps(zero_signal())
    sig, flags = parse_signal(f"here you go: {good} thanks")
    assert sig is not None and flags["json_ok"] and flags["schema_ok"] and flags["range_ok"]
    bad_range = dict(zero_signal()); bad_range["market_sentiment"] = 3.0
    sig2, flags2 = parse_signal(json.dumps(bad_range))
    assert flags2["json_ok"] and not flags2["range_ok"] and sig2 is not None  # clamped
    sig3, flags3 = parse_signal("no json here at all")
    assert sig3 is None and not flags3["json_ok"]
    print("[selftest] parse_signal: valid / clamped / garbage OK")

    # 3) evaluate metrics via a stubbed generator (no model needed).
    ref = dict(zero_signal()); ref.update({"market_sentiment": 0.8, "tech_signal": 0.6,
                                           "confidence": 0.9})
    test_rows = [{"messages": [{"role": "system", "content": SIGNAL_SYSTEM_PROMPT},
                               {"role": "user", "content": "Tech rallies; stocks surge"},
                               {"role": "assistant", "content": json.dumps(ref)}]}] * 5

    import sys
    mod = sys.modules[__name__]
    orig = mod.generate_signals_batched
    # Stub: fine-tuned returns near-perfect JSON; base returns messy/wrong text.
    def stub_good(model, tok, users, mc, ec):
        pred = dict(ref); pred["market_sentiment"] = 0.7      # slightly off => small MAE
        return [json.dumps(pred) for _ in users]
    def stub_bad(model, tok, users, mc, ec):
        return ["I think the market seems positive maybe?" for _ in users]
    mod.generate_signals_batched = stub_good
    tuned = evaluate_model(None, None, test_rows, ModelConfig(), EvalConfig(), "fine-tuned")
    mod.generate_signals_batched = stub_bad
    base = evaluate_model(None, None, test_rows, ModelConfig(), EvalConfig(), "base")
    mod.generate_signals_batched = orig

    assert tuned["json_success_rate"] == 1.0 and base["json_success_rate"] == 0.0
    assert tuned["mae_overall"] is not None and base["mae_overall"] is None
    assert tuned["directional_acc_per_signal"]["market_sentiment"] == 1.0
    print("[selftest] evaluate_model metrics (json/schema/MAE/directional) OK")

    print_eval_report(base, tuned)
    print("\n[selftest] ALL OFFLINE CHECKS PASSED")


# ============================================================================
# ENTRY POINT
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Chapter 12 -- Step 4: fine-tune the LLM signal extractor (LoRA + SFT).")
    parser.add_argument("--selftest", action="store_true", help="run offline logic checks and exit")
    parser.add_argument("--data_dir", type=str, default="./signal_dataset")
    parser.add_argument("--output_dir", type=str, default="./signal_llm_lora")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--num_eval_samples", type=int, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="train on only the first N examples (smoke test)")
    parser.add_argument("--skip_train", action="store_true", help="load saved adapter, eval only")
    args = parser.parse_args()

    if args.selftest:
        stage_selftest()
        return

    model_cfg = ModelConfig()
    if args.model_name:
        model_cfg.model_name = args.model_name
    lora_cfg = LoRAConfig()
    if args.lora_r:
        lora_cfg.r = args.lora_r
    sft_cfg = SFTConfig()
    if args.num_epochs:
        sft_cfg.num_epochs = args.num_epochs
    if args.batch_size:
        sft_cfg.batch_size = args.batch_size
    if args.learning_rate:
        sft_cfg.learning_rate = args.learning_rate
    eval_cfg = EvalConfig()
    if args.num_eval_samples:
        eval_cfg.num_eval_samples = args.num_eval_samples

    print("=" * 70)
    print("CHAPTER 12 -- STEP 4: FINE-TUNE LLM SIGNAL EXTRACTOR (LoRA + SFT)")
    print("=" * 70)
    run_pipeline(args.data_dir, args.output_dir, model_cfg, lora_cfg, sft_cfg,
                 eval_cfg, skip_train=args.skip_train,
                 max_train_samples=args.max_train_samples)


if __name__ == "__main__":
    main()
