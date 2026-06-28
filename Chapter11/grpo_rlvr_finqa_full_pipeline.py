"""
Chapter 11: GRPO + LoRA + RLVR for Financial Question Answering (FinQA)
======================================================================

This module implements an *advanced* reinforcement-learning fine-tuning pipeline
for a large language model, built entirely from scratch in PyTorch. It is the
sequel to Chapter 10, which taught RLHF with PPO (SFT -> reward model -> PPO).

Chapter 11 swaps in three modern ideas, each implemented by hand so the code
stays readable and dependency-light for the next decade (only `torch` and
`transformers` are needed -- no `trl`, no `peft`):

1. GRPO (Group Relative Policy Optimization)
   A *critic-free* policy-gradient method. PPO (Chapter 10) trained a separate
   value head to estimate "how good is this state". GRPO throws that away.
   Instead, for each prompt it samples a *group* of G answers, scores them, and
   uses the group's own mean/std as the baseline. The advantage of an answer is
   simply "how much better than its sibling answers was it?". No value network,
   no GAE -- just relative comparison inside a group.

2. LoRA (Low-Rank Adaptation)
   We freeze the entire 8B base model and inject tiny trainable low-rank
   matrices into its linear layers. Only those adapters (a fraction of a percent
   of the weights) receive gradients, so an 8B model fits and trains on a single
   GPU. We write the LoRA layer ourselves.

3. RLVR (Reinforcement Learning with Verifiable Rewards)
   Chapter 10 had to *learn* a reward model from human preference pairs. FinQA
   gives us something better: a ground-truth answer for every question. So the
   reward is computed by a deterministic Python *verifier* -- did the model's
   final number match the gold answer? -- with no learned reward model at all.

The task: FinQA (Chen et al., 2021). Each example is a snippet of a financial
report (text + a table) and a numerical question. The model must read the
context, reason step by step, and produce a final number. We verify that number
against the dataset's `exe_ans` field.

What the model is asked to produce
----------------------------------
Free-form chain-of-thought reasoning, then the final answer wrapped in a LaTeX
box, e.g. "... so the interest expense is \\boxed{3.8}". The verifier extracts
the boxed number and compares it to the gold answer.

Pipeline (much simpler than Chapter 10 -- no SFT, no reward model)
------------------------------------------------------------------
  Stage 1: Load the instruct base model and inject LoRA adapters.
  Stage 2: GRPO training loop driven purely by the RLVR verifier.
  Stage 3: Evaluate accuracy of the BASE model vs. the GRPO-tuned model on the
           held-out test set.

Dependencies (minimal for longevity):
- torch >= 2.0
- transformers >= 4.51   (Qwen3 support)
- numpy
- tqdm
"""

import os
import re
import json
import math
import random
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from tqdm import tqdm


# ============================================================================
# CONFIGURATION
# ============================================================================
# As in Chapter 10, every knob lives in a dataclass. Device and dtype are
# *derived* from the hardware at import time -- never hardcoded -- so the same
# file runs on any NVIDIA GPU (bf16) or falls back to CPU (fp32) unchanged.

@dataclass
class ModelConfig:
    """Configuration for the base language model."""
    # Default is Qwen2.5-3B-Instruct: capable enough to solve many FinQA
    # questions (so GRPO gets a useful reward signal) yet not already at ceiling
    # (so fine-tuning has visible headroom to improve). You can swap in another
    # model without touching any other code, e.g.:
    #   "Qwen/Qwen3-8B"                   (strongest, but often near-ceiling on FinQA)
    #   "Qwen/Qwen2.5-Math-1.5B-Instruct" (math-specialized, smallest/fastest)
    #   "Qwen/Qwen2.5-1.5B-Instruct"      (small / laptop GPU)
    #   "Qwen/Qwen2.5-0.5B-Instruct"      (smoke-test only)
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"

    # The prompt (financial report + table + question) can be long, so we give
    # it a generous budget and truncate anything beyond it.
    max_prompt_length: int = 1536
    # How many tokens of reasoning + answer the model may generate. Multi-step
    # financial reasoning can be long; if this is too small the model runs out
    # of room before emitting its \boxed{} answer and gets zero reward even when
    # its reasoning was on track, so we keep a generous budget.
    max_new_tokens: int = 640

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float32
    )


@dataclass
class LoRAConfig:
    """Configuration for the hand-written LoRA adapters."""
    r: int = 32                       # rank of the low-rank update (the "bottleneck")
    alpha: int = 64                   # scaling numerator; effective scale = alpha / r
    dropout: float = 0.05             # dropout applied to the adapter input
    # Which linear layers to adapt. The attention projections alone are the
    # cheapest choice, but for an RL fine-tune we want enough capacity for the
    # policy to actually change behavior, so we also adapt the MLP projections
    # ("gate_proj", "up_proj", "down_proj"). This is still a tiny fraction of the
    # model's parameters.
    target_modules: Tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""
    # --- the "group" in Group Relative Policy Optimization ---
    group_size: int = 8               # G: answers sampled per prompt
    prompts_per_batch: int = 8        # prompts processed before an optimizer step
    # During the training forward/backward we process this many sequences at a
    # time. The full-vocabulary logits for a large-vocab model (Qwen3 ~152k) are
    # several GB per sequence, so chunking keeps peak memory bounded regardless
    # of group_size. The optimizer math is identical to processing the whole
    # group at once (we just normalize by the total number of responses).
    micro_batch_size: int = 2

    # --- optimization ---
    # LoRA adapters need a MUCH larger learning rate than full fine-tuning:
    # classic full-model RL values (1e-6, even 1e-5) barely move the tiny
    # adapters (the KL to the reference stays ~0 and nothing is learned).
    # ~1e-4 is a typical, working value for LoRA + GRPO. Raise it if KL stays
    # near zero; lower it if KL explodes (>0.1) and accuracy collapses.
    learning_rate: float = 1e-4
    ppo_epochs: int = 1               # times we reuse each batch of rollouts
    grad_accum: int = 1               # gradient-accumulation micro-steps
    max_grad_norm: float = 1.0
    num_iterations: int = 300         # total training iterations (1 iter = 1 batch)

    # --- the GRPO objective ---
    clip_epsilon: float = 0.2         # PPO-style ratio clip
    # beta: weight of the KL-to-reference penalty. Recent practice (and TRL's
    # default) sets this to 0 -- the KL term is not essential for GRPO and it
    # drags on learning. We keep computing the KL as a *diagnostic* (to watch how
    # far the policy drifts) but do not penalize it. Set >0 to re-enable.
    kl_coef: float = 0.0

    # --- sampling during rollouts ---
    temperature: float = 0.9          # >0 so the group is diverse (see STEP 2)
    top_p: float = 1.0

    # --- RLVR reward shaping ---
    reward_correct: float = 1.0       # given when the final answer matches gold
    reward_format: float = 0.1        # small bonus for emitting a \boxed{} answer
    answer_tol: float = 1e-3          # relative tolerance for numeric matching

    # --- logging / checkpointing ---
    log_every: int = 5
    save_every: int = 50


@dataclass
class EvalConfig:
    """Configuration for evaluation (base vs. fine-tuned)."""
    num_eval_samples: int = 500       # how many test questions to score (<=1147)
    eval_temperature: float = 0.0     # 0.0 => greedy decoding (deterministic)
    max_new_tokens: int = 640
    num_examples_to_print: int = 4    # qualitative side-by-side examples


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility (identical to Chapter 10)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_parameter_count(model: nn.Module) -> Dict[str, int]:
    """Count total vs. trainable parameters.

    With LoRA we expect `trainable` to be a tiny fraction of `total`. For
    Qwen3-8B with rank-16 adapters on the 4 attention projections this is
    roughly ~10M trainable out of ~8B total (about 0.1%).
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


# ============================================================================
# LORA FROM SCRATCH
# ============================================================================
# LoRA replaces a frozen weight matrix W (out x in) with W + (alpha/r) * B @ A,
# where A is (r x in) and B is (out x r). Because r is tiny (e.g. 16), B @ A is
# a *low-rank* update with very few parameters. W itself never changes.
#
# Crucial initialization detail: A is random (kaiming) but B starts at ZERO.
# Therefore at step 0 the update B @ A is exactly the zero matrix, so the
# adapted layer outputs exactly what the frozen layer would. The model starts
# as an identical copy of the base model and only drifts as B learns.
#
# That zero-init is also what lets us avoid loading a second 8B "reference"
# model: disabling the adapters (see `set_lora_enabled`) turns the policy back
# into the frozen base model on demand -- and that frozen base IS our reference.


class LoRALinear(nn.Module):
    """A drop-in wrapper around a frozen `nn.Linear` that adds a LoRA update."""

    def __init__(self, base_linear: nn.Linear, r: int, alpha: int, dropout: float):
        super().__init__()
        self.base = base_linear            # the original, frozen projection
        self.r = r
        self.scaling = alpha / r           # e.g. 32 / 16 = 2.0
        self.enabled = True                # toggled off to recover the base model

        in_features = base_linear.in_features
        out_features = base_linear.out_features

        # Freeze the original weight (and bias, if any). Only A and B will train.
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        # Low-rank factors. We keep them in fp32 for stable optimization even
        # when the frozen base runs in bf16; we cast A,B to the input dtype on
        # the fly inside forward().
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.dropout = nn.Dropout(p=dropout)

        # A ~ kaiming-uniform (same convention the LoRA paper uses), B = 0.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Always run the frozen base projection.
        result = self.base(x)
        if not self.enabled:
            # Adapters off => behave exactly like the original frozen layer.
            return result

        # Low-rank path: x -> dropout -> @A^T -> @B^T, scaled.
        # Shapes: x (..., in) @ A^T (in, r) -> (..., r) @ B^T (r, out) -> (..., out)
        lora_A = self.lora_A.to(x.dtype)
        lora_B = self.lora_B.to(x.dtype)
        update = self.dropout(x) @ lora_A.t() @ lora_B.t()
        return result + self.scaling * update


def inject_lora(model: nn.Module, lora_config: LoRAConfig) -> List[nn.Parameter]:
    """Replace targeted `nn.Linear` layers in-place with `LoRALinear` wrappers.

    Returns the list of trainable LoRA parameters (to hand to the optimizer).
    All non-LoRA parameters are frozen here, so the optimizer only ever touches
    the adapters.
    """
    # First freeze everything; the LoRALinear constructor will re-freeze the
    # base weights it wraps, and A/B remain trainable by default.
    for param in model.parameters():
        param.requires_grad = False

    # We collect (parent_module, attribute_name, child_linear) for every layer
    # we want to wrap, then mutate after iterating (mutating during the walk is
    # unsafe). `named_modules()` gives dotted paths like
    #   "model.layers.0.self_attn.q_proj".
    to_replace = []
    for module_name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear) and child_name in lora_config.target_modules:
                to_replace.append((module, child_name, child))

    for parent, child_name, child in to_replace:
        wrapped = LoRALinear(
            child,
            r=lora_config.r,
            alpha=lora_config.alpha,
            dropout=lora_config.dropout,
        )
        setattr(parent, child_name, wrapped)

    lora_params = [p for p in model.parameters() if p.requires_grad]
    return lora_params


def set_lora_enabled(model: nn.Module, enabled: bool):
    """Turn every LoRA adapter on or off.

    Disabling makes the model behave exactly like the frozen base model, which
    is how we get our "reference policy" for the KL penalty and how we evaluate
    the base model -- no second copy of the 8B weights in memory.
    """
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.enabled = enabled


# ============================================================================
# DATA: FinQA PROMPT BUILDING + DATASET
# ============================================================================
# A FinQA record looks like:
#   {
#     "pre_text":  [ ...sentences before the table... ],
#     "post_text": [ ...sentences after the table... ],
#     "table":     [ ["", "2009", "2008"], ["revenue", "$ 6427", ...], ... ],
#     "qa": { "question": "...", "exe_ans": 3.8, ... },
#     "id": "ADI/2009/page_49.pdf-1",
#   }
# `exe_ans` is the ground truth: a float for numeric questions, or the string
# "yes"/"no" for comparison questions. That field is our entire reward signal.


def table_row_to_text(header: List[str], row: List[str]) -> str:
    """Serialize one table row into a sentence.

    Ported from the official FinQA repo (code/utils/general_utils.py) so our
    table reading matches how the dataset authors intended tables to be read.

    Example:
      header = ["", "october 31 2009", "november 1 2008"]
      row    = ["fair value", "$ 6427", "$ -23158"]
      ->  "fair value of october 31 2009 is $ 6427 ; "
          "fair value of november 1 2008 is $ -23158 ;"
    """
    res = ""
    if header[0]:
        res += header[0] + " "
    for head, cell in zip(header[1:], row[1:]):
        res += "the " + row[0] + " of " + head + " is " + cell + " ; "
    return res.strip()


def serialize_table(table: List[List[str]]) -> str:
    """Turn a full table (list of rows, row 0 = header) into text lines."""
    if not table:
        return ""
    header = table[0]
    lines = [table_row_to_text(header, row) for row in table[1:]]
    return "\n".join(lines)


# System prompt: tells the model the task and -- importantly for RLVR -- the
# exact output format the verifier expects (the \boxed{} answer).
SYSTEM_PROMPT = (
    "You are a meticulous financial analyst. You are given an excerpt from a "
    "financial report (text and a table) followed by a question. Reason step by "
    "step using the numbers in the context, then give the final answer. "
    "Put ONLY the final numeric answer inside \\boxed{}, for example "
    "\\boxed{3.8}. For yes/no questions, put \\boxed{yes} or \\boxed{no}."
)


def build_prompt(record: Dict[str, Any], tokenizer, max_prompt_length: int) -> str:
    """Build the full chat-formatted prompt string for one FinQA record.

    Context = pre_text + serialized table + post_text + question. We truncate
    the *context* (not the instructions) by token count so the prompt fits,
    then apply the tokenizer's chat template.
    """
    pre_text = " ".join(record.get("pre_text", []))
    post_text = " ".join(record.get("post_text", []))
    table_text = serialize_table(record.get("table", []))
    question = record["qa"]["question"]

    context = (
        f"{pre_text}\n\n"
        f"Table:\n{table_text}\n\n"
        f"{post_text}"
    ).strip()

    # Truncate the context by tokens. We reserve headroom for the system prompt,
    # the question, and chat-template special tokens.
    context_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
    headroom = 256  # rough budget for instructions/question/template tokens
    budget = max(0, max_prompt_length - headroom)
    if len(context_ids) > budget:
        context_ids = context_ids[:budget]
        context = tokenizer.decode(context_ids)

    user_message = (
        f"{context}\n\n"
        f"Question: {question}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    # `add_generation_prompt=True` appends the assistant turn header so the
    # model continues as the assistant. For Qwen3 we disable "thinking" mode so
    # the model writes its reasoning in plain text (which our verifier reads)
    # rather than inside hidden <think> tags; the kwarg is ignored by tokenizers
    # that do not support it.
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


class FinQADataset(Dataset):
    """Wraps FinQA records and yields ready-to-use prompts + gold answers."""

    def __init__(self, records: List[Dict[str, Any]], tokenizer, max_prompt_length: int):
        self.records = records
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        return {
            "prompt": build_prompt(record, self.tokenizer, self.max_prompt_length),
            "gold": record["qa"]["exe_ans"],
            "question": record["qa"]["question"],
            "id": record.get("id", str(idx)),
        }


def load_finqa(path: str) -> List[Dict[str, Any]]:
    """Load a FinQA JSON split and drop any record missing a usable answer."""
    with open(path, "r") as f:
        data = json.load(f)

    clean = []
    for record in data:
        qa = record.get("qa", {})
        if "exe_ans" not in qa or "question" not in qa:
            continue
        clean.append(record)
    return clean


# ============================================================================
# RLVR VERIFIER (THE REWARD FUNCTION)
# ============================================================================
# This is the heart of "RL with Verifiable Rewards". There is NO neural reward
# model here -- just deterministic Python that decides whether the model's
# final answer matches the gold answer. Because the reward is computed from the
# dataset's ground truth, it cannot be "hacked" the way a learned reward model
# sometimes can.


_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_NUMBER_RE = re.compile(r"-?\$?\(?-?\d[\d,]*\.?\d*\)?%?")


def _to_number(text: str) -> Optional[float]:
    """Parse a messy financial string into a float, or None if not a number.

    Handles dollar signs, thousands commas, trailing percent signs, and the
    accounting convention where parentheses mean a negative number.

    Examples:
      "$ 6,427"   -> 6427.0
      "(23158)"   -> -23158.0
      "3.8%"      -> 3.8     (the percent sign is stripped; scale handled later)
    """
    if text is None:
        return None
    s = text.strip().lower()

    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative = True
        s = s[1:-1]
    s = s.replace("$", "").replace(",", "").replace("%", "").replace(" ", "")
    if s.startswith("-"):
        negative = True
        s = s[1:]
    if s == "":
        return None
    try:
        value = float(s)
    except ValueError:
        return None
    return -value if negative else value


def extract_answer(text: str) -> Dict[str, Any]:
    """Extract the model's final answer from its generated text.

    Returns a dict:
      {"has_box": bool, "raw": str|None, "number": float|None, "yesno": str|None}

    Priority:
      1. The LAST \\boxed{...} (the model may show intermediate boxes; the final
         one is the answer).
      2. Fallback: a trailing "the answer is X" phrase.
      3. Fallback: the last number-looking token in the text.
    """
    raw = None
    has_box = False

    boxes = _BOXED_RE.findall(text)
    if boxes:
        has_box = True
        raw = boxes[-1].strip()
    else:
        m = re.search(r"answer\s*(?:is|:)\s*([^\n]+)", text, flags=re.IGNORECASE)
        if m:
            raw = m.group(1).strip()
        else:
            nums = _NUMBER_RE.findall(text)
            raw = nums[-1].strip() if nums else None

    yesno = None
    if raw is not None:
        low = raw.strip().lower()
        if low in ("yes", "no"):
            yesno = low

    # Turn the raw answer string into a number. Models often add words around
    # the value, e.g. "\boxed{3.8 million}" or "answer is 3.8 (approx)". So if a
    # direct parse fails, fall back to the FIRST number-looking token inside it.
    number = None
    if raw is not None and yesno is None:
        number = _to_number(raw)
        if number is None:
            inner = _NUMBER_RE.findall(raw)
            if inner:
                number = _to_number(inner[0])
    return {"has_box": has_box, "raw": raw, "number": number, "yesno": yesno}


def numbers_match(pred: float, gold: float, tol: float) -> bool:
    """Numeric match with relative tolerance AND percentage-scale variants.

    FinQA answers frequently differ from a model's natural output by a factor
    of 100 (a ratio of 0.038 vs. the percentage 3.8) or 1/100. We accept any of
    those scalings so we reward correct *reasoning* rather than punishing a
    cosmetic percent-vs-fraction mismatch.

    Examples (tol = 1e-3):
      pred=3.80,  gold=3.8   -> match (relative diff 0)
      pred=380,   gold=3.8   -> match (380 / 100 = 3.8)
      pred=0.038, gold=3.8   -> match (0.038 * 100 = 3.8)
    """
    def close(a: float, b: float) -> bool:
        denom = max(1.0, abs(b))
        return abs(a - b) <= tol * denom

    candidates = (pred, pred * 100.0, pred / 100.0)
    return any(close(c, gold) for c in candidates)


def compute_reward(generated_text: str, gold: Any, config: GRPOConfig) -> Dict[str, Any]:
    """Score one generated answer against the gold answer.

    Reward = (reward_correct if the answer is correct)
             + (reward_format if a well-formed \\boxed{} answer was emitted).

    The format bonus is small but important: it nudges the model to ALWAYS
    produce a parseable \\boxed{} answer so the verifier can read it. Without it,
    a model that reasons correctly but forgets the box would get zero reward and
    the wrong learning signal.

    Returns {"reward": float, "correct": bool, "has_box": bool}.
    """
    parsed = extract_answer(generated_text)
    reward = 0.0
    correct = False

    gold_is_str = isinstance(gold, str)
    if gold_is_str:
        # yes/no question: compare strings.
        gold_norm = gold.strip().lower()
        if parsed["yesno"] is not None and parsed["yesno"] == gold_norm:
            correct = True
    else:
        # numeric question.
        gold_val = float(gold)
        if parsed["number"] is not None and numbers_match(parsed["number"], gold_val, config.answer_tol):
            correct = True

    if correct:
        reward += config.reward_correct
    if parsed["has_box"]:
        reward += config.reward_format

    return {"reward": reward, "correct": correct, "has_box": parsed["has_box"]}


# ============================================================================
# GRPO TRAINER
# ============================================================================
# How GRPO differs from the PPO of Chapter 10, at a glance:
#
#   PPO (Ch.10)                         GRPO (Ch.11)
#   -----------                         ------------
#   value head V(s) estimates baseline  group mean reward is the baseline
#   advantage A = R - V(s), then GAE    advantage A_i = (r_i - mean) / std
#   one sample per prompt               G samples per prompt (a "group")
#   reward from a learned reward model  reward from the RLVR verifier
#
# The clipped surrogate objective and the KL-to-reference penalty are shared
# with PPO, so most of this will feel familiar.


class GRPOTrainer:
    """Trains LoRA adapters on a policy model with GRPO + RLVR."""

    def __init__(self, model, tokenizer, lora_params, model_config: ModelConfig,
                 grpo_config: GRPOConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.config = grpo_config
        self.device = model_config.device

        # Optimizer touches ONLY the LoRA parameters; the 8B base is frozen.
        self.optimizer = torch.optim.AdamW(lora_params, lr=grpo_config.learning_rate)

        # Mixed precision exactly as in Chapter 10.
        self.use_amp = self.device == "cuda" and model_config.dtype != torch.float32

        self.stats = {
            "reward": [], "accuracy": [], "format_rate": [],
            "kl": [], "loss": [],
        }

    # ----- log-prob helper (shared shape with Chapter 10's _compute_log_probs) -----
    def _token_log_probs(self, logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Per-token log p(token_t | token_<t) for a batch.

        Standard next-token shift: the logits at position t predict token t+1.
        Returns a (batch, seq_len-1) tensor of log-probs for the *realized*
        tokens (not the mean -- we keep per-token values so we can mask the
        prompt and average only over response tokens).

        Memory note: a naive `log_softmax(logits)` allocates a full
        (batch, seq_len, vocab) tensor -- for a 152k-vocab model and a batch of
        long sequences that is many GB, and the backward pass must keep it. We
        avoid that using the identity
            log p(label) = logit[label] - logsumexp(all logits)
        `logsumexp` reduces over the vocabulary to a (batch, seq_len) tensor
        without ever materializing the giant softmax, so peak memory drops a lot
        and we never need to upcast the whole logit tensor to float32.
        """
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        # logit value of the realized next token at each position.
        gathered = shift_logits.gather(
            dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        # Normalizer: log sum_v exp(logit_v). Computed in float32 for stability
        # but as a (batch, seq_len) result, not a (batch, seq_len, vocab) tensor.
        logsumexp = torch.logsumexp(shift_logits.float(), dim=-1)
        return gathered.float() - logsumexp

    # ----- rollout generation -----
    @torch.no_grad()
    def generate_group(self, prompt: str) -> Dict[str, torch.Tensor]:
        """Sample `group_size` completions for one prompt (LoRA enabled).

        We expand the single prompt to G copies and generate in one batched
        call. Sampling (temperature > 0) is what makes the G answers *different*
        from each other -- that diversity is the whole point of GRPO, because a
        group of identical answers would have zero reward variance and thus zero
        advantage.

        Returns padded tensors covering prompt+response, plus the prompt length
        so the caller can mask out prompt tokens.
        """
        self.model.eval()
        set_lora_enabled(self.model, True)

        enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        prompt_len = input_ids.shape[1]

        out = self.model.generate(
            input_ids=input_ids.repeat(self.config.group_size, 1),
            attention_mask=attention_mask.repeat(self.config.group_size, 1),
            max_new_tokens=self.model_config.max_new_tokens,
            do_sample=True,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        # `out` is (G, prompt_len + response_len), already padded to a common
        # length by generate(). Build a mask over response tokens only.
        full_ids = out
        seq_len = full_ids.shape[1]
        response_mask = torch.zeros_like(full_ids, dtype=torch.float32)
        response_mask[:, prompt_len:] = 1.0
        # Don't count padding that generate() may have appended after EOS.
        pad_id = self.tokenizer.pad_token_id
        response_mask[full_ids == pad_id] = 0.0

        # Decode just the response part for the verifier.
        responses = self.tokenizer.batch_decode(
            full_ids[:, prompt_len:], skip_special_tokens=True
        )
        return {
            "full_ids": full_ids,
            "response_mask": response_mask,
            "prompt_len": prompt_len,
            "responses": responses,
        }

    # ----- one GRPO update over a batch of prompts -----
    def grpo_step(self, batch: List[Dict[str, Any]]) -> Dict[str, float]:
        """Run rollouts for a batch of prompts and take one optimizer step."""
        all_rewards, all_correct, all_boxes = [], [], []
        groups = []  # per-prompt rollout tensors + advantages

        # ================================================================
        # STEP 1: Rollouts -- sample a group per prompt and score each answer.
        # ================================================================
        for item in batch:
            roll = self.generate_group(item["prompt"])
            rewards, corrects, boxes = [], [], []
            for resp in roll["responses"]:
                r = compute_reward(resp, item["gold"], self.config)
                rewards.append(r["reward"])
                corrects.append(r["correct"])
                boxes.append(r["has_box"])

            rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)

            # ============================================================
            # STEP 2: Group-relative advantage.
            # A_i = (r_i - mean(group)) / (std(group) + eps), the SAME scalar
            # for every token of answer i. This is GRPO's critic-free baseline:
            # "how much better than my sibling answers was I?"
            #
            # Numerical example with G=4 rewards [1.1, 0.1, 1.1, 0.0]:
            #   mean = 0.575, std ~= 0.567
            #   advantages ~= [0.93, -0.84, 0.93, -1.01]
            # If all four answers were identical (e.g. all wrong -> all 0.0),
            # std = 0 and every advantage is 0: no gradient. That is expected,
            # and it is why GRPO needs a base model strong enough to get *some*
            # answers right -- otherwise there is nothing to learn from.
            # ============================================================
            advantages = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)

            groups.append({
                "full_ids": roll["full_ids"],
                "response_mask": roll["response_mask"],
                "advantages": advantages,
            })
            all_rewards.extend(rewards)
            all_correct.extend(corrects)
            all_boxes.extend(boxes)

        # ================================================================
        # STEP 3: Old + reference log-probs (no grad).
        # old_logp : current policy's log-probs, frozen as the PPO "behavior"
        #            policy that the ratio is measured against.
        # ref_logp : reference policy = base model = adapters DISABLED. Used for
        #            the KL penalty that keeps the policy from drifting too far.
        # ================================================================
        for g in groups:
            full_ids = g["full_ids"]
            attn = (full_ids != self.tokenizer.pad_token_id).long()

            with torch.no_grad():
                set_lora_enabled(self.model, True)
                with torch.autocast(device_type="cuda", dtype=self.model_config.dtype,
                                    enabled=self.use_amp):
                    logits = self.model(input_ids=full_ids, attention_mask=attn).logits
                g["old_logp"] = self._token_log_probs(logits, full_ids)
                del logits

                set_lora_enabled(self.model, False)
                with torch.autocast(device_type="cuda", dtype=self.model_config.dtype,
                                    enabled=self.use_amp):
                    ref_logits = self.model(input_ids=full_ids, attention_mask=attn).logits
                g["ref_logp"] = self._token_log_probs(ref_logits, full_ids)
                del ref_logits
            g["attn"] = attn

        # ================================================================
        # STEP 4: Optimize the clipped surrogate + KL penalty.
        # ================================================================
        self.model.train()
        set_lora_enabled(self.model, True)
        total_loss_val, total_kl_val, n_terms = 0.0, 0.0, 0

        # Normalizer so the accumulated micro-batch gradients equal the mean
        # objective over EVERY response in the batch: each response contributes
        # 1 / (total_responses * grad_accum).
        total_responses = sum(g["full_ids"].shape[0] for g in groups)
        norm = max(1, total_responses) * self.config.grad_accum
        mb = self.config.micro_batch_size

        for _ in range(self.config.ppo_epochs):
            self.optimizer.zero_grad()
            for g in groups:
                full_ids = g["full_ids"]
                attn = g["attn"]
                # Response mask aligns with the SHIFTED tokens (positions 1..L-1).
                resp_mask = g["response_mask"][:, 1:]
                advantages = g["advantages"].unsqueeze(1)  # (G, 1) broadcasts over tokens
                G = full_ids.shape[0]

                # Micro-batch the group so we never hold more than `mb`
                # sequences' worth of (seq x vocab) logits at once.
                for s in range(0, G, mb):
                    e = min(s + mb, G)
                    ids_chunk = full_ids[s:e]
                    attn_chunk = attn[s:e]
                    rmask = resp_mask[s:e]
                    adv_chunk = advantages[s:e]
                    old_chunk = g["old_logp"][s:e]
                    ref_chunk = g["ref_logp"][s:e]

                    with torch.autocast(device_type="cuda", dtype=self.model_config.dtype,
                                        enabled=self.use_amp):
                        logits = self.model(input_ids=ids_chunk,
                                            attention_mask=attn_chunk).logits
                    new_logp = self._token_log_probs(logits, ids_chunk)
                    del logits  # free the big (chunk x seq x vocab) tensor early

                    # PPO-style probability ratio pi_new / pi_old (per token).
                    ratio = torch.exp(new_logp - old_chunk)
                    surr1 = ratio * adv_chunk
                    surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon,
                                        1 + self.config.clip_epsilon) * adv_chunk
                    surrogate = torch.min(surr1, surr2)

                    # KL(policy || reference) via the unbiased "k3" estimator:
                    #   kl = exp(ref - new) - (ref - new) - 1   (>= 0 per token)
                    # This is the low-variance estimator GRPO/DeepSeek use; it
                    # adds the KL straight into the loss, not into the reward.
                    diff = ref_chunk - new_logp
                    per_token_kl = torch.exp(diff) - diff - 1.0

                    per_token_obj = surrogate - self.config.kl_coef * per_token_kl

                    # Average over RESPONSE tokens only (mask out prompt +
                    # padding) to get one objective per response in the chunk.
                    # eps guards a degenerate all-padding row.
                    per_resp = (per_token_obj * rmask).sum(dim=1) / (rmask.sum(dim=1) + 1e-8)
                    # Sum (not mean) here, then divide by the global normalizer,
                    # so summing across all chunks reproduces the batch mean.
                    loss = -per_resp.sum() / norm
                    loss.backward()

                    total_loss_val += (-per_resp).sum().item()
                    per_resp_kl = (per_token_kl * rmask).sum(dim=1) / (rmask.sum(dim=1) + 1e-8)
                    total_kl_val += per_resp_kl.sum().item()
                    n_terms += per_resp.shape[0]

            # ============================================================
            # STEP 5: Clip gradients and step (LoRA params only).
            # ============================================================
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.config.max_grad_norm,
            )
            self.optimizer.step()

        n = max(1, len(all_rewards))
        return {
            "reward": float(np.mean(all_rewards)),
            "accuracy": float(sum(all_correct) / n),
            "format_rate": float(sum(all_boxes) / n),
            "kl": total_kl_val / max(1, n_terms),
            "loss": total_loss_val / max(1, n_terms),
        }

    def train(self, dataset: FinQADataset, output_dir: str) -> Dict[str, List[float]]:
        """Main GRPO loop: repeatedly sample a batch of prompts and update."""
        os.makedirs(output_dir, exist_ok=True)
        num_prompts = len(dataset)
        indices = list(range(num_prompts))

        print("\n" + "=" * 60)
        print("GRPO + RLVR TRAINING")
        print("=" * 60)
        print(f"Prompts available : {num_prompts}")
        print(f"Group size (G)    : {self.config.group_size}")
        print(f"Prompts per batch : {self.config.prompts_per_batch}")
        print(f"Iterations        : {self.config.num_iterations}")

        pbar = tqdm(range(self.config.num_iterations), desc="GRPO")
        for it in pbar:
            # Sample a fresh batch of prompts (with replacement across the run).
            batch_idx = random.sample(indices, min(self.config.prompts_per_batch, num_prompts))
            batch = [dataset[i] for i in batch_idx]

            metrics = self.grpo_step(batch)
            for k, v in metrics.items():
                self.stats[k].append(v)

            pbar.set_postfix(
                reward=f"{metrics['reward']:.3f}",
                acc=f"{metrics['accuracy']:.2f}",
                kl=f"{metrics['kl']:.3f}",
            )

            if (it + 1) % self.config.log_every == 0:
                recent = lambda key: float(np.mean(self.stats[key][-self.config.log_every:]))
                # flush=True so these summary lines appear immediately in a
                # redirected/teed log (Python block-buffers stdout when it is not
                # a terminal, which would otherwise hide them until the buffer fills).
                print(
                    f"\n[iter {it+1}] reward={recent('reward'):.3f} "
                    f"acc={recent('accuracy'):.3f} format={recent('format_rate'):.3f} "
                    f"kl={recent('kl'):.4f} loss={recent('loss'):.4f}",
                    flush=True,
                )

            if (it + 1) % self.config.save_every == 0:
                self.save_adapter(output_dir)

        self.save_adapter(output_dir)
        return self.stats

    def save_adapter(self, output_dir: str):
        """Save just the LoRA adapter weights (tiny) and the tokenizer."""
        os.makedirs(output_dir, exist_ok=True)
        adapter_state = {
            name: param.detach().cpu()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        torch.save(adapter_state, os.path.join(output_dir, "lora_adapter.pt"))
        self.tokenizer.save_pretrained(output_dir)


# ============================================================================
# EVALUATION (BASE vs. FINE-TUNED)
# ============================================================================

@torch.no_grad()
def evaluate(model, tokenizer, dataset: FinQADataset, model_config: ModelConfig,
             eval_config: EvalConfig, grpo_config: GRPOConfig, label: str) -> Dict[str, Any]:
    """Greedy-decode one answer per question and measure accuracy.

    The accuracy here is the FinQA "execution accuracy" analog: did the model's
    final number match the gold `exe_ans`? We reuse the same RLVR verifier used
    in training so the train and eval definitions of "correct" are identical.
    """
    model.eval()
    n = min(eval_config.num_eval_samples, len(dataset))
    correct, has_box = 0, 0
    examples = []

    use_amp = model_config.device == "cuda" and model_config.dtype != torch.float32

    for idx in tqdm(range(n), desc=f"Eval[{label}]"):
        item = dataset[idx]
        enc = tokenizer(item["prompt"], return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(model_config.device)
        attention_mask = enc["attention_mask"].to(model_config.device)
        prompt_len = input_ids.shape[1]

        # Build generation kwargs. For greedy decoding (temperature == 0) we do
        # NOT pass sampling-only flags like `temperature`, otherwise newer
        # transformers warns that they are ignored.
        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=eval_config.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
        if eval_config.eval_temperature > 0.0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = eval_config.eval_temperature
        else:
            gen_kwargs["do_sample"] = False

        with torch.autocast(device_type="cuda", dtype=model_config.dtype, enabled=use_amp):
            out = model.generate(**gen_kwargs)
        response = tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True)
        result = compute_reward(response, item["gold"], grpo_config)
        correct += int(result["correct"])
        has_box += int(result["has_box"])

        if len(examples) < eval_config.num_examples_to_print:
            examples.append({
                "question": item["question"],
                "gold": item["gold"],
                "response": response,
                "correct": result["correct"],
            })

    return {
        "label": label,
        "accuracy": correct / max(1, n),
        "format_rate": has_box / max(1, n),
        "num_evaluated": n,
        "examples": examples,
    }


def compare_models(model, tokenizer, dataset: FinQADataset, model_config: ModelConfig,
                   eval_config: EvalConfig, grpo_config: GRPOConfig) -> Dict[str, Any]:
    """Evaluate base (adapters off) vs. fine-tuned (adapters on) on the same set.

    Because the adapters live inside one model, we just toggle them -- the base
    model needs no separate weights in memory.
    """
    print("\n" + "=" * 60)
    print("EVALUATION: BASE vs. GRPO-FINE-TUNED")
    print("=" * 60)

    set_lora_enabled(model, False)
    base_results = evaluate(model, tokenizer, dataset, model_config, eval_config,
                            grpo_config, label="base")

    set_lora_enabled(model, True)
    tuned_results = evaluate(model, tokenizer, dataset, model_config, eval_config,
                             grpo_config, label="grpo")

    print(f"\nBase  model accuracy : {base_results['accuracy']:.4f} "
          f"(format {base_results['format_rate']:.3f})")
    print(f"GRPO  model accuracy : {tuned_results['accuracy']:.4f} "
          f"(format {tuned_results['format_rate']:.3f})")
    delta = tuned_results["accuracy"] - base_results["accuracy"]
    print(f"Improvement          : {delta:+.4f} "
          f"({delta * 100:+.2f} percentage points)")

    return {"base": base_results, "grpo": tuned_results, "improvement": delta}


def print_comparison_examples(results: Dict[str, Any], num_examples: int = 4):
    """Print a few base-vs-tuned generations side by side."""
    print("\n" + "=" * 60)
    print("QUALITATIVE EXAMPLES")
    print("=" * 60)
    base_ex = results["base"]["examples"]
    grpo_ex = results["grpo"]["examples"]
    for i in range(min(num_examples, len(base_ex), len(grpo_ex))):
        print(f"\n--- Example {i+1} ---")
        print(f"Q: {base_ex[i]['question']}")
        print(f"Gold: {base_ex[i]['gold']}")
        print(f"[BASE  | correct={base_ex[i]['correct']}] {base_ex[i]['response'][:400]}")
        print(f"[GRPO  | correct={grpo_ex[i]['correct']}] {grpo_ex[i]['response'][:400]}")


def create_evaluation_report(results: Dict[str, Any], output_path: str) -> Dict[str, Any]:
    """Write a JSON report of the base-vs-tuned comparison."""
    report = {
        "base_accuracy": results["base"]["accuracy"],
        "grpo_accuracy": results["grpo"]["accuracy"],
        "improvement": results["improvement"],
        "base_format_rate": results["base"]["format_rate"],
        "grpo_format_rate": results["grpo"]["format_rate"],
        "num_evaluated": results["base"]["num_evaluated"],
    }
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved evaluation report to {output_path}")
    return report


# ============================================================================
# ENTRY POINT
# ============================================================================

def run_full_pipeline(
    data_path: str,
    output_dir: str = "./finqa_grpo_lora",
    model_name: Optional[str] = None,
    num_eval_samples: Optional[int] = None,
    num_iterations: Optional[int] = None,
    max_new_tokens: Optional[int] = None,
    group_size: Optional[int] = None,
    prompts_per_batch: Optional[int] = None,
    micro_batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    kl_coef: Optional[float] = None,
    skip_train: bool = False,
) -> Dict[str, Any]:
    """End-to-end: load model + LoRA, GRPO-train, then compare base vs. tuned."""
    set_seed(42)
    os.makedirs(output_dir, exist_ok=True)

    # ---- configs (allow a few CLI overrides) ----
    model_config = ModelConfig()
    if model_name:
        model_config.model_name = model_name
    if max_new_tokens is not None:
        model_config.max_new_tokens = max_new_tokens
    grpo_config = GRPOConfig()
    if num_iterations is not None:
        grpo_config.num_iterations = num_iterations
    if group_size is not None:
        grpo_config.group_size = group_size
    if prompts_per_batch is not None:
        grpo_config.prompts_per_batch = prompts_per_batch
    if micro_batch_size is not None:
        grpo_config.micro_batch_size = micro_batch_size
    if learning_rate is not None:
        grpo_config.learning_rate = learning_rate
    if kl_coef is not None:
        grpo_config.kl_coef = kl_coef
    lora_config = LoRAConfig()
    eval_config = EvalConfig()
    if num_eval_samples is not None:
        eval_config.num_eval_samples = num_eval_samples
    if max_new_tokens is not None:
        eval_config.max_new_tokens = max_new_tokens

    print("=" * 60)
    print("CHAPTER 11: GRPO + LoRA + RLVR on FinQA")
    print("=" * 60)
    print(f"Model  : {model_config.model_name}")
    print(f"Device : {model_config.device}  dtype: {model_config.dtype}")

    # ---- tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
    if tokenizer.pad_token is None:
        # Causal LMs often ship without a pad token; reuse EOS (Chapter 10 too).
        tokenizer.pad_token = tokenizer.eos_token

    # ---- base model + LoRA injection ----
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name,
        dtype=model_config.dtype,
    )
    lora_params = inject_lora(model, lora_config)
    model.to(model_config.device)

    # Gradient checkpointing: trade compute for memory. Instead of storing every
    # transformer layer's activations for the backward pass, recompute them on
    # the fly during backprop. This is the key lever that lets an 8B model's
    # backward pass fit on a single GPU.
    #
    # Why we still need it WITH LoRA: LoRA shrinks the optimizer state and the
    # gradients (only the tiny adapters train), but the adapters live inside
    # every attention block, so autograd must still keep the *whole* network's
    # forward activations to reach them -- and those activations, not the
    # weights, are what dominate memory for long sequences. Checkpointing cuts
    # exactly that cost. `use_reentrant=False` is the modern variant and works
    # even though our embedding inputs are frozen (require no grad).
    #
    # Note on caching: checkpointing only activates in train() mode, so our
    # generation passes (run in eval() mode under no_grad) keep their fast KV
    # cache; only the training forward recomputes.
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    counts = get_parameter_count(model)
    print(f"Parameters: {counts['trainable']:,} trainable / "
          f"{counts['total']:,} total "
          f"({100 * counts['trainable'] / counts['total']:.3f}% trainable)")

    # ---- data ----
    train_path = os.path.join(data_path, "train.json")
    test_path = os.path.join(data_path, "test.json")
    train_records = load_finqa(train_path)
    test_records = load_finqa(test_path)
    print(f"Loaded {len(train_records)} train / {len(test_records)} test records")

    train_dataset = FinQADataset(train_records, tokenizer, model_config.max_prompt_length)
    test_dataset = FinQADataset(test_records, tokenizer, model_config.max_prompt_length)

    # ---- Stage 2: GRPO training ----
    trainer = GRPOTrainer(model, tokenizer, lora_params, model_config, grpo_config)
    if not skip_train:
        trainer.train(train_dataset, output_dir)
    else:
        # Eval-only mode: try to load a previously trained adapter.
        adapter_path = os.path.join(output_dir, "lora_adapter.pt")
        if os.path.exists(adapter_path):
            state = torch.load(adapter_path, map_location=model_config.device)
            own = dict(model.named_parameters())
            for name, tensor in state.items():
                if name in own:
                    own[name].data.copy_(tensor.to(model_config.device))
            print(f"Loaded adapter from {adapter_path}")

    # ---- Stage 3: evaluation ----
    results = compare_models(model, tokenizer, test_dataset, model_config,
                             eval_config, grpo_config)
    print_comparison_examples(results, eval_config.num_examples_to_print)
    create_evaluation_report(results, os.path.join(output_dir, "evaluation_report.json"))

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Chapter 11: GRPO + LoRA + RLVR fine-tuning on FinQA."
    )
    parser.add_argument("--data_path", type=str, default="./FinQA_dataset",
                        help="Folder containing train.json / test.json.")
    parser.add_argument("--output_dir", type=str, default="./finqa_grpo_lora")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Override the base model (e.g. Qwen/Qwen2.5-3B-Instruct).")
    parser.add_argument("--num_eval_samples", type=int, default=None,
                        help="How many test questions to score.")
    parser.add_argument("--num_iterations", type=int, default=None,
                        help="Number of GRPO iterations.")
    parser.add_argument("--max_new_tokens", type=int, default=None,
                        help="Max tokens the model may generate (reasoning + answer).")
    parser.add_argument("--group_size", type=int, default=None,
                        help="G: answers sampled per prompt (GRPO group size).")
    parser.add_argument("--prompts_per_batch", type=int, default=None,
                        help="Prompts processed before each optimizer step.")
    parser.add_argument("--micro_batch_size", type=int, default=None,
                        help="Sequences per training forward/backward (memory vs speed knob).")
    parser.add_argument("--learning_rate", type=float, default=None,
                        help="LoRA learning rate.")
    parser.add_argument("--kl_coef", type=float, default=None,
                        help="Weight of the KL-to-reference penalty (0 = off; >0 anchors the policy).")
    parser.add_argument("--skip_train", action="store_true",
                        help="Skip training and evaluate (loads saved adapter if present).")
    args = parser.parse_args()

    run_full_pipeline(
        data_path=args.data_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        num_eval_samples=args.num_eval_samples,
        num_iterations=args.num_iterations,
        max_new_tokens=args.max_new_tokens,
        group_size=args.group_size,
        prompts_per_batch=args.prompts_per_batch,
        micro_batch_size=args.micro_batch_size,
        learning_rate=args.learning_rate,
        kl_coef=args.kl_coef,
        skip_train=args.skip_train,
    )


if __name__ == "__main__":
    main()
