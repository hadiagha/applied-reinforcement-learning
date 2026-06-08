# Chapter 10: RLHF with PPO — Training a Medical Support Chatbot

This chapter implements a full **Reinforcement Learning from Human Feedback (RLHF)** pipeline applied to a medical patient support chatbot. The system walks through all three canonical RLHF stages — supervised fine-tuning, reward modeling, and PPO-based RL training — using a custom-built medical dialogue dataset.

---

## Overview

The chatbot is designed to assist patients with health-related questions while remaining safe: it avoids specific diagnoses, recognizes emergencies, consistently recommends professional consultation, and responds with empathy. Safety is not bolted on — it is baked into the reward signal and evaluated explicitly throughout.

The base model is [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B), chosen for its manageable size while still demonstrating meaningful learning across all three training stages.

---

## Dataset

All data in `medical_chatbot_data/json/` was created specifically for this chapter. There are three datasets, each serving a distinct stage of the RLHF pipeline.

### `sft_dataset.json` — Supervised Fine-Tuning (600 records)

Used to teach the model the basic format and quality bar of a good medical response before any RL training begins.

| Field | Description |
|---|---|
| `id` | Unique record identifier |
| `category` | One of six medical dialogue categories |
| `query` | Patient question |
| `response` | High-quality reference response |
| `split` | `train` (540) or `validation` (60) |

**Categories covered**: symptom inquiry, emergency recognition, medication questions, mental health support, appointment logistics, chronic condition management.

Each response is carefully crafted to be empathetic, include appropriate emergency warning signs where relevant, avoid diagnosing, and end with a recommendation to seek professional care.

### `preference_dataset_cleaned.json` — Reward Model Training (2,400 records)

Used to train the reward model by showing it pairs of responses and labeling which one is safer and more helpful.

| Field | Description |
|---|---|
| `id` | Unique record identifier |
| `category` | Medical dialogue category |
| `query` | Patient question |
| `chosen` | The preferred (safer, more appropriate) response |
| `rejected` | The dispreferred response |
| `preference_reason` | Explanation of why chosen is better |
| `split` | `train` (2,160) or `validation` (240) |

The `preference_reason` field is particularly useful for understanding what the reward model is being trained to value: avoiding specific diagnoses, recognizing urgency, recommending timely professional care, and showing empathy without false reassurance.

A raw version (`preference_dataset.json`) also exists; the `_cleaned` version is what the pipeline uses.

### `eval_dataset.json` — Model Evaluation (305 records)

Used to compare all three trained models (base, SFT, PPO) on realistic patient queries with ground-truth labels.

| Field | Description |
|---|---|
| `id` | Unique record identifier |
| `category` | Medical dialogue category |
| `query` | Patient question |
| `expected_urgency` | `emergency`, `urgent`, `moderate`, or `routine` |
| `safety_critical` | Whether the query requires particular safety care |

**Urgency distribution**: routine 50%, moderate 30%, emergency 17%, urgent 3%.  
**Safety critical**: 155 true, 150 false — an intentionally balanced split for meaningful safety metrics.

---

## Code Design

All logic lives in a single file: `rlhf_fine_tuning_full_pipeline.py` (~2,600 lines). It is organized into configuration dataclasses, dataset classes, trainer classes, and a top-level pipeline runner.

### Configuration

Four dataclasses control every hyperparameter without requiring command-line argument parsing:

| Class | Purpose |
|---|---|
| `ModelConfig` | Base model name, sequence length, device/dtype |
| `SFTConfig` | SFT learning rate, batch size, epochs, warmup |
| `RewardModelConfig` | Reward model LR, margin for ranking loss |
| `PPOConfig` | Clip epsilon, GAE lambda, KL penalty, entropy coefficient |
| `SafetyConfig` | Emergency keywords, empathy keywords, dangerous content patterns |

### Stage 1 — Supervised Fine-Tuning (`SFTTrainer`)

Trains the base model on `sft_dataset.json` using standard causal language modeling. The dataset class (`MedicalSFTDataset`) formats each example as a structured prompt with a system instruction, then masks the prompt tokens in the loss (labels set to -100) so the model only learns to generate the response.

Training uses gradient accumulation, linear warmup, and cosine annealing. The best checkpoint (by validation loss) is saved and used as the starting point for Stage 2.

### Stage 2 — Reward Model (`RewardModel`, `RewardModelTrainer`)

The reward model wraps the SFT model backbone and adds a single linear head that maps the last non-padding token's hidden state to a scalar reward score.

It is trained on `preference_dataset_cleaned.json` using a pairwise ranking loss:

```
loss = -log_sigmoid(reward_chosen - reward_rejected - margin)
```

The margin pushes the model to not just rank correctly but by a meaningful gap. Preference accuracy (how often chosen > rejected) is tracked on the validation split.

### Stage 3 — PPO Training (`PPOTrainer`, `PolicyWithValueHead`)

The PPO trainer wraps the SFT model with a value head (`ValueHead`) that estimates the expected cumulative reward from a given state. A frozen copy of the SFT model serves as the reference policy for computing KL divergence.

**Rollout generation** (`generate_rollouts`):
1. Sample a query batch and generate responses with the current policy.
2. Score each response with the frozen reward model.
3. Apply reward shaping:
   - Language bonus: penalizes non-ASCII output (prevents language drift in multilingual models like Qwen).
   - Safety bonus: rewards empathy keywords (+0.2), professional referral language (+0.2), and appropriate response length (+0.1–0.2).
4. Subtract KL penalty: `shaped_reward - kl_coeff × KL(policy || reference)`.

**PPO update** (`ppo_update`):
- Computes probability ratios: `ratio = exp(log_π_new - log_π_old)`
- Clips with the standard surrogate: `min(ratio × A, clip(ratio, 1-ε, 1+ε) × A)`
- Updates value head with clipped MSE loss.
- Adds entropy bonus to encourage exploration.
- Adjusts the KL coefficient adaptively based on observed KL against the reference policy.

Advantages are computed using **Generalized Advantage Estimation (GAE)** for multi-step rollouts or a simplified single-step estimate `A = R - V` depending on the setting.

### Stage 4 — Evaluation (`SafetyEvaluator`, `compare_models`)

The safety evaluator scores every generated response on five dimensions:

| Dimension | What it measures |
|---|---|
| Emergency recognition | Does it recommend 911 or ER when appropriate? |
| Empathy | Does it use empathetic language? |
| Professional referral | Does it include a disclaimer to see a doctor? |
| Dangerous content | Does it give specific diagnoses or dosages? |
| Urgency matching | Does its urgency match the expected label? |

`compare_models()` runs all three models (base, SFT, PPO) on the eval set and `create_evaluation_report()` writes a JSON report with per-model metrics and computed improvements.

---

## How to Run

### Requirements

```bash
pip install torch transformers datasets accelerate
```

A GPU is strongly recommended. The pipeline is tested with Qwen2.5-0.5B; the `ModelConfig` handles device detection and dtype selection automatically.

### Run the Full Pipeline

```bash
python rlhf_fine_tuning_full_pipeline.py
```

This runs all four stages sequentially:

1. SFT training on the 540-example training split.
2. Reward model training on the 2,160-example preference training split.
3. PPO training using the reward model signal.
4. Evaluation comparing base, SFT, and PPO models on the 305-example eval set.

Checkpoints for each stage are saved to disk as each stage completes, so a crash mid-pipeline does not require starting over from scratch.

### Inspect PPO Mechanics

The file includes a standalone educational function:

```python
demonstrate_ppo_clipping()
```

Call this directly to step through the PPO clipping mechanism with concrete numerical examples — useful for building intuition before reading the full training loop.
