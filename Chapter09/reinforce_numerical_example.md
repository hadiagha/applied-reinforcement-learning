# REINFORCE Algorithm: A Complete Numerical Example

This document provides a step-by-step numerical walkthrough of the REINFORCE algorithm, demonstrating exactly how policy gradients update the policy parameters after observing a trajectory.

## Problem Setup

**Scenario**: An inventory manager must decide how many units to order each day. There are 10 possible actions (order quantities 0 through 9 units).

**Action Space**: A = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} — 10 discrete actions

**Episode Length**: 5 time steps (a short planning horizon)

**Policy Representation**: The policy is a neural network that outputs 10 logits, which are converted to probabilities via softmax. For simplicity, we assume a single state (or that the state is the same throughout this example).

---

## Initial Policy

Suppose the policy network currently outputs the following **logits** (raw scores before softmax):

| Action | Logit (z) |
|--------|-----------|
| 0 | 0.5 |
| 1 | 1.0 |
| 2 | 1.5 |
| 3 | 2.0 |
| 4 | 1.8 |
| 5 | 1.2 |
| 6 | 0.8 |
| 7 | 0.3 |
| 8 | 0.1 |
| 9 | -0.2 |

### Step 1: Convert Logits to Probabilities (Softmax)

The softmax function converts logits to probabilities:

$$\pi_\theta(a_k|s) = \frac{\exp(z_k)}{\sum_{j=0}^{9} \exp(z_j)}$$

First, compute the exponentials:

| Action | Logit (z) | exp(z) |
|--------|-----------|--------|
| 0 | 0.5 | 1.649 |
| 1 | 1.0 | 2.718 |
| 2 | 1.5 | 4.482 |
| 3 | 2.0 | 7.389 |
| 4 | 1.8 | 6.050 |
| 5 | 1.2 | 3.320 |
| 6 | 0.8 | 2.226 |
| 7 | 0.3 | 1.350 |
| 8 | 0.1 | 1.105 |
| 9 | -0.2 | 0.819 |

**Sum of exponentials**: 1.649 + 2.718 + 4.482 + 7.389 + 6.050 + 3.320 + 2.226 + 1.350 + 1.105 + 0.819 = **31.108**

Now divide each exp(z) by the sum to get probabilities:

| Action | exp(z) | Probability π(a\|s) |
|--------|--------|---------------------|
| 0 | 1.649 | 0.053 (5.3%) |
| 1 | 2.718 | 0.087 (8.7%) |
| 2 | 4.482 | 0.144 (14.4%) |
| 3 | 2.0 | 0.237 (23.7%) |
| 4 | 6.050 | 0.194 (19.4%) |
| 5 | 3.320 | 0.107 (10.7%) |
| 6 | 2.226 | 0.072 (7.2%) |
| 7 | 1.350 | 0.043 (4.3%) |
| 8 | 1.105 | 0.036 (3.6%) |
| 9 | 0.819 | 0.026 (2.6%) |

**Interpretation**: The current policy favors action 3 (order 3 units) with 23.7% probability, followed by action 4 (19.4%) and action 2 (14.4%).

---

## The Trajectory

The agent runs one episode and takes the following actions:

| Time Step (t) | Action Taken | Reward Received |
|---------------|--------------|-----------------|
| 0 | 3 | +2 |
| 1 | 2 | +1 |
| 2 | 3 | +3 |
| 3 | 5 | -1 |
| 4 | 4 | +2 |

**Key observation**: 
- 4 unique actions were taken: {2, 3, 4, 5}
- Action 3 was taken **twice** (at t=0 and t=2)
- Actions {0, 1, 6, 7, 8, 9} were never taken

---

## Step 2: Compute the Return-to-Go G(t)

In REINFORCE, each action at time t is weighted by the **return-to-go** G(t)—the sum of rewards from time t onward. This is more principled than using the total trajectory return R(τ), because an action at time t can only affect rewards *after* it was taken, not before.

For simplicity, we use **undiscounted returns** (γ = 1). The return-to-go is computed by summing rewards from each time step to the end:

$$G_t = \sum_{k=t}^{T} r_k$$

Let's compute G(t) for each time step, working backwards:

| Time Step (t) | Reward rₜ | Calculation | G(t) |
|---------------|-----------|-------------|------|
| 4 | +2 | G₄ = r₄ = 2 | **+2** |
| 3 | -1 | G₃ = r₃ + G₄ = -1 + 2 | **+1** |
| 2 | +3 | G₂ = r₂ + G₃ = 3 + 1 | **+4** |
| 1 | +1 | G₁ = r₁ + G₂ = 1 + 4 | **+5** |
| 0 | +2 | G₀ = r₀ + G₁ = 2 + 5 | **+7** |

**Key insight**: Each action is now weighted by a *different* return:
- Action at t=0 is weighted by G₀ = +7 (it affects all future rewards)
- Action at t=4 is weighted by G₄ = +2 (it only affects the final reward)

This is more accurate credit assignment than using R(τ) = +7 for every action.

---

## Step 3: Compute Log-Probabilities

For each action taken, we need the log-probability under the current policy.

The log-probability for a softmax policy is:

$$\log \pi_\theta(a_k|s) = z_k - \log \sum_{j} \exp(z_j)$$

We already computed $\sum_j \exp(z_j) = 31.108$, so $\log(31.108) = 3.438$.

| Time Step | Action | Logit (z) | Log-Prob = z - 3.438 |
|-----------|--------|-----------|----------------------|
| 0 | 3 | 2.0 | 2.0 - 3.438 = **-1.438** |
| 1 | 2 | 1.5 | 1.5 - 3.438 = **-1.938** |
| 2 | 3 | 2.0 | 2.0 - 3.438 = **-1.438** |
| 3 | 5 | 1.2 | 1.2 - 3.438 = **-2.238** |
| 4 | 4 | 1.8 | 1.8 - 3.438 = **-1.638** |

**Verification**: Log-probabilities should equal log(probability). For action 3: log(0.237) ≈ -1.44 ✓

---

## Step 4: Compute the Policy Gradient

The REINFORCE gradient using return-to-go is:

$$\nabla_\theta J(\theta) = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t$$

Notice that each action is weighted by its **own** G(t), not a shared R(τ).

### Understanding the Gradient for Softmax

For a softmax policy, the gradient of the log-probability with respect to the logits has a beautiful form:

$$\frac{\partial \log \pi(a_k|s)}{\partial z_j} = \begin{cases} 1 - \pi(a_k|s) & \text{if } j = k \text{ (the chosen action)} \\ -\pi(a_j|s) & \text{if } j \neq k \text{ (all other actions)} \end{cases}$$

**Intuition**: 
- For the **chosen action**: the gradient is positive (1 - π), pushing to increase its probability
- For **all other actions**: the gradient is negative (-π), pushing to decrease their probabilities
- The magnitudes ensure that probability mass is redistributed (probabilities must sum to 1)

### Computing Gradients for Each Time Step

Let's compute the gradient contribution from each time step. **Crucially, each time step uses its own G(t)**.

#### Time Step 0: Action 3 taken, G₀ = +7

Gradient w.r.t. each logit:

| Logit | Gradient Formula | Value | × G₀ = +7 |
|-------|------------------|-------|-------------|
| z₀ | -π(0) | -0.053 | -0.371 |
| z₁ | -π(1) | -0.087 | -0.609 |
| z₂ | -π(2) | -0.144 | -1.008 |
| z₃ | 1 - π(3) | +0.763 | **+5.341** |
| z₄ | -π(4) | -0.194 | -1.358 |
| z₅ | -π(5) | -0.107 | -0.749 |
| z₆ | -π(6) | -0.072 | -0.504 |
| z₇ | -π(7) | -0.043 | -0.301 |
| z₈ | -π(8) | -0.036 | -0.252 |
| z₉ | -π(9) | -0.026 | -0.182 |

#### Time Step 1: Action 2 taken, G₁ = +5

| Logit | Gradient Formula | Value | × G₁ = +5 |
|-------|------------------|-------|-------------|
| z₀ | -π(0) | -0.053 | -0.265 |
| z₁ | -π(1) | -0.087 | -0.435 |
| z₂ | 1 - π(2) | +0.856 | **+4.280** |
| z₃ | -π(3) | -0.237 | -1.185 |
| z₄ | -π(4) | -0.194 | -0.970 |
| z₅ | -π(5) | -0.107 | -0.535 |
| z₆ | -π(6) | -0.072 | -0.360 |
| z₇ | -π(7) | -0.043 | -0.215 |
| z₈ | -π(8) | -0.036 | -0.180 |
| z₉ | -π(9) | -0.026 | -0.130 |

#### Time Step 2: Action 3 taken (again), G₂ = +4

| Logit | Gradient Formula | Value | × G₂ = +4 |
|-------|------------------|-------|-------------|
| z₀ | -π(0) | -0.053 | -0.212 |
| z₁ | -π(1) | -0.087 | -0.348 |
| z₂ | -π(2) | -0.144 | -0.576 |
| z₃ | 1 - π(3) | +0.763 | **+3.052** |
| z₄ | -π(4) | -0.194 | -0.776 |
| z₅ | -π(5) | -0.107 | -0.428 |
| z₆ | -π(6) | -0.072 | -0.288 |
| z₇ | -π(7) | -0.043 | -0.172 |
| z₈ | -π(8) | -0.036 | -0.144 |
| z₉ | -π(9) | -0.026 | -0.104 |

#### Time Step 3: Action 5 taken, G₃ = +1

| Logit | Gradient Formula | Value | × G₃ = +1 |
|-------|------------------|-------|-------------|
| z₀ | -π(0) | -0.053 | -0.053 |
| z₁ | -π(1) | -0.087 | -0.087 |
| z₂ | -π(2) | -0.144 | -0.144 |
| z₃ | -π(3) | -0.237 | -0.237 |
| z₄ | -π(4) | -0.194 | -0.194 |
| z₅ | 1 - π(5) | +0.893 | **+0.893** |
| z₆ | -π(6) | -0.072 | -0.072 |
| z₇ | -π(7) | -0.043 | -0.043 |
| z₈ | -π(8) | -0.036 | -0.036 |
| z₉ | -π(9) | -0.026 | -0.026 |

#### Time Step 4: Action 4 taken, G₄ = +2

| Logit | Gradient Formula | Value | × G₄ = +2 |
|-------|------------------|-------|-------------|
| z₀ | -π(0) | -0.053 | -0.106 |
| z₁ | -π(1) | -0.087 | -0.174 |
| z₂ | -π(2) | -0.144 | -0.288 |
| z₃ | -π(3) | -0.237 | -0.474 |
| z₄ | 1 - π(4) | +0.806 | **+1.612** |
| z₅ | -π(5) | -0.107 | -0.214 |
| z₆ | -π(6) | -0.072 | -0.144 |
| z₇ | -π(7) | -0.043 | -0.086 |
| z₈ | -π(8) | -0.036 | -0.072 |
| z₉ | -π(9) | -0.026 | -0.052 |

---

## Step 5: Sum the Gradients Across All Time Steps

Now we sum the gradient contributions from all 5 time steps. Note how each column uses a different G(t):

| Logit | t=0 (G₀=7) | t=1 (G₁=5) | t=2 (G₂=4) | t=3 (G₃=1) | t=4 (G₄=2) | **Total Gradient** |
|-------|------------|------------|------------|------------|------------|-------------------|
| z₀ | -0.371 | -0.265 | -0.212 | -0.053 | -0.106 | **-1.007** |
| z₁ | -0.609 | -0.435 | -0.348 | -0.087 | -0.174 | **-1.653** |
| z₂ | -1.008 | +4.280 | -0.576 | -0.144 | -0.288 | **+2.264** |
| z₃ | +5.341 | -1.185 | +3.052 | -0.237 | -0.474 | **+6.497** |
| z₄ | -1.358 | -0.970 | -0.776 | -0.194 | +1.612 | **-1.686** |
| z₅ | -0.749 | -0.535 | -0.428 | +0.893 | -0.214 | **-1.033** |
| z₆ | -0.504 | -0.360 | -0.288 | -0.072 | -0.144 | **-1.368** |
| z₇ | -0.301 | -0.215 | -0.172 | -0.043 | -0.086 | **-0.817** |
| z₈ | -0.252 | -0.180 | -0.144 | -0.036 | -0.072 | **-0.684** |
| z₉ | -0.182 | -0.130 | -0.104 | -0.026 | -0.052 | **-0.494** |

### Key Observations

1. **Action 3 has the largest positive gradient (+6.497)** because it was taken twice (at t=0 with G₀=7, and at t=2 with G₂=4). REINFORCE will strongly increase its probability.

2. **Action 2 also has a positive gradient (+2.264)** because it was taken at t=1 when G₁=5 was still relatively high.

3. **Actions 4 and 5 have negative gradients despite being taken!** This is the key difference from using R(τ). Action 5 was taken at t=3 when G₃=+1 (small), and action 4 was taken at t=4 when G₄=+2 (also small). The positive contributions from being chosen are outweighed by the negative contributions from other time steps when they weren't chosen.

4. **Using G(t) provides better credit assignment**: Early actions (t=0, t=1) that could influence more future rewards get weighted more heavily. Late actions (t=3, t=4) get weighted less because they only affect a few remaining rewards.

5. **Actions 0, 1, 6, 7, 8, 9 all have negative gradients** because they were never taken. Probability mass must be redistributed to make the taken actions more likely.

---

## Step 6: Apply the Update

The REINFORCE update rule is:

$$\theta \leftarrow \theta + \alpha \cdot \nabla_\theta J(\theta)$$

Let's use a learning rate of **α = 0.1**.

### New Logits

| Action | Old Logit | Gradient | α × Gradient | **New Logit** |
|--------|-----------|----------|--------------|---------------|
| 0 | 0.5 | -1.007 | -0.101 | **0.399** |
| 1 | 1.0 | -1.653 | -0.165 | **0.835** |
| 2 | 1.5 | +2.264 | +0.226 | **1.726** |
| 3 | 2.0 | +6.497 | +0.650 | **2.650** |
| 4 | 1.8 | -1.686 | -0.169 | **1.631** |
| 5 | 1.2 | -1.033 | -0.103 | **1.097** |
| 6 | 0.8 | -1.368 | -0.137 | **0.663** |
| 7 | 0.3 | -0.817 | -0.082 | **0.218** |
| 8 | 0.1 | -0.684 | -0.068 | **0.032** |
| 9 | -0.2 | -0.494 | -0.049 | **-0.249** |

### New Probabilities (After Softmax)

Computing softmax on the new logits:

| Action | New Logit | exp(new logit) | **New Probability** | Change |
|--------|-----------|----------------|---------------------|--------|
| 0 | 0.399 | 1.490 | 0.046 (4.6%) | ↓ from 5.3% |
| 1 | 0.835 | 2.305 | 0.071 (7.1%) | ↓ from 8.7% |
| 2 | 1.726 | 5.618 | 0.173 (17.3%) | ↑ from 14.4% |
| 3 | 2.650 | 14.15 | 0.436 (43.6%) | ↑↑ from 23.7% |
| 4 | 1.631 | 5.109 | 0.157 (15.7%) | ↓ from 19.4% |
| 5 | 1.097 | 2.995 | 0.092 (9.2%) | ↓ from 10.7% |
| 6 | 0.663 | 1.941 | 0.060 (6.0%) | ↓ from 7.2% |
| 7 | 0.218 | 1.244 | 0.038 (3.8%) | ↓ from 4.3% |
| 8 | 0.032 | 1.033 | 0.032 (3.2%) | ↓ from 3.6% |
| 9 | -0.249 | 0.780 | 0.024 (2.4%) | ↓ from 2.6% |

**Sum of new exp values**: 32.47 (used for normalization)

---

## Summary of Changes

```
BEFORE                              AFTER
Action 3: 23.7%  ──────────────►  Action 3: 43.6%  (+19.9%)  ★ Taken twice (t=0, t=2)
Action 4: 19.4%  ──────────────►  Action 4: 15.7%  (-3.7%)   Taken at t=4 (low G₄=2)
Action 2: 14.4%  ──────────────►  Action 2: 17.3%  (+2.9%)   Taken at t=1 (high G₁=5)
Action 5: 10.7%  ──────────────►  Action 5:  9.2%  (-1.5%)   Taken at t=3 (low G₃=1)
Action 1:  8.7%  ──────────────►  Action 1:  7.1%  (-1.6%)   Never taken
Action 6:  7.2%  ──────────────►  Action 6:  6.0%  (-1.2%)   Never taken
Action 0:  5.3%  ──────────────►  Action 0:  4.6%  (-0.7%)   Never taken
Action 7:  4.3%  ──────────────►  Action 7:  3.8%  (-0.5%)   Never taken
Action 8:  3.6%  ──────────────►  Action 8:  3.2%  (-0.4%)   Never taken
Action 9:  2.6%  ──────────────►  Action 9:  2.4%  (-0.2%)   Never taken
```

---

## Key Takeaways

### 1. G(t) provides proper credit assignment

Using return-to-go G(t) instead of total return R(τ) means each action is weighted by the rewards it could actually influence. Action at t=0 is weighted by G₀=7 (all future rewards), while action at t=4 is weighted by only G₄=2 (just the final reward).

### 2. Actions taken early with high G(t) get reinforced most

Action 3 was taken at t=0 (G₀=7) and t=2 (G₂=4), receiving strong positive gradients both times. Its probability jumped from 23.7% to 43.6%—the largest increase.

### 3. Actions taken late with low G(t) may not be reinforced

Action 4 was taken at t=4 when G₄=+2, and action 5 was taken at t=3 when G₃=+1. Despite being chosen, their probabilities *decreased* because the small positive contribution from being selected was outweighed by negative contributions from other time steps.

### 4. This is more accurate than using R(τ) for all actions

If we had used R(τ)=+7 for every action (the naive approach), actions 4 and 5 would have been reinforced equally to actions taken earlier. But that's wrong—action 4 at t=4 couldn't have caused the rewards at t=0, t=1, t=2, or t=3. G(t) correctly ignores those past rewards.

### 5. Untaken actions are still suppressed

Actions 0, 1, 6, 7, 8, 9 were never taken, so their probabilities all decreased. This is a mathematical consequence of softmax: to increase some probabilities, others must decrease.

---

## What If the Returns Were Negative?

Suppose all the rewards were negative (a bad trajectory). The G(t) values would be negative, and the gradients would flip:

| Action | Gradient (positive G) | Gradient (negative G) |
|--------|----------------------|----------------------|
| z₃ | +6.497 | **-6.497** |
| z₂ | +2.264 | **-2.264** |
| z₄ | -1.686 | **+1.686** |
| z₅ | -1.033 | **+1.033** |
| z₀ | -1.007 | **+1.007** |
| z₁ | -1.653 | **+1.653** |
| ... | ... | ... |

Now the taken actions (especially those taken early when G(t) was large) would be **suppressed**, and the untaken actions would **relatively increase**. The agent would learn: "that combination of actions led to a bad outcome—try something different next time."

---

## Connection to the Chapter

This example demonstrates the core mechanics of REINFORCE with return-to-go:

1. **Sample a trajectory** using the current policy
2. **Compute G(t) for each time step** (the return-to-go from that point onward)
3. **Compute log-probability gradients** for each action taken
4. **Scale each gradient by its own G(t)** (proper credit assignment)
5. **Update the policy** to make successful early actions more likely

The key insight is that G(t) provides better credit assignment than using R(τ) for all actions. Actions taken early in the trajectory (when G(t) is large) get stronger updates, while actions taken late (when G(t) is small) get weaker updates. This matches our intuition: early decisions have more influence on the total outcome.

The same logic applies to continuous actions (Gaussian policies) and to more sophisticated variants like REINFORCE with baseline—the baseline simply centers the returns around zero to reduce variance, but the fundamental update mechanism remains the same.
