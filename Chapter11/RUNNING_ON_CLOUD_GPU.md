# Running Chapter 11 on a Cloud GPU

This chapter's training code (`grpo_rlvr_finqa_full_pipeline.py`) needs an **NVIDIA GPU**.
You don't need anything exotic — it runs on a single GPU and scales with the model you pick:

| Model (`--model_name`) | Rough GPU memory | Notes |
|---|---|---|
| `Qwen/Qwen2.5-0.5B-Instruct` | ~8 GB | smoke-testing only |
| `Qwen/Qwen2.5-1.5B-Instruct` | ~16–24 GB | the chapter's main model |
| `Qwen/Qwen2.5-3B-Instruct` | ~24–40 GB | larger demo |
| `Qwen/Qwen3-8B` | ~40–80 GB | A100-class |

A 24–48 GB GPU (e.g. L4, A10, RTX 4090/6000, A100) is plenty for the 1.5B/3B models. The
code auto-detects the GPU and uses bf16 if available, so the same file runs unchanged on any
NVIDIA card or falls back to CPU.

The guide has two parts:
- **Part A — Brev** (NVIDIA's managed GPU service), step by step.
- **Part B — Any other cloud GPU** (Lambda, RunPod, Vast.ai, Paperspace, AWS/GCP, a lab
  cluster) — the workflow is identical once you can `ssh` and `scp`.

Two universal ideas you'll use everywhere:
- **Interactive ("normal") run** — you watch the logs live; the job stops if your laptop
  disconnects. Good for short tests.
- **Detached run (tmux or `nohup`)** — the job keeps running after you close your laptop.
  Use this for the multi-hour training runs.

---

## Part A — Brev

### A.1 Install the CLI and log in (on your laptop)
```bash
# macOS / Linux
brew install brevdev/homebrew-brev/brev   # or see https://docs.nvidia.com/brev
brev login
```

### A.2 Find and create a GPU instance
```bash
brev ls                       # your existing instances
brev create ch11-gpu          # create with smart defaults
# ...or pick a GPU explicitly:
brev create ch11-gpu --gpu-name A100 --min-vram 40
```
Give it a clear name (e.g. `ch11-grpo`). Creation + boot takes a few minutes — wait until
`brev ls` shows the instance **RUNNING** with **SHELL: READY** before connecting.

> 💡 GPUs bill by the hour. **Stop or delete the instance when you're done** (see A.9).

### A.3 Connect
```bash
brev shell ch11-gpu           # interactive SSH shell on the box
# or run a single command without an interactive shell:
brev exec ch11-gpu "nvidia-smi"
```

> The remote username/home differ by provider image (it might be `ubuntu`, `shadeform`,
> etc.). Find your home once and use it consistently:
> ```bash
> brev exec ch11-gpu 'whoami; echo $HOME'
> ```
> Below we assume the project lives in `$HOME/ch11` on the box.

### A.4 Copy the code and data up (from your laptop)
```bash
cd /path/to/coding_ch11           # the folder with the .py, requirements.txt, FinQA_dataset/

brev exec ch11-gpu "mkdir -p \$HOME/ch11/FinQA_dataset"
brev copy ./grpo_rlvr_finqa_full_pipeline.py ch11-gpu:\$HOME/ch11/
brev copy ./requirements.txt                  ch11-gpu:\$HOME/ch11/
brev copy ./FinQA_dataset/train.json          ch11-gpu:\$HOME/ch11/FinQA_dataset/
brev copy ./FinQA_dataset/test.json           ch11-gpu:\$HOME/ch11/FinQA_dataset/
```
(`brev copy` is `scp` under the hood; it also copies directories.)

### A.5 Set up the Python environment — and the one gotcha that bites everyone
```bash
brev shell ch11-gpu
cd ~/ch11
pip install -r requirements.txt
```

⚠️ **CUDA/torch version mismatch.** `pip install torch` grabs the *newest* PyTorch wheel,
which may bundle a CUDA runtime newer than the instance's driver supports. The symptom:
```python
>>> import torch; torch.cuda.is_available()
False        # with a "CUDA driver is too old" warning
```
Fix it by installing the torch build that matches your **driver's** CUDA version (the
"CUDA Version" shown top-right in `nvidia-smi`). For example, a CUDA 12.8 driver:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```
See https://pytorch.org for the right index URL (`cu121`, `cu124`, `cu128`, …). Then verify:
```bash
python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# want: ... True
```
(One more: `transformers`' chat templates need `jinja2>=3.1` — already pinned in
`requirements.txt`, but if an old system `jinja2` shadows it, `pip install --user 'jinja2>=3.1'`.)

### A.6 Run — interactive ("normal") way
Best for a quick smoke test where you want to see logs scroll live:
```bash
cd ~/ch11
python3 grpo_rlvr_finqa_full_pipeline.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --num_iterations 2 --num_eval_samples 8
```
You'll see the parameter counts, the training progress bar, and the base-vs-fine-tuned eval.
**If you close your laptop or lose Wi-Fi, this job dies.** That's fine for a 2-minute test,
but not for real training — use the detached way below.

### A.7 Run — detached, so it survives disconnects

**Option 1: tmux** (a terminal that keeps running on the server)
```bash
brev shell ch11-gpu
tmux new -s train                       # start a named session
cd ~/ch11
python3 grpo_rlvr_finqa_full_pipeline.py \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --num_iterations 150 --num_eval_samples 300 \
  --learning_rate 5e-5 --kl_coef 0.04 --max_new_tokens 512 \
  --output_dir ./finqa_1.5b 2>&1 | tee run.log
# DETACH (leave it running): press  Ctrl-b , release, then press  d
```
Now you can close everything. To come back later:
```bash
brev shell ch11-gpu
tmux attach -t train        # reattach;  detach again with Ctrl-b then d
tmux ls                     # list sessions
```

**Option 2: `nohup`** (simpler — no detach keystrokes, backgrounds immediately)
```bash
brev shell ch11-gpu
cd ~/ch11
nohup python3 grpo_rlvr_finqa_full_pipeline.py \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --num_iterations 150 --num_eval_samples 300 \
  --learning_rate 5e-5 --kl_coef 0.04 --max_new_tokens 512 \
  --output_dir ./finqa_1.5b \
  > run.log 2>&1 &
# prints a PID and returns your prompt; safe to close the laptop
```

> Tip: to keep the live progress bar flowing into the log under `nohup`/`tee`, run Python
> unbuffered with `python3 -u ...` (the periodic `[iter]` summary lines already flush).
> On a fragmentation-prone large run you can also prefix
> `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

### A.8 Monitor a running job
From the **box** (e.g. a second `brev shell`):
```bash
grep '\[iter' ~/ch11/run.log | tail -20     # smoothed per-5-iter metrics
tail -f ~/ch11/run.log                        # live stream (Ctrl-c stops watching, NOT training)
pgrep -af grpo_rlvr_finqa                     # confirm the process is alive
nvidia-smi                                    # GPU utilization / memory
```
From your **laptop** without logging in (`brev` only exists on your laptop, not the box):
```bash
brev exec ch11-gpu "grep '\[iter' ~/ch11/run.log | tail -20"
```
Watch that **`kl`** rises into a stable band and **`reward`/`acc`** trend up (see the chapter
for what healthy training looks like).

### A.9 Stop / delete when done (avoid surprise bills)
**First copy any results back to your laptop:**
```bash
brev copy ch11-gpu:\$HOME/ch11/finqa_1.5b/    ./results/finqa_1.5b/   # adapter + report
brev copy ch11-gpu:\$HOME/ch11/run.log        ./results/run.log
```
Then:
```bash
brev stop ch11-gpu      # pause (keeps disk, cheaper, can restart)
brev delete ch11-gpu    # destroy completely (stops all billing)
```

---

## Part B — Any other cloud GPU (Lambda, RunPod, Vast.ai, Paperspace, AWS/GCP, lab cluster)

Once you have **`ssh` access** to a GPU box, the workflow is the same as Part A — only the
"create instance" and "copy files" commands change. The provider gives you a host/IP, a user,
and either a password or an SSH key.

### B.1 Connect and copy files (standard SSH/SCP)
```bash
# connect
ssh user@HOST                                  # (RunPod/Vast often give: ssh root@HOST -p PORT -i key)

# copy code + data up (from your laptop)
scp grpo_rlvr_finqa_full_pipeline.py requirements.txt user@HOST:~/ch11/
scp -r FinQA_dataset user@HOST:~/ch11/
# (rsync is nicer for big/resumable transfers:)
rsync -avP FinQA_dataset grpo_rlvr_finqa_full_pipeline.py requirements.txt user@HOST:~/ch11/
```
Provider quick-notes:
- **Lambda Cloud / Paperspace / AWS-GCP-Azure VMs** — plain `ssh user@IP` with your key; user
  is usually `ubuntu`.
- **RunPod / Vast.ai** — they give a custom `ssh root@HOST -p <port> -i ~/.ssh/key`; pass the
  same `-p <port>` to `scp`. Many of their images are pre-loaded with CUDA + PyTorch.
- **University / Slurm clusters** — `ssh` to the login node, then request a GPU
  (`srun --gres=gpu:1 --pty bash`) before running; copy with `scp`/`rsync` as above.

### B.2 Environment, run, monitor — identical to Part A
Everything from **A.5 onward is provider-independent** because it's just Python on Linux:
- Install deps and **fix the torch/CUDA match** (A.5) — this gotcha applies on *every*
  provider; always check `nvidia-smi` for the driver's CUDA version and install the matching
  torch wheel.
- Use **tmux** or **`nohup`** for long runs (A.7). These are standard Linux tools present on
  essentially every box — nothing brev-specific about them.
- Monitor with `grep`/`tail -f`/`nvidia-smi` on the box (A.8).
- **Copy results back with `scp`/`rsync`, then terminate the instance** to stop billing (A.9).
  On hourly providers this is the step people forget — set a reminder.

---

## Quick reference card

```text
CONNECT            brev shell <name>            |  ssh user@HOST [-p PORT -i key]
COPY UP            brev copy ./f <name>:~/ch11/ |  scp ./f user@HOST:~/ch11/   (rsync -avP for big)
GPU CHECK          nvidia-smi                   (note the CUDA Version, top-right)
TORCH FIX          pip install torch --index-url https://download.pytorch.org/whl/cuXXX
START (detached)   tmux new -s train  →  run  →  Ctrl-b then d
   or              nohup python3 -u ... > run.log 2>&1 &
REATTACH           tmux attach -t train
MONITOR            grep '\[iter' run.log | tail ;  tail -f run.log ;  nvidia-smi
IS IT ALIVE?       pgrep -af grpo_rlvr_finqa
COPY RESULTS BACK  brev copy <name>:~/ch11/<outdir>/ ./   |  scp -r user@HOST:~/ch11/<outdir> ./
SHUT DOWN          brev stop/delete <name>      |  provider's terminate button  (STOPS BILLING)
```

### tmux survival kit
| Action | Keys / command |
|---|---|
| New named session | `tmux new -s train` |
| **Detach** (leave running) | `Ctrl-b` then `d` |
| Reattach | `tmux attach -t train` |
| List sessions | `tmux ls` |
| New window inside tmux | `Ctrl-b` then `c` |
| Scroll (then `q` to exit) | `Ctrl-b` then `[` |
| Kill the session | `tmux kill-session -t train` |

If `tmux` isn't installed: `sudo apt-get install -y tmux` (or just use the `nohup` approach,
which needs nothing extra).
