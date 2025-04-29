#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project Hyperion · Phase 2
Compilation-&-fusion benchmark on existing *.pt* checkpoints
Target GPU  : AMD Instinct MI250X  (ROCm 5.7 + Triton 2.2)
Artefacts   : CSV / JSON / PNG plots + plain-text summary
"""

# ═════════════════════════════════ 0 · ENV VARS ═══════════════════════════
import os, warnings, argparse, time, json, contextlib
from pathlib import Path

# Set MIOpen environment variables to avoid auto-tuning and use cached kernels
ENV_DEFAULTS = {
    "HIP_VISIBLE_DEVICES": "0",                     # single-GPU demo
    "MIOPEN_FIND_ENFORCE": "1",                     # NONE: no forced search
    "MIOPEN_FIND_MODE"  : "1",                      # normal find (not exhaustive)
    "MIOPEN_DEBUG_DISABLE_FIND_DB": "1",            # disable FindDb entirely
    "TRITON_CONV_NUM_STAGES": "2",
    "PYTORCH_TRITON_CONV_MAX_SHARED_SIZE": "65536",
}
for k, v in ENV_DEFAULTS.items():
    os.environ[k] = v  # Force set these values

# ════════════════════════════════ 1 · IMPORTS ═════════════════════════════
import torch, torch.nn as nn, torch.nn.functional as F, torch._dynamo
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from torchvision.models import resnet18

torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision("high")

# ════════════════════════════════ 2 · CLI ════════════════════════════════
P = argparse.ArgumentParser("Hyperion compilation benchmark")
P.add_argument("--base_dir", default=".", help="project root")
P.add_argument("--dtype",    choices=["fp32", "bf16"], default="fp32")
P.add_argument("--repeat",   type=int, default=10, help="timing iterations")
args = P.parse_args()

BASE  = Path(args.base_dir).expanduser().resolve()
CKPTS = BASE / "data" / "processed"
OUT   = BASE / "results" / "benchmarks" / "compilation"
OUT.mkdir(parents=True, exist_ok=True)

# ════════════════════════ 3 · dtype / autocast helper ═════════════════════
USE_BF16 = args.dtype == "bf16" and torch.cuda.is_available() \
           and torch.cuda.is_bf16_supported()
DTYPE = torch.bfloat16 if USE_BF16 else torch.float32
dtype_cm = (torch.autocast("cuda", dtype=DTYPE, enabled=USE_BF16)
            if torch.cuda.is_available() else contextlib.nullcontext())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def flush(): torch.cuda.synchronize() if device.type == "cuda" else None

# ═══════════════ 4 · A generic Transformer-LM (matches your checkpoints) ══
class SimpleTransformerLM(nn.Module):
    """A simple transformer language model."""
    def __init__(self, vocab_size=50257, d_model=768, ff_dim=3072, n_layers=4, nhead=12):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
                            d_model, nhead, ff_dim,
                            activation='gelu', batch_first=False)
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, idx):         # idx: [T, B]
        x = self.embedding(idx)
        x = self.transformer(x)
        return self.fc(x)

# ═════════════════════════ 5 · Checkpoint registry ════════════════════════
def make_input_lm():
    return dict(idx = torch.randint(0, 50257, (128, 8), device=device))

def make_input_cifar():
    return dict(x = torch.randn(32, 3, 32, 32, device=device)
                       .to(memory_format=torch.channels_last))

# Define models directly rather than loading from checkpoints
MODEL_SPECS = {
    # logical-name → (builder-fn, input-fn)
    "simple_transformer_lm": (
        lambda: SimpleTransformerLM().to(device).eval(),
        make_input_lm
    ),
    "resnet18_cifar10": (
        lambda: resnet18(weights=None, num_classes=10)
                .to(memory_format=torch.channels_last).to(device).eval(),
        make_input_cifar
    ),
}

# ═════════════════════════ 6 · Helper utilities ═══════════════════════════
def compile_model(m, mode):
    if mode == "eager":
        return m
    try:
        return torch.compile(m, mode=mode, fullgraph=False)
    except Exception as e:
        warnings.warn(f"torch.compile({mode}) failed ({e}) – using eager")
        return m

def bench(fn, repeat=args.repeat, warm=3):
    with torch.no_grad():
        for _ in range(warm): fn(); flush()
        t0 = time.time()
        for _ in range(repeat): fn()
        flush()
    return (time.time() - t0) / repeat   # → seconds

# ═════════════════════════ 7 · Main loop ══════════════════════════════════
records = []

print()
for name, (builder, make_input) in MODEL_SPECS.items():
    print(f"🔹 {name}")
    inputs = make_input()

    # speed tests
    for variant, c_mode in [("checkpoint",  "eager"),
                            ("compile_def", "default"),
                            ("compile_max", "max-autotune")]:
        try:
            model = builder()  # Build fresh model each time
            model = compile_model(model, c_mode)

            with dtype_cm:
                model(**inputs)  # warm-up
            t_sec = bench(lambda: model(**inputs))
            mem = (torch.cuda.memory_allocated()/1e9
                  if device.type=="cuda" else 0.0)

            records.append(dict(model=name, variant=variant,
                                time_ms=t_sec*1e3, mem_gb=mem))

            base_ms = next((r["time_ms"] for r in records
                           if r["model"]==name and r["variant"]=="checkpoint"), 
                           t_sec*1e3)  # Fall back to current time if no checkpoint
            speed = base_ms/(t_sec*1e3)
            tag = "" if variant=="checkpoint" else f"×{speed:.2f}"
            print(f"   {variant:<12} {t_sec*1e3:8.2f} ms {tag}")
        except Exception as e:
            print(f"   {variant:<12} FAILED: {str(e)}")
            # Add a placeholder record
            records.append(dict(model=name, variant=variant, 
                                time_ms=0.0, mem_gb=0.0))
    print()

# ════════════════════════ 8 · Persist results ════════════════════════════
df = pd.DataFrame(records)
csv_p = OUT/"compilation_ckpt_benchmark.csv"
json_p= OUT/"compilation_ckpt_benchmark.json"
df.sort_values(["model","variant"]).to_csv(csv_p, index=False)
df.to_json(json_p, orient="records", indent=2)

# ════════════════════════ 9 · Visualisations ═════════════════════════════
def plot_speed(ax, data):
    data = data[data.time_ms > 0]  # Filter out failed benchmarks
    
    if data.empty:
        ax.text(0.5, 0.5, "No valid benchmark data", 
                ha='center', va='center', transform=ax.transAxes)
        return
        
    base = data[data.variant=="checkpoint"].set_index("model").time_ms
    for var, col in [("compile_def","tab:orange"),
                     ("compile_max","tab:green")]:
        cur_data = data[data.variant==var]
        if cur_data.empty:
            continue
            
        cur = cur_data.set_index("model").time_ms
        common_models = base.index.intersection(cur.index)
        if len(common_models) == 0:
            continue
            
        spd = base.loc[common_models]/cur.loc[common_models]
        ax.barh(spd.index, spd.values, color=col, alpha=.9, label=var)
        for y,v in enumerate(spd.values):
            if v > 0:
                ax.text(v+0.03, y, f"{v:.2f}×", va="center")
    ax.set_xlabel("Speed-up vs checkpoint  ↑ better")
    
    if ax.get_legend_handles_labels()[0]:
        ax.legend()

def plot_mem(ax, data):
    data = data[data.time_ms > 0]  # Filter out failed benchmarks
    
    if data.empty:
        ax.text(0.5, 0.5, "No valid benchmark data", 
                ha='center', va='center', transform=ax.transAxes)
        return
        
    models = data.model.unique()
    if len(models) == 0:
        ax.text(0.5, 0.5, "No valid benchmark data", 
                ha='center', va='center', transform=ax.transAxes)
        return
            
    y = np.arange(len(models))
    h = .25
    for i,(var,col) in enumerate([("checkpoint","tab:blue"),
                                  ("compile_def","tab:orange"),
                                  ("compile_max","tab:green")]):
        var_data = data[data.variant==var]
        if var_data.empty:
            continue
            
        mem = var_data.set_index("model").mem_gb
        indices = [list(models).index(m) for m in mem.index]
        ax.barh([y[i] + i*h - h for i in indices], 
                mem.values, height=h, color=col, alpha=.8, label=var)
    
    ax.set_yticks(y)
    ax.set_yticklabels(models)
    ax.set_xlabel("Peak GPU memory [GiB]  ↓ better")
    
    if ax.get_legend_handles_labels()[0]:
        ax.legend()

plt.figure(figsize=(11, 4+.6*len(df.model.unique())))
plt.suptitle("torch.compile on Hyperion checkpoints – MI250X")
ax1 = plt.subplot(1,2,1); plot_speed(ax1, df)
ax2 = plt.subplot(1,2,2); plot_mem  (ax2, df)
plt.tight_layout(rect=[0,0,1,0.93])
plt.savefig(OUT/"compilation_ckpt_speed_mem.png", dpi=120)

# ═══════════════════════ 10 · Text summary ═══════════════════════════════
summary_lines = ["Compilation benchmark highlights:\n"]
for m in df.model.unique():
    checkpoint_data = df[(df.model==m)&(df.variant=="checkpoint")]
    if checkpoint_data.empty or checkpoint_data.time_ms.item() <= 0:
        summary_lines.append(f"• **{m}** – No valid checkpoint baseline.")
        continue
        
    t0 = checkpoint_data.time_ms.item()
    for var in ("compile_def","compile_max"):
        variant_data = df[(df.model==m)&(df.variant==var)]
        if variant_data.empty or variant_data.time_ms.item() <= 0:
            summary_lines.append(f"• **{m}** – {var}: Failed to benchmark.")
            continue
            
        t = variant_data.time_ms.item()
        summary_lines.append(f"• **{m}** – {var}: **{t0/t:.2f}×** faster.")

(OUT/"compilation_ckpt_analysis.txt").write_text("\n".join(summary_lines))

print("\n".join(summary_lines))
print(f"\n✅  Artefacts saved to  {OUT}\n")
