# Hyperion-Accelerated-Deep-Learning-and-Distributed-Performance-on-MI250X

---
## 🚀 Introduction

**Project Hyperion** is my personal journey and showcase in modern deep learning systems engineering. My primary aim with this project is simple: _to demonstrate not just how to train a big AI model, but to show what it means to deeply understand, profile, scale, and optimize every layer - from hardware to distributed software, from raw tensors to state-of-the-art compiler fusion - on real, modern hardware._

This repository is my attempt at a truly **comprehensive, production-grade ML systems project** - with a focus on AMD Instinct MI250X, but written to be insightful for anyone in ML infrastructure, research engineering, or high-performance AI. Every step, every file, every result is reproducible and explained. If you want to see what it takes to move from "vibe coding" to serious, rigorous ML systems engineering, read on!

---

## 🧭 Table of Contents

- [Project Structure](#project-structure)
- [Hardware and Software Requirements](#hardware-and-software-requirements)
- [Quickstart Installation](#quickstart-installation)
- [1. Hardware Exploration](#1-hardware-exploration)
- [2. Data Engineering](#2-data-engineering)
- [3. Baseline Models & Training](#3-baseline-models--training)
- [4. Mixed Precision and Memory Techniques](#4-mixed-precision-and-memory-techniques)
- [5. Distributed Training: DDP, FSDP, LoRA, Llama](#5-distributed-training-ddp-fsdp-lora-llama)
  - [A Deeper Dive: FSDP, DDP, NCCL, RCCL](#a-deeper-dive-fsdp-ddp-nccl-rccl)
- [6. Compiler & Kernel Fusion Optimization](#6-compiler--kernel-fusion-optimization)
- [7. Results, Metrics & Analysis](#7-results-metrics--analysis)
- [8. Scaling and Visualization](#8-scaling-and-visualization)
- [9. Advanced: Configuration, Troubleshooting, Reproducibility](#9-advanced-configuration-troubleshooting-reproducibility)
- [Personal Reflections](#personal-reflections)
- [License & Acknowledgments](#license--acknowledgments)

---


---

## Project Structure

```
project-hyperion/
├── Phase 1/
│   ├── 01_hardware_exploration.ipynb   # Hardware benchmarking & GPU profiling
│   ├── baseline_performance.ipynb      # Baseline model speed/memory benchmarks
│   └── benchmarking.py                 # General benchmarking utilities
│
├── 02_development/
│   ├── dataset_preparation.ipynb       # End-to-end data pipelines (WikiText-2, CIFAR-10)
│   ├── core_framework.ipynb            # Modular DataLoader/models/training
│   ├── mixed_precision.ipynb           # AMP, bfloat16, memory profiling
│   ├── memory_optimization.ipynb       # Activation checkpointing, peak memory stats
│   ├── distributed_utils.py            # All DDP/FSDP/LoRA/LLama/metrics
│   ├── run_distributed.py              # Main distributed launcher script
│   ├── compilation_optimization.py     # torch.compile, kernel fusion, profiling
│   ├── test_nccl.py                    # NCCL/RCCL communication sanity check
│   └── run_language_fsdp.sh            # Example bash launcher for 4-GPU FSDP
│
├── data/
│   ├── raw/                            # Raw datasets: WikiText, CIFAR-10, etc.
│   ├── processed/                      # Tokenized/cleaned datasets, model checkpoints
│   └── distributed/                    # Scaling metrics, distributed model checkpoints
│
├── results/
│   └── benchmarks/                     # All experiment results, CSVs, plots
│       ├── hardware/
│       ├── baseline/
│       ├── scaling/
│       ├── compilation/
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Hardware and Software Requirements

- **Hardware:** AMD Instinct MI250X GPU(s) (ideally 4+), 32GB+ RAM, fast SSD
- **OS:** Ubuntu 20.04/22.04 LTS (ROCm recommended)
- **ROCm:** 5.7+ (full support for `torch`, `torchvision`, `nccl`, `rccl` etc.)
- **Python:** 3.10+
- **PyTorch:** 2.3.0a0+ (ROCm build - check with `torch.version.hip`)
- **Others:** See requirements.txt (HuggingFace, torchvision, datasets, peft, matplotlib, tqdm etc.)

---

## Quickstart Installation

```
git clone [https://github.com/yourusername/project-hyperion.git](https://github.com/SaiDhanushKolla777/Hyperion-Accelerated-Deep-Learning-and-Distributed-Performance-on-MI250X)
cd project-hyperion

conda create -n hyperion python=3.10
conda activate hyperion

pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/rocm5.7
pip install -r requirements.txt
```

---

## 1. Hardware Exploration

**Goal:** See how close you can actually get to the theoretical peak of MI250X, and not just trust vendor documentation.

- **Check versions:** ROCm, PyTorch, Python, and enumerate all available GPUs.
- **Memory benchmarks:** Matrix multiplication and memory bandwidth for FP32, FP16, BF16.
- **Result:** On MI250X, you will see **FP16/BF16 matmul at ~128 TFLOPS, memory bandwidth up to ~1270 GB/s** for large enough matrices.
- **Script/notebook:** `Phase 1/01_hardware_exploration.ipynb`

**Why it matters:** Knowing your hardware’s *actual* measured performance is the foundation for real systems tuning. "If you don't measure, you don't know!"

---

## 2. Data Engineering

**Goal:** Build *robust, repeatable, and verified* data pipelines for both language and vision tasks.

- **WikiText-2:** Download (HuggingFace), filter empty lines, tokenize (GPT-2, with pad_token as eos_token), pad to sequence length, save as arrow files.
- **CIFAR-10:** Download, normalize, filter corrupted/empty images, convert for PyTorch usage, save as tensors.
- **Location:** `02_development/dataset_preparation.ipynb`
- **Verification:** Every step validates the data via print/debug/visualization.

**Personal note:** Too many ML projects fumble here - I wanted my data pipeline to be bulletproof, not a black box.

---

## 3. Baseline Models & Training

**Goal:** Implement clean, performant training loops for language and vision, then rigorously *profile* them.

- **Models:** Minimalist Transformer LM (nice for token-level language modeling), ResNet-18 (for CIFAR-10).
- **Loaders:** Modular PyTorch DataLoader wrappers for maximal flexibility.
- **Training:** Standard best-practice training loops, checkpoints, reproducible seeds.
- **Metrics:** Per-epoch average loss, accuracy, throughput, and memory.
- **Location:** `02_development/core_framework.ipynb`
- **Results:** Logs, checkpoints in `data/processed/`.

---

## 4. Mixed Precision and Memory Techniques

**Goal:** Squeeze more performance out of MI250X GPUs - not just with brute force, but with smarter memory usage.

- **Mixed Precision (AMP):** Enable bf16/FP16 via torch.cuda.amp.autocast and GradScaler, with full memory and loss profiling. Reduces memory by 40%+ with minimal accuracy change.
- **Activation Checkpointing:** Use `torch.utils.checkpoint` on both transformer stacks and ResNet blocks; dramatically reduce peak memory with small run-time penalty.
- **Profiling:** Print and log peak memory before/after forward and backward.
- **Location:** `02_development/mixed_precision.ipynb`, `memory_optimization.ipynb`

**Personal viewpoint:** These optimizations are undervalued - you don’t need to throw more hardware at the problem if you know how to unlock more from what you have.

---

## 5. Distributed Training: DDP, FSDP, LoRA, Llama

**Goal:** Move beyond single-GPU training. Efficiently scale deep learning models across multiple AMD MI250X GPUs, while keeping code flexible for NVIDIA as well. This section covers how I implemented true multi-GPU support in this project, including robust, real-world usage of DDP, FSDP, LoRA, Llama, and proper NCCL/RCCL communication setup.

---

### DDP (DistributedDataParallel)

**How I used it:**

- **SPMD Parallelism:** Each GPU (process) runs the same model and data split. After each backward pass, gradients are *automatically synchronized* across GPUs.
- **Implementation:**  
  - The core logic is in `distributed_utils.py` with modular initialization for DDP, barrier synchronization, and cleanup.
  - Every run via DDP writes per-rank metrics to CSV (loss, accuracy, epoch time), so I could analyze scaling *per GPU*.
  - Example command used:
    ```
    torchrun --standalone --nproc_per_node=4 run_distributed.py --model language_ddp --epochs 25
    ```
- **Data Loading:** Used `DistributedSampler` to ensure that each GPU only processes its unique image/text batch, preventing overlap and ensuring correctness.
- **Post-run Analysis:** Aggregated all-rank results for speedup/efficiency plots in results.

---

### FSDP (Fully Sharded Data Parallel)

**How I used it:**

- **What It Does:**  
  FSDP shards (splits) *model parameters*, gradients, and optimizer states across all GPUs. You can train models much larger than would fit on a single card, with each device only storing a piece of the whole model at any time.
- **Why I Chose FSDP:**  
  - For LLM-scale models, DDP runs out of memory.  
  - FSDP enables both memory and communication efficiency, and is natively supported in PyTorch 2.x for ROCm.
- **Features Implemented:**
  - **Mixed Precision:** BF16 everywhere for stability and speed; config in `run_distributed.py` lets you toggle this.
  - **Full State-Dict Checkpointing:** My implementation saves FSDP checkpoints in a way that can be safely resumed and inspected, avoiding silent corruption or partial saves.
  - **Distributed Logging:** As with DDP, FSDP runs log per-rank metrics and aggregate to CSV for comprehensive scaling analysis.
  - **Flexibility:** The code allows switching models between DDP and FSDP with just a command-line flag.
- **Example Command:**
  ```
  torchrun --standalone --nproc_per_node=4 run_distributed.py --model language_fsdp --epochs 25
  ```
- **Scaling:** Tested up to 4 GPUs for language and vision models and Llama-2.

---

### LoRA and Llama Fine-Tuning

**How I used it:**

- **LoRA (Low-Rank Adapters):**  
  - Integrated through Hugging Face's `peft` library and transformers, allowing the project to fine-tune extremely large Llama-2 models by only adapting small adapter weights (instead of all parameters).
  - This drastically reduces GPU memory needs and makes LLM fine-tuning feasible on MI250X without proprietary infra.
- **FSDP + Llama:**  
  - My scripts (`run_distributed.py` and associated configs) setup FSDP wrapping of Hugging Face’s Llama layers transparently.
  - Checkpoint logic is robust to the complex module wrapping required by FSDP + LoRA + HuggingFace.
- **Logging:**  
  - All runs log memory stats, throughput, and accuracy for later comparison.
- **Example Command:**
  ```
  torchrun --standalone --nproc_per_node=4 run_distributed.py --model llama --lora --epochs 3
  ```

---

### NCCL and RCCL: Communication Backbone

**How I used them:**

- **NCCL (NVIDIA):**  
  The standard backend for PyTorch-distributed on CUDA.
- **RCCL (AMD):**  
  AMD’s drop-in compatible library, fully supported for MI250X and ROCm 5.7+.
- **How I Managed Both:**  
  - My distributed code (`distributed_utils.py`) defers to the available backend automatically.
  - Set key environment variables in the script (`run_language_fsdp.sh`) to avoid known MI250X issues:
    ```
    export NCCL_COLLNET_ENABLE=0
    export RCCL_P2P_ENABLE=1
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=0
    export TORCH_NCCL_TIMEOUT=7200
    export RCCL_TIMEOUT=7200
    ```
  - Ensures high-throughput collectives and avoids hangs on failed connections or xGMI path.
- **Sanity Check:**  
  - Provided a `test_nccl.py` script so you can verify NCCL/RCCL communication is working before running big jobs-a must for reliable scaling.
- **Portability:**  
  - All distributed scripts and logic are written so that you can switch between NVIDIA and AMD GPUs without code changes-just select the correct PyTorch and backend.

---

### Why This Matters: Real-World Robustness

In my experience, distributed training **fails most often** due to subtle but critical mistakes:
- Silent parameter desync
- Bad communication environments
- Checkpoint corruption or loss of model state
- Inconsistent scaling between debug/small/large runs

By fully integrating DDP, FSDP, and LoRA with rigorous NCCL/RCCL logging, checkpointing, and validation, I ensured that **every training run is robust, scalable, and debuggable** from your laptop to a cluster. This is the foundation real ML infrastructure is built upon.

---

---

## 6. Compiler & Kernel Fusion Optimization

**Goal:** Leverage the latest PyTorch 2.x compiler stack (`torch.compile`) and underlying Triton kernel fusion for real speed and memory savings.

- **Method:** Benchmark all model checkpoints in both eager (checkpointed), compile with default fusion, and max-autotune modes.
- **Result:** 
  - **ResNet-18:** Up to **1.68×** speedup with Inductor kernel fusion, huge peak memory reduction (from 0.47 GB to 0.05 GB!)
  - **Transformer LM:** About **1.07×** speedup with fusion.
- **All artefacts:** CSV, JSON, PNG plots in `results/benchmarks/compilation/`.
- **Quick script:** `python 02_development/compilation_optimization.py --base_dir . --dtype bf16 --repeat 10`

**Takeaway:** The *promise* of ML compilers is real, but it’s not "plug and play" - careful profiling is needed to realize the gains.

---

## 7. Results, Metrics & Analysis

### 7.1 Hardware Benchmarks

#### Precision Performance

| Matrix Size | Precision | Time (s)    | TFLOPS   |
|-------------|-----------|-------------|----------|
| 1024×1024   | FP32      | 0.000172    | 12.46    |
| 1024×1024   | FP16      | 0.000139    | 15.40    |
| 1024×1024   | BF16      | 0.000127    | 16.93    |
| 2048×2048   | FP32      | 0.000578    | 29.74    |
| 2048×2048   | FP16      | 0.000240    | 71.56    |
| 2048×2048   | BF16      | 0.000266    | 64.51    |
| 4096×4096   | FP32      | 0.004079    | 33.69    |
| 4096×4096   | FP16      | 0.001214    | 113.21   |
| 4096×4096   | BF16      | 0.001225    | 112.22   |
| 8192×8192   | FP32      | 0.030176    | 36.44    |
| 8192×8192   | FP16      | 0.008575    | 128.22   |
| 8192×8192   | BF16      | 0.009082    | 121.07   |

#### Memory Bandwidth

| Size (Millions of elements) | Bandwidth (GB/s) |
|-----------------------------:|-----------------:|
|                           10 |           796.39 |
|                           20 |           991.76 |
|                           50 |          1248.92 |
|                          100 |          1266.20 |
|                          200 |          1261.13 |
|                          500 |          1268.89 |

---

### 7.2 Baseline Model Benchmarks

| Model                     | Forward (ms) | Backward (ms) | Total (ms) | Memory (MB) | Throughput (samples/s) |
|---------------------------|-------------:|--------------:|-----------:|------------:|-----------------------:|
| create_resnet50           |        18.66 |         34.83 |      56.32 |      3230.98 |                568.22 |
| create_vit_model          |         1.78 |          3.53 |       5.44 |       514.87 |               5883.44 |
| create_custom_transformer |         3.87 |          8.18 |      12.52 |       617.17 |               2555.90 |

---

### 7.3 Distributed Scaling

- **DDP** (Data‐Parallel):  
  Near‐linear speedup up to 4 GPUs on both the transformer LM and ResNet-18 benchmarks.  
- **FSDP** (Fully‐Sharded DataParallel):  
  Shards parameters and grads across devices, slashes per‐GPU memory, and makes very large models trainable.  
- **LoRA** (Low-Rank Adapters):  
  Efficient fine-tuning on Llama-2 with minimal extra parameters, easily scaled across GPUs.  

_All detailed CSV logs and speed/efficiency plots are in_  
`data/distributed/scaling_analysis.csv` / `scaling_analysis.png`.

---

### 7.4 Compilation & Fusion Benchmarks

| Model                   | Variant      | Time (ms) | Speedup | Peak Mem (GB) |
|-------------------------|-------------:|----------:|--------:|--------------:|
| resnet18_cifar10        | checkpoint   |      2.55 |   1.00× |          0.47 |
| resnet18_cifar10        | compile_def  |      1.51 |   1.68× |          0.05 |
| resnet18_cifar10        | compile_max  |      2.40 |   1.06× |          0.05 |
| simple_transformer_lm   | checkpoint   |      5.99 |   1.00× |          0.42 |
| simple_transformer_lm   | compile_def  |      5.60 |   1.07× |          0.43 |
| simple_transformer_lm   | compile_max  |      5.56 |   1.08× |          0.43 |

![image](https://github.com/user-attachments/assets/5b2e4045-e944-4e0f-b7f3-c342160b937d)

---

## 8. Scaling and Visualization

**Scaling analysis and efficiency plots** are all automatically generated from CSV logs in `data/distributed/` using the helper in `distributed_utils.py`. You’ll find:
- **Plots for speedup/efficiency** vs. GPU count (for DDP, FSDP, and Llama).
- **CSV summaries** for recordkeeping and further analysis.

---

## 9. Advanced: Configuration, Troubleshooting, Reproducibility

### Environment knobs for best MI250X performance

```
export NCCL_COLLNET_ENABLE=0
export RCCL_P2P_ENABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=0
export TORCH_NCCL_TIMEOUT=7200
export RCCL_TIMEOUT=7200
```

### Common Issues & Solutions

- **ROCm not detected:** Double check kernel+ROCm version, set HIP_VISIBLE_DEVICES. Use `torch.version.hip` to verify ROCm build of PyTorch.
- **OOM:** Lower batch size, enable AMP and activation checkpointing.
- **Distributed bugs:** Use `test_nccl.py` for communications; check firewalls, env vars.
- **Compilation issues:** Try with and without different `torch.compile` modes; ensure Triton is available.
- **Dataset problems:** Validate via dataset_preparation and core_framework notebooks.

**Reproducibility:** Everything is checkpointed and serialized, with hashes and file size checks.

---

## Personal Reflections

Building Project Hyperion made me appreciate systems ML at a whole new level. Anyone can train a model, but to *really* understand, you need to work across all of:
- Hardware quirks
- Real (not idealized) memory and bandwidth bottlenecks
- Compiler interactions
- Distributed barriers and weird collective bugs
- The pain and satisfaction of getting FSDP checkpoints to work on every run!

I believe my project stands out for its **breadth**, **depth**, and **practical rigor**. It’s not enough to "just run", you must be able to explain, measure, and debug *every* stage. Hyperion is my answer to the challenge: "What does it look like when a single engineer owns the full ML system stack and leaves nothing as a black box?"

---

## License & Acknowledgments

**MIT License** - see LICENSE file.

**Thanks to:**
- AMD for MI250X hardware and ROCm
- PyTorch and Hugging Face teams for open-source excellence
- ML systems community for pushing the state-of-the-art

---

### If you have read this far: Thank you! Open an issue or pull request if you spot a bug, want new features, or just want to chat about ML systems engineering.

*- [Your Name]*
```
This comprehensive README includes **every detail of the project**, full technical explanations, personal perspective, and in-depth coverage of main distributed and systems concepts such as FSDP and NCCL/RCCL. It is designed to impress ML infrastructure teams and clearly demonstrate your expertise.
```

Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/31094950/aca333f8-d218-4f6d-bf66-3866f7e861c8/paste-1.txt
[2] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/31094950/1f6e33a9-7c93-4cda-a802-6adbea25da9e/paste-2.txt

---
Answer from Perplexity: pplx.ai/share
