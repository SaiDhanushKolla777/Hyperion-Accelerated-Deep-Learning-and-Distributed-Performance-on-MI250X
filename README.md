# Project-Hyperion-Scaling-Distributed-Deep-Learning-Through-Communication-Optimized-GPU-Frameworks


# Project Hyperion

*A Comprehensive ML Systems Benchmark and Optimization Suite for AMD Instinct MI250X*

---

## Introduction

Project Hyperion is a deep dive into machine learning systems engineering, designed to **benchmark, optimize, and scale deep learning models on AMD Instinct MI250X GPUs**. From precise hardware exploration to large-scale distributed training (DDP, FSDP, LoRA), it demonstrates best practices and hands-on mastery of systems-level ML. This repository is a meticulously documented, reproducible pipeline that covers everything from data cleaning to compiler-level kernel fusion, and is ready for both academic benchmarking and industry deployment.

---

## Table of Contents

- [Project Motivation](#project-motivation)
- [Key Features](#key-features)
- [Directory Structure](#directory-structure)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
- [1. Hardware Benchmarking](#1-hardware-benchmarking)
- [2. Data Preparation](#2-data-preparation)
- [3. Model Frameworks & Training](#3-model-frameworks--training)
- [4. Mixed Precision & Memory Optimization](#4-mixed-precision--memory-optimization)
- [5. Distributed Training & Scaling](#5-distributed-training--scaling)
- [6. Compiler & Kernel Fusion](#6-compiler--kernel-fusion)
- [7. Results & Metrics](#7-results--metrics)
- [8. Advanced Configuration](#8-advanced-configuration)
- [9. Troubleshooting & FAQ](#9-troubleshooting--faq)
- [10. License and Acknowledgements](#10-license-and-acknowledgements)
- [11. Contact and Contributions](#11-contact-and-contributions)

---

## Project Motivation

- **ML Systems:** Provide a full-stack demonstration of ML systems engineering – from raw hardware to distributed optimization and compiler acceleration.
- **Open-Source Reproducibility:** All code, datasets, and results are open and reproducible, ensuring full transparency for researchers and engineers.
- **Industry-Grade Readiness:** The codebase adheres to best practices in modularity, metrics logging, and scaling, making it production-ready.

---

## Key Features

- **Hardware Exploration:** In-depth analysis of MI250X memory, TFLOPS, bandwidth, and precision performance.
- **Data Engineering:** Robust pipelines for WikiText-2 and CIFAR-10 – download, clean, tokenize, filter, visualize, and save.
- **Model Development:** Ready-to-train language and vision models (Transformer LM, ResNet-18) with modular PyTorch code.
- **Performance Optimization:** Mixed-precision (AMP, BF16/FP16), activation checkpointing, gradient accumulation, and memory profiling.
- **Distributed Training:** Turnkey support for DDP, FSDP, and LoRA; multi-GPU and multi-node scaling with detailed metrics.
- **Compiler Fusion:** End-to-end benchmarks with `torch.compile` (Inductor kernel fusion), including speedup and memory metrics.
- **Scaling Visualization:** Automated plots and CSVs for speedup, scaling efficiency, and memory reduction.

---

## Directory Structure

```
project-hyperion/
│
├── Phase 1/
│   ├── 01_hardware_exploration.ipynb    # GPU/precision/memory benchmarks
│   ├── baseline_performance.ipynb       # Model and batch size benchmarks
│   └── benchmarking.py                  # Benchmarking utilities
│
├── 02_development/
│   ├── dataset_preparation.ipynb        # WikiText-2/CIFAR-10 ETL pipelines
│   ├── core_framework.ipynb             # Data loaders, model definition/training
│   ├── mixed_precision.ipynb            # Mixed/bfloat16, loss scaling
│   ├── memory_optimization.ipynb        # Activation checkpointing, profile
│   ├── compilation_optimization.py      # torch.compile fusion runner
│   ├── distributed_utils.py             # DDP, FSDP, LoRA, metrics, scaling analysis
│   ├── run_distributed.py               # CLI launcher for training at scale
│   ├── run_language_fsdp.sh             # Example bash launcher with optimal RCCL/NCCL
│   ├── test_nccl.py                     # GPU communication/NCCL sanity test
│
├── data/
│   ├── raw/                             # Untouched datasets
│   ├── processed/                       # Tokenized/filtered/tensorized datasets, model checkpoints
│   └── distributed/                     # DDP/FSDP/LoRA metrics, checkpoints
│
├── results/
│   └── benchmarks/                      # Hardware, baseline, scaling, and compilation results
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Prerequisites

- **Hardware:** AMD Instinct MI250X GPU(s), 32GB+ RAM, fast SSD
- **OS:** Ubuntu 20.04/22.04 LTS or compatible Linux
- **ROCm stack:** 5.7+ (MI250X support, see [official docs](https://docs.amd.com/))
- **Python:** 3.10+
- **PyTorch:** 2.3.0a0+ (ROCm build)
- **Other Python dependencies:** See `requirements.txt` (`torchvision`, `transformers`, `datasets`, `matplotlib`, `pandas`, `peft`, etc.)

---

## Installation & Setup

```
git clone https://github.com/yourusername/project-hyperion.git
cd project-hyperion

conda create -n hyperion python=3.10
conda activate hyperion

pip install torch==2.3.0 torchvision --extra-index-url https://download.pytorch.org/whl/rocm5.7
pip install -r requirements.txt
```

---

## 1. Hardware Benchmarking

- **Explore GPU capabilities:**
  - Validate PyTorch + ROCm installation, check GPU(s), memory availability.
  - Benchmark tensor/matrix operations in FP32, FP16, BF16.
  - Max out MI250X: observe **~128 TFLOPS** (FP16/BF16, 8192x8192 matrix).
  - Memory bandwidth: sustained **~1270 GB/s**.
- **Usage:** Run `Phase 1/01_hardware_exploration.ipynb`
- **Results:** Saved as CSV and PNG under `results/benchmarks/hardware/`.

---

## 2. Data Preparation

- **WikiText-2:**
  - Download from HuggingFace, filter empty lines, tokenize with GPT-2, pad/truncate, save all splits with attention masks.
- **CIFAR-10:**
  - Download, normalize, filter corrupted/empty images, save as PyTorch tensor lists.
- **Verification:** Includes sample visualization.
- **Usage:** `02_development/dataset_preparation.ipynb`

---

## 3. Model Frameworks & Training

- **Language Model:** Minimalist Transformer LM (configurable embed dim, heads, layers, max_len).
- **Vision Model:** ResNet-18 for CIFAR-10.
- **Training:** Modular scripts, standard PyTorch training loops, logging average loss/accuracy per epoch.
- **Checkpointing:** Models saved after training for further use.
- **Usage:** `core_framework.ipynb`

---

## 4. Mixed Precision & Memory Optimization

- **Mixed Precision:** AMP (Autocast + GradScaler) for language and vision models; supports BF16/FP16.
- **Results:** Demonstrated 40%+ memory reduction, identical/faster convergence.
- **Activation Checkpointing:** Supported for both transformer and ResNet; saves memory on forward, transparent for training.
- **Profiling:** Usage of `torch.cuda.memory_allocated()` and `torch.cuda.memory_reserved()` before/after forward/backward.
- **Usage:** `mixed_precision.ipynb`, `memory_optimization.ipynb`

---

## 5. Distributed Training & Scaling

- **DDP (DistributedDataParallel):** WikiText-2 and CIFAR-10 models; each worker logs per-epoch metrics, aggregates loss/accuracy across GPUs.
- **FSDP (FullyShardedDataParallel):** Enables >1B parameter models; supports sharding of transformer LMs.
- **LoRA + Llama-2 (HuggingFace):** Efficient fine-tuning with LoRA adapters or full FSDP sharding.
- **Metrics Logging:** Each run writes a CSV (loss/accuracy/duration/GPU count) for exact scaling analysis (found in `data/distributed/`).
- **CLI Launch:** `torchrun --standalone --nproc_per_node=4 run_distributed.py --model language_fsdp --epochs 25`
- **Scaling Report:** Automated plot/CSV of speedup and efficiency vs. GPU count (`scaling_analysis.png`).
- **Environment Guidance:** Example shell launcher (`run_language_fsdp.sh`) with optimal NCCL/RCCL variables for MI250X.

---

## 6. Compiler & Kernel Fusion

- **torch.compile:** Benchmarks with and without kernel fusion/compilation for both transformer and vision models.
- **Speedup:** Up to **1.68×** for ResNet-18; **1.07×** for transformer LM (see `compilation_ckpt_analysis.txt`).
- **Memory Reduction:** Major savings for models using channels_last layout.
- **Usage:** `compilation_optimization.py`
- **Results:** CSV, JSON, PNG plots in `results/benchmarks/compilation/`

---

## 7. Results & Metrics

### Hardware Benchmarks

| Matrix Size | Precision | Time (s)   | TFLOPS  |
|-------------|-----------|------------|---------|
| 8192x8192   | FP32      | 0.030      | 36.44   |
| 8192x8192   | FP16      | 0.0086     | 128.2   |
| 8192x8192   | BF16      | 0.0091     | 121.1   |

### Model Benchmarks

| Model               | Fwd (ms) | Bwd (ms) | Memory (MB) | Throughput (samples/s) |
|---------------------|----------|----------|-------------|------------------------|
| create_resnet50     | 18.66    | 34.83    | 3231        | 568.2                  |
| custom_transformer  | 3.87     | 8.18     | 617         | 2555.9                 |

### Distributed Scaling

- **Near-linear DDP/FSDP scaling** for both language and vision – up to 4 GPUs.
- **Automated plots:** Speedup vs. GPU count and efficiency vs. ideal.

### Compilation Speedup

| Model            | Variant      | Time (ms) | Speedup  | Peak Mem (GB) |
|------------------|-------------|-----------|----------|---------------|
| resnet18_cifar10 | checkpoint  | 2.55      | 1.0×     | 0.47          |
| resnet18_cifar10 | compile_def | 1.51      | 1.68×    | 0.05          |
| simple_transformer_lm | compile_def | 5.60 | 1.07×    | 0.43          |

---

## 8. Advanced Configuration

For optimal multi-GPU MI250X performance, use these env vars:
```
export NCCL_COLLNET_ENABLE=0
export RCCL_P2P_ENABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=0
export TORCH_NCCL_TIMEOUT=7200
export RCCL_TIMEOUT=7200
```

Run `run_language_fsdp.sh` or `torchrun` as shown above.

---

## 9. Troubleshooting & FAQ

- **ROCm not detected:** Check driver install, ensure HIP_VISIBLE_DEVICES, update conda/pip package paths.
- **OOM Errors:** Lower batch size, enable activation checkpointing, ensure AMP enabled.
- **Distributed communication issues:** Verify NCCL/RCCL settings, run `test_nccl.py` as a sanity check.
- **Compilation failures:** Check PyTorch and Triton versions, try disabling kernel fusion.
- **Dataset errors:** Check data/processed directory, rerun dataset preparation if files are missing.

---

## 10. License and Acknowledgements

**License:** MIT License (see [LICENSE](LICENSE))

**Thanks to:**
- AMD for MI250X and ROCm ecosystem
- PyTorch core & distributed teams
- Hugging Face for datasets, models, and peft (LoRA)
- The open-source ML and systems community

---

## 11. Contact and Contributions

- **Issues & Pull Requests:** Please submit on GitHub.
- **Custom benchmarks/expansion ideas:** Welcome! This project is modular-submit your own datasets, models, or distributed experiments.
- **Email:** [Your address or link]

---

**Project Hyperion** is built to demonstrate not just ML model performance, but the craft, reproducibility, and depth required of true ML systems engineering. It is a showcase of your ability to build, measure, and optimize *all* layers of a modern, hardware-aware AI pipeline.

