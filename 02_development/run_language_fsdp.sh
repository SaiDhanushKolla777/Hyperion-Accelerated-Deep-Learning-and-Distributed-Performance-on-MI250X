#!/usr/bin/env bash
# ── run_language_fsdp.sh ───────────────────────────────────────────────
set -euo pipefail

################################################################################
# 1. Robust yet FAST NCCL/RCCL settings for MI250 X                            #
################################################################################
export NCCL_COLLNET_ENABLE=0          # sidestep reverse-path bug
export RCCL_P2P_ENABLE=1              # keep xGMI fast path
export TORCH_NCCL_ASYNC_ERROR_HANDLING=0    # prevent watchdog autokill
export TORCH_NCCL_TIMEOUT=7200              # 2-hour watchdog (instead of 300 s)
export RCCL_TIMEOUT=7200              # Set for AMD GPUs as well

# (If you ever really need blocking mode, use the new name:
#  export TORCH_NCCL_BLOCKING_WAIT=1)

################################################################################
# 2. Launch the 4-GPU FSDP language-model training                             #
################################################################################
torchrun --standalone --nproc_per_node=4 \
         run_distributed.py \
         --model language_fsdp \
         --epochs 25
