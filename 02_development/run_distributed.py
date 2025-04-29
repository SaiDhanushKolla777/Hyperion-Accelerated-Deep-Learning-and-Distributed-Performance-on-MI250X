#!/usr/bin/env python3
# run_distributed.py
#
# Helper-launcher for the training routines defined in distributed_utils.py
# ─────────────────────────────────────────────────────────────────────────────
# • Start with:  torchrun --standalone --nproc_per_node=N run_distributed.py …
# • torchrun already sets $RANK, $LOCAL_RANK, $WORLD_SIZE  – do **not** spawn
#   the workers yourself.
# ─────────────────────────────────────────────────────────────────────────────
import os, sys, argparse
import torch
from huggingface_hub import login

from distributed_utils import (
    train_language_model_ddp,
    train_cifar_model_ddp,
    train_language_model_fsdp,
    train_llama_fsdp,
    create_scaling_report,
    run_scaling_experiment,
)

# ╭─────────────────────────────────────────────────────────────────────────╮
# │ 0. (optional) – Hugging Face authentication                            │
# ╰─────────────────────────────────────────────────────────────────────────╯
HF_TOKEN = ""        # <-- put yours here
if HF_TOKEN:
    try:
        login(token=HF_TOKEN,
              add_to_git_credential=False,
              write_permission=False)
    except Exception as e:
        print("HF login failed – continuing without auth:", e)

# ╭─────────────────────────────────────────────────────────────────────────╮
# │ 1. CLI arguments                                                        │
# ╰─────────────────────────────────────────────────────────────────────────╯
parser = argparse.ArgumentParser(description="Project-Hyperion launcher")

parser.add_argument("--model",
                    choices=["language_ddp", "cifar",
                             "language_fsdp", "llama",
                             "all", "scaling"],
                    default="language_ddp")
parser.add_argument("--epochs",        type=int, default=5)
parser.add_argument("--base_dir",      type=str,
                    default="/home/aac/project-hyperion")

parser.add_argument("--hf_token",      type=str, default=None,
                    help="HF token (overrides the constant above)")

# ── Llama-specific knobs ──────────────────────────────────────────────────
parser.add_argument("--model_id",      type=str,
                    default="NousResearch/Llama-2-7b-hf",
                    help="HF repo to download when --model=llama")
parser.add_argument("--lora",          action="store_true",
                    help="Use LoRA adapters instead of FSDP sharding")
parser.add_argument("--batch_size",    type=int, default=1,
                    help="Per-GPU batch size for Llama")
parser.add_argument("--progress_every", type=int, default=50,
                    help="TQDM refresh interval (steps) for Llama")

# ── scaling utility ───────────────────────────────────────────────────────
parser.add_argument("--scaling_gpus",  type=str, default="1,2,4,8",
                    help="Comma-separated GPU counts for the scaling test")

args = parser.parse_args()
args.hf_token = args.hf_token or HF_TOKEN

# ╭─────────────────────────────────────────────────────────────────────────╮
# │ 2. discover rank / world-size provided by torchrun                      │
# ╰─────────────────────────────────────────────────────────────────────────╯
try:
    RANK       = int(os.environ["RANK"])
    WORLD_SIZE = int(os.environ["WORLD_SIZE"])
except KeyError:
    sys.exit("❌  Please launch me with torchrun – RANK/WORLD_SIZE not found.")

LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
if LOCAL_RANK == 0:
    print(f"torch {torch.__version__} | rank {RANK}/{WORLD_SIZE} | "
          f"GPU {LOCAL_RANK} / node has {torch.cuda.device_count()} devices")

# ╭─────────────────────────────────────────────────────────────────────────╮
# │ 3. helper – run ONE job on THIS rank                                    │
# ╰─────────────────────────────────────────────────────────────────────────╯
def run_selected(which: str) -> None:
    if   which == "language_ddp":
        train_language_model_ddp(RANK, WORLD_SIZE,
                                 epochs=args.epochs,
                                 base_dir=args.base_dir)

    elif which == "cifar":
        train_cifar_model_ddp   (RANK, WORLD_SIZE,
                                 epochs=args.epochs,
                                 base_dir=args.base_dir)

    elif which == "language_fsdp":
        train_language_model_fsdp(RANK, WORLD_SIZE,
                                  epochs=args.epochs,
                                  base_dir=args.base_dir)

    elif which == "llama":
        train_llama_fsdp        (RANK, WORLD_SIZE,
                                  epochs         = args.epochs,
                                  base_dir       = args.base_dir,
                                  hf_token       = args.hf_token,
                                  model_id       = args.model_id,
                                  lora           = args.lora,
                                  batch_size     = args.batch_size,
                                  progress_every = args.progress_every)
    else:
        raise ValueError(f"unknown model: {which}")

# ╭─────────────────────────────────────────────────────────────────────────╮
# │ 4. main dispatch                                                        │
# ╰─────────────────────────────────────────────────────────────────────────╯
if args.model == "scaling":
    # Only rank-0 orchestrates the outer loop; others exit immediately.
    if RANK == 0:
        gpu_list = [int(x) for x in args.scaling_gpus.split(",")]

        for mdl in ["language_ddp", "cifar", "language_fsdp", "llama"]:
            run_scaling_experiment(
                mdl, gpu_list,
                epochs     = args.epochs,
                base_dir   = args.base_dir,
                hf_token   = args.hf_token,
            )

        create_scaling_report(os.path.join(args.base_dir, "data/distributed"))
    sys.exit(0)

elif args.model == "all":
    # Every rank runs the four jobs sequentially.
    for mdl in ["language_ddp", "cifar", "language_fsdp", "llama"]:
        if LOCAL_RANK == 0:
            print(f"\n=== [{mdl}] starting ===\n")
        run_selected(mdl)

    if RANK == 0:
        create_scaling_report(os.path.join(args.base_dir, "data/distributed"))

else:
    # one specific model
    run_selected(args.model)

    if RANK == 0:
        create_scaling_report(os.path.join(args.base_dir, "data/distributed"))
