# test_nccl.py
import os
import sys
import torch
import torch.distributed as dist
import datetime

def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    # Basic information
    if local_rank == 0:
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA/ROCm available: {torch.cuda.is_available()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Testing NCCL with {world_size} processes")
    
    # Print local rank information
    device_name = torch.cuda.get_device_name(local_rank)
    print(f"Process {local_rank}: Using device {device_name}")
    
    # Initialize NCCL with timeout
    try:
        print(f"Process {local_rank}: Initializing process group...")
        dist.init_process_group(
            "nccl", 
            timeout=datetime.timedelta(seconds=30),
        )
        print(f"Process {local_rank}: Process group initialized successfully!")
        
        # Test simple allreduce to verify communication
        tensor = torch.ones(1).to(local_rank) * local_rank
        print(f"Process {local_rank}: Before allreduce: {tensor}")
        dist.all_reduce(tensor)
        print(f"Process {local_rank}: After allreduce: {tensor}")
        
        # Clean up
        dist.destroy_process_group()
        print(f"Process {local_rank}: Destroyed process group")
        
    except Exception as e:
        print(f"Process {local_rank}: Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
