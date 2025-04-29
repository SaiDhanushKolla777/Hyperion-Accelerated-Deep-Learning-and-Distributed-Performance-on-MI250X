"""
Benchmarking utilities for Project Hyperion.
"""

import torch
import time
import numpy as np
import pandas as pd
import os
from typing import Dict, Any, Callable, List, Tuple, Optional, Union

def benchmark_forward_pass(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    num_iterations: int = 50,
    warmup: int = 10
) -> Dict[str, float]:
    """
    Benchmark a model's forward pass.
    
    Args:
        model: The PyTorch model to benchmark
        inputs: Input tensor
        num_iterations: Number of iterations for benchmarking
        warmup: Number of warmup iterations
        
    Returns:
        Dictionary with benchmark results
    """
    device = next(model.parameters()).device
    
    # Warmup
    for _ in range(warmup):
        _ = model(inputs)
    
    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(num_iterations):
        _ = model(inputs)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    # Calculate metrics
    elapsed_time = end_time - start_time
    avg_time = elapsed_time / num_iterations
    throughput = inputs.size(0) / avg_time
    
    return {
        "total_time": elapsed_time,
        "avg_time": avg_time,
        "throughput": throughput
    }

def benchmark_training_step(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: Callable,
    num_iterations: int = 50,
    warmup: int = 10,
    use_amp: bool = False
) -> Dict[str, float]:
    """
    Benchmark a full training step (forward + backward + optimizer).
    
    Args:
        model: The PyTorch model to benchmark
        inputs: Input tensor
        targets: Target tensor
        optimizer: The optimizer
        criterion: The loss function
        num_iterations: Number of iterations for benchmarking
        warmup: Number of warmup iterations
        use_amp: Whether to use automatic mixed precision
        
    Returns:
        Dictionary with benchmark results
    """
    device = next(model.parameters()).device
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    # Warmup
    for _ in range(warmup):
        optimizer.zero_grad()
        
        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    
    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(num_iterations):
        optimizer.zero_grad()
        
        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    # Calculate metrics
    elapsed_time = end_time - start_time
    avg_time = elapsed_time / num_iterations
    throughput = inputs.size(0) / avg_time
    
    # Calculate memory usage
    torch.cuda.reset_peak_memory_stats()
    
    if use_amp:
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
    else:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
    
    memory_usage = torch.cuda.max_memory_allocated() / 1024**2  # MB
    
    return {
        "total_time": elapsed_time,
        "avg_time": avg_time,
        "throughput": throughput,
        "memory_usage": memory_usage
    }

def compare_precision_formats(
    model_fn: Callable,
    input_shape: Tuple[int, ...],
    batch_sizes: List[int] = [1, 2, 4, 8, 16, 32, 64],
    num_iterations: int = 20,
    warmup: int = 5
) -> pd.DataFrame:
    """
    Compare performance of different precision formats.
    
    Args:
        model_fn: Function that creates the model
        input_shape: Base input shape (without batch dimension)
        batch_sizes: List of batch sizes to test
        num_iterations: Number of iterations for benchmarking
        warmup: Number of warmup iterations
        
    Returns:
        DataFrame with comparison results
    """
    results = []
    
    precisions = [
        ("fp32", torch.float32),
        ("fp16", torch.float16),
        ("bf16", torch.bfloat16)
    ]
    
    for batch_size in batch_sizes:
        for name, dtype in precisions:
            try:
                # Skip if not supported
                if not hasattr(torch, dtype.__str__().split('.')[-1]):
                    continue
                
                # Create model
                model = model_fn()
                
                # Move model to GPU and convert to dtype
                if torch.cuda.is_available():
                    model = model.to('cuda').to(dtype)
                
                # Create input tensor
                full_input_shape = (batch_size,) + input_shape
                inputs = torch.rand(*full_input_shape)
                
                if torch.cuda.is_available():
                    inputs = inputs.to('cuda').to(dtype)
                
                # Benchmark forward pass
                benchmark = benchmark_forward_pass(
                    model=model,
                    inputs=inputs,
                    num_iterations=num_iterations,
                    warmup=warmup
                )
                
                # Add to results
                results.append({
                    "Batch Size": batch_size,
                    "Precision": name,
                    "Avg Time (ms)": benchmark["avg_time"] * 1000,
                    "Throughput (samples/s)": benchmark["throughput"]
                })
                
            except Exception as e:
                print(f"Error with {name} at batch size {batch_size}: {e}")
    
    return pd.DataFrame(results)

def save_benchmark_results(
    results: pd.DataFrame,
    filename: str,
    results_dir: str = "/home/aac/results/benchmarks"
) -> None:
    """
    Save benchmark results to a CSV file.
    
    Args:
        results: DataFrame with benchmark results
        filename: Name of the file to save
        results_dir: Directory to save the results
    """
    os.makedirs(results_dir, exist_ok=True)
    file_path = os.path.join(results_dir, filename)
    results.to_csv(file_path, index=False)
    print(f"Results saved to {file_path}")
