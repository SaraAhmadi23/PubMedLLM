"""
Helper utilities for the PubMedLLM project.
"""

import os
import json
import logging
import torch
from typing import Dict, List, Any, Optional, Union

# Set up logging
def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Configure logging with the specified log level.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    return logging.getLogger()

def check_hardware_compatibility() -> Dict[str, bool]:
    """
    Check hardware compatibility for model training and inference.
    
    Returns:
        Dictionary with hardware compatibility flags
    """
    results = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "bfloat16_supported": torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
        "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
    }
    
    if results["cuda_available"] and results["device_count"] > 0:
        results["device_name"] = torch.cuda.get_device_name(0)
        results["device_capability"] = torch.cuda.get_device_capability(0)
        results["memory_allocated"] = torch.cuda.memory_allocated(0) / (1024 ** 3)  # GB
        results["memory_reserved"] = torch.cuda.memory_reserved(0) / (1024 ** 3)  # GB
        results["max_memory"] = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # GB
    
    return results

def save_json(data: Any, filepath: str, indent: int = 2) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        filepath: Path to save the file
        indent: JSON indentation level
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)

def load_json(filepath: str) -> Any:
    """
    Load data from a JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        Loaded data
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_available_devices() -> Dict[str, Any]:
    """
    Get information about available computing devices.
    
    Returns:
        Dictionary with device information
    """
    devices = {"cpu": True}
    
    # Check for CUDA devices
    if torch.cuda.is_available():
        devices["cuda"] = True
        devices["cuda_devices"] = []
        for i in range(torch.cuda.device_count()):
            device_props = torch.cuda.get_device_properties(i)
            devices["cuda_devices"].append({
                "index": i,
                "name": device_props.name,
                "total_memory_gb": round(device_props.total_memory / (1024**3), 2),
                "compute_capability": f"{device_props.major}.{device_props.minor}"
            })
    else:
        devices["cuda"] = False
    
    # Check for MPS (Apple Silicon)
    devices["mps"] = torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False
    
    return devices

def estimate_memory_requirements(
    model_params: int,
    batch_size: int,
    seq_length: int,
    bits: int = 16
) -> Dict[str, float]:
    """
    Estimate memory requirements for model training.
    
    Args:
        model_params: Number of model parameters
        batch_size: Training batch size
        seq_length: Maximum sequence length
        bits: Bits per parameter (16 for fp16/bf16, 32 for fp32, 8 for int8, 4 for int4)
        
    Returns:
        Dictionary with memory estimates in GB
    """
    bytes_per_param = bits / 8
    
    # Model memory
    model_memory_gb = model_params * bytes_per_param / (1024**3)
    
    # Optimizer states (Adam uses 8 bytes per parameter)
    optimizer_memory_gb = model_params * 8 / (1024**3) if bits >= 16 else 0
    
    # Activations memory (rough estimate)
    token_embedding_size = 4096  # Assuming 4096 hidden size for Llama models
    activation_memory_gb = batch_size * seq_length * token_embedding_size * 4 / (1024**3)
    
    # Gradient accumulation and other overhead (rough estimate)
    overhead_gb = model_memory_gb * 0.1
    
    # Total estimate
    total_memory_gb = model_memory_gb + optimizer_memory_gb + activation_memory_gb + overhead_gb
    
    return {
        "model_memory_gb": round(model_memory_gb, 2),
        "optimizer_memory_gb": round(optimizer_memory_gb, 2),
        "activation_memory_gb": round(activation_memory_gb, 2),
        "overhead_gb": round(overhead_gb, 2),
        "total_memory_gb": round(total_memory_gb, 2)
    } 