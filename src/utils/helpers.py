"""

Helper utilities for the PubMedLLM project.

"""



import os

import json

import logging

import torch

from typing import Dict, List, Any, Optional, Union



# Set up logging

def setup_logging(level: str = "INFO") -> logging.Logger:

    """

    Setup logging configuration

    

    Args:

        level: Logging level

        

    Returns:

        Configured logger

    """

    logging.basicConfig(

        level=getattr(logging, level.upper()),

        format='[%(asctime)s] %(levelname)s: %(message)s'

    )

    

    return logging.getLogger()



def check_hardware_compatibility() -> Dict[str, bool]:

    """

    Check hardware compatibility for training

    

    Returns:

        Dict of hardware capabilities

    """

    return {

        "cuda_available": torch.cuda.is_available(),

        "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False,

        "bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,

    }



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



def call_critic_with_retry(critic_prompt: str, max_retries: int = 3) -> str:

    """

    Call GPT-4 critic with retry logic

    

    Args:

        critic_prompt: Prompt for the critic

        max_retries: Maximum number of retries

        

    Returns:

        Critic response

    """

    import time
    import os
    import openai
    from openai import OpenAI
    


    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    


    for attempt in range(max_retries):

        try:

            response = client.chat.completions.create(

                model="gpt-4",

                messages=[

                    {"role": "system", "content": "You are a medical critic. Please evaluate correctness."},

                    {"role": "user", "content": critic_prompt}

                ],

                stream=True

            )
            


            # Accumulate streamed response

            return ''.join(

                choice.delta.content or ""

                for chunk in response

                for choice in chunk.choices

            )
            


        except Exception as e:

            if attempt < max_retries - 1:

                time.sleep(1.0 * (attempt + 1))

                continue

            return "Score: 0\nExplanation: Error calling critic."
            


    return "Score: 0\nExplanation: Maximum retries exceeded." 