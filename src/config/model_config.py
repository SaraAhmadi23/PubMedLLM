"""
Model configuration for Llama 3.1 8B
"""

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"
MAX_SEQ_LENGTH = 512  # Can increase for longer reasoning traces
LORA_RANK = 32  # Larger rank = smarter, but slower

# Model loading configurations
MODEL_CONFIG = {
    "load_in_4bit": True,  # False for LoRA 16bit
    "fast_inference": True,  # Enable vLLM fast inference
    "max_lora_rank": LORA_RANK,
    "gpu_memory_utilization": 0.5,  # Reduce if out of memory
}

# LoRA configurations
LORA_CONFIG = {
    "r": LORA_RANK,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],  # Remove QKVO if out of memory
    "lora_alpha": LORA_RANK,
    "use_gradient_checkpointing": "unsloth",  # Enable long context finetuning
    "random_state": 3407,
}

# Target LoRA modules
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# System prompt template
SYSTEM_PROMPT = """
Analyze the patient case and follow clinical guidelines to generate a structured response.
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

# Function to check if bfloat16 is supported
def is_bfloat16_supported():
    """
    Check if hardware supports bfloat16.
    """
    import torch
    return hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported() 