"""
Configuration settings for the Llama 3.1 8B model.
"""

# Model parameters
MODEL_NAME = "meta-llama/meta-Llama-3.1-8B-Instruct"
MAX_SEQ_LENGTH = 512
LORA_RANK = 32
LOAD_IN_4BIT = True
FAST_INFERENCE = True

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