"""
Llama 3.1 model initialization and configuration.
"""

import torch
from unsloth import FastLanguageModel, PatchFastRL
from ..config.model_config import (
    MODEL_NAME, 
    MAX_SEQ_LENGTH, 
    LORA_RANK, 
    LOAD_IN_4BIT, 
    FAST_INFERENCE,
    TARGET_MODULES
)

def initialize_model_and_tokenizer():
    """
    Initialize the Llama 3.1 model and tokenizer with unsloth optimizations.
    
    Returns:
        Tuple of (model, tokenizer)
    """
    # Apply the GRPO patch to FastLanguageModel
    PatchFastRL("GRPO", FastLanguageModel)

    # Initialize the model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
        fast_inference=FAST_INFERENCE,
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=0.5,  # Adjust based on available GPU memory
    )

    # Configure the model for LoRA fine-tuning
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_RANK,
        use_gradient_checkpointing="unsloth",  # Enable long context fine-tuning
        random_state=3407,
    )
    
    return model, tokenizer

def save_model(model, save_path, tokenizer=None, save_type="lora"):
    """
    Save the trained model in the specified format.
    
    Args:
        model: The trained model
        save_path: Path to save the model
        tokenizer: The tokenizer (optional for some save methods)
        save_type: Type of save ('lora', 'merged_16bit', 'merged_4bit', or 'gguf')
    """
    if save_type == "lora":
        model.save_lora(save_path)
        print(f"Saved LoRA adapters to {save_path}")
    
    elif save_type == "merged_16bit":
        if tokenizer is None:
            raise ValueError("Tokenizer is required for merged_16bit save format")
        model.save_pretrained_merged(save_path, tokenizer, save_method="merged_16bit")
        print(f"Saved merged 16-bit model to {save_path}")
    
    elif save_type == "merged_4bit":
        if tokenizer is None:
            raise ValueError("Tokenizer is required for merged_4bit save format")
        model.save_pretrained_merged(save_path, tokenizer, save_method="merged_4bit")
        print(f"Saved merged 4-bit model to {save_path}")
    
    elif save_type == "gguf":
        if tokenizer is None:
            raise ValueError("Tokenizer is required for GGUF save format")
        model.save_pretrained_gguf(save_path, tokenizer)
        print(f"Saved GGUF format model to {save_path}")
    
    else:
        raise ValueError(f"Unsupported save_type: {save_type}")

def load_saved_lora(model, lora_path):
    """
    Load a saved LoRA adapter.
    
    Args:
        model: The base model
        lora_path: Path to the saved LoRA adapter
        
    Returns:
        LoRA request object to be used with model.fast_generate()
    """
    return model.load_lora(lora_path) 