"""
Training configuration for GRPO training
"""
from dataclasses import dataclass
from typing import Optional
from .model_config import is_bfloat16_supported

@dataclass
class TrainingConfig:
    """Training configuration parameters"""
    use_vllm: bool = True  # use vLLM for fast inference!
    learning_rate: float = 3e-5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    weight_decay: float = 0.05
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    optim: str = "paged_adamw_8bit"
    logging_steps: int = 1
    bf16: bool = is_bfloat16_supported()
    fp16: bool = not is_bfloat16_supported()
    per_device_train_batch_size: int = 250
    gradient_accumulation_steps: int = 2  # Increase if you want more stable updates
    num_generations: int = 6
    max_prompt_length: int = 2048
    max_completion_length: int = 2048
    max_steps: int = 250
    save_steps: int = 250
    max_grad_norm: float = 1
    report_to: str = "none"
    output_dir: str = "outputs"

def get_training_args(
    learning_rate: Optional[float] = None,
    per_device_batch_size: Optional[int] = None,
    num_train_epochs: Optional[int] = None,
) -> TrainingConfig:
    """
    Get training arguments with optional overrides.
    
    Args:
        learning_rate: Learning rate override
        per_device_batch_size: Batch size override
        num_train_epochs: Number of epochs override
        
    Returns:
        TrainingConfig with specified overrides
    """
    config = TrainingConfig()
    
    if learning_rate is not None:
        config.learning_rate = learning_rate
        
    if per_device_batch_size is not None:
        config.per_device_train_batch_size = per_device_batch_size
        
    if num_train_epochs is not None:
        config.num_train_epochs = num_train_epochs
        
    return config

# Environment configuration

WANDB_DISABLED = True

PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"



# GRPO training configuration

def get_training_args(learning_rate=3e-5, per_device_batch_size=8, num_train_epochs=3):

    """

    Get configuration for the GRPO trainer.

    

    Args:

        learning_rate: Learning rate for training

        per_device_batch_size: Batch size per device

        num_train_epochs: Number of training epochs

        

    Returns:

        GRPOConfig object with training settings

    """

    from trl import GRPOConfig

    

    return GRPOConfig(

        use_vllm=True,                  # Use vLLM for fast inference

        learning_rate=learning_rate,    # Learning rate

        adam_beta1=0.9,                 # Adam beta1

        adam_beta2=0.99,                # Adam beta2

        weight_decay=0.05,              # Weight decay

        warmup_ratio=0.05,              # Warmup ratio

        lr_scheduler_type="cosine",     # Learning rate scheduler

        optim="paged_adamw_8bit",       # Optimizer

        logging_steps=1,                # Logging frequency

        bf16=is_bfloat16_supported(),   # Use bfloat16 if supported

        fp16=not is_bfloat16_supported(), # Use fp16 if bfloat16 not supported

        per_device_train_batch_size=per_device_batch_size, # Batch size per device

        gradient_accumulation_steps=2,  # Gradient accumulation steps

        num_train_epochs=num_train_epochs, # Number of training epochs

        num_generations=6,              # Number of generations for GRPO

        max_prompt_length=2048,         # Maximum prompt length

        max_completion_length=2048,     # Maximum completion length

        save_steps=250,                 # Save checkpoint frequency

        max_grad_norm=1,                # Maximum gradient norm

        report_to="none",               # Disable reporting

        output_dir="outputs",           # Output directory

    )



# Critic configuration

CRITIC_MODEL = "gpt-4o"                 # Model to use for critique

CRITIC_MAX_RETRIES = 3                  # Maximum retries for critic API calls

CRITIC_SYSTEM_PROMPT = "You are a medical critic. Please evaluate correctness." 