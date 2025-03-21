"""
Configuration settings for the GRPO training process.
"""

from .model_config import is_bfloat16_supported

# Environment configuration
WANDB_DISABLED = True
PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

# GRPO training configuration
def get_training_args():
    """
    Get configuration for the GRPO trainer.
    """
    from trl import GRPOConfig
    
    return GRPOConfig(
        use_vllm=True,                  # Use vLLM for fast inference
        learning_rate=3e-5,             # Learning rate
        adam_beta1=0.9,                 # Adam beta1
        adam_beta2=0.99,                # Adam beta2
        weight_decay=0.05,              # Weight decay
        warmup_ratio=0.05,              # Warmup ratio
        lr_scheduler_type="cosine",     # Learning rate scheduler
        optim="paged_adamw_8bit",       # Optimizer
        logging_steps=1,                # Logging frequency
        bf16=is_bfloat16_supported(),   # Use bfloat16 if supported
        fp16=not is_bfloat16_supported(), # Use fp16 if bfloat16 not supported
        per_device_train_batch_size=250, # Batch size per device
        gradient_accumulation_steps=2,  # Gradient accumulation steps
        num_generations=6,              # Number of generations for GRPO
        max_prompt_length=2048,         # Maximum prompt length
        max_completion_length=2048,     # Maximum completion length
        max_steps=250,                  # Maximum number of training steps
        save_steps=250,                 # Save checkpoint frequency
        max_grad_norm=1,                # Maximum gradient norm
        report_to="none",               # Disable reporting
        output_dir="outputs",           # Output directory
    )

# Critic configuration
CRITIC_MODEL = "gpt-4o"                 # Model to use for critique
CRITIC_MAX_RETRIES = 3                  # Maximum retries for critic API calls
CRITIC_SYSTEM_PROMPT = "You are a medical critic. Please evaluate correctness." 