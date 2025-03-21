"""
GRPO training loop and trainer setup.
"""

import os
import logging
from typing import Dict, Any, List, Optional
from openai import OpenAI
from trl import GRPOTrainer
from datasets import Dataset
from ..config.training_config import get_training_args
from .reward_functions import get_reward_functions

# Set up logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

def setup_environment():
    """
    Configure environment variables for training.
    """
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    logging.info("Environment variables configured for training")

def create_trainer(
    model, 
    tokenizer, 
    dataset: Dataset,
    openai_api_key: Optional[str] = None
) -> GRPOTrainer:
    """
    Set up the GRPOTrainer with model, dataset, and reward functions.
    
    Args:
        model: The model to train
        tokenizer: The tokenizer
        dataset: The training dataset
        openai_api_key: OpenAI API key for critic (optional)
        
    Returns:
        Configured GRPOTrainer
    """
    # Initialize OpenAI client if API key is provided
    openai_client = None
    if openai_api_key:
        openai_client = OpenAI(api_key=openai_api_key)
        logging.info("OpenAI client initialized for critic rewards")
    else:
        logging.warning("No OpenAI API key provided. Critic rewards will not be available.")
    
    # Get GRPO configuration
    training_args = get_training_args()
    
    # Get reward functions (with or without critic)
    reward_functions = get_reward_functions(openai_client) if openai_client else [get_reward_functions(None)[0]]
    
    # Create and return the trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_functions,
        args=training_args,
        train_dataset=dataset,
    )
    
    logging.info("GRPOTrainer initialized successfully")
    return trainer

def train_model(trainer: GRPOTrainer, output_dir: str = "outputs"):
    """
    Run the training loop with error handling and checkpointing.
    
    Args:
        trainer: Configured GRPOTrainer
        output_dir: Directory to save checkpoints
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    logging.info("Starting training...")
    try:
        trainer.train()
        logging.info("Training completed successfully")
    except KeyboardInterrupt:
        logging.warning("Training interrupted by user")
        # Save checkpoint on interruption
        trainer.save_model(f"{output_dir}/checkpoint-interrupted")
        logging.info(f"Saved interrupted checkpoint to {output_dir}/checkpoint-interrupted")
    except Exception as e:
        logging.error(f"Training failed with error: {str(e)}")
        # Save emergency checkpoint
        try:
            trainer.save_model(f"{output_dir}/checkpoint-emergency")
            logging.info(f"Saved emergency checkpoint to {output_dir}/checkpoint-emergency")
        except Exception as save_error:
            logging.error(f"Failed to save emergency checkpoint: {str(save_error)}")
    finally:
        logging.info("Training process completed")

def monitor_training_metrics(logs: Dict[str, Any]):
    """
    Process and log training metrics.
    
    Args:
        logs: Dictionary of training metrics
    """
    # Extract important metrics
    reward = logs.get("reward", 0)
    reward_std = logs.get("reward_std", 0)
    completion_length = logs.get("completion_length", 0)
    kl = logs.get("kl", 0)
    step = logs.get("step", 0)
    
    # Log metrics in a formatted way
    logging.info(f"Step {step}: reward={reward:.4f}, reward_std={reward_std:.4f}, "
                 f"completion_length={completion_length:.2f}, kl={kl:.6f}")
    
    # You could add custom metric tracking here (e.g., to a database or file) 