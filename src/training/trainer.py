"""
Training utilities and GRPO trainer setup
"""
import os
import logging
from typing import List, Dict, Any, Optional
import torch
from trl import GRPOTrainer
from ..config.model_config import SYSTEM_PROMPT

def setup_environment(enable_wandb: bool = False):
    """
    Setup training environment
    
    Args:
        enable_wandb: Whether to enable Weights & Biases logging
    """
    os.environ["WANDB_DISABLED"] = str(not enable_wandb).lower()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
def create_trainer(
    model,
    tokenizer,
    dataset,
    training_args,
    openai_api_key: Optional[str] = None,
    output_dir: str = "outputs"
) -> GRPOTrainer:
    """
    Create GRPO trainer instance
    
    Args:
        model: The model to train
        tokenizer: The tokenizer
        dataset: Training dataset
        training_args: Training configuration
        openai_api_key: Optional OpenAI API key for critic
        output_dir: Output directory
        
    Returns:
        GRPOTrainer instance
    """
    # Setup reward functions
    reward_funcs = [
        combined_format_reward,
    ]
    
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
        reward_funcs.append(critic_reward)
    
    return GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
    )

def train_model(trainer: GRPOTrainer, output_dir: str = "outputs"):
    """
    Run model training
    
    Args:
        trainer: The trainer instance
        output_dir: Output directory
    """
    try:
        trainer.train()
    except KeyboardInterrupt:
        logging.warning("Training interrupted by user")
    finally:
        logging.info("Training session completed")
        
def combined_format_reward(completions: List[Dict[str, Any]], **kwargs) -> List[float]:
    """
    Reward function for output format adherence
    
    Args:
        completions: List of model completions
        **kwargs: Additional arguments
        
    Returns:
        List of reward scores
    """
    rewards = []
    for completion in completions:
        text = completion[0]["content"]
        score = 0.0
        
        # Structure check
        if validate_xml_structure(text):
            score += 0.5
            
            # Check if <answer> is non-empty
            answer_text = extract_xml_answer(text)
            if answer_text:
                score += 0.5
                
        rewards.append(score)
    return rewards

def critic_reward(
    prompts: List[Dict[str, Any]],
    completions: List[Dict[str, Any]], 
    answers: List[str],
    **kwargs
) -> List[float]:
    """
    GPT-4 critic reward function
    
    Args:
        prompts: Input prompts
        completions: Model completions
        answers: Reference answers
        **kwargs: Additional arguments
        
    Returns:
        List of reward scores
    """
    rewards = []
    for prompt, completion, reference in zip(prompts, completions, answers):
        model_output = completion[0]["content"]
        
        # Build critic prompt
        critic_prompt = f"""
You are a clinical expert. Evaluate this model-generated response for correctness.

Patient prompt:
{prompt[-1]["content"]}

Model's response:
{model_output}

Reference guidelines or info:
{reference}

Please provide a numeric "Score: X" from 0 to 10,
where 0 means completely incorrect, 10 means perfectly correct,
followed by a brief explanation.
""".strip()

        # Call GPT-4 with streaming & retry logic
        critic_response = call_critic_with_retry(critic_prompt, max_retries=3)
        
        # Parse score and scale to 0-2 range
        raw_score = parse_critic_score(critic_response)
        scaled_reward = raw_score / 5.0
        rewards.append(scaled_reward)
        
    return rewards

def validate_xml_structure(text: str) -> bool:
    """Check if text has valid XML structure"""
    import re
    return bool(re.search(r"<reasoning>.*?</reasoning>.*?<answer>.*?</answer>", text, re.DOTALL))

def extract_xml_answer(text: str) -> str:
    """Extract answer from XML structure"""
    import re
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else ""

def parse_critic_score(critic_response: str) -> float:
    """Parse numeric score from critic response"""
    import re
    match = re.search(r"Score:\s*([\d\.]+)", critic_response)
    return float(match.group(1)) if match else 0.0 