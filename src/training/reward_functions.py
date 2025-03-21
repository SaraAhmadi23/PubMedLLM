"""
Reward functions for GRPO training.
"""

import re
import time
import logging
import httpx
from typing import List, Dict, Any, Tuple, Optional
from openai import OpenAI
from ..config.training_config import CRITIC_MODEL, CRITIC_MAX_RETRIES, CRITIC_SYSTEM_PROMPT

# ================================
# ✅ Logging Configuration
# ================================
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

# ================================
# ✅ XML Handling Functions
# ================================
def extract_xml_answer(text: str) -> str:
    """
    Extract the content between <answer> tags.
    
    Args:
        text: Text containing XML tags
        
    Returns:
        Content between <answer> tags or empty string if not found
    """
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else ""

def validate_xml_structure(text: str) -> bool:
    """
    Check if text contains proper reasoning and answer tags.
    
    Args:
        text: Text to validate
        
    Returns:
        True if text contains valid XML structure, False otherwise
    """
    return bool(re.search(r"<reasoning>.*?</reasoning>.*?<answer>.*?</answer>", text, re.DOTALL))

# ================================
# ✅ OpenAI API Functions
# ================================
def process_response(response):
    """
    Accumulate content from streaming response.
    
    Args:
        response: OpenAI streaming response
        
    Returns:
        Concatenated content string
    """
    return ''.join(
        choice.delta.content or ""
        for chunk in response
        for choice in chunk.choices
    )

def call_critic_llm(client: OpenAI, critic_prompt: str) -> str:
    """
    Call OpenAI API to evaluate model output.
    
    Args:
        client: OpenAI client
        critic_prompt: Prompt for the critic model
        
    Returns:
        Critic's response
    """
    conversation_history = [
        {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
        {"role": "user", "content": critic_prompt}
    ]

    response = client.chat.completions.create(
        model=CRITIC_MODEL,
        messages=conversation_history,
        stream=True
    )

    return process_response(response)

def parse_critic_score(critic_response: str) -> float:
    """
    Extract numeric score from critic's response.
    
    Args:
        critic_response: Response from the critic
        
    Returns:
        Numeric score (0-10) or 0 if no score found
    """
    match = re.search(r"Score:\\s*([\\d\\.]+)", critic_response)
    if match:
        return float(match.group(1))
    return 0.0

def call_critic_with_retry(client: OpenAI, critic_prompt: str, max_retries: int = CRITIC_MAX_RETRIES) -> str:
    """
    Call critic API with retry mechanism.
    
    Args:
        client: OpenAI client
        critic_prompt: Prompt for the critic
        max_retries: Maximum number of retry attempts
        
    Returns:
        Critic's response or fallback message if all retries fail
    """
    for attempt in range(max_retries):
        try:
            return call_critic_llm(client, critic_prompt)
        except httpx.RemoteProtocolError as e:
            if attempt < max_retries - 1:
                # Incremental backoff
                time.sleep(1.0 * (attempt + 1))
            else:
                logging.warning(f"RemoteProtocolError after {max_retries} attempts. Returning fallback content.")
                return "Score: 0\nExplanation: RemoteProtocolError fallback."
        except Exception as e:
            if attempt < max_retries - 1:
                logging.warning(f"Error encountered: {e}. Retrying...")
                time.sleep(1.0 * (attempt + 1))
            else:
                logging.warning(f"Error after {max_retries} attempts. Returning fallback content.")
                return "Score: 0\nExplanation: Fallback due to error."

# ================================
# ✅ Reward Functions
# ================================
def combined_format_reward(completions: List[List[Dict[str, str]]], **kwargs) -> List[float]:
    """
    Reward based on formatting adherence to expected structure.
    
    Args:
        completions: List of model completions
        
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
    client: OpenAI,
    prompts: List[List[Dict[str, str]]], 
    completions: List[List[Dict[str, str]]], 
    answers: List[str], 
    **kwargs
) -> List[float]:
    """
    Reward based on external critic evaluation.
    
    Args:
        client: OpenAI client
        prompts: List of input prompts
        completions: List of model completions
        answers: List of reference answers
        
    Returns:
        List of reward scores
    """
    rewards = []
    for prompt, completion, reference_answer in zip(prompts, completions, answers):
        model_output = completion[0]["content"]

        # Build a prompt for the critic
        critic_prompt = f"""
You are a clinical expert. Evaluate this model-generated response for correctness.

Patient prompt:
{prompt[-1]["content"]}

Model's response:
{model_output}

Reference guidelines or info:
{reference_answer}

Please provide a numeric "Score: X" from 0 to 10,
where 0 means completely incorrect, 10 means perfectly correct,
followed by a brief explanation.
""".strip()

        # Call GPT-4 with streaming & retry logic
        critic_response = call_critic_with_retry(client, critic_prompt)

        # Parse out the numeric score (e.g. "Score: 7")
        raw_score = parse_critic_score(critic_response)

        # Scale 0..10 -> 0..2
        scaled_reward = raw_score / 5.0
        rewards.append(scaled_reward)
    return rewards

# Wrapper function to simplify trainer setup
def get_reward_functions(openai_client: OpenAI) -> List[callable]:
    """
    Get list of reward functions for GRPO training.
    
    Args:
        openai_client: OpenAI client for critic API
        
    Returns:
        List of reward functions
    """
    return [
        combined_format_reward,
        lambda prompts, completions, answers, **kwargs: 
            critic_reward(openai_client, prompts, completions, answers, **kwargs)
    ] 