"""
Model inference and prediction utilities.
"""

from typing import List, Dict, Any, Optional, Union
from vllm import SamplingParams
from ..config.model_config import SYSTEM_PROMPT

def create_sampling_params(
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_tokens: int = 1024,
    **kwargs
) -> SamplingParams:
    """
    Create sampling parameters for text generation.
    
    Args:
        temperature: Controls randomness (lower = more deterministic)
        top_p: Nucleus sampling parameter
        max_tokens: Maximum number of tokens to generate
        **kwargs: Additional parameters for SamplingParams
        
    Returns:
        Configured SamplingParams object
    """
    return SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        **kwargs
    )

def generate_response(
    model,
    tokenizer,
    prompt: str,
    system_prompt: str = SYSTEM_PROMPT,
    sampling_params: Optional[SamplingParams] = None,
    lora_path: Optional[str] = None
) -> str:
    """
    Generate a response for a given prompt.
    
    Args:
        model: The model to use for generation
        tokenizer: The tokenizer
        prompt: User prompt
        system_prompt: System prompt to use (defaults to medical reasoning prompt)
        sampling_params: Generation parameters (uses defaults if None)
        lora_path: Path to LoRA weights (optional)
        
    Returns:
        Generated text
    """
    # Format the input as a chat
    chat_input = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    # Apply chat template
    text = tokenizer.apply_chat_template(
        chat_input,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Set default sampling parameters if not provided
    if sampling_params is None:
        sampling_params = create_sampling_params()
    
    # Prepare LoRA request if path is provided
    lora_request = model.load_lora(lora_path) if lora_path else None
    
    # Generate response
    output = model.fast_generate(
        [text],
        sampling_params=sampling_params,
        lora_request=lora_request
    )[0].outputs[0].text
    
    return output

def batch_generate_responses(
    model,
    tokenizer,
    prompts: List[str],
    system_prompt: str = SYSTEM_PROMPT,
    sampling_params: Optional[SamplingParams] = None,
    lora_path: Optional[str] = None
) -> List[str]:
    """
    Generate responses for multiple prompts in batch.
    
    Args:
        model: The model to use for generation
        tokenizer: The tokenizer
        prompts: List of user prompts
        system_prompt: System prompt to use
        sampling_params: Generation parameters
        lora_path: Path to LoRA weights (optional)
        
    Returns:
        List of generated responses
    """
    # Format all inputs as chats
    chat_inputs = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            tokenize=False,
            add_generation_prompt=True
        )
        for prompt in prompts
    ]
    
    # Set default sampling parameters if not provided
    if sampling_params is None:
        sampling_params = create_sampling_params()
    
    # Prepare LoRA request if path is provided
    lora_request = model.load_lora(lora_path) if lora_path else None
    
    # Generate responses in batch
    outputs = model.fast_generate(
        chat_inputs,
        sampling_params=sampling_params,
        lora_request=lora_request
    )
    
    # Extract text from outputs
    responses = [output.outputs[0].text for output in outputs]
    
    return responses

def parse_structured_response(response: str) -> Dict[str, str]:
    """
    Parse a structured response with <reasoning> and <answer> tags.
    
    Args:
        response: The generated response text
        
    Returns:
        Dictionary containing reasoning and answer components
    """
    import re
    
    result = {"reasoning": "", "answer": ""}
    
    # Extract reasoning
    reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", response, re.DOTALL)
    if reasoning_match:
        result["reasoning"] = reasoning_match.group(1).strip()
    
    # Extract answer
    answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if answer_match:
        result["answer"] = answer_match.group(1).strip()
    
    return result 