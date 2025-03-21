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

    TARGET_MODULES,

    MODEL_CONFIG,

    LORA_CONFIG

)



def initialize_model_and_tokenizer(

    model_name: str,

    max_seq_length: int,

    lora_path: str = None,

    **kwargs

):

    """

    Initialize model and tokenizer with given configuration

    

    Args:

        model_name: Name or path of the model

        max_seq_length: Maximum sequence length

        lora_path: Optional path to LoRA weights

        **kwargs: Additional arguments to override default config

        

    Returns:

        tuple: (model, tokenizer)

    """

    # Update config with any overrides

    config = MODEL_CONFIG.copy()

    config.update(kwargs)

    

    # Initialize model and tokenizer

    model, tokenizer = FastLanguageModel.from_pretrained(

        model_name=model_name,

        max_seq_length=max_seq_length,

        **config

    )

    

    # Apply LoRA configuration

    model = FastLanguageModel.get_peft_model(

        model,

        **LORA_CONFIG

    )

    

    if lora_path:

        model.load_lora(lora_path)

    

    return model, tokenizer



def save_model(model, path: str, save_format: str = "lora"):

    """

    Save the model weights

    

    Args:

        model: The model to save

        path: Path to save to

        save_format: Format to save in ("lora", "merged_16bit", "merged_4bit")

    """

    if save_format == "lora":

        model.save_lora(path)

    elif save_format == "merged_16bit":

        model.save_pretrained_merged(path, save_method="merged_16bit")

    elif save_format == "merged_4bit":

        model.save_pretrained_merged(path, save_method="merged_4bit")

    else:

        raise ValueError(f"Unknown save format: {save_format}")



def load_saved_lora(model, path: str):

    """

    Load saved LoRA weights

    

    Args:

        model: The model to load weights into

        path: Path to the saved weights

        

    Returns:

        The loaded weights request object

    """

    return model.load_lora(path) 