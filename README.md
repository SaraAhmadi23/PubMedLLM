# PubMedLLM

A modular project for fine-tuning Llama 3.1 8B with medical data for PubMed tasks.

## Project Description

This project provides a modular implementation for fine-tuning and inference with the Llama 3.1 8B model on medical datasets. It is designed to be deployed on high-performance computing clusters like Compute Canada, with a focus on modular architecture, scalability, and optimization.

## Project Structure

```
src/
├── config/          # Configuration settings
│   ├── model_config.py      # Model parameters
│   └── training_config.py   # Training parameters
├── data/            # Data loading and processing
│   └── data_processor.py    # Medical data processing 
├── models/          # Model definitions
│   └── llama_model.py       # Llama model initialization
├── training/        # Training logic
│   ├── reward_functions.py  # GRPO reward functions
│   └── trainer.py           # Training loop
├── utils/           # Utility functions
│   ├── helpers.py           # Common utilities
│   └── evaluation.py        # Evaluation metrics
├── inference/       # Inference capabilities
│   └── inference.py         # Generation utilities
└── main.py          # Entry point
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/PubMedLLM.git
cd PubMedLLM

# Install dependencies
pip install -r requirements.txt
```

## Usage

The project provides a command-line interface for training, inference, and evaluation.

### Training

To fine-tune the model on medical data:

```bash
python -m src.main --mode train \
  --model_path meta-llama/Meta-Llama-3.1-8B \
  --data_path path/to/training_data.json \
  --output_dir ./outputs \
  --batch_size 8 \
  --epochs 3 \
  --lr 2e-5
```

With OpenAI critic rewards:

```bash
python -m src.main --mode train \
  --model_path meta-llama/Meta-Llama-3.1-8B \
  --data_path path/to/training_data.json \
  --output_dir ./outputs \
  --openai_api_key your_api_key
```

### Inference

Single prompt inference:

```bash
python -m src.main --mode inference \
  --model_path meta-llama/Meta-Llama-3.1-8B \
  --lora_path ./outputs/adapter \
  --prompt "Analyze this patient case: 45-year-old male with chest pain..." \
  --temperature 0.7
```

Batch inference:

```bash
python -m src.main --mode inference \
  --model_path meta-llama/Meta-Llama-3.1-8B \
  --lora_path ./outputs/adapter \
  --prompt_file path/to/prompts.txt \
  --output_dir ./results
```

### Evaluation

To evaluate model performance on a test dataset:

```bash
python -m src.main --mode evaluate \
  --model_path meta-llama/Meta-Llama-3.1-8B \
  --lora_path ./outputs/adapter \
  --data_path path/to/test_data.json \
  --output_dir ./evaluation
```

## Features

- Modular architecture with clean separation of components
- GRPO (Gradient Reinforcement Policy Optimization) training with custom rewards
- Efficient LoRA fine-tuning optimized for medical tasks
- Optimized for high-performance computing environments
- Structured medical response format with reasoning and answers
- Comprehensive evaluation metrics for medical reasoning quality
- Command-line interface for training, inference, and evaluation

## Requirements

See `requirements.txt` for detailed dependencies. Key requirements:

- unsloth>=2023.12.1.0
- transformers>=4.35.0
- peft>=0.8.2
- trl>=0.7.10
- datasets>=2.14.0
- vllm>=0.3.0
- torch>=2.0.0 