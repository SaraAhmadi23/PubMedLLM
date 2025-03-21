# PubMedLLM

Fine-tuning Llama 3.1 for medical tasks using GRPO (Guided Reward Policy Optimization).

## Features

- Fine-tune Llama 3.1 models on medical datasets
- GRPO training with format and critic rewards
- Structured medical reasoning output
- Evaluation metrics for format adherence and answer quality
- Support for both training and inference modes

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/PubMedLLM.git
cd PubMedLLM
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the model:

```bash
python -m src.main \
    --mode train \
    --model_path meta-llama/Meta-Llama-3.1-8B \
    --data_path data/datasets/merged_medical_data.json \
    --output_dir ./outputs
```

Optional training arguments:
- `--epochs`: Number of training epochs (default: 3)
- `--batch_size`: Training batch size (default: 8)
- `--lr`: Learning rate (default: 2e-5)
- `--wandb`: Enable Weights & Biases logging

### Inference

To run inference:

```bash
python -m src.main \
    --mode inference \
    --model_path meta-llama/Meta-Llama-3.1-8B \
    --prompt "Your medical question here" \
    --output_dir ./results
```

For batch inference from a file:
```bash
python -m src.main \
    --mode inference \
    --model_path meta-llama/Meta-Llama-3.1-8B \
    --prompt_file prompts.txt \
    --output_dir ./results
```

### Evaluation

To evaluate the model:

```bash
python -m src.main \
    --mode evaluate \
    --model_path meta-llama/Meta-Llama-3.1-8B \
    --data_path data/datasets/test_data.json \
    --output_dir ./evaluation
```

## Project Structure

```
PubMedLLM/
├── src/
│   ├── config/
│   │   ├── model_config.py
│   │   └── training_config.py
│   ├── data/
│   │   └── data_processor.py
│   ├── models/
│   │   └── llama_model.py
│   ├── training/
│   │   └── trainer.py
│   ├── inference/
│   │   └── inference.py
│   ├── utils/
│   │   ├── helpers.py
│   │   └── evaluation.py
│   └── main.py
├── data/
│   └── datasets/
│       └── merged_medical_data.json
├── outputs/
├── requirements.txt
└── README.md
```

## Model Output Format

The model generates structured responses in the following format:

```xml
<reasoning>
Clinical Reasoning Steps:
- Step 1: ...
- Step 2: ...

Confidence Levels:
- Diagnosis: 8/10
- Treatment: 7/10

Specificity:
- Gender-specific: Yes
- Age-specific: Yes
- Comorbidity-specific: No

Alternative Treatment Options:
- Option A: Rejected because...
- Option B: Rejected because...

Expected Outcomes:
- Short-term: ...
- Long-term: ...

Assumptions:
- Assumption 1
- Assumption 2

Error Handling:
- Case 1: Action 1
- Case 2: Action 2

References:
- Source 1: Details
- Source 2: Details
</reasoning>
<answer>
Final clinical decision or recommendation
</answer>
```

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details. 