"""
Main entry point for the PubMedLLM system.
This module provides a command-line interface for training and inference.
"""

import os
import sys
import argparse
import logging
import json
from typing import Dict, List, Any, Optional

# Local imports
from src.utils.helpers import setup_logging, check_hardware_compatibility
from src.config.model_config import MODEL_NAME, MAX_SEQ_LENGTH
from src.config.training_config import get_training_args
from src.data.data_processor import get_medical_dataset, load_json_data
from src.models.llama_model import initialize_model_and_tokenizer, save_model, load_saved_lora
from src.training.trainer import setup_environment, create_trainer, train_model
from src.inference.inference import generate_response, batch_generate_responses, parse_structured_response

# Set up argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="PubMedLLM: Fine-tuning Llama 3.1 for medical tasks")
    
    # Common arguments
    parser.add_argument("--mode", type=str, choices=["train", "inference", "evaluate"], required=True,
                        help="Operation mode: train, inference, or evaluate")
    parser.add_argument("--model_path", type=str, default=MODEL_NAME,
                        help="Path to the base model or model identifier from Hugging Face")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Path to saved LoRA weights (for inference or continued training)")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Directory to save outputs")
    parser.add_argument("--log_level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        default="INFO", help="Logging level")
    
    # Training arguments
    training_group = parser.add_argument_group("Training arguments")
    training_group.add_argument("--data_path", type=str, default=None,
                                help="Path to training data (JSON file)")
    training_group.add_argument("--epochs", type=int, default=3,
                                help="Number of training epochs")
    training_group.add_argument("--batch_size", type=int, default=8,
                                help="Training batch size")
    training_group.add_argument("--lr", type=float, default=2e-5,
                                help="Learning rate")
    training_group.add_argument("--wandb", action="store_true",
                                help="Enable Weights & Biases logging")
    training_group.add_argument("--openai_api_key", type=str, default=None,
                                help="OpenAI API key for critic reward")
    
    # Inference arguments
    inference_group = parser.add_argument_group("Inference arguments")
    inference_group.add_argument("--prompt", type=str, default=None,
                                help="Input prompt for inference")
    inference_group.add_argument("--prompt_file", type=str, default=None,
                                help="File containing prompts, one per line")
    inference_group.add_argument("--temperature", type=float, default=0.8,
                                help="Sampling temperature")
    inference_group.add_argument("--max_tokens", type=int, default=1024,
                                help="Maximum number of tokens to generate")
    inference_group.add_argument("--system_prompt", type=str, default=None,
                                help="Custom system prompt (overrides default)")
    
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Set up logging
    logger = setup_logging(args.log_level)
    logger.info("Starting PubMedLLM application")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check hardware compatibility
    hardware_info = check_hardware_compatibility()
    logger.info(f"Hardware info: {hardware_info}")
    if not hardware_info["cuda_available"]:
        logger.warning("CUDA is not available. Running on CPU will be very slow!")
    
    # Execute requested mode
    if args.mode == "train":
        run_training(args, logger)
    elif args.mode == "inference":
        run_inference(args, logger)
    elif args.mode == "evaluate":
        run_evaluation(args, logger)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        return 1
    
    return 0

def run_training(args, logger):
    """Run the training process"""
    logger.info("Starting training mode")
    
    # Configure environment
    setup_environment(enable_wandb=args.wandb)
    
    # Check for training data
    if not args.data_path:
        logger.error("No training data provided. Use --data_path to specify training data.")
        return
    
    logger.info(f"Loading training data from {args.data_path}")
    try:
        # Get dataset
        dataset = get_medical_dataset(args.data_path)
        logger.info(f"Loaded dataset with {len(dataset)} examples")
        
        # Initialize model and tokenizer
        logger.info(f"Initializing model: {args.model_path}")
        model, tokenizer = initialize_model_and_tokenizer(
            model_name=args.model_path, 
            lora_path=args.lora_path
        )
        
        # Get training arguments
        training_args = get_training_args(
            learning_rate=args.lr,
            per_device_batch_size=args.batch_size,
            num_train_epochs=args.epochs
        )
        
        # Create trainer
        trainer = create_trainer(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            training_args=training_args,
            openai_api_key=args.openai_api_key,
            output_dir=args.output_dir
        )
        
        # Start training
        logger.info("Starting training")
        train_model(trainer, output_dir=args.output_dir)
        
        # Save the model
        logger.info(f"Saving model to {args.output_dir}")
        save_model(model, tokenizer, path=args.output_dir, save_format="lora")
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)

def run_inference(args, logger):
    """Run inference with the model"""
    logger.info("Starting inference mode")
    
    # Initialize model and tokenizer
    logger.info(f"Initializing model: {args.model_path}")
    model, tokenizer = initialize_model_and_tokenizer(
        model_name=args.model_path,
        load_in_4bit=True,
        fast_inference=True,
        lora_path=args.lora_path
    )
    
    # Get inputs
    prompts = []
    if args.prompt:
        prompts.append(args.prompt)
    
    if args.prompt_file:
        try:
            with open(args.prompt_file, 'r', encoding='utf-8') as f:
                file_prompts = [line.strip() for line in f if line.strip()]
                prompts.extend(file_prompts)
        except Exception as e:
            logger.error(f"Error reading prompts file: {str(e)}")
            return
    
    if not prompts:
        logger.error("No prompts provided. Use --prompt or --prompt_file.")
        return
    
    # Configure generation parameters
    from vllm import SamplingParams
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=0.95
    )
    
    # Set up system prompt if provided
    system_prompt = args.system_prompt
    
    # Run inference
    logger.info(f"Running inference on {len(prompts)} prompts")
    try:
        if len(prompts) == 1:
            # Single prompt mode
            response = generate_response(
                model=model, 
                tokenizer=tokenizer,
                prompt=prompts[0],
                system_prompt=system_prompt,
                sampling_params=sampling_params,
                lora_path=args.lora_path
            )
            
            # Parse structured response if present
            structured = parse_structured_response(response)
            
            # Output results
            print("\n== PROMPT ==")
            print(prompts[0])
            print("\n== RESPONSE ==")
            print(response)
            
            # Save to file
            output_file = os.path.join(args.output_dir, "inference_result.txt")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("== PROMPT ==\n")
                f.write(prompts[0])
                f.write("\n\n== RESPONSE ==\n")
                f.write(response)
            
            logger.info(f"Results saved to {output_file}")
            
        else:
            # Batch mode
            responses = batch_generate_responses(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                system_prompt=system_prompt,
                sampling_params=sampling_params,
                lora_path=args.lora_path
            )
            
            # Save to file
            output_file = os.path.join(args.output_dir, "batch_inference_results.jsonl")
            with open(output_file, 'w', encoding='utf-8') as f:
                for prompt, response in zip(prompts, responses):
                    result = {
                        "prompt": prompt,
                        "response": response,
                        "structured": parse_structured_response(response)
                    }
                    f.write(json.dumps(result) + '\n')
            
            logger.info(f"Batch results saved to {output_file}")
            
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}", exc_info=True)

def run_evaluation(args, logger):
    """Evaluate the model on a test dataset"""
    logger.info("Starting evaluation mode")
    
    # Check for test data
    if not args.data_path:
        logger.error("No test data provided. Use --data_path to specify test data.")
        return
    
    try:
        # Load test data
        logger.info(f"Loading test data from {args.data_path}")
        test_data = load_json_data(args.data_path)
        
        # Initialize model and tokenizer
        logger.info(f"Initializing model: {args.model_path}")
        model, tokenizer = initialize_model_and_tokenizer(
            model_name=args.model_path,
            load_in_4bit=True,
            fast_inference=True,
            lora_path=args.lora_path
        )
        
        # Extract test prompts
        test_prompts = [item.get("prompt", "") for item in test_data]
        if not test_prompts:
            logger.error("No prompts found in test data")
            return
        
        # Configure generation parameters
        from vllm import SamplingParams
        sampling_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=0.95
        )
        
        # Run batch inference
        logger.info(f"Running evaluation on {len(test_prompts)} test examples")
        responses = batch_generate_responses(
            model=model,
            tokenizer=tokenizer,
            prompts=test_prompts,
            system_prompt=args.system_prompt,
            sampling_params=sampling_params,
            lora_path=args.lora_path
        )
        
        # Calculate metrics
        from src.utils.evaluation import (
            extract_structured_parts,
            calculate_format_adherence,
            calculate_response_metrics,
            compare_answers,
            evaluate_reasoning_quality
        )
        
        # Calculate overall metrics
        overall_metrics = calculate_response_metrics(responses, tokenizer)
        
        # Calculate per-example metrics
        example_metrics = []
        for i, (test_item, response) in enumerate(zip(test_data, responses)):
            # Extract reference answer if available
            reference_answer = test_item.get("reference", "")
            
            # Parse response
            structured = extract_structured_parts(response)
            
            # Format adherence
            format_metrics = calculate_format_adherence(
                response, 
                ["reasoning", "answer"]
            )
            
            # Answer comparison if reference exists
            answer_comparison = {}
            if reference_answer:
                answer_comparison = compare_answers(
                    structured.get("answer", ""),
                    reference_answer
                )
            
            # Reasoning quality
            reasoning_metrics = evaluate_reasoning_quality(
                structured.get("reasoning", "")
            )
            
            # Combine metrics
            example_metric = {
                "example_id": i,
                "format_score": format_metrics["format_score"],
                "reasoning_metrics": reasoning_metrics,
                **answer_comparison
            }
            example_metrics.append(example_metric)
        
        # Save results
        results = {
            "overall_metrics": overall_metrics,
            "example_metrics": example_metrics,
        }
        
        output_file = os.path.join(args.output_dir, "evaluation_results.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        # Save responses
        responses_file = os.path.join(args.output_dir, "evaluation_responses.jsonl")
        with open(responses_file, 'w', encoding='utf-8') as f:
            for test_item, response in zip(test_data, responses):
                result = {
                    "prompt": test_item.get("prompt", ""),
                    "reference": test_item.get("reference", ""),
                    "response": response,
                    "structured": extract_structured_parts(response)
                }
                f.write(json.dumps(result) + '\n')
        
        logger.info(f"Evaluation metrics saved to {output_file}")
        logger.info(f"Evaluation responses saved to {responses_file}")
        
        # Print summary
        print("\n== EVALUATION SUMMARY ==")
        print(f"Total examples: {len(test_prompts)}")
        print(f"Average format score: {overall_metrics['avg_format_score']:.2f}")
        print(f"Perfect format responses: {overall_metrics['perfect_format_count']} ({overall_metrics['perfect_format_percentage']:.1f}%)")
        
        if "exact_match" in example_metrics[0]:
            exact_matches = sum(m.get("exact_match", 0) for m in example_metrics)
            print(f"Exact answer matches: {exact_matches} ({exact_matches/len(example_metrics)*100:.1f}%)")
            
            avg_overlap = sum(m.get("word_overlap", 0) for m in example_metrics) / len(example_metrics)
            print(f"Average word overlap: {avg_overlap:.2f}")
            
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}", exc_info=True)

if __name__ == "__main__":
    sys.exit(main()) 