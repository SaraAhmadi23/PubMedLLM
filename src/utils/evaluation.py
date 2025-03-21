"""
Evaluation utilities and metrics for assessing model performance.
"""

import re
import json
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple

def extract_structured_parts(response: str) -> Dict[str, str]:
    """
    Extract structured parts from a model response.
    
    Args:
        response: Model-generated text response
        
    Returns:
        Dictionary with extracted components
    """
    result = {}
    
    # Extract content between XML tags
    for tag in ["reasoning", "answer", "confidence", "specificity", "alternatives", 
                "outcomes", "assumptions", "errors", "references"]:
        pattern = f"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            result[tag] = match.group(1).strip()
        else:
            result[tag] = ""
    
    return result

def calculate_format_adherence(response: str, required_tags: List[str]) -> Dict[str, Union[float, Dict[str, bool]]]:
    """
    Calculate how well the response adheres to the required format.
    
    Args:
        response: Model-generated text response
        required_tags: List of XML tags that should be in the response
        
    Returns:
        Dictionary with format adherence metrics
    """
    tag_presence = {}
    total_found = 0
    
    for tag in required_tags:
        pattern = f"<{tag}>.*?</{tag}>"
        found = bool(re.search(pattern, response, re.DOTALL))
        tag_presence[tag] = found
        if found:
            total_found += 1
    
    format_score = total_found / len(required_tags) if required_tags else 0.0
    
    return {
        "format_score": format_score,
        "tag_presence": tag_presence
    }

def count_tokens(text: str, tokenizer) -> int:
    """
    Count the number of tokens in a text string.
    
    Args:
        text: Input text
        tokenizer: Tokenizer to use for counting
        
    Returns:
        Number of tokens
    """
    return len(tokenizer.encode(text))

def calculate_response_metrics(responses: List[str], tokenizer) -> Dict[str, Union[float, int]]:
    """
    Calculate various metrics on a list of model responses.
    
    Args:
        responses: List of model-generated responses
        tokenizer: Tokenizer for token counting
        
    Returns:
        Dictionary with response metrics
    """
    metrics = {}
    
    # Token statistics
    token_counts = [count_tokens(r, tokenizer) for r in responses]
    metrics["avg_tokens"] = np.mean(token_counts)
    metrics["min_tokens"] = min(token_counts)
    metrics["max_tokens"] = max(token_counts)
    metrics["std_tokens"] = np.std(token_counts)
    
    # Format adherence statistics
    required_tags = ["reasoning", "answer"]
    format_scores = [
        calculate_format_adherence(r, required_tags)["format_score"] 
        for r in responses
    ]
    metrics["avg_format_score"] = np.mean(format_scores)
    metrics["perfect_format_count"] = sum(1 for s in format_scores if s == 1.0)
    metrics["perfect_format_percentage"] = (metrics["perfect_format_count"] / len(responses)) * 100 if responses else 0
    
    return metrics

def compare_answers(generated: str, reference: str) -> Dict[str, float]:
    """
    Compare generated answer with reference answer.
    This is a placeholder for more sophisticated metrics.
    
    Args:
        generated: Generated answer text
        reference: Reference answer text
        
    Returns:
        Dictionary with comparison metrics
    """
    # Extract just the answer part if XML tags are present
    generated_match = re.search(r"<answer>(.*?)</answer>", generated, re.DOTALL)
    if generated_match:
        generated = generated_match.group(1).strip()
    
    reference_match = re.search(r"<answer>(.*?)</answer>", reference, re.DOTALL)
    if reference_match:
        reference = reference_match.group(1).strip()
    
    # Simple exact match
    exact_match = generated.lower() == reference.lower()
    
    # Word overlap (very simple metric)
    generated_words = set(generated.lower().split())
    reference_words = set(reference.lower().split())
    
    if not reference_words:
        word_overlap = 0.0
    else:
        word_overlap = len(generated_words.intersection(reference_words)) / len(reference_words)
    
    return {
        "exact_match": float(exact_match),
        "word_overlap": word_overlap
    }

def evaluate_reasoning_quality(reasoning: str) -> Dict[str, float]:
    """
    Evaluate the quality of clinical reasoning.
    This is a placeholder for more sophisticated evaluation.
    
    Args:
        reasoning: The reasoning text to evaluate
        
    Returns:
        Dictionary with reasoning quality metrics
    """
    # Extract just the reasoning part if XML tags are present
    reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", reasoning, re.DOTALL)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
    
    # Count sentences as a proxy for reasoning depth
    sentences = re.split(r'[.!?]+', reasoning)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Count medical terms (very simplistic approach)
    medical_terms_pattern = r'\b(diagnosis|treatment|symptom|patient|disease|clinical|therapy|medication|dose|drug|prognosis|assessment|evaluation)\b'
    medical_terms_count = len(re.findall(medical_terms_pattern, reasoning.lower()))
    
    # Simple heuristic for structure
    has_structure = any(indicator in reasoning.lower() for indicator in 
                         ["first", "second", "third", "finally", "conclusion", "assessment", 
                          "plan", "recommendation", "differential diagnosis"])
    
    return {
        "sentence_count": len(sentences),
        "word_count": len(reasoning.split()),
        "medical_terms_count": medical_terms_count,
        "has_structure": float(has_structure),
        "avg_sentence_length": len(reasoning.split()) / len(sentences) if sentences else 0
    } 