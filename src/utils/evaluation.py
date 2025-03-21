"""

Evaluation utilities and metrics for assessing model performance.

"""



import re

import json

import numpy as np

from typing import List, Dict, Any, Optional, Union, Tuple



def extract_structured_parts(text: str) -> Dict[str, str]:

    """

    Extract structured parts from model output

    

    Args:

        text: Model output text

        

    Returns:

        Dict with reasoning and answer

    """

    parts = {}

    

    # Extract reasoning

    reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)

    parts["reasoning"] = reasoning_match.group(1).strip() if reasoning_match else ""

    

    # Extract answer

    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)

    parts["answer"] = answer_match.group(1).strip() if answer_match else ""

    

    return parts



def calculate_format_adherence(text: str, required_sections: List[str]) -> Dict[str, float]:

    """

    Calculate format adherence metrics

    

    Args:

        text: Model output text

        required_sections: List of required sections

        

    Returns:

        Dict of format metrics

    """

    metrics = {

        "format_score": 0.0,

        "missing_sections": [],

        "empty_sections": []

    }

    

    # Check each required section

    for section in required_sections:

        pattern = f"<{section}>(.*?)</{section}>"

        match = re.search(pattern, text, re.DOTALL)

        

        if not match:

            metrics["missing_sections"].append(section)

            continue

            

        content = match.group(1).strip()

        if not content:

            metrics["empty_sections"].append(section)

            

    # Calculate score

    total_sections = len(required_sections)

    present_sections = total_sections - len(metrics["missing_sections"])

    non_empty_sections = present_sections - len(metrics["empty_sections"])

    

    metrics["format_score"] = non_empty_sections / total_sections

    

    return metrics



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



def calculate_response_metrics(responses: List[str], tokenizer) -> Dict[str, Any]:

    """

    Calculate overall response metrics

    

    Args:

        responses: List of model responses

        tokenizer: Tokenizer for length calculation

        

    Returns:

        Dict of response metrics

    """

    metrics = {

        "total_responses": len(responses),

        "avg_length": 0,

        "perfect_format_count": 0,

        "perfect_format_percentage": 0,

        "avg_format_score": 0,

    }

    

    format_scores = []

    lengths = []

    

    for response in responses:

        # Calculate length

        length = len(tokenizer.encode(response))

        lengths.append(length)

        

        # Check format

        format_metrics = calculate_format_adherence(

            response,

            required_sections=["reasoning", "answer"]

        )

        format_scores.append(format_metrics["format_score"])

        

        if format_metrics["format_score"] == 1.0:

            metrics["perfect_format_count"] += 1

            

    # Calculate averages

    metrics["avg_length"] = sum(lengths) / len(lengths)

    metrics["avg_format_score"] = sum(format_scores) / len(format_scores)

    metrics["perfect_format_percentage"] = (

        metrics["perfect_format_count"] / metrics["total_responses"] * 100

    )

    

    return metrics



def compare_answers(generated: str, reference: str) -> Dict[str, float]:

    """

    Compare generated answer with reference

    

    Args:

        generated: Generated answer

        reference: Reference answer

        

    Returns:

        Dict of comparison metrics

    """

    from difflib import SequenceMatcher

    

    metrics = {

        "exact_match": generated.strip() == reference.strip(),

        "word_overlap": 0.0,

    }

    

    # Calculate word overlap

    gen_words = set(generated.lower().split())

    ref_words = set(reference.lower().split())

    

    if ref_words:

        overlap = len(gen_words.intersection(ref_words))

        metrics["word_overlap"] = overlap / len(ref_words)

        

    # Calculate sequence similarity

    metrics["sequence_similarity"] = SequenceMatcher(

        None,

        generated.lower(),

        reference.lower()

    ).ratio()

    

    return metrics



def evaluate_reasoning_quality(reasoning: str) -> Dict[str, Any]:

    """

    Evaluate quality of reasoning

    

    Args:

        reasoning: Reasoning text

        

    Returns:

        Dict of quality metrics

    """

    metrics = {

        "length": len(reasoning.split()),

        "has_structure": bool(re.search(r"^\s*[-•*]\s+", reasoning, re.MULTILINE)),

        "num_points": len(re.findall(r"^\s*[-•*]\s+", reasoning, re.MULTILINE)),

    }

    

    # Check for clinical terms

    clinical_terms = [

        r"diagnosis",

        r"treatment",

        r"symptoms?",

        r"patient",

        r"clinical",

        r"medical",

        r"health",

        r"care",

        r"condition",

        r"assessment"

    ]

    

    metrics["clinical_terms_count"] = sum(

        1 for term in clinical_terms

        if re.search(rf"\b{term}\b", reasoning.lower())

    )

    

    # Check for evidence references

    evidence_patterns = [

        r"study shows",

        r"research indicates",

        r"evidence suggests",

        r"according to",

        r"based on",

        r"literature",

        r"guidelines?"

    ]

    

    metrics["evidence_references"] = sum(

        1 for pattern in evidence_patterns

        if re.search(pattern, reasoning.lower())

    )

    

    return metrics 