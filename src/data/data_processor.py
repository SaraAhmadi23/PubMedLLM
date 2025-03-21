"""
Data processing utilities for medical dataset.
"""

import json
from typing import Dict, List, Any, Optional
from datasets import Dataset
from ..config.model_config import SYSTEM_PROMPT

def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load and parse JSON data from a file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        List of parsed JSON objects
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = f.read()
        
        data = json.loads(raw_data)
        print("✅ Successfully loaded JSON!")
        return data
    except Exception as e:
        print(f"Error loading JSON data: {e}")
        return []

def process_medical_data(entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract structured medical reasoning data from a dataset entry.
    
    Args:
        entry: Dictionary containing patient information and reference answer
        
    Returns:
        Processed data in the format expected by the model
    """
    # Handle missing fields gracefully
    patient_info = entry.get("patient_info", {})
    reference_answer = entry.get("reference_answer", {})
    
    # Safely extract patient information with defaults
    age = patient_info.get("age", "Unknown")
    bmi = patient_info.get("bmi", "N/A")
    comorbidities = ", ".join(patient_info.get("comorbidities", [])) if patient_info.get("comorbidities") else "None"
    diet_habits = patient_info.get("diet_habits", "N/A")
    current_meds = ", ".join(patient_info.get("current_meds", [])) if patient_info.get("current_meds") else "None"
    allergies = ", ".join(patient_info.get("allergies", [])) if patient_info.get("allergies") else "None"
    
    # Construct patient case description
    prompt_text = f"""
    Patient Case:
    - Age: {age}
    - BMI: {bmi}
    - Comorbidities: {comorbidities}
    - Diet habits: {diet_habits}
    - Medications: {current_meds}
    - Allergies: {allergies}

    Based on the above, provide a structured clinical decision-making approach.
    """
    
    # Extract reasoning steps
    reasoning_steps = "\n".join([
        f"- {step['action']}: {step['details']}"
        for step in reference_answer.get("reasoning_steps", [])
    ]) or "N/A"
    
    # Extract alternative treatment options
    alternative_options = "\n".join([
        f"- {option['option']}: {option['rationale_for_rejection']}"
        for option in reference_answer.get("alternative_options", [])
    ]) or "N/A"
    
    # Extract confidence levels
    confidence_levels = "\n".join([
        f"- {key.replace('_', ' ').title()}: {value}/10"
        for key, value in reference_answer.get("confidence_levels", {}).items()
    ]) or "N/A"
    
    # Extract specificity
    specificity = reference_answer.get("specificity", {})
    specificity_text = f"""
    - Gender-specific: {specificity.get("is_gender_specific", "Unknown")}
    - Age-specific: {specificity.get("is_age_specific", "Unknown")}
    - Comorbidity-specific: {specificity.get("is_comorbidity_specific", "Unknown")}
    """
    
    # Extract expected outcomes
    expected_outcomes = "\n".join([
        f"- {stage}: {outcome}"
        for stage, outcome in reference_answer.get("expected_outcomes", {}).items()
    ]) or "N/A"
    
    # Extract assumptions
    assumptions = "\n".join([
        f"- {assumption}" 
        for assumption in reference_answer.get("assumptions", [])
    ]) or "N/A"
    
    # Extract error handling
    error_handling = "\n".join([
        f"- {case['scenario']}: {case['action']}"
        for case in reference_answer.get("error_handling", {}).get("edge_cases", [])
    ]) or "N/A"
    
    # Extract references
    references = "\n".join([
        f"- {stage}: {info['source']} - {info['details']}"
        for stage, info in reference_answer.get("evidence", {}).items()
    ]) or "N/A"
    
    # Construct the formatted answer
    formatted_answer = f"""
    <reasoning>
    Clinical Reasoning Steps:
    {reasoning_steps}

    Confidence Levels:
    {confidence_levels}

    Specificity:
    {specificity_text}

    Alternative Treatment Options:
    {alternative_options}

    Expected Outcomes:
    {expected_outcomes}

    Assumptions:
    {assumptions}

    Error Handling:
    {error_handling}

    References:
    {references}
    </reasoning>
    <answer>
    {reference_answer.get("stages", {}).get("stage_5_final_plan", {}).get("description", "N/A")}
    </answer>
    """
    
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text}
        ],
        "answers": formatted_answer.strip()
    }

def get_medical_dataset(data: List[Dict[str, Any]]) -> Dataset:
    """
    Process the entire medical dataset for training.
    
    Args:
        data: List of raw data entries
        
    Returns:
        HuggingFace Dataset object
    """
    processed_data = [process_medical_data(entry) for entry in data]
    return Dataset.from_list(processed_data) 