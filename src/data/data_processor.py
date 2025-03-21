"""

Data processing utilities for medical dataset.

"""



import json

from typing import Dict, List, Any, Optional

from datasets import Dataset

from ..config.model_config import SYSTEM_PROMPT



def load_json_data(file_path: str) -> List[Dict[str, Any]]:

    """

    Load JSON data from file

    

    Args:

        file_path: Path to JSON file

        

    Returns:

        List of data entries

    """

    with open(file_path, "r", encoding="utf-8") as f:

        return json.load(f)



def process_medical_data(entry: Dict[str, Any]) -> Dict[str, Any]:

    """

    Process a single medical data entry

    

    Args:

        entry: Raw data entry

        

    Returns:

        Processed entry with prompt and answers

    """

    # Extract patient info

    patient_info = entry.get("patient_info", {})

    reference_answer = entry.get("reference_answer", {})

    

    # Build patient description

    patient_desc = []

    

    if age := patient_info.get("age"):

        patient_desc.append(f"Age: {age}")

        

    if bmi := patient_info.get("bmi"):

        patient_desc.append(f"BMI: {bmi}")

        

    if comorbidities := patient_info.get("comorbidities", []):

        patient_desc.append(f"Comorbidities: {', '.join(comorbidities)}")

        

    if diet := patient_info.get("diet_habits"):

        patient_desc.append(f"Diet habits: {diet}")

        

    if meds := patient_info.get("current_meds", []):

        patient_desc.append(f"Medications: {', '.join(meds)}")

        

    if allergies := patient_info.get("allergies", []):

        patient_desc.append(f"Allergies: {', '.join(allergies)}")

        

    prompt_text = "Patient Case:\n" + "\n".join(patient_desc)

    prompt_text += "\n\nBased on the above, provide a structured clinical decision-making approach."

    

    # Build structured answer

    answer_parts = []

    

    # Reasoning steps

    if steps := reference_answer.get("reasoning_steps", []):

        answer_parts.append("Clinical Reasoning Steps:")

        for step in steps:

            answer_parts.append(f"- {step['action']}: {step['details']}")

            

    # Confidence levels

    if levels := reference_answer.get("confidence_levels", {}):

        answer_parts.append("\nConfidence Levels:")

        for key, value in levels.items():

            answer_parts.append(f"- {key.replace('_', ' ').title()}: {value}/10")

            

    # Specificity

    if spec := reference_answer.get("specificity", {}):

        answer_parts.append("\nSpecificity:")

        answer_parts.append(f"- Gender-specific: {spec.get('is_gender_specific', 'Unknown')}")

        answer_parts.append(f"- Age-specific: {spec.get('is_age_specific', 'Unknown')}")

        answer_parts.append(f"- Comorbidity-specific: {spec.get('is_comorbidity_specific', 'Unknown')}")

        

    # Alternative options

    if options := reference_answer.get("alternative_options", []):

        answer_parts.append("\nAlternative Treatment Options:")

        for opt in options:

            answer_parts.append(f"- {opt['option']}: {opt['rationale_for_rejection']}")

            

    # Expected outcomes

    if outcomes := reference_answer.get("expected_outcomes", {}):

        answer_parts.append("\nExpected Outcomes:")

        for stage, outcome in outcomes.items():

            answer_parts.append(f"- {stage}: {outcome}")

            

    # Assumptions

    if assumptions := reference_answer.get("assumptions", []):

        answer_parts.append("\nAssumptions:")

        for assumption in assumptions:

            answer_parts.append(f"- {assumption}")

            

    # Error handling

    if errors := reference_answer.get("error_handling", {}).get("edge_cases", []):

        answer_parts.append("\nError Handling:")

        for case in errors:

            answer_parts.append(f"- {case['scenario']}: {case['action']}")

            

    # References

    if refs := reference_answer.get("evidence", {}):

        answer_parts.append("\nReferences:")

        for stage, info in refs.items():

            answer_parts.append(f"- {stage}: {info['source']} - {info['details']}")

            

    # Final answer

    final_plan = reference_answer.get("stages", {}).get("stage_5_final_plan", {}).get("description", "N/A")

    

    formatted_answer = f"""

<reasoning>

{chr(10).join(answer_parts)}

</reasoning>

<answer>

{final_plan}

</answer>

""".strip()

    

    return {

        "prompt": [

            {"role": "system", "content": SYSTEM_PROMPT},

            {"role": "user", "content": prompt_text}

        ],

        "answers": formatted_answer

    }



def get_medical_dataset(data_path: str) -> Dataset:

    """

    Create dataset from medical data file

    

    Args:

        data_path: Path to data file

        

    Returns:

        HuggingFace dataset

    """

    data = load_json_data(data_path)

    processed = [process_medical_data(entry) for entry in data]

    return Dataset.from_list(processed) 