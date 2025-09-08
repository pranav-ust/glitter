#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import logging
from typing import Dict, List, Any, Union
from vllm import LLM, SamplingParams
from pathlib import Path # Added for output path handling

# logger is now configured by the main runner, but we can get it here
logger = logging.getLogger(__name__) # Changed to get logger by name


def load_json_data(file_path: str) -> Any:
    """
    Load JSON data from a file.

    Args:
        file_path: Path to the JSON file.

    Returns:
        Loaded JSON data.
    
    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Error: File not found at {file_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Error: Could not decode JSON from {file_path}")
        raise


def create_chat_prompt(text: str, system_prompt: str) -> List[Dict[str, str]]:
    """
    Create chat-format prompts for the model.
    
    Args:
        text: Text to be translated.
        system_prompt: System prompt with instructions.
        
    Returns:
        List of message dictionaries in chat format.
    """
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user", 
            "content": text
        },
    ]
    return messages

# Refactored main function to be callable
def run_experiment(
    llm: LLM, # Added pre-initialized LLM
    tokenizer: Any, # Added pre-initialized tokenizer
    model_name_to_run: str, # Kept for logging/identification
    num_samples_to_process: Union[int, str],
    dataset_path: str = "dataset.json",
    prompts_path: str = "prompts.json",
    temperature: float = 0.2, # Default temperature
    max_tokens: int = 1024,   # Default max_tokens
    output_dir: Path = Path("outputs"),
    model_identifier: str = "model", # For naming output file
    # Removed model_save_dir, max_model_len as LLM is pre-configured
    # max_model_len might still be passed if it's used for other logic by the experiment itself,
    # but not for LLM init here. For zero_shot, it seems it was only for LLM.
    max_model_len: int = 4096 # Kept in signature if run_single_experiment passes it, but not used for LLM init here
) -> List[Dict[str, str]]:
    """
    Runs the Zero-Shot translation experiment using a pre-initialized LLM and tokenizer.

    Args:
        llm: Pre-initialized vLLM LLM instance.
        tokenizer: Pre-initialized tokenizer from the LLM instance.
        model_name_to_run: Name or path of the HuggingFace model (for logging).
        num_samples_to_process: Number of items from the dataset to process or "all".
        dataset_path: Path to the dataset JSON file.
        prompts_path: Path to the prompts JSON file.
        temperature: Sampling temperature.
        max_tokens: Maximum number of tokens to generate.
        output_dir: Directory to save the output JSON file.
        model_identifier: Short name for the model (e.g., "gemma") for file naming.
        max_model_len: Max model length (passed from main, not used for LLM init here).

    Returns:
        A list of dictionaries, each containing the original text and its translation.
    """
    experiment_type = "zero-shot"
    logger.info(f"Starting Zero-Shot experiment for model: {model_name_to_run} using provided LLM instance.")

    # Load prompts
    logger.info(f"Loading prompts from {prompts_path}")
    prompts_data = load_json_data(prompts_path)
    zero_shot_system_prompt = prompts_data.get("zero-shot-prompt")

    if not zero_shot_system_prompt:
        logger.error("Could not find 'zero-shot-prompt' in the prompts file.")
        return []

    # Load dataset
    logger.info(f"Loading dataset from {dataset_path}")
    dataset = load_json_data(dataset_path)

    if not isinstance(dataset, list):
        logger.error("Dataset is not a list. Please ensure it's a JSON array of items.")
        return []

    # Select a subset of the dataset
    if isinstance(num_samples_to_process, int):
        if num_samples_to_process <= 0:
            logger.error("Number of samples must be positive.")
            return []
        items_to_process = dataset[:num_samples_to_process]
    elif num_samples_to_process == "all":
        items_to_process = dataset
    else:
        logger.error(f"Invalid num_samples_to_process: {num_samples_to_process}. Must be int or 'all'.")
        return []
        
    logger.info(f"Selected {len(items_to_process)} items for translation.")

    if not items_to_process:
        logger.info("No items to process. Exiting experiment.")
        return []

    # Prepare prompts for vLLM
    formatted_prompts_for_vllm = []
    original_texts = []
    processed_items_info = [] # To store original item and its translation

    for item in items_to_process:
        text_to_translate = item.get("text")
        if not text_to_translate:
            logger.warning(f"Item with ID {item.get('id', 'N/A')} has no 'text' field. Skipping.")
            continue
        
        original_texts.append(text_to_translate)
        chat_messages = create_chat_prompt(text_to_translate, zero_shot_system_prompt)
        
        try:
            # Using the tokenizer from the llm engine
            # The passed tokenizer should be llm.llm_engine.tokenizer.tokenizer or compatible
            formatted_prompt = tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            formatted_prompts_for_vllm.append(formatted_prompt)
        except AttributeError as e:
            logger.error(f"Could not apply chat template for model {model_name_to_run} using provided tokenizer: {e}. Check tokenizer compatibility.")
            # Fallback for older vLLM or tokenizer versions (less ideal)
            simple_prompt = f"System: {zero_shot_system_prompt}\\nUser: {text_to_translate}\\nAssistant:"
            logger.warning(f"Using fallback simple prompt format for item: {item.get('id', 'N/A')}")
            formatted_prompts_for_vllm.append(simple_prompt)


    if not formatted_prompts_for_vllm:
        logger.error(f"No valid prompts could be created for model {model_name_to_run}. Exiting.")
        return []

    sampling_params = SamplingParams(
        temperature=temperature, 
        max_tokens=max_tokens
    )

    logger.info(f"Generating translations for {len(formatted_prompts_for_vllm)} prompts with {model_name_to_run}...")
    try:
        outputs = llm.generate(formatted_prompts_for_vllm, sampling_params)
    except Exception as e:
        logger.error(f"Error during generation with {model_name_to_run}: {e}")
        return []

    # Process and store results
    results_to_save = []
    logger.info(f"--- Translation Results ({model_identifier} - {experiment_type}) ---")
    for i, output_obj in enumerate(outputs): # Renamed 'output' to 'output_obj' to avoid conflict
        original_text = original_texts[i]
        # Assuming the first output is the one we want
        generated_text = output_obj.outputs[0].text.strip() if output_obj.outputs else "Error: No output generated"
        
        # Minimal print to console, detailed output will be in JSON
        # logger.info(f"Original: '{original_text[:50]}...' -> Translated: '{generated_text[:50]}...'")
        
        item_id = items_to_process[i].get("id", f"item_{i}")
        results_to_save.append({
            "id": item_id,
            "original_text": original_text,
            "translated_text": generated_text
        })

    # Save results to a JSON file
    output_file_name = f"{model_identifier}-{experiment_type}.json"
    output_file_path = output_dir / output_file_name
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, indent=4, ensure_ascii=False)
        logger.info(f"Successfully saved results for {experiment_type} with {model_identifier} to {output_file_path}")
    except IOError as e:
        logger.error(f"Error saving results to {output_file_path}: {e}")

    logger.info(f"Zero-Shot experiment for {model_identifier} finished. {len(results_to_save)} items processed.")
    return results_to_save

# Removed argparse and if __name__ == "__main__": block
# The script will now be imported and run_experiment will be called by run_experiments.py 