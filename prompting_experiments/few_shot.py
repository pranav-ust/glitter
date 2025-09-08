#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import logging
from typing import Dict, List, Any, Union
from dataclasses import dataclass
from vllm import LLM, SamplingParams
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class FewShotExample:
    english_text: str
    german_translation: str
    
    def to_chat_format(self) -> List[Dict[str, str]]:
        """Convert few-shot example to chat format"""
        return [
            {"role": "user", "content": self.english_text},
            {"role": "assistant", "content": self.german_translation}
        ]

# Few-shot examples from the prompt - kept internal to this script for now
# These could be loaded from prompts_path in a future enhancement
FEW_SHOT_EXAMPLES = [
    FewShotExample(
        english_text="The ascent challenges both skill and endurance. Weather conditions added another layer of complexity. Experienced mountaineers undertook the task despite the risks involved. Their journey will serve as inspiration for future climbers.",
        german_translation="Der Aufstieg stellt sowohl Geschick als auch Ausdauer auf die Probe. Die Wetterbedingungen fügten eine weitere Komplexitätsebene hinzu. Erfahrene Bergsteiger*innen unternahmen die Aufgabe trotz der damit verbundenen Risiken. Ihre Reise wird als Inspiration für zukünftige Bergsteiger*innen dienen."
    ),
    FewShotExample(
        english_text="The ascent challenges both skill and endurance. Weather conditions added another layer of complexity. Experienced mountaineers undertook the task despite the risks involved. Their journey will serve as inspiration for future climbers.",
        german_translation="Der Aufstieg stellt sowohl Geschick als auch Ausdauer auf die Probe. Die Wetterbedingungen fügten eine weitere Komplexitätsebene hinzu. Erfahren Bergsteigens unternahmen die Aufgabe trotz der damit verbundenen Risiken. Ihrens Reise wird als Inspiration für zukünftig Bergsteigens dienen."
    ),
    FewShotExample(
        english_text="The ascent challenges both skill and endurance. Weather conditions added another layer of complexity. Experienced mountaineers undertook the task despite the risks involved. Their journey will serve as inspiration for future climbers.",
        german_translation="Der Aufstieg stellt sowohl Geschick als auch Ausdauer auf die Probe. Die Wetterbedingungen fügten eine weitere Komplexitätsebene hinzu. Erfahrene Bergsteigende unternahmen die Aufgabe trotz der damit verbundenen Risiken. Ihre Reise wird als Inspiration für zukünftige Bergsteigende dienen."
    ),
]

# System prompt - kept internal for now
# This could also be loaded from prompts_path
DEFAULT_FEW_SHOT_SYSTEM_PROMPT = """You are an experienced translator specializing in gender-fair language. In the following English passage, different word classes such as nouns and adjectives need to be inflected for gender.
Translate the passage into German, using a gender-fair strategy when the gender identity of the referents in the passage is unknown, non-binary or it encompasses more than one gender identity. When translating, you might want to modify gendered referents by either using the gender star (*), ending with -ens, rephrasing with participles, using inherently neutral words, or keeping unchanged if the gender context is already specified in the sentence. 
Here are a few examples on how you can translate an English passage into German with said strategies."""

# Final instruction (to append to user message) - kept internal
DEFAULT_FINAL_INSTRUCTION = "Now translate the following English passage and provide only the gender-fair German translation using any ONE of the strategies above which you think is most appropriate."

def load_json_data(file_path: str) -> Any:
    """Load JSON data from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Error: File not found at {file_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Error: Could not decode JSON from {file_path}")
        raise

def create_few_shot_chat_prompt(
    text_to_translate: str, 
    tokenizer, 
    system_prompt: str, 
    few_shot_examples: List[FewShotExample],
    final_instruction: str
) -> str:
    """Create a few-shot prompt with examples and the text to translate"""
    
    messages = [{"role": "system", "content": system_prompt}]
    
    for shot in few_shot_examples:
        messages.extend(shot.to_chat_format())
    
    user_message_content = f"{final_instruction}\n\n{text_to_translate}"
    messages.append({"role": "user", "content": user_message_content})
    
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

def run_experiment(
    llm: LLM,
    tokenizer: Any,
    model_name_to_run: str,
    num_samples_to_process: Union[int, str],
    dataset_path: str = "dataset.json",
    prompts_path: str = "prompts.json",
    max_model_len: int = 8192,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    output_dir: Path = Path("outputs"),
    model_identifier: str = "model"
) -> List[Dict[str, str]]:
    """
    Runs the Few-Shot translation experiment using a pre-initialized LLM and tokenizer.
    Args:
        llm: Pre-initialized vLLM LLM instance.
        tokenizer: Pre-initialized tokenizer from the LLM instance.
        model_name_to_run: Name or path of the HuggingFace model (for logging).
        num_samples_to_process: Number of items from dataset or "all".
        dataset_path: Path to the dataset JSON file.
        prompts_path: Path to the prompts JSON file (currently not used for system/few-shot examples).
        max_model_len: Maximum model sequence length (for non-LLM config if any).
        temperature: Sampling temperature.
        max_tokens: Maximum number of tokens for generation.
        output_dir: Directory to save the output JSON.
        model_identifier: Short name for the model (for file naming).

    Returns:
        List of dictionaries with original text and translation.
    """
    experiment_type = "few-shot"
    logger.info(f"Starting {experiment_type} experiment for model: {model_name_to_run} using provided LLM instance.")

    # These could be loaded from prompts_path if needed by looking up keys.
    # For now, using the defaults defined in this script.
    system_prompt_to_use = DEFAULT_FEW_SHOT_SYSTEM_PROMPT
    few_shot_examples_to_use = FEW_SHOT_EXAMPLES
    final_instruction_to_use = DEFAULT_FINAL_INSTRUCTION
    
    # Example of how prompts_path could be used (if it contained these):
    # prompts_content = load_json_data(prompts_path)
    # system_prompt_to_use = prompts_content.get("few_shot_system_prompt", DEFAULT_FEW_SHOT_SYSTEM_PROMPT)
    # few_shot_examples_to_use = # ... logic to parse examples from prompts_content ...
    # final_instruction_to_use = prompts_content.get("few_shot_final_instruction", DEFAULT_FINAL_INSTRUCTION)

    logger.info(f"Loading dataset from {dataset_path}")
    dataset = load_json_data(dataset_path)

    if not isinstance(dataset, list):
        logger.error("Dataset is not a list. Ensure it's a JSON array.")
        return []

    if isinstance(num_samples_to_process, int):
        if num_samples_to_process <= 0:
            logger.error("Number of samples must be positive.")
            return []
        items_to_process = dataset[:num_samples_to_process]
    elif num_samples_to_process == "all":
        items_to_process = dataset
    else:
        logger.error(f"Invalid num_samples_to_process: {num_samples_to_process}")
        return []
        
    logger.info(f"Selected {len(items_to_process)} items for translation.")

    if not items_to_process:
        logger.info("No items to process. Exiting experiment.")
        return []

    formatted_prompts_for_vllm = []
    original_texts = []

    for item in items_to_process:
        text_to_translate = item.get("text")
        if not text_to_translate:
            logger.warning(f"Item ID {item.get('id', 'N/A')} has no 'text'. Skipping.")
            continue
        
        original_texts.append(text_to_translate)
        try:
            formatted_prompt = create_few_shot_chat_prompt(
                text_to_translate, 
                tokenizer,
                system_prompt_to_use,
                few_shot_examples_to_use,
                final_instruction_to_use
            )
            formatted_prompts_for_vllm.append(formatted_prompt)
        except Exception as e:
            logger.error(f"Error creating {experiment_type} prompt for item ID {item.get('id', 'N/A')}: {e}")
            continue 

    if not formatted_prompts_for_vllm:
        logger.error(f"No valid {experiment_type} prompts created. Exiting.")
        return []

    sampling_params = SamplingParams(
        temperature=temperature, 
        max_tokens=max_tokens,
        stop_token_ids=[tokenizer.eos_token_id], # Specific to this experiment type
        skip_special_tokens=True # Specific to this experiment type
    )

    logger.info(f"Generating translations for {len(formatted_prompts_for_vllm)} prompts ({experiment_type}, {model_identifier})...")
    try:
        outputs = llm.generate(formatted_prompts_for_vllm, sampling_params)
    except Exception as e:
        logger.error(f"Error during generation ({experiment_type}, {model_identifier}): {e}")
        return []

    results_to_save = []
    logger.info(f"--- {experiment_type.capitalize()} Translation Results ({model_identifier}) ---")
    for i, output_obj in enumerate(outputs):
        original_text = original_texts[i]
        generated_text = output_obj.outputs[0].text.strip() if output_obj.outputs else "Error: No output generated"
        
        item_id = items_to_process[i].get("id", f"item_{i}")
        results_to_save.append({
            "id": item_id,
            "original_text": original_text,
            "translated_text": generated_text
        })

    output_file_name = f"{model_identifier}-{experiment_type}.json"
    output_file_path = output_dir / output_file_name
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, indent=4, ensure_ascii=False)
        logger.info(f"Saved results for {experiment_type} ({model_identifier}) to {output_file_path}")
    except IOError as e:
        logger.error(f"Error saving results to {output_file_path}: {e}")

    logger.info(f"{experiment_type.capitalize()} experiment for {model_identifier} finished. {len(results_to_save)} items processed.")
    return results_to_save