#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import logging
from typing import List, Any, Dict, Union
from dataclasses import dataclass
from pathlib import Path
from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)

# --- Prompt Components for Contrastive Few-Shot (Kept internal for now) ---
DEFAULT_SYSTEM_PROMPT = """You are an experienced translator specializing in gender-fair language. In the following English passage, different word classes such as nouns and adjectives need to be inflected for gender.
Translate the passage into German, using a gender-fair strategy when the gender identity of the referents in the passage is unknown, non-binary or it encompasses more than one gender identity. When translating, you might want to modify gendered referents by either using the gender star (*), ending with -ens, rephrasing with participles, using inherently neutral words, or keeping unchanged if the gender context is already specified in the sentence."""
DEFAULT_EXAMPLE_ENGLISH_TEXT = "The ascent challenges both skill and endurance. Weather conditions added another layer of complexity. Experienced mountaineers undertook the task despite the risks involved. Their journey will serve as inspiration for future climbers."
DEFAULT_EXAMPLE_STANDARD_GERMAN = "Der Aufstieg stellt sowohl Geschick als auch Ausdauer auf die Probe. Die Wetterbedingungen fügten eine weitere Komplexitätsebene hinzu. Erfahrene Bergsteiger unternahmen die Aufgabe trotz der damit verbundenen Risiken. Ihre Reise wird als Inspiration für zukünftige Bergsteiger dienen."
DEFAULT_EXAMPLE_CRITIQUE = "However, there are no clues as to what the gender identity of the referents is. To avoid linguistic sexism, gender-fair language should be preferred over generic masculine."
DEFAULT_FINAL_INSTRUCTION = "Now translate the following English passage and provide only the gender-fair German translation using any ONE of the strategies above which you think is most appropriate. Answer only with the gender-fair German translation, no other text."

@dataclass
class ContrastiveExampleShot: # Renamed for clarity
    english_text: str
    gender_fair_german_translation: str

    def to_chat_format(self) -> List[Dict[str, str]]:
        return [
            {"role": "user", "content": self.english_text},
            {"role": "assistant", "content": self.gender_fair_german_translation}
        ]

DEFAULT_CONTRASTIVE_SHOT_EXAMPLES = [
    ContrastiveExampleShot(
        english_text=DEFAULT_EXAMPLE_ENGLISH_TEXT,
        gender_fair_german_translation="Gender star: \"Der Aufstieg stellt sowohl Geschick als auch Ausdauer auf die Probe. Die Wetterbedingungen fügten eine weitere Komplexitätsebene hinzu. Erfahrene Bergsteiger*innen unternahmen die Aufgabe trotz der damit verbundenen Risiken. Ihre Reise wird als Inspiration für zukünftige Bergsteiger*innen dienen.\""
    ),
    ContrastiveExampleShot(
        english_text=DEFAULT_EXAMPLE_ENGLISH_TEXT,
        gender_fair_german_translation="Ens-forms: \"Der Aufstieg stellt sowohl Geschick als auch Ausdauer auf die Probe. Die Wetterbedingungen fügten eine weitere Komplexitätsebene hinzu. Erfahren Bergsteigens unternahmen die Aufgabe trotz der damit verbundenen Risiken. Ihrens Reise wird als Inspiration für zukünftig Bergsteigens dienen.\""
    ),
    ContrastiveExampleShot(
        english_text=DEFAULT_EXAMPLE_ENGLISH_TEXT,
        gender_fair_german_translation="Gender-neutral rewording: \"Der Aufstieg stellt sowohl Geschick als auch Ausdauer auf die Probe. Die Wetterbedingungen fügten eine weitere Komplexitätsebene hinzu. Erfahrene Bergsteigende unternahmen die Aufgabe trotz der damit verbundenen Risiken. Ihre Reise wird als Inspiration für zukünftige Bergsteigende dienen.\""
    ),
]

def load_json_data(file_path: str) -> Any:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Error: File not found at {file_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Error: Could not decode JSON from {file_path}")
        raise

def create_contrastive_chat_prompt(
    text_to_translate: str, 
    tokenizer: Any,
    system_prompt: str,
    example_english_text: str,
    example_standard_german: str,
    example_critique: str,
    contrastive_examples: List[ContrastiveExampleShot],
    final_instruction: str
) -> str:
    messages: List[Dict[str, str]] = []
    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": f"Consider this English passage that needs gender-fair translation:\n\nEnglish: \"{example_english_text}\""})
    assistant_critique_content = (
        f"The standard translation of this passage uses the masculine generic and is: \"{example_standard_german}\"\n\n"
        f"{example_critique}\n\n"
        f"Here are some ways to make the translation of the above English passage gender-fair:"
    )
    messages.append({"role": "assistant", "content": assistant_critique_content})
    for shot in contrastive_examples:
        messages.extend(shot.to_chat_format())
    user_message_for_translation = f"{final_instruction}\n\n{text_to_translate}"
    messages.append({"role": "user", "content": user_message_for_translation})
    
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception as e:
        logger.error(f"Error applying chat template for contrastive prompt: {e}")
        raise

def run_experiment(
    llm: LLM,
    tokenizer: Any,
    model_name_to_run: str,
    num_samples_to_process: Union[int, str],
    dataset_path: str = "dataset.json",
    prompts_path: str = "prompts.json",
    max_model_len: int = 8192,
    temperature: float = 0.1,
    max_tokens: int = 1024,
    output_dir: Path = Path("outputs"),
    model_identifier: str = "model"
) -> List[Dict[str, Any]]:
    """
    Runs the Contrastive Few-Shot translation experiment using a pre-initialized LLM and tokenizer.

    Args:
        llm: Pre-initialized vLLM LLM instance.
        tokenizer: Pre-initialized tokenizer from the LLM instance.
        model_name_to_run: Name or path of the HuggingFace model (for logging).
        num_samples_to_process: Number of items from dataset or "all".
        dataset_path: Path to the dataset JSON file.
        prompts_path: Path to the prompts JSON file (not used for main contrastive prompts).
        max_model_len: Max model sequence length (for non-LLM config if any).
        temperature: Sampling temperature.
        max_tokens: Max tokens for generation.
        output_dir: Directory for output JSON.
        model_identifier: Short model name for file naming.

    Returns:
        List of dictionaries with original text, translated text, and prompt used.
    """
    experiment_type = "contrastive-few-shot"
    logger.info(f"Starting {experiment_type} experiment for model: {model_name_to_run} using provided LLM instance.")

    # Using default internal prompts
    system_prompt_to_use = DEFAULT_SYSTEM_PROMPT
    example_english_text_to_use = DEFAULT_EXAMPLE_ENGLISH_TEXT
    example_standard_german_to_use = DEFAULT_EXAMPLE_STANDARD_GERMAN
    example_critique_to_use = DEFAULT_EXAMPLE_CRITIQUE
    contrastive_examples_to_use = DEFAULT_CONTRASTIVE_SHOT_EXAMPLES
    final_instruction_to_use = DEFAULT_FINAL_INSTRUCTION

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
        
    logger.info(f"Selected {len(items_to_process)} items for {experiment_type} translation.")
    if not items_to_process:
        logger.info("No items to process. Exiting experiment.")
        return []

    formatted_prompts_for_vllm = []
    original_texts_map = {} # Using dict to map prompt to original text and id

    for item_idx, item in enumerate(items_to_process):
        text_to_translate = item.get("text")
        item_id = item.get("id", f"item_{item_idx}")
        if not text_to_translate:
            logger.warning(f"Item ID {item_id} has no 'text' field. Skipping.")
            continue
        
        try:
            formatted_prompt = create_contrastive_chat_prompt(
                text_to_translate, tokenizer,
                system_prompt_to_use,
                example_english_text_to_use,
                example_standard_german_to_use,
                example_critique_to_use,
                contrastive_examples_to_use,
                final_instruction_to_use
            )
            formatted_prompts_for_vllm.append(formatted_prompt)
            original_texts_map[formatted_prompt] = {"text": text_to_translate, "id": item_id}
        except Exception as e:
            logger.error(f"Error creating {experiment_type} prompt for item ID {item_id}: {e}")
            continue

    if not formatted_prompts_for_vllm:
        logger.error(f"No valid {experiment_type} prompts created. Exiting.")
        return []

    sampling_params = SamplingParams(
        temperature=temperature, 
        max_tokens=max_tokens,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id else [],
        skip_special_tokens=True
    )

    logger.info(f"Generating translations for {len(formatted_prompts_for_vllm)} prompts ({experiment_type}, {model_identifier})...")
    try:
        llm_outputs = llm.generate(formatted_prompts_for_vllm, sampling_params)
    except Exception as e:
        logger.error(f"Error during generation ({experiment_type}, {model_identifier}): {e}")
        return []

    results_to_save = []
    logger.info(f"--- {experiment_type.capitalize()} Results ({model_identifier}) ---")
    for output_obj in llm_outputs:
        prompt_text = output_obj.prompt
        original_info = original_texts_map.get(prompt_text)
        if not original_info:
            logger.warning(f"Could not find original text for prompt: {prompt_text[:100]}... Skipping result.")
            continue

        original_text = original_info["text"]
        item_id = original_info["id"]
        generated_text = output_obj.outputs[0].text.strip() if output_obj.outputs else "Error: No output generated"
        
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