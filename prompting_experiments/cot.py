import json
import logging
import re # Add re import for regex operations
from typing import Dict, List, Any, Union, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

from pydantic import BaseModel, ValidationError
from vllm import LLM, SamplingParams

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Pydantic model for validating the final JSON output from the LLM
class TranslationResponse(BaseModel):
    seed_word: str
    reasoning: str
    final_translation: str

# Dataclass for our JSON-based few-shot examples
@dataclass
class JsonFewShotExample:
    english_text: str
    assistant_json_response: Dict[str, str]

    def to_chat_format(self) -> List[Dict[str, str]]:
        """Convert few-shot example to chat format."""
        return [
            {"role": "user", "content": f"Translate the following English passage:\n{self.english_text}"},
            {"role": "assistant", "content": json.dumps(self.assistant_json_response, ensure_ascii=False)}
        ]

# Define Few-Shot Examples - kept internal for now
DEFAULT_FEW_SHOT_EXAMPLES = [
    JsonFewShotExample(
        english_text=(
            "The ascent challenges both skill and endurance. Weather conditions added another layer of complexity. "
            "Experienced mountaineers undertook the task despite the risks involved. "
            "Their journey will serve as inspiration for future climbers."
        ),
        assistant_json_response={
            "seed_word": "mountaineers",
            "reasoning": "In the passage, there are no clues as to what the gender of \"mountaineers\". In such cases, the generic masculine \"Bergsteiger\" is a common translation in German. This, however, is an example of gender-exclusive language. To avoid exclusion, one could opt for gender-fair alternatives such as gender star as in \"Bergsteiger*innen\", ens-forms as in \"Bergsteigens\", or a neutral alternative as in \"Bergsteigende\". The gender star is chosen here.",
            "final_translation": "Der Aufstieg stellt sowohl Geschick als auch Ausdauer auf die Probe. Die Wetterbedingungen fügten eine weitere Komplexitätsebene hinzu. Erfahrene Bergsteiger*innen unternahmen die Aufgabe trotz der damit verbundenen Risiken. Ihre Reise wird als Inspiration für zukünftige Bergsteiger*innen dienen."
        }
    )
]

# System prompt - kept internal for now
DEFAULT_SYSTEM_PROMPT = ('''You are an expert translator specializing in gender-fair German translations. '''
                 '''Translate the given English passage into German. '''
                 '''When the gender identity of referents is unknown, non-binary, or encompasses multiple identities, '''
                 '''apply a gender-fair strategy (e.g., Gendersternchen *, -ens forms, participles, neutral words). '''
                 '''Your response MUST be a single JSON object with three fields: '''
                 '''1. "seed_word": A string indicating the key English word(s) that required gender-fair consideration (comma-separated if multiple). '''
                 '''2. "reasoning": A string explaining the gender-fair choices made for the seed words. '''
                 '''3. "final_translation": A string containing the complete gender-fair German translation. '''
                 '''Refer to the examples provided on how to structure your JSON response.''')

# Final instruction - kept internal for now
DEFAULT_FINAL_INSTRUCTION = (
    "Now, carefully analyze the following English passage. Identify relevant seed words, formulate your reasoning, "
    "and provide the complete gender-fair German translation. Ensure your entire output is a single, valid JSON object structured as described and exemplified above."
)

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

def create_cot_chat_prompt(
    text_to_translate: str, 
    tokenizer: Any, 
    system_prompt: str,
    few_shot_examples: List[JsonFewShotExample],
    final_instruction: str
) -> str:
    """Create a CoT few-shot prompt using chat template for vLLM."""
    messages = [{"role": "system", "content": system_prompt}]

    for shot in few_shot_examples:
        messages.extend(shot.to_chat_format())

    user_content = f"{final_instruction}\n\nEnglish passage to translate:\n{text_to_translate}"
    messages.append({"role": "user", "content": user_content})

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception as e:
        logger.error(f"Error applying chat template: {e}")
        raise

def parse_llm_output_to_json(generated_text: str) -> Optional[Dict[str, Any]]:
    """Attempts to parse the LLM's text output into a JSON object.
    Includes a regex-based fallback if standard JSON parsing fails.
    """
    original_generated_text = generated_text # Keep a copy for regex fallback
    json_str_to_parse = None

    # Attempt to find and clean JSON string
    if generated_text.startswith("```json"):
        json_str_to_parse = generated_text[7:-3].strip()
    elif generated_text.startswith("```"): # Handle cases like ``` { ... } ```
        # Find the first occurrence of { and the last occurrence of } after that
        first_brace = generated_text.find("{")
        if first_brace != -1:
            # Try to find the matching closing brace for the JSON object
            # This is a simplified approach; a robust parser would be needed for complex nested structures outside the main JSON
            last_brace = generated_text.rfind("}")
            if last_brace > first_brace:
                json_str_to_parse = generated_text[first_brace : last_brace + 1].strip()
        if not json_str_to_parse: # Fallback if ``` stripping is complex
             # Remove ``` and then try to find JSON
            processed_text = generated_text.replace("```json", "").replace("```", "").strip()
            json_start_index = processed_text.find("{")
            json_end_index = processed_text.rfind("}") + 1
            if json_start_index != -1 and json_end_index != 0 and json_start_index < json_end_index:
                json_str_to_parse = processed_text[json_start_index:json_end_index]
    elif generated_text.startswith("{") and generated_text.endswith("}"):
        json_str_to_parse = generated_text
    else:
        json_start_index = generated_text.find("{")
        json_end_index = generated_text.rfind("}") + 1
        if json_start_index != -1 and json_end_index != 0 and json_start_index < json_end_index:
            json_str_to_parse = generated_text[json_start_index:json_end_index]

    if json_str_to_parse:
        try:
            # First attempt: standard JSON parsing
            return json.loads(json_str_to_parse)
        except json.JSONDecodeError as e:
            logger.warning(f"Initial JSONDecodeError: {e}. Output: {json_str_to_parse[:100]}...")
            # Attempt 2: Try to fix common escape issues before regex
            try:
                # Replace \\" with " and \\* with * as these seem to be common culprits
                # Be cautious with this, as it might overly simplify or break valid escapes.
                # This is a targeted fix for observed issues.
                fixed_json_str = json_str_to_parse.replace("\\\\*", "*").replace("\\\\\\\"", "\\\"") # More specific
                # A simpler, more general replacement might be:
                # fixed_json_str = json_str_to_parse.replace('\\\\', '\\') # if \\ is consistently an issue
                return json.loads(fixed_json_str)
            except json.JSONDecodeError as e2:
                logger.warning(f"JSONDecodeError after attempting to fix escapes: {e2}. Output: {fixed_json_str[:100]}...")
                # Proceed to regex fallback if fixing escapes also fails
    else:
        # If json_str_to_parse is None, it means we couldn't reliably find a JSON-like structure.
        # We can still try regex on the original_generated_text as a last resort.
        logger.warning(f"Could not reliably find JSON object in output: {original_generated_text[:100]}... Proceeding to regex fallback.")


    # Fallback: Regex-based extraction
    # This regex attempts to capture content within quotes, being mindful of escaped quotes.
    # It's made more robust to handle newlines and other characters within the fields.
    # (?s) makes . match newlines.
    # "key":\\s*"( (?:\\\\"|[^"])* )" -> captures content of "key": "..."
    # Breakdown of (?:\\\\"|[^"])* :
    #   \\\\": matches an escaped quote \\"
    #   [^"]: matches any character that is not a quote
    #   (?:...)*: matches zero or more of the preceding group (escaped quotes or non-quote characters)
    
    # Strip markdown ```json ... ``` if present, for regex
    text_for_regex = original_generated_text
    if text_for_regex.startswith("```json"):
        text_for_regex = text_for_regex[7:-3].strip()
    elif text_for_regex.startswith("```"):
        text_for_regex = text_for_regex[3:-3].strip()


    patterns = {
        "seed_word": r'"seed_word":\s*"(.*?)(?<!\\)"', # Simpler, might fail with internal escaped quotes
        # More robust for internal escaped quotes: r'"seed_word":\s*"((?:\\.|[^"\\])*)"'
        "reasoning": r'"reasoning":\s*"((?:\\.|[^"\\])*)"',
        "final_translation": r'"final_translation":\s*"((?:\\.|[^"\\])*)"'
    }
    
    # More robust patterns to handle escaped quotes and newlines within the string values.
    # The pattern ((?:\\.|[^"\\])*) attempts to correctly capture strings that may contain escaped quotes.
    # Using (?s) to make . match newlines if your values can span multiple lines.
    # However, json.loads handles newlines in strings correctly, so regex primarily for structure.

    extracted_data = {}
    try:
        # Using re.DOTALL to make . match newlines within the fields if necessary,
        # though JSON strings themselves handle internal newlines.
        # The primary challenge is the overall structure and specific invalid escapes.

        # Regex for "seed_word"
        seed_word_match = re.search(patterns["seed_word"], text_for_regex, re.DOTALL)
        if seed_word_match:
            # The simple (.*?) might over-unescape. If \\" was meant to be a literal backslash then quote.
            # json.loads would handle this. If regex is used, we get the raw string content.
            # For "reasoning" and "final_translation", we use the more robust ((?:\\.|[^"\\])*)
            extracted_data["seed_word"] = seed_word_match.group(1).replace('\\\\"', '"').replace('\\\\*', '*')
        
        reasoning_match = re.search(patterns["reasoning"], text_for_regex, re.DOTALL)
        if reasoning_match:
            extracted_data["reasoning"] = reasoning_match.group(1).replace('\\\\"', '"').replace('\\\\*', '*')

        final_translation_match = re.search(patterns["final_translation"], text_for_regex, re.DOTALL)
        if final_translation_match:
            extracted_data["final_translation"] = final_translation_match.group(1).replace('\\\\"', '"').replace('\\\\*', '*')

        if len(extracted_data) == 3: # Check if all keys were found
            logger.info(f"Successfully parsed LLM output using regex fallback for: {original_generated_text[:50]}...")
            return extracted_data
        else:
            missing_keys = set(patterns.keys()) - set(extracted_data.keys())
            logger.warning(f"Regex fallback failed to find all keys. Missing: {missing_keys}. Output: {original_generated_text[:100]}...")
            return None

    except Exception as regex_exc: # Catch any unexpected regex errors
        logger.error(f"Error during regex fallback: {regex_exc}. Output: {original_generated_text[:100]}...")
        return None

    logger.warning(f"Failed to parse LLM output as JSON and regex fallback also failed. Output: {original_generated_text[:100]}...")
    return None

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
    model_identifier: str = "model",
    gpu_memory_utilization: float = 0.9,
    top_p_value: float = 0.95
) -> List[Dict[str, Any]]:
    """
    Runs the Chain-of-Thought (CoT) translation experiment using a pre-initialized LLM.
    Args:
        llm: Pre-initialized vLLM LLM instance.
        tokenizer: Pre-initialized tokenizer from the LLM instance.
        model_name_to_run: Name or path of the HuggingFace model (for logging).
        num_samples_to_process: Number of items from dataset or "all".
        dataset_path: Path to the dataset JSON file.
        prompts_path: Path to prompts JSON file (not currently used for main CoT prompts).
        max_model_len: Max model sequence length (for non-LLM config if any).
        temperature: Sampling temperature.
        max_tokens: Max tokens for generation.
        output_dir: Directory for output JSON.
        model_identifier: Short model name for file naming.
        gpu_memory_utilization: GPU memory utilization (LLM is already configured with this from main).
        top_p_value: Nucleus sampling top_p value.

    Returns:
        List of dictionaries with original text, raw output, and parsed/validated translation.
    """
    experiment_type = "cot"
    logger.info(f"Starting {experiment_type} experiment for model: {model_name_to_run} using provided LLM instance.")

    # Using internally defined prompts for CoT
    system_prompt_to_use = DEFAULT_SYSTEM_PROMPT
    few_shot_examples_to_use = DEFAULT_FEW_SHOT_EXAMPLES
    final_instruction_to_use = DEFAULT_FINAL_INSTRUCTION

    logger.info(f"Loading dataset from: {dataset_path}")
    dataset = load_json_data(dataset_path)
    if not isinstance(dataset, list):
        logger.error("Dataset must be a JSON array. Exiting.")
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

    logger.info(f"Processing {len(items_to_process)} items from the dataset for {experiment_type}.")
    if not items_to_process:
        logger.info("No items to process. Exiting experiment.")
        return []

    vllm_prompts = []
    original_english_texts = [] # To keep track of original texts corresponding to prompts
    item_ids = [] # To keep track of item IDs

    for item in items_to_process:
        english_text = item.get("text")
        if not english_text:
            logger.warning(f"Item ID {item.get('id', 'N/A')} missing 'text'. Skipping.")
            continue
        
        original_english_texts.append(english_text)
        item_ids.append(item.get("id", f"item_{len(original_english_texts)-1}"))
        try:
            formatted_prompt = create_cot_chat_prompt(
                english_text, 
                tokenizer, 
                system_prompt_to_use, 
                few_shot_examples_to_use,
                final_instruction_to_use
            )
            vllm_prompts.append(formatted_prompt)
        except Exception as e:
            logger.error(f"Failed to create {experiment_type} prompt for '{english_text[:50]}...': {e}")
            # Remove corresponding original_text and id if prompt creation failed
            original_english_texts.pop()
            item_ids.pop()
            continue
    
    if not vllm_prompts:
        logger.error(f"No valid {experiment_type} prompts created. Exiting.")
        return []

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p_value, # Use the specific top_p for CoT
        max_tokens=max_tokens,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id else [],
        skip_special_tokens=True 
    )

    logger.info(f"Generating {experiment_type} outputs for {len(vllm_prompts)} prompts with {model_identifier}...")
    llm_outputs = llm.generate(vllm_prompts, sampling_params)
    logger.info(f"{experiment_type} generation complete.")

    results_to_save = []
    for i, output_obj in enumerate(llm_outputs):
        item_id = item_ids[i]
        original_text = original_english_texts[i]
        raw_generated_output = output_obj.outputs[0].text.strip() if output_obj.outputs else "Error: No output generated"
        
        current_result = {
            "id": item_id,
            "original_text": original_text,
            "raw_generated_output": raw_generated_output,
            "parsed_translation_response": None,
            "parsing_error": None
        }

        parsed_json = parse_llm_output_to_json(raw_generated_output)
        if parsed_json:
            try:
                validated_response = TranslationResponse.model_validate(parsed_json) # Changed from model_validate_json
                current_result["parsed_translation_response"] = validated_response.model_dump()
            except ValidationError as e:
                logger.warning(f"Pydantic validation error for item {item_id}: {e}. Raw JSON: {parsed_json}")
                current_result["parsing_error"] = str(e)
            except Exception as e:
                logger.error(f"Unexpected error during Pydantic validation for item {item_id}: {e}")
                current_result["parsing_error"] = f"Unexpected validation error: {str(e)}"
        else:
            current_result["parsing_error"] = "Failed to parse LLM output as JSON or JSON was malformed."
            logger.warning(f"Could not parse LLM output to JSON for item {item_id}. Raw output: {raw_generated_output[:100]}...")

        results_to_save.append(current_result)

    output_file_name = f"{model_identifier}-{experiment_type}.json"
    output_file_path = output_dir / output_file_name
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, indent=4, ensure_ascii=False)
        logger.info(f"Saved {experiment_type} results for {model_identifier} to {output_file_path}")
    except IOError as e:
        logger.error(f"Error saving {experiment_type} results to {output_file_path}: {e}")

    logger.info(f"{experiment_type.capitalize()} experiment for {model_identifier} finished. {len(results_to_save)} items processed.")
    return results_to_save 