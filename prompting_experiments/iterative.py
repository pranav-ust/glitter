from vllm import LLM, SamplingParams
import json # Added for output
import logging # Added for logging
from typing import Dict, List, Any, Union, Tuple # Added for typing
from pathlib import Path # Added for output path handling
from tqdm import tqdm

logger = logging.getLogger(__name__)

class IterativeGenderFairTranslator: # Renamed for clarity
    def __init__(self, llm: LLM, tokenizer: Any, model_name_for_logging: str = "Unknown Model"):
        """Initialize with LLM and tokenizer (potentially less state needed)."""
        logger.info(f"Initializing translator utilities for {model_name_for_logging}")
        self.llm = llm # Keep LLM reference if batch generation happens within class
        self.tokenizer = tokenizer

    # Static method for prompt formatting might be cleaner for batching
    @staticmethod
    def format_prompt_with_history(tokenizer: Any, chat_history: List[Dict[str, str]]) -> str:
        """Format chat history using the tokenizer's chat template."""
        try:
            # Important: Ensure add_generation_prompt=True is appropriate for the model.
            # Some models might expect the prompt to end differently.
            return tokenizer.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            logger.error(f"Error applying chat template: {e}. Check tokenizer compatibility.")
            # Re-raise or return an error indicator depending on desired handling
            raise


    # --- Methods below are specific turn prompt builders ---

    @staticmethod
    def build_turn1_initial_prompt(text_to_translate: str) -> List[Dict[str, str]]:
        """Builds the chat history for the initial translation prompt."""
        initial_prompt_text = f"""
Translate the following English passage into German. For example:

English text: The ascent challenges both skill and endurance. Experienced mountaineers undertook the task despite the risks involved.
German translation: Der Aufstieg stellt sowohl Geschick als auch Ausdauer auf die Probe. Erfahrene Bergsteiger unternahmen die Aufgabe trotz der damit verbundenen Risiken.

Now, translate the following English passage into German. Provide only the translation, no other text.
English text: {text_to_translate}"""
        return [{"role": "user", "content": initial_prompt_text}]

    @staticmethod
    def build_turn2_reasoning_prompt(text_to_translate: str, initial_translation: str, turn1_history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Builds the chat history for the reasoning prompt, including previous turns."""
        reasoning_prompt_text = f"""Consider the English passage: '{text_to_translate}' and your German translation: '{initial_translation}'.

In the English passage, identify any words or phrases (seed words) that might require gender-fair language considerations when translated into German. For example, "mountaineers" could be translated to "Bergsteiger" (masculine), "Bergsteiger*innen" (gender star), "Bergsteigens" (ens-form), or "Bergsteigende" (neutral participle).

For your identified seed words from the original English text, explain your reasoning for the gender-fair choices or why a particular form was chosen in your initial translation, and list potential gender-fair alternatives if applicable.
Structure your response clearly. Respond in under 150 words."""

        # Append the assistant's response from turn 1 and the new user prompt
        turn2_history = turn1_history + [
            {"role": "assistant", "content": initial_translation},
            {"role": "user", "content": reasoning_prompt_text}
        ]
        return turn2_history

    @staticmethod
    def build_turn3_final_translation_prompt(text_to_translate: str, reasoning_and_alternatives: str, turn2_history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Builds the chat history for the final translation prompt."""
        final_translation_prompt_text = f"""Based on the original English passage '{text_to_translate}', your initial translation, and your reasoning about gender-fair alternatives: '{reasoning_and_alternatives}'.

Now, provide the final, revised German translation that incorporates the most appropriate gender-fair language. For example, if "Bergsteiger" was identified and "Bergsteiger*innen" was chosen as a better alternative, use that in the final translation.

Respond only with the complete final German translation, no other text."""

        # Append the assistant's response from turn 2 and the new user prompt
        turn3_history = turn2_history + [
            {"role": "assistant", "content": reasoning_and_alternatives},
            {"role": "user", "content": final_translation_prompt_text}
        ]
        return turn3_history

    # --- Original iterative method is removed as logic moves to run_experiment ---
    # def run_iterative_translation(self, ...) -> ...:
    #     ... (Removed) ...

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

def run_experiment(
    llm: LLM, # Added pre-initialized LLM
    tokenizer: Any, # Added pre-initialized tokenizer
    model_name_to_run: str, # Kept for logging
    num_samples_to_process: Union[int, str],
    dataset_path: str = "dataset.json",
    prompts_path: str = "prompts.json", # Not used directly here
    temperature: float = 0.2,
    max_tokens: int = 1024,
    output_dir: Path = Path("outputs"),
    model_identifier: str = "model",
    gpu_memory_utilization: float = 0.9, # Not used for LLM init here
    max_model_len: int = 4096 # Not used for LLM init here
) -> List[Dict[str, Any]]:
    """
    Runs the Iterative translation experiment using batch processing per turn.

    Args:
        llm: Pre-initialized vLLM LLM instance.
        tokenizer: Pre-initialized tokenizer from the LLM instance.
        model_name_to_run: Name or path of the HuggingFace model (for logging).
        num_samples_to_process: Number of items from the dataset to process or "all".
        dataset_path: Path to the dataset JSON file.
        prompts_path: Path to the prompts JSON file (currently unused).
        temperature: Sampling temperature.
        max_tokens: Maximum number of tokens to generate per turn.
        output_dir: Directory to save the output JSON file.
        model_identifier: Short name for the model (for file naming).
        gpu_memory_utilization: GPU memory utilization (LLM already configured).
        max_model_len: Max model length (passed from main, not used for LLM init here).

    Returns:
        A list of dictionaries, each containing iterative translation details.
    """
    experiment_type = "iterative" # Changed identifier
    logger.info(f"Starting {experiment_type} experiment for model: {model_name_to_run} using provided LLM and tokenizer.")

    # We don't strictly need the translator instance if we use static methods,
    # but keep it for now if we need LLM/tokenizer access easily.
    # translator = IterativeGenderFairTranslator(llm, tokenizer, model_name_to_run)

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

    logger.info(f"Preparing {len(items_to_process)} items for {experiment_type} with {model_identifier}.")
    if not items_to_process:
        logger.info("No items to process. Exiting experiment.")
        return []

    # --- Store intermediate results ---
    # Use item ID as key for easier recombination later
    intermediate_results: Dict[str, Dict[str, Any]] = {
        item.get("id", f"item_{i}"): {"original_text": item.get("text"), "error": None}
        for i, item in enumerate(items_to_process)
    }
    item_ids_in_order = [item.get("id", f"item_{i}") for i, item in enumerate(items_to_process)] # Maintain order

    # --- Sampling Params ---
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        # Consider skip_special_tokens=True if tokenizer adds them and they interfere
    )

    # === Turn 1: Initial Translation (Batch) ===
    logger.info("Starting Turn 1: Initial Translation (Batch)...")
    turn1_prompts_formatted = []
    turn1_histories = {} # Store history for next turn {item_id: history_list}
    valid_ids_turn1 = [] # Keep track of items without errors so far

    for item_id in item_ids_in_order:
        original_text = intermediate_results[item_id].get("original_text")
        if not original_text:
            logger.warning(f"Item ID {item_id} missing 'text'. Skipping.")
            intermediate_results[item_id]["error"] = "Missing 'text' field"
            continue

        try:
            chat_history = IterativeGenderFairTranslator.build_turn1_initial_prompt(original_text)
            formatted_prompt = IterativeGenderFairTranslator.format_prompt_with_history(tokenizer, chat_history)
            turn1_prompts_formatted.append(formatted_prompt)
            turn1_histories[item_id] = chat_history # Save history for Turn 2
            valid_ids_turn1.append(item_id) # Mark as valid for this turn's batch
        except Exception as e:
            logger.error(f"Error preparing Turn 1 prompt for item {item_id}: {e}")
            intermediate_results[item_id]["error"] = f"Turn 1 Prompt Error: {e}"

    if not turn1_prompts_formatted:
        logger.warning("No valid prompts generated for Turn 1. Ending experiment.")
        # Combine results (which will mostly contain errors) before returning
    else:
        try:
            logger.info(f"Generating {len(turn1_prompts_formatted)} initial translations...")
            outputs_turn1 = llm.generate(prompts=turn1_prompts_formatted, sampling_params=sampling_params)
            logger.info("Turn 1 generation complete.")

            # Process outputs_turn1 and store results
            for i, item_id in enumerate(valid_ids_turn1):
                 # Check if the output corresponds to the input
                 # Assuming outputs_turn1[i].prompt == turn1_prompts_formatted[i]
                 # Or rely on the order being preserved.
                 if i < len(outputs_turn1):
                     # TODO: Add check for generation errors if vLLM provides them per output
                     translation = outputs_turn1[i].outputs[0].text.strip()
                     intermediate_results[item_id]["initial_translation"] = translation
                 else:
                     logger.error(f"Mismatch in Turn 1 output length for item {item_id}. Expected {len(valid_ids_turn1)} outputs, got {len(outputs_turn1)}.")
                     intermediate_results[item_id]["error"] = "Turn 1 Generation Error: Output mismatch"
                     # Remove from histories if generation failed?
                     if item_id in turn1_histories: del turn1_histories[item_id]


        except Exception as e:
            logger.error(f"Error during vLLM generation in Turn 1: {e}", exc_info=True)
            # Mark all items in this batch as failed for this turn
            for item_id in valid_ids_turn1:
                 if intermediate_results[item_id].get("error") is None: # Don't overwrite previous errors
                    intermediate_results[item_id]["error"] = f"Turn 1 Batch Generation Error: {e}"
            turn1_histories = {} # Clear histories as Turn 1 failed


    # === Turn 2: Reasoning (Batch) ===
    logger.info("Starting Turn 2: Reasoning (Batch)...")
    turn2_prompts_formatted = []
    turn2_histories = {} # Store history for next turn {item_id: history_list}
    valid_ids_turn2 = []

    # Only process items that succeeded or didn't error out before Turn 1 generation AND have history
    items_for_turn2 = [item_id for item_id in valid_ids_turn1 if item_id in turn1_histories and intermediate_results[item_id].get("error") is None]

    for item_id in items_for_turn2:
        original_text = intermediate_results[item_id]["original_text"]
        initial_translation = intermediate_results[item_id].get("initial_translation")
        turn1_history = turn1_histories[item_id]

        if initial_translation is None: # Should have been caught by error check, but belts and suspenders
            logger.warning(f"Item {item_id} missing initial translation for Turn 2. Skipping.")
            intermediate_results[item_id]["error"] = "Turn 2 Pre-check Error: Missing initial translation"
            continue

        try:
            chat_history = IterativeGenderFairTranslator.build_turn2_reasoning_prompt(
                original_text, initial_translation, turn1_history
            )
            formatted_prompt = IterativeGenderFairTranslator.format_prompt_with_history(tokenizer, chat_history)
            turn2_prompts_formatted.append(formatted_prompt)
            turn2_histories[item_id] = chat_history # Save history for Turn 3
            valid_ids_turn2.append(item_id)
        except Exception as e:
            logger.error(f"Error preparing Turn 2 prompt for item {item_id}: {e}")
            intermediate_results[item_id]["error"] = f"Turn 2 Prompt Error: {e}"
            if item_id in turn1_histories: del turn1_histories[item_id] # Ensure inconsistent state isn't used

    if not turn2_prompts_formatted:
        logger.warning("No valid prompts generated for Turn 2.")
    else:
        try:
            logger.info(f"Generating {len(turn2_prompts_formatted)} reasoning responses...")
            # Adjust max_tokens if reasoning needs fewer tokens? sampling_params_turn2?
            outputs_turn2 = llm.generate(prompts=turn2_prompts_formatted, sampling_params=sampling_params)
            logger.info("Turn 2 generation complete.")

            for i, item_id in enumerate(valid_ids_turn2):
                 if i < len(outputs_turn2):
                     reasoning = outputs_turn2[i].outputs[0].text.strip()
                     intermediate_results[item_id]["reasoning_and_alternatives"] = reasoning
                 else:
                     logger.error(f"Mismatch in Turn 2 output length for item {item_id}.")
                     intermediate_results[item_id]["error"] = "Turn 2 Generation Error: Output mismatch"
                     if item_id in turn2_histories: del turn2_histories[item_id]

        except Exception as e:
            logger.error(f"Error during vLLM generation in Turn 2: {e}", exc_info=True)
            for item_id in valid_ids_turn2:
                 if intermediate_results[item_id].get("error") is None:
                    intermediate_results[item_id]["error"] = f"Turn 2 Batch Generation Error: {e}"
            turn2_histories = {}

    # === Turn 3: Final Translation (Batch) ===
    logger.info("Starting Turn 3: Final Translation (Batch)...")
    turn3_prompts_formatted = []
    valid_ids_turn3 = [] # IDs successfully processed up to start of turn 3 batch

    items_for_turn3 = [item_id for item_id in valid_ids_turn2 if item_id in turn2_histories and intermediate_results[item_id].get("error") is None]

    for item_id in items_for_turn3:
        original_text = intermediate_results[item_id]["original_text"]
        reasoning = intermediate_results[item_id].get("reasoning_and_alternatives")
        turn2_history = turn2_histories[item_id]

        if reasoning is None:
            logger.warning(f"Item {item_id} missing reasoning for Turn 3. Skipping.")
            intermediate_results[item_id]["error"] = "Turn 3 Pre-check Error: Missing reasoning"
            continue

        try:
            chat_history = IterativeGenderFairTranslator.build_turn3_final_translation_prompt(
                original_text, reasoning, turn2_history
            )
            formatted_prompt = IterativeGenderFairTranslator.format_prompt_with_history(tokenizer, chat_history)
            turn3_prompts_formatted.append(formatted_prompt)
            # No need to save history after turn 3
            valid_ids_turn3.append(item_id)
        except Exception as e:
            logger.error(f"Error preparing Turn 3 prompt for item {item_id}: {e}")
            intermediate_results[item_id]["error"] = f"Turn 3 Prompt Error: {e}"
            # No history to delete here

    if not turn3_prompts_formatted:
        logger.warning("No valid prompts generated for Turn 3.")
    else:
        try:
            logger.info(f"Generating {len(turn3_prompts_formatted)} final translations...")
            outputs_turn3 = llm.generate(prompts=turn3_prompts_formatted, sampling_params=sampling_params)
            logger.info("Turn 3 generation complete.")

            for i, item_id in enumerate(valid_ids_turn3):
                if i < len(outputs_turn3):
                     final_translation = outputs_turn3[i].outputs[0].text.strip()
                     intermediate_results[item_id]["final_gender_fair_translation"] = final_translation
                else:
                     logger.error(f"Mismatch in Turn 3 output length for item {item_id}.")
                     intermediate_results[item_id]["error"] = "Turn 3 Generation Error: Output mismatch"

        except Exception as e:
            logger.error(f"Error during vLLM generation in Turn 3: {e}", exc_info=True)
            for item_id in valid_ids_turn3:
                 if intermediate_results[item_id].get("error") is None:
                     intermediate_results[item_id]["error"] = f"Turn 3 Batch Generation Error: {e}"

    # --- Combine results ---
    logger.info("Combining results from all turns...")
    results_to_save = []
    processed_count = 0
    for item_id in item_ids_in_order:
        result_data = intermediate_results[item_id]
        # Structure the final output per item
        final_item_output = {
            "id": item_id,
            "original_text": result_data.get("original_text"),
            "iterative_translation_details": { # Keep structure similar to original
                "initial_translation": result_data.get("initial_translation"),
                "reasoning_and_alternatives": result_data.get("reasoning_and_alternatives"),
                "final_gender_fair_translation": result_data.get("final_gender_fair_translation")
            },
             "error": result_data.get("error") # Add error field at top level
        }
        # Clean up details if error occurred preventing completion
        if result_data.get("error"):
            if final_item_output["iterative_translation_details"]["initial_translation"] is None and \
               final_item_output["iterative_translation_details"]["reasoning_and_alternatives"] is None and \
               final_item_output["iterative_translation_details"]["final_gender_fair_translation"] is None:
                final_item_output["iterative_translation_details"] = None # Set to None if completely failed
        else:
             processed_count +=1 # Count successfully processed items


        results_to_save.append(final_item_output)

    # --- Save results ---
    output_file_name = f"{model_identifier}-{experiment_type}.json"
    output_file_path = output_dir / output_file_name
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, indent=4, ensure_ascii=False)
        logger.info(f"Saved {experiment_type} results for {model_identifier} to {output_file_path}")
    except IOError as e:
        logger.error(f"Error saving {experiment_type} results to {output_file_path}: {e}")

    logger.info(f"{experiment_type.capitalize()} experiment for {model_identifier} finished. {processed_count}/{len(items_to_process)} items processed successfully.")
    return results_to_save

# Example of how you might call this from a main script (conceptual)
# if __name__ == "__main__":
#     # Setup logging
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#
#     # !! IMPORTANT !!: LLM and Tokenizer must be initialized *before* calling run_experiment
#     # This usually happens in your main script that orchestrates different experiments.
#     # Example placeholder:
#     model_name = "mistralai/Mistral-7B-Instruct-v0.2" # Or your actual model
#     logger.info(f"Loading LLM and tokenizer for {model_name}...")
#     # Adjust parameters as needed (tensor_parallel_size, gpu_memory_utilization, etc.)
#     llm_instance = LLM(model=model_name, trust_remote_code=True, gpu_memory_utilization=0.9, max_model_len=4096)
#     tokenizer_instance = llm_instance.get_tokenizer()
#     logger.info("LLM and tokenizer loaded.")
#
#     # Define experiment parameters
#     output_directory = Path("./experiment_outputs")
#     model_short_name = "mistral-7b-instruct" # For filenames
#     num_samples = 3 # Process first 3 samples for testing
#     dataset_file = "path/to/your/dataset.json" # Ensure this exists
#
#     # Run the batched experiment
#     run_experiment(
#         llm=llm_instance,
#         tokenizer=tokenizer_instance,
#         model_name_to_run=model_name,
#         num_samples_to_process=num_samples,
#         dataset_path=dataset_file,
#         # prompts_path="prompts.json", # Optional if needed elsewhere
#         temperature=0.1,
#         max_tokens=512, # Adjust max tokens per turn if needed
#         output_dir=output_directory,
#         model_identifier=model_short_name
#         # gpu_memory_utilization, max_model_len passed implicitly via llm_instance
#     )
#     logger.info("Batched iterative experiment finished.")