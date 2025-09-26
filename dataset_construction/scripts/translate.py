import json
import os
from typing import Optional

import pandas as pd
import torch

# import deep_translator as dt
from dotenv import load_dotenv
import numpy as np
from utils import (
    DeeplTranslator,
    OpenWeightTranslator,
    OpenAITranslator,
    TowerTranslator,
)
import logging
import tyro

load_dotenv()


def get_translator(model_name_or_path: str):
    """Instantiate a translator for commercial systems."""

    if model_name_or_path == "google-translate":
        # return dt.GoogleTranslator(**args)
        pass
    elif model_name_or_path == "deepl":
        return DeeplTranslator()
    elif "gpt" in model_name_or_path:
        return OpenAITranslator(model_name_or_path)
    elif model_name_or_path in ["anthill", "sugarloaf", "vesuvius"]:
        return TowerTranslator(model_name_or_path)
        # return dt.DeeplTranslator(
        #     **args, api_key=os.environ.get("DEEPL_API_KEY"), use_free_api=True
        # )
    else:
        return OpenWeightTranslator(model_name_or_path)


def get_prompt_template(template_id: str):
    if template_id == "instruction":
        return 'Translate the following sentence into German. Reply only with the translation. Sentence: "{sentence}"'
    elif template_id == "tower":
        return "Translate the following text from English into German.\nEnglish: {sentence}\nGerman:"
    elif template_id == "flan":
        return "{sentence}\n\nTranslate this to German?"
    else:
        raise NotImplementedError()


# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main(
    dataset_file: str,
    model_name_or_path: str,
    output_file: str,
    dry_run: bool = False,
    prompt_template: str = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
):
    df = pd.read_csv(dataset_file, index_col="id", sep="\t")
    logger.info(f"Loaded dataset with {len(df)} rows.")

    if "translation" in df.columns:
        logger.warning("Dataset already contains a 'translation' column.")
        logger.warning("Filtering only rows with translations set to ERROR.")
        df = df.loc[df["translation"] == "ERROR"]
        logger.info(f"Filtered to {len(df)} rows.")

    if dry_run:
        df = df.head(100)

    input_texts = df.apply(
        lambda row: f"{row['preceding_context']} {row['matching_sentence']} {row['trailing_context']}",
        axis=1,
    )
    char_count = input_texts.apply(len).sum()

    if prompt_template is not None:  # format using the prompt template
        logger.info(f"Using prompt template: {prompt_template}")
        template_formatter = get_prompt_template(prompt_template)
        input_texts = [
            template_formatter.format(sentence=p) for p in input_texts.tolist()
        ]

    logger.info(f"Loaded {len(df)} rows with {char_count} characters.")
    logger.info(
        f"Average words per passage: {np.mean([len(t.split(' ')) for t in input_texts])}"
    )
    logger.info("Some input texts...")
    logger.info(input_texts[:3])

    #############
    # TRANSLATION
    #############
    # if "google-translate" in model_name_or_path or "deepl" in model_name_or_path:
    translator = get_translator(model_name_or_path)
    logger.info(f"Instantiated: {translator}")

    def _is_deepl_or_google():
        return "google" in model_name_or_path or "deepl" in model_name_or_path

    if not _is_deepl_or_google():
        kwargs = {"temperature": temperature, "max_tokens": max_tokens}
    else:
        kwargs = {"source_lang": "en", "target_lang": "de"}

    completions = translator.translate(input_texts, **kwargs)

    ##############################
    # POSTPROCESSING AND SAVING
    ##############################
    # basic post processing
    completions = [c.strip() for c in completions]

    # model_id = model_name_or_path.replace("/", "--")

    df["translation"] = completions
    df.to_csv(output_file, sep="\t")

    # df.to_csv(f"./results/translations_{model_id}.tsv")
    # df.to_json(f"./results/translations_{model_id}.json", orient="records", indent=2)


if __name__ == "__main__":
    tyro.cli(main)
