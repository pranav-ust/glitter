"""
This module implements an attention-based analysis pipeline for identifying and analyzing
gender bias in text using BERT attention mechanisms. It processes text documents to:
1. Identify passages containing specific seed words
2. Analyze attention patterns between seed words and gendered terms
3. Extract relevant context and metadata for further analysis

The pipeline uses HuggingFace's BERT model with explicit attention implementation
and SpaCy for text processing.
"""

import glob
import logging
import os
import re
import shutil
from os.path import join as ospj
from typing import List, Dict, Set, Generator, Optional
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

import fire
import spacy
import torch
from torch.cuda import is_available
from transformers import AutoTokenizer, AutoModel

from datatrove.data import Document, DocumentsPipeline
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import (
    ParquetWriter,
    JsonlWriter,
    HuggingFaceDatasetWriter,
)


# def ensure_directories() -> None:
#     """Create necessary directories and clean up existing outputs.

#     Creates 'logs', 'data', and 'output' directories if they don't exist.
#     Removes existing 'output' directory to ensure clean pipeline runs.
#     """
#     if os.path.exists("output"):
#         logging.info("Removing existing output directory")
#         shutil.rmtree("output")

#     directories = ["logs", "data", "output"]
#     for directory in directories:
#         os.makedirs(directory, exist_ok=True)
#         logging.info(f"Ensured directory exists: {directory}")


# # Create directories before setting up logging
# ensure_directories()

# Configure logging with both file and console handlers
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/find_unambiguous_passages.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)

# Set device for PyTorch operations
DEVICE = torch.device("cuda" if is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# Print a warning if the device is not CUDA
if DEVICE.type != "cuda":
    print("Warning: CUDA is not available. This script requires CUDA to run.")


def load_gendered_words() -> List[str]:
    """Load gendered words from text files in the biased-words directory.

    Returns:
        List[str]: A deduplicated list of gendered words loaded from all .txt files.
    """
    bias_files = glob.glob("data/gendered-words/*.txt")
    gendered_words: Set[str] = set()

    for file_path in bias_files:
        logger.debug(f"Loading gendered words from: {file_path}")
        with open(file_path) as f:
            words = {line.strip() for line in f if line.strip()}
            gendered_words.update(words)

    logger.info(f"Loaded {len(gendered_words)} unique gendered words")
    return list(gendered_words)


def split_into_sentences(text: str, nlp) -> List[str]:
    """Split text into sentences using spaCy's sentencizer and filter for quality.

    Applies the following heuristics to ensure sentence quality:
    - Minimum 10 words
    - Contains at least one verb
    - Not a list item (doesn't start with bullet points, numbers, etc.)
    - Not too short or too long

    Args:
        text (str): Input text to be split into sentences.
        nlp: Loaded spaCy model

    Returns:
        List[str]: List of cleaned, high-quality sentences.
    """
    doc = nlp(text)

    quality_sentences = []
    for sent in doc.sents:
        sent_text = str(sent).strip()

        # Skip if too short or empty
        if len(sent_text.split()) < 10:
            continue

        # Skip if starts with list markers
        if re.match(r"^[\*\-\•\d]+[\.\)\]]*\s", sent_text):
            continue

        # Check for at least one verb
        has_verb = False
        has_noun = False
        for token in sent:
            if token.pos_ in {"VERB", "AUX"}:
                has_verb = True
            if token.pos_ in {"NOUN", "PROPN"}:
                has_noun = True

        if not (has_verb and has_noun):
            continue

        quality_sentences.append(sent_text)

    return quality_sentences


def quality_check_fails(
    prev_context: str, matching_sentence: str, trailing_context: str, nlp
) -> bool:
    """Check if the sentence context meets quality criteria.

    Applies various heuristics to ensure high-quality, meaningful sentences.

    Args:
        prev_context (str): The preceding context
        matching_sentence (str): The main sentence being analyzed
        trailing_context (str): The following context
        nlp: Loaded spaCy model

    Returns:
        bool: True if any quality check fails, False if all checks pass
    """
    # Check each context section
    for text in [prev_context, matching_sentence, trailing_context]:
        if not text or not isinstance(text, str):
            return True

        doc = nlp(text)

        # Length checks
        word_count = len(text.split())
        if word_count < 10 or word_count > 300:  # Skip too short or too long
            return True

        # Character checks
        char_count = len(text)
        if char_count < 50 or char_count > 1500:  # Reasonable character length
            return True

        # List and formatting checks
        if re.match(r"^[\*\-\•\d]+[\.\)\]]*\s", text):  # Skip list items
            return True
        if text.count("\n") > 1:  # Skip multi-line text
            return True
        if text.count("|") > 0:  # Skip table-like content
            return True
        if re.search(r"={2,}", text):  # Skip wiki headers
            return True

        # Content checks
        has_verb = False
        has_noun = False
        has_proper_capitalization = text[0].isupper() if text else False
        has_proper_ending = text[-1] in {".", "!", "?"} if text else False

        for token in doc:
            if token.pos_ in {"VERB", "AUX"}:
                has_verb = True
            if token.pos_ in {"NOUN", "PROPN"}:
                has_noun = True

        # Skip if missing essential elements
        if not (
            has_verb and has_noun and has_proper_capitalization and has_proper_ending
        ):
            return True

        # Skip if contains unwanted patterns
        unwanted_patterns = [
            r"http[s]?://",  # URLs
            r"\[\[",  # Wiki markup
            r"\]\]",  # Wiki markup
            r"\{\{",  # Templates
            r"\}\}",  # Templates
            r"<[^>]+>",  # HTML tags
            r"\s{3,}",  # Excessive whitespace
            r"\([^\)]{100,}\)",  # Very long parentheticals
            r"^See also",  # Reference phrases
            r"^References",
            r"^Bibliography",
            r"^\s*Source:",
        ]

        for pattern in unwanted_patterns:
            if re.search(pattern, text):
                return True

        # Content ratio checks
        alpha_chars = sum(c.isalpha() for c in text)
        total_chars = len(text)
        if total_chars > 0:
            alpha_ratio = alpha_chars / total_chars
            if alpha_ratio < 0.7:  # Ensure text is mostly alphabetic
                return True

        # Check for balanced quotes and parentheses
        if text.count('"') % 2 != 0 or text.count("(") != text.count(")"):
            return True

    return False  # All checks passed


class PassageExtractor(PipelineStep):
    """Extract and analyze passages containing seed words with attention-based analysis.

    This class implements attention-based analysis to:
    1. Find passages containing seed words
    2. Analyze attention patterns between seed words and gendered terms
    3. Extract contextual information and metadata

    Attributes:
        name (str): Name identifier for the pipeline step
        model: BERT model for attention analysis
        tokenizer: Associated tokenizer for the BERT model
        nlp: SpaCy model for text processing
        gendered_words (List[str]): List of words to analyze for gender bias
    """

    name = "⛏ Passage Filter with Attention"

    def __init__(
        self,
        apply_quality_filters: bool = True,
        add_coref_attention_scores: bool = True,
    ):
        """Initialize the passage extractor with necessary models and word lists.

        Args:
            seed_file (str): Path to file containing seed words
        """
        super().__init__()
        logger.info("Initializing PassageFilter")
        self.apply_quality_filters = apply_quality_filters

        self.add_coref_attention_scores = add_coref_attention_scores
        if add_coref_attention_scores:
            # Initialize BERT model and tokenizer
            logger.info("Loading BERT model and tokenizer")
            self.model = AutoModel.from_pretrained(
                "nielsr/coref-bert-base", attn_implementation="eager"
            ).to(DEVICE)
            self.tokenizer = AutoTokenizer.from_pretrained("nielsr/coref-bert-base")
            self.gendered_words = load_gendered_words()

        # Load SpaCy model
        logger.info("Loading SpaCy model")
        self.nlp = spacy.load("en_core_web_sm")

    def get_word_correlations(
        self, text: str, seed_word: str, doc_metadata: dict
    ) -> list[dict]:
        """Calculate attention scores between seed word and gendered words."""
        doc = self.nlp(text)

        word_positions = []
        gendered_positions = []

        # Calculate section boundaries from the original document metadata
        preceding_len = len(doc_metadata.get("preceding_context", "").split())
        matching_len = len(doc_metadata.get("matching_sentence", "").split())

        for i, token in enumerate(doc):
            if token.text.lower() == seed_word.lower():
                word_positions.append(i)
            if token.text.lower() in self.gendered_words:
                # Determine location of gendered word
                location = "trailing"
                if i < preceding_len:
                    location = "preceding"
                elif i < preceding_len + matching_len:
                    location = "matching"
                gendered_positions.append((i, token.text, location))

        if not word_positions or not gendered_positions:
            return []

        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512, padding=True
        )

        if len(inputs["input_ids"][0]) == 512:
            decoded_text = self.tokenizer.decode(inputs["input_ids"][0])
            if any(pos >= len(decoded_text) for pos, _, _ in gendered_positions) or any(
                pos >= len(decoded_text) for pos in word_positions
            ):
                return []

        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)

        attention = outputs.attentions[-1].mean(dim=1).cpu()
        bert_tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        spacy_to_bert = {}
        current_pos = 0
        for i, token in enumerate(bert_tokens):
            if not token.startswith("##"):
                spacy_to_bert[current_pos] = i
                current_pos += 1

        results = []
        for word_pos in word_positions:
            word_bert_pos = spacy_to_bert.get(word_pos)
            if word_bert_pos is None:
                continue

            for gendered_pos, gendered_text, location in gendered_positions:
                gendered_bert_pos = spacy_to_bert.get(gendered_pos)
                if gendered_bert_pos is None:
                    continue

                score = attention[0, gendered_bert_pos, word_bert_pos].item()
                results.append(
                    {
                        "gendered_word": gendered_text,
                        "attention_score": score,
                        "word_pos": word_pos,
                        "gendered_pos": gendered_pos,
                        "word_pos_tag": doc[word_pos].pos_,
                        "distance": abs(word_pos - gendered_pos),
                        "gendered_location": location,
                    }
                )

        return results

    def run(
        self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1  # type: ignore
    ) -> DocumentsPipeline:  # type: ignore
        logging.info("Starting Passage Filter with Attention Analysis")

        # Known total documents
        # total_docs = 2553960
        # docs_per_worker = total_docs // world_size
        # logging.info(f"Processing {docs_per_worker} documents per worker")

        doc_count = 0
        match_count = 0

        for doc in data:
            doc_count += 1

            # Get document components
            main_id = doc.id
            seed = doc.metadata["seed"]
            prev_context = doc.metadata["preceding_context"]
            trailing_context = doc.metadata["trailing_context"]
            matching_sentence = doc.metadata["matching_sentence"]

            # Quality check the contexts
            if self.apply_quality_filters:
                if quality_check_fails(
                    prev_context, matching_sentence, trailing_context, self.nlp
                ):
                    continue

            if self.add_coref_attention_scores:
                # Get attention correlations
                query_text = f"{prev_context} {matching_sentence} {trailing_context}"
                correlations = self.get_word_correlations(
                    query_text, seed, doc.metadata
                )

                if correlations:
                    match_count += 1
                    logging.debug(
                        f"Found match with seed {seed} and {len(correlations)} correlations"
                    )

                    # Find correlation with maximum attention score
                    max_correlation = max(
                        correlations, key=lambda x: x["attention_score"]
                    )

                    nd = Document(
                        id=f"{main_id}",
                        text=doc.text,
                        metadata={
                            "seed": seed,
                            "preceding_context": prev_context,
                            "trailing_context": trailing_context,
                            "matching_sentence": matching_sentence,
                            "correlations": correlations,
                            "max_attention_score": max_correlation["attention_score"],
                            "max_found_at": max_correlation["gendered_location"],
                        },
                    )
                    yield nd

        logging.info(
            f"Finished Passage Extractor. Processed {doc_count} documents, found {match_count} matches"
        )


def main(
    n_tasks: int = 4,
    n_workers: int = 4,
    dry_run: bool = False,
    output_dir: str = "~/myscratch/german-gnt",
) -> None:
    """Run the gender bias analysis pipeline.

    Args:
        n_tasks (int): Number of parallel tasks to run
        n_workers (int): Number of worker processes
        dry_run (bool): If True, runs on limited dataset for testing
        output_dir (str): Directory for output files
    """
    logger.info(f"Starting pipeline with {n_tasks} tasks and {n_workers} workers")
    logger.info(f"Dry run: {dry_run}, Output directory: {output_dir}")

    # Define pipeline steps
    pipeline = [
        HuggingFaceDatasetReader(
            dataset="Bainbridge/filtered_wikipedia_gnt-v2",
            dataset_options={"split": "train"},
            limit=4000 if dry_run else -1,
            doc_progress=True,
        ),
        PassageExtractor(),
        # ParquetWriter(output_folder=ospj(output_dir, "data")),
        # JsonlWriter(output_folder=ospj(output_dir, "data"), compression=None),
        HuggingFaceDatasetWriter(
            dataset="Bainbridge/wikipedia_gnt_v2",
            output_filename="qfilters_and_gwords/${rank}.parquet",
            private=False,
            local_working_dir=ospj(output_dir, "samples-v3-gendered_words"),
        ),
    ]

    # Initialize and run pipeline
    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        logging_dir=ospj(output_dir, "logs_unambiguous"),
        tasks=n_tasks,
        workers=n_workers,
    )

    logger.info("Starting pipeline execution")
    executor.run()
    logger.info("Pipeline execution completed")


if __name__ == "__main__":
    fire.Fire(main)
