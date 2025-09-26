# Dataset Construction and Automatic Translation

Running the code within this section of the repository, you can replicate our filtering approach to select passages on Wikipedia, as well as the automatic translation experiments.

## Getting Started

Create a new python environment (we ran our code with Python 3.10 and 3.11) and run:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

> [!IMPORTANT]
> Each script will require minimal changes to adapt to your setup, e.g., correct input/output directories, etc. We used a SLURM-based HPC to run our experiments. Some bash script and organization require you to be in the same situation or minimal changes to be run on a standard workstation. If anything is not clear, please open an issue on this repository.

## 1. Passages Extraction From Wikipedia

Visit the `scripts` folder. Following the order given by filenames, the pipeline

- Extracts passages containing seed terms from Wikipedia 
- Validates terms using SpaCy POS tagging
- Captures surrounding context (2 sentences before, 1 after)
- Pre-annotates the passages extracted with the ambiguity label of the english seed
- Cleans and postprocess such automatic labels. 
- Samples a final set of passages that match the criteria. Maily, we 1) analyze gender correlations and attention patterns, 2) generate statistics, 3) balance the dataset across categories (ambiguous, unambiguous male/female/both), and 4) ensure LGBTQIA+ content representation and uniform seed word distribution
- Outputs the resulting output to a local jsonl file. The final output (`outputs/sampled_passages_[timestamp].tsv`) contains two passages per seed word, full context and attention scores, gendered word correlations and their locations, and additional metadata for analysis.

## 2. Translate the Passages

Use the `bash\translate.sh` runner script you can trigger the translation of the collected sample using the Vesuvius model through Tower's API.
The script also supports running other models, if you do not have access to a TransCreate API key (we conducted this experiments when the API was provided by Widn AI).

## 3. Use an LLM-Critic to Detect the Translation Gender

Use the script `bash\run_gpt4.1_critic_gender_label.sh` to run an automatic LLM-based evaluation of which German gender form was used in the translations.

> [!IMPORTANT]
> This bash script has been re-adapted several times to run the LLM-critic on several files resulting from the prompting analysis. As such, be sure to re-adapt input and output files in the script as it best fits your use case. In case of any doubts, do not esistate to open an issue on this repository. 