---
license: cc-by-4.0
task_categories:
- translation
language:
- de
- en
size_categories:
- n<1K
tags:
- gender-fair-translation
- machine-translation
- multi-reference
---
# GLITTER


GLITTER (Gender-Fair Language in German Machine Translation) is a multi-sentence, multi-reference benchmark designed to evaluate machine translation systems' ability to produce gender-fair German translations from English sources.

![GLITTER Process](glitter.png)

### Details of the Dataset

| Field                                    | Type                       | Description                                                                                                                                              |
| ---------------------------------------- | -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `id`                                     | integer                    | Unique row identifier.                                                                                                                                   |
| `seed`                                   | string                     | English seed noun used to find/build the passage (often an occupation or human group).                                                                   |
| `preceding_context`                      | text                       | Two sentences before the focal sentence; carries intra-/extra-sentential cues.                                                                           |
| `matching_sentence`                      | text                       | Focal sentence containing the seed occurrence.                                                                                                           |
| `trailing_context`                       | text                       | One sentence after the focal sentence; may contain late disambiguation cues.                                                                             |
| `type`                                   | categorical                | Source type of the English passage (e.g., `natural`, `synthetic`).                                                                                       |
| `ambiguity`                              | categorical                | Whether the seed’s referential gender is determined by context (`ambiguous`, `unambiguous_male`, `unambiguous_female`, `unambiguous_all`).               |
| `contextual_cue`                         | categorical / nullable     | Location of the disambiguating cue when unambiguous (`preceding`, `matching`, `trailing`; empty if `ambiguous`).                                         |
| `queer_related`                          | categorical (yes/no)       | Flags passages from LGBTQ+-tagged topics.                                                                                                                |
| `translation`                            | text (DE)                  | German hypothesis from the baseline MT system (pre–post-edit).                                                                                           |
| `vesuvius_human_annotations_finegrained` | categorical                | Human label of how the **seed** was rendered in `translation` (e.g., gendered/neutral categories).                                                       |
| `EuroLLM_human_annotations_finegrained`  | categorical                | Parallel fine-grained label from an LLM/hybrid judging setup.                                                                                            |
| `gender-star_PE`                         | text (DE) / nullable       | Professionally post-edited German reference using typographical inclusion (Gendersternchen `*`).                                                         |
| `ens_PE`                                 | text (DE) / nullable       | Post-edited reference using **ens-forms** (inclusive neomorphemes).                                                                                      |
| `neutral_PE`                             | text (DE) / nullable       | Post-edited **gender-neutral rewording** reference.                                                                                                      |
| `extra_PE`                               | number / nullable          | Optional slot for extra post-edit metadata.                                                                                                              |
| `seed-gender_PE`                         | text (DE) / nullable       | Post-edit variant specifically targeting the seed term’s gender rendering.                                                                               |
| `human_entities_span`                    | JSON (list of dicts)       | Character spans over the German text marking human entity mentions relevant to gender analysis. Keys typically include `start`, `end`, `text`, `labels`. |
| `disambiguating_t_span`                  | JSON / nullable            | Spans for **trailing**-context cues that disambiguate the seed’s gender.                                                                                 |
| `disambiguating_m_span`                  | JSON / nullable            | Spans for **matching**-sentence cues.                                                                                                                    |
| `disambiguating_p_span`                  | JSON / nullable            | Spans for **preceding**-context cues.                                                                                                                    |
| `post-edits`                             | JSON (list of strings, DE) | Array of one or more concrete post-edited outputs used to build references.                                                                              |
| `comments`                               | text / nullable            | Curator/translator notes, rationale, or edge cases.                                                                                                      |


### Supported Tasks

- **Translation**: English to German translation with focus on gender-fair language
- **Evaluation**: Benchmarking MT systems on gender-fair translation capabilities

### Languages

- Source: English (en)
- Target: German (de)

## Dataset Details

### Dataset Description

GLITTER provides comprehensive coverage of gender phenomena through:
- **Extended context**: Four-sentence passages (vs. single sentences in existing benchmarks)
- **Multiple reference translations**: Three gender-fair alternatives per source
  - 565 neutral-rewording forms
  - 562 gender-star (*) forms  
  - 545 ens-forms
- **Diverse scenarios**: Ambiguous and unambiguous gender contexts with disambiguation cues in different locations (preceding, matching, trailing sentences)
- **Professional curation**: All references professionally translated and post-edited by expert translators

### Dataset Composition

- **Total pairs**: 2,010 EN-DE parallel pairs
- **Unique English sources**: 980
- **Source types**: 
  - Natural (567): Wikipedia extracts with quotations and direct speech
  - Synthetic (430): GPT-4o-generated passages post-edited for quality
- **Gender phenomena coverage**:
  - Ambiguous cases (505): No gender cues in context
  - Unambiguous cases (495): Male (219), Female (189), All genders (87)
  - Queer-related content (112 instances) for representation of LGBTQ+ contexts

### Funded by

- European Association for Machine Translation (EAMT) - sponsorship of activities in 2024
- The Research Foundation – Flanders (FWO), research project 1SH5V24N
- VSC (Flemish Supercomputer Center), funded by Ghent University, FWO and Flemish Government
- Portuguese Recovery and Resilience Plan (projects C645008882-00000055 and UID/50008)

### Language(s)

- Source: English
- Target: German (with gender-fair language strategies)

### License

Creative Commons Attribution 4.0 International (cc-by-4.0)

## Uses

### Direct Use

This dataset is designed for:
- Evaluating machine translation systems on gender-fair German translation
- Training or fine-tuning MT models to produce gender-inclusive translations
- Research on gender phenomena in neural machine translation
- Analyzing MT systems' robustness to different contextual scenarios
- Benchmarking automatic evaluation metrics for gender-fair translation

### Out-of-Scope Use

- Translation for language pairs other than English-German
- Single-sentence translation evaluation (dataset emphasizes multi-sentence context)
- General-purpose German translation without consideration of gender-fair language

## Dataset Structure

Each instance contains:
- **English source**: Four-sentence passage with gender-ambiguous plural noun (seed word)
- **German references**: Up to three professionally post-edited gender-fair alternatives
- **Annotations**: 
  - Gender ambiguity type (ambiguous/unambiguous)
  - For unambiguous cases: gender of referent (male/female/all)
  - Location of disambiguation cues (preceding/matching/trailing sentence)
  - Source type (natural from Wikipedia/synthetic)
  - Queer-related content flag

The median length of source passages is 71 words, significantly longer than existing benchmarks (GeNTE: 26 words, GenderQueer: 15 words).

## Dataset Creation

### Curation Rationale

Existing gender-fair translation benchmarks have two main limitations:
1. Limited context beyond sentence-level, preventing analysis of extra-sentential gender phenomena
2. Insufficient diversity of gender-fair alternatives (typically only one strategy)

GLITTER addresses these gaps by providing multi-sentence passages with professionally curated references using three distinct gender-fair strategies, enabling comprehensive evaluation of MT systems' gender-fair translation capabilities.

### Source Data

#### Data Collection and Processing

**Wikipedia Collection:**
- Source: Wikipedia English (version 20231101.en, "train" split)
- Seed words: 115 gender-ambiguous plural nouns (e.g., "deputies", "participants")
- Extraction: Four-sentence passages (two preceding, one matching, one trailing sentence)
- Filtering: Quality checks for POS tagging, word counts, and presence of gendered referential expressions

**Synthetic Augmentation:**
- Generator: GPT-4o (gpt-4o-2024-08-06)
- Purpose: Address underrepresented scenarios (especially trailing context disambiguation)
- Post-editing: Manual correction to improve quality, diversity, and balanced gender representation
- Style: Wikipedia-like passages

**Queer Content:**
- Source: Wikipedia pages tagged with LGBTQ+ topics
- Count: 112 instances (56 ambiguous, 56 unambiguous)
- Purpose: Ensure representation and assess MT handling of gender in queer contexts

#### Who are the source data producers?

- **Natural data**: Wikipedia contributors (various authors providing lexical, stylistic, and content variation)
- **Synthetic data**: GPT-4o with human post-editing by the research team
- **Queer content**: Wikipedia LGBTQ+ topic contributors

### Annotations

#### Annotation Process

Professional translators with expertise in gender-fair German were hired to:
1. Translate English passages using Tower Vesuvius commercial MT system
2. Annotate translation hypotheses on gender aspects
3. Collect span-level annotations for disambiguation patterns
4. Post-edit passages to create three gender-fair reference variants:
   - Gender-neutral rewording (minimizing gender markers)
   - Gender star (*) typographical solution
   - Ens-forms (neologistic gender-inclusive forms)

Additional annotations by research team:
- Gender ambiguity classification
- Disambiguation cue location
- Gender form usage in MT outputs (manual analysis of 2,010 translations)

#### Who are the annotators?

- Four professional translators with expertise in gender-fair German

#### Personal and Sensitive Information

The dataset includes content from Wikipedia pages tagged with LGBTQ+ topics to ensure representation of queer contexts. This content is publicly available and does not contain personal identifying information. The dataset focuses on evaluating gender-fair language usage rather than individual identification.

## Bias, Risks, and Limitations

### Known Limitations

- **Language pair**: Limited to English-German translation only
- **Data sources**: Primarily Wikipedia (encyclopedic style) and synthetic data, which may not represent all text types

### Bias Considerations

- **Synthetic data bias**: LLM-generated passages initially defaulted to male references, requiring manual correction
- **Geographic representation**: German gender-fair language strategies may vary regionally; dataset uses practices common in Germany
- **Gender representation**: Dataset explicitly includes queer content but may not cover all gender identities comprehensively


## Citation

**BibTeX:**

```bibtex
@inproceedings{pranav-etal-2025-glitter,
    title = "GLITTER: A Multi-Sentence, Multi-Reference Benchmark for Gender-Fair German Machine Translation",
    author = "Pranav, A and Hackenbuchner, Jani\c{c}a and Attanasio, Giuseppe and Lardelli, Manuel and Lauscher, Anne",
    booktitle = "Findings of EMNLP",
    year = "2025",
    publisher = "Association for Computational Linguistics"
}
```

**APA:**

Pranav, A., Hackenbuchner, J., Attanasio, G., Lardelli, M., & Lauscher, A. (2025). GLITTER: A Multi-Sentence, Multi-Reference Benchmark for Gender-Fair German Machine Translation. In Proceedings of the Association for Computational Linguistics.

## Dataset Card Authors

- A Pranav (University of Hamburg, co-first author)
- Janiça Hackenbuchner (Ghent University, co-first author)
- Giuseppe Attanasio (Instituto de Telecomunicações)
- Manuel Lardelli (University of Padua)
- Anne Lauscher (University of Hamburg)
