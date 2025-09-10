# Synthetic Data Creation

Complementary to the naturally sampled data, we prompted GPT-4o `gpt-4o-2024-11-20` to generate synthetic passages.

GPT-4o struggled in generating requested samples, often not disambiguating the gender at all or solely within in the matching sentence. 
We, therefore, compiled our synthetic data based on a combination of a few different few-shot prompts, an example of which is shown in the file 'prompt'.
The prompts followed the same overall pattern and ensured that we cover a range of linguistic features (ambiguous vs. unambiguous, for which the gender was either disambiguated in preceding, matching or trailing context) 
and gender phenomena (female, male or inclusive) as needed for this dataset.

As LLM-generated passages predominantly defaulted to male gender references and intra-sentential disambiguation, we conducted **light post-editing** to improve quality and diversity.
Through targeted editing, we ensure balanced representation of disambiguation locations (preceding, matching, and trailing sentences) and diverse gender references (female, inclusive, masculine and feminine)
