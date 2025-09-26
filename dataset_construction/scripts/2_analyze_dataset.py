from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re
import random

def analyze_dataset():
    # Load the dataset
    print("Loading dataset...")
    ds = load_dataset("Bainbridge/wikipedia_gnt_v2", "qfilters_and_gwords_labeled")
    
    # Convert to pandas DataFrame for easier analysis
    df = pd.DataFrame(ds['train'])
    
    print("\nDataset Overview:")
    print(f"Number of samples: {len(df)}")

    def extract_final_label(text):
        """Extract the final label decision from the text using regex."""
        # Convert to lowercase for consistent matching
        text = text.lower()
        
        # Look for corrected output or final decision markers
        final_markers = [
            r"\*\*corrected output:\*\*(.*?)(?=\n|$)",
            r"(?:therefore|thus|hence|in conclusion|final decision|final output|corrected to).*?(unambiguous|ambiguous).*?(?=\n|$)",
            r"(?:^|\n)(?!.*(?:however|but|although)).*?(unambiguous|ambiguous)[^.]*?(?:\.|$)",  # Last instance not followed by contrasting words
        ]
        
        # Try to find the final decision first
        for pattern in final_markers:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            final_matches = list(matches)
            if final_matches:
                final_text = final_matches[-1].group(0)  # Take the last match
                
                # Now extract the specific label type
                if 'ambiguous' in final_text.lower() and '(' not in final_text.lower():
                    return 'ambiguous'
                elif 'unambiguous' in final_text.lower():
                    if '(male)' in final_text.lower():
                        return 'unambiguous_male'
                    elif '(female)' in final_text.lower():
                        return 'unambiguous_female'
                    elif '(both)' in final_text.lower():
                        return 'unambiguous_both'
        
        # If no final decision marker found, fall back to last occurrence
        patterns = {
            'ambiguous': r'\bambiguous\b(?!\s*\([^)]*\))',
            'unambiguous_male': r'\bunambiguous\s*\(\s*male\s*\)',
            'unambiguous_female': r'\bunambiguous\s*\(\s*female\s*\)',
            'unambiguous_both': r'\bunambiguous\s*\(\s*both\s*\)'
        }
        
        last_label = None
        last_pos = -1
        
        for label_type, pattern in patterns.items():
            matches = list(re.finditer(pattern, text))
            if matches:
                pos = matches[-1].start()
                if pos > last_pos:
                    last_pos = pos
                    last_label = label_type
        
        return last_label

    # Add extracted labels to the dataframe
    df['extracted_label'] = df['output'].apply(extract_final_label)
    df['is_consistent'] = df.apply(lambda x: x['extracted_label'] == x['label'].lower().replace('(', '_').replace(')', '').replace(' ', '') if pd.notna(x['extracted_label']) else None, axis=1)
    
    # Save the full analyzed dataset to jsonl
    df.to_json('analyzed_dataset.jsonl', orient='records', lines=True)
    
    # Print sample rows from different categories
    print("\nSample rows from different categories:")
    
    # Get inconsistent samples
    inconsistent_samples = df[df['is_consistent'] == False].sample(n=min(5, len(df[df['is_consistent'] == False])))
    print("\n=== Inconsistent Samples ===")
    for _, row in inconsistent_samples.iterrows():
        print(f"\nRow Index: {row.name}")
        print(f"Text: {row['text'][:200]}...")  # First 200 chars of text
        print(f"Output: {row['output']}")
        print(f"Extracted Label: {row['extracted_label']}")
        print(f"Assigned Label: {row['label']}")
        print("-" * 80)
    
    # Get consistent samples for each label type
    label_types = ['ambiguous', 'unambiguous_male', 'unambiguous_female', 'unambiguous_both']
    for label in label_types:
        consistent_samples = df[(df['extracted_label'] == label) & (df['is_consistent'] == True)].sample(n=min(2, len(df[(df['extracted_label'] == label) & (df['is_consistent'] == True)])))
        print(f"\n=== Consistent {label.upper()} Samples ===")
        for _, row in consistent_samples.iterrows():
            print(f"\nRow Index: {row.name}")
            print(f"Text: {row['text'][:200]}...")  # First 200 chars of text
            print(f"Output: {row['output']}")
            print(f"Extracted Label: {row['extracted_label']}")
            print(f"Assigned Label: {row['label']}")
            print("-" * 80)

    # Calculate and print statistics
    total_analyzed = len(df[df['extracted_label'].notna()])
    total_inconsistent = len(df[df['is_consistent'] == False])
    
    print(f"\nSummary:")
    print(f"Total rows analyzed: {total_analyzed}")
    print(f"Total inconsistent labels: {total_inconsistent}")
    print(f"Consistency rate: {((total_analyzed - total_inconsistent) / total_analyzed):.2%}")
    
    # Show distribution of extracted labels
    extracted_labels = Counter(df['extracted_label'].dropna())
    print("\nDistribution of extracted labels:")
    for label, count in extracted_labels.most_common():
        print(f"{label}: {count}")

if __name__ == "__main__":
    analyze_dataset() 