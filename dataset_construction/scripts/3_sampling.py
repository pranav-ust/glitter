import json
import pandas as pd
import re
from collections import defaultdict, Counter
import random
import math

# Define LGBTQIA+ related terms
LGBTQ_TERMS = [
    'lgbt', 'lgbtq', 'lgbtqia', 'transgender', 'transsexual', 'trans', 
    'gay', 'lesbian', 'bisexual', 'queer', 'homosexual', 'non-binary',
    'nonbinary', 'genderqueer', 'intersex', 'asexual', 'pansexual',
    'gender fluid', 'genderfluid', 'gender non-conforming', 'same-sex'
]

# Target numbers for each category
TARGETS = {
    'ambiguous': 991,
    'unambiguous_male': 625,
    'unambiguous_female': 417,
    'unambiguous_both': 100
}

# Number of LGBT samples to keep for ambiguous category
LGBT_AMBIG_TARGET = 300

def load_jsonl(file_path):
    """Load JSONL file into a list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def contains_lgbtq_terms(text):
    """Check if text contains any LGBTQ+ related terms."""
    text_lower = text.lower().strip().split()
    return any(term in text_lower for term in LGBTQ_TERMS)

def get_seed_distribution(samples):
    """Get distribution of seed words in samples."""
    return Counter(sample['seed'] for sample in samples)

def print_seed_stats(seed_dist, category):
    """Print statistics about seed word distribution."""
    print(f"\nSeed word distribution for {category}:")
    total = sum(seed_dist.values())
    sorted_seeds = sorted(seed_dist.items(), key=lambda x: x[1], reverse=True)
    print(f"Total unique seed words: {len(seed_dist)}")
    print(f"Top 10 most common seed words:")
    for seed, count in sorted_seeds[:10]:
        print(f"  {seed}: {count} ({count/total*100:.2f}%)")

def calculate_seed_quotas(grouped_data, target_size, min_samples_per_seed=1):
    """Calculate target number of samples per seed word with better distribution."""
    total_seeds = len(grouped_data)
    if total_seeds == 0:
        return {}
    
    # Calculate available samples for each seed
    available_samples = {seed: len(samples) for seed, samples in grouped_data.items()}
    total_available = sum(available_samples.values())
    
    # Initial calculation of samples per seed
    base_per_seed = max(min_samples_per_seed, target_size // total_seeds)
    
    # First pass: allocate base amounts
    quotas = {}
    remaining_target = target_size
    remaining_seeds = set(grouped_data.keys())
    
    while remaining_target > 0 and remaining_seeds:
        # Calculate quota for this round
        current_quota = max(min_samples_per_seed, remaining_target // len(remaining_seeds))
        
        # Try to allocate quota to each remaining seed
        allocated_seeds = set()
        for seed in remaining_seeds:
            available = available_samples[seed]
            if available >= current_quota:
                quotas[seed] = current_quota
                remaining_target -= current_quota
                allocated_seeds.add(seed)
            elif available > 0:
                quotas[seed] = available
                remaining_target -= available
                allocated_seeds.add(seed)
        
        # Remove allocated seeds
        remaining_seeds -= allocated_seeds
        
        # If we couldn't allocate to any seeds, break
        if not allocated_seeds:
            break
    
    # If we still have remaining target, distribute it among seeds with extra samples
    if remaining_target > 0:
        extra_available = {seed: available_samples[seed] - quotas.get(seed, 0) 
                         for seed in grouped_data.keys()}
        extra_available = {k: v for k, v in extra_available.items() if v > 0}
        
        while remaining_target > 0 and extra_available:
            # Find seeds with fewest samples so far
            min_quota = min((quotas.get(seed, 0) for seed in extra_available), default=0)
            candidates = [seed for seed in extra_available if quotas.get(seed, 0) == min_quota]
            
            # Allocate one more sample to a candidate
            seed = random.choice(candidates)
            quotas[seed] = quotas.get(seed, 0) + 1
            remaining_target -= 1
            extra_available[seed] -= 1
            if extra_available[seed] == 0:
                del extra_available[seed]
    
    return quotas

def sample_with_quotas(grouped_data, quotas):
    """Sample from groups according to specified quotas with backfilling."""
    result = []
    unused_samples = []
    
    # First pass: try to meet quotas exactly
    for seed, quota in quotas.items():
        if seed in grouped_data:
            available = grouped_data[seed]
            if len(available) >= quota:
                sampled = random.sample(available, quota)
                # Store unused samples for potential backfilling
                unused_samples.extend([s for s in available if s not in sampled])
            else:
                sampled = available
                print(f"Warning: Not enough samples for seed '{seed}'. Needed {quota}, found {len(available)}")
            result.extend(sampled)
    
    # Calculate how many more samples we need
    total_quota = sum(quotas.values())
    deficit = total_quota - len(result)
    
    # If we have a deficit and unused samples, use them to backfill
    if deficit > 0 and unused_samples:
        backfill = random.sample(unused_samples, min(deficit, len(unused_samples)))
        result.extend(backfill)
        if len(backfill) < deficit:
            print(f"Warning: Could not fully backfill. Still short by {deficit - len(backfill)} samples")
    
    return result

def rebalance_samples(samples, target_size):
    """Rebalance samples to reach target size while maintaining seed word distribution."""
    if len(samples) == target_size:
        return samples
    
    # Group by seed word
    by_seed = defaultdict(list)
    for sample in samples:
        by_seed[sample['seed']].append(sample)
    
    if len(samples) < target_size:
        # Need to add more samples
        deficit = target_size - len(samples)
        while deficit > 0:
            # Find seeds with the least samples
            seed_counts = {seed: len(group) for seed, group in by_seed.items()}
            min_count = min(seed_counts.values())
            candidates = [seed for seed, count in seed_counts.items() if count == min_count]
            
            # Pick a random seed and add a sample if available
            seed = random.choice(candidates)
            if len(by_seed[seed]) > min_count:  # If we have more samples available
                sample = random.choice(by_seed[seed])
                samples.append(sample)
                deficit -= 1
            else:
                break  # No more samples available
    else:
        # Need to remove samples
        excess = len(samples) - target_size
        while excess > 0:
            # Find seeds with the most samples
            seed_counts = {seed: len(group) for seed, group in by_seed.items()}
            max_count = max(seed_counts.values())
            candidates = [seed for seed, count in seed_counts.items() if count == max_count]
            
            # Pick a random seed and remove a sample
            seed = random.choice(candidates)
            sample = random.choice(by_seed[seed])
            samples.remove(sample)
            by_seed[seed].remove(sample)
            excess -= 1
    
    return samples

def uniform_sample_from_groups(grouped_data, target_size):
    """Sample uniformly from groups to reach target size."""
    # Calculate quotas for each seed
    quotas = calculate_seed_quotas(grouped_data, target_size)
    
    # Sample according to quotas
    samples = sample_with_quotas(grouped_data, quotas)
    
    # Rebalance to reach target size
    return rebalance_samples(samples, target_size)

def sample_and_analyze():
    # Load the dataset
    print("Loading dataset...")
    data = load_jsonl('analyzed_dataset.jsonl')
    df = pd.DataFrame(data)
    
    # Create dictionaries to store samples for each label
    final_samples = defaultdict(list)
    
    # First pass: Separate LGBT and non-LGBT content and group by seed
    lgbt_by_seed = defaultdict(lambda: defaultdict(list))
    non_lgbt_by_seed = defaultdict(lambda: defaultdict(list))
    
    # Filter out rows where extracted_label is None
    df = df[df['extracted_label'].notna()]
    
    for _, row in df.iterrows():
        label = row['extracted_label']  # Use extracted_label directly
        row_dict = row.to_dict()
        seed = row_dict['seed']
        
        if contains_lgbtq_terms(row['text']):
            lgbt_by_seed[label][seed].append(row_dict)
        else:
            non_lgbt_by_seed[label][seed].append(row_dict)
    
    # Process each category
    for label, target in TARGETS.items():
        print(f"\nProcessing {label}...")
        
        # Handle LGBT samples first
        lgbt_available = lgbt_by_seed[label]
        if label == 'ambiguous':
            # For ambiguous, sample uniformly to get LGBT_AMBIG_TARGET samples
            lgbt_to_keep = uniform_sample_from_groups(lgbt_available, LGBT_AMBIG_TARGET)
        else:
            # For other categories, keep all LGBT samples but try to keep them uniform
            all_lgbt = []
            for seed_group in lgbt_available.values():
                all_lgbt.extend(seed_group)
            lgbt_to_keep = all_lgbt
        
        # Calculate how many non-LGBT samples we need
        non_lgbt_needed = target - len(lgbt_to_keep)
        
        # Sample non-LGBT entries uniformly across seeds
        if non_lgbt_needed > 0:
            non_lgbt_to_keep = uniform_sample_from_groups(non_lgbt_by_seed[label], non_lgbt_needed)
        else:
            non_lgbt_to_keep = []
        
        # Combine samples and mark LGBT status
        final_samples[label] = []
        for sample in lgbt_to_keep:
            sample['queer_related'] = 'yes'
            final_samples[label].append(sample)
        for sample in non_lgbt_to_keep:
            sample['queer_related'] = 'no'
            final_samples[label].append(sample)
        
        # Print statistics
        print(f"\nStatistics for {label}:")
        print(f"Total samples: {len(final_samples[label])}")
        print(f"LGBT samples: {len(lgbt_to_keep)}")
        print(f"Non-LGBT samples: {len(non_lgbt_to_keep)}")
        
        # Print seed word distribution
        print_seed_stats(get_seed_distribution(final_samples[label]), label)
    
    # Combine all samples into a DataFrame and save
    all_samples = []
    for label, samples in final_samples.items():
        for sample in samples:
            sample['final_label'] = label  # Store the final label we used for sampling
            all_samples.append(sample)
    
    # Convert to DataFrame and save
    final_df = pd.DataFrame(all_samples)
    
    # Print final statistics
    print("\nFinal Dataset Statistics:")
    print("-------------------------")
    print(f"Total samples: {len(final_df)}")
    print("\nSamples per category:")
    print(final_df['final_label'].value_counts())
    print("\nQueer-related content distribution:")
    print(final_df['queer_related'].value_counts())
    print("\nOverall seed word distribution:")
    print_seed_stats(get_seed_distribution(all_samples), "all categories")
    
    # Save to TSV
    final_df.to_csv('sample.csv', sep='\t', index=False)
    print("\nSamples saved to sample.tsv")

if __name__ == "__main__":
    sample_and_analyze() 