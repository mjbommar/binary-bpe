#!/usr/bin/env python3
"""
Extract text field from Hugging Face dataset and save to file.
"""
from datasets import load_dataset
import os

# Load the dataset
print("Loading dataset...")
ds = load_dataset('alea-institute/kl3m-sft-hearings-sample-001')

# Get the output path
output_path = os.path.expanduser('~/sample-001.txt')

# Extract all text fields and write to file
print(f"Extracting text from {len(ds['train'])} examples...")
with open(output_path, 'w', encoding='utf-8') as f:
    for i, example in enumerate(ds['train']):
        # Write the text field
        f.write(example['text'])
        # Add double newline separator between texts
        f.write('\n\n')

        # Progress indicator every 100,000 examples
        if (i + 1) % 100000 == 0:
            print(f"Processed {i + 1:,} / {len(ds['train']):,} examples...")

print(f"Successfully wrote {len(ds['train']):,} text examples to {output_path}")