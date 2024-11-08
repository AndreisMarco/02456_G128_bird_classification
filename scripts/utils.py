import os
import re
from datasets import Dataset, concatenate_datasets

def sort_numerically(batch_paths):
    def extract_number(batch_dir):
        # Extract the numerical part from the batch directory name
        match = re.search(r'(\d+)', batch_dir)
        return int(match.group(1)) if match else 0
    
    # Sort batch directories by the extracted number
    return sorted(batch_paths, key=extract_number)

def load_and_merge_batches(batch_folder):
    # Get a list of all (batch folders in the batch folder
    batch_paths = [f for f in os.listdir(batch_folder) if os.path.isdir(os.path.join(batch_folder, f))]
    
    # Sort the directories numerically
    batch_paths = sort_numerically(batch_paths)
    
    # List to store individual datasets
    datasets_list = []
    
    # Loop through each batch folder
    for batch_dir in batch_paths:
        batch_path = os.path.join(batch_folder, batch_dir)
        # Load the batch folder as a Hugging Face dataset
        dataset = Dataset.load_from_disk(batch_path)
        datasets_list.append(dataset)
        print(f"Loaded: {batch_path}")
    
    # Concatenate all datasets into a single dataset
    merged_dataset = concatenate_datasets(datasets_list)
    print(f"Merged {len(datasets_list)} batches into a single dataset.")
    
    return merged_dataset
