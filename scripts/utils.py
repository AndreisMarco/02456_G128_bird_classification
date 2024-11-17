import re
import os
from datetime import datetime
from collections import Counter

from datasets import Dataset, concatenate_datasets

def log_message(message):
    '''
    Home-made simple function to add the times to prints
    '''
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{current_time} - {message}")

def sort_numerically(batch_paths):
    '''
    Necessary for standardizing the batches import order
    '''
    def extract_number(batch_dir):
        match = re.search(r'(\d+)', batch_dir)
        return int(match.group(1)) if match else 0
    sorted_paths = sorted(batch_paths, key=extract_number)
    return sorted_paths

def load_and_merge_batches(batch_folder):
    '''
    Loads all .arrow files and merges them in a single dataset
    '''
    log_message(f"Loading and merging batches from folder: {batch_folder}")
    batch_paths = [f for f in os.listdir(batch_folder) if os.path.isdir(os.path.join(batch_folder, f))]
    batch_paths = sort_numerically(batch_paths)
    datasets_list = []

    for batch_dir in batch_paths:
        batch_path = os.path.join(batch_folder, batch_dir)
        dataset = Dataset.load_from_disk(batch_path)
        datasets_list.append(dataset)
    
    merged_dataset = concatenate_datasets(datasets_list)
    log_message(f"Merged {len(datasets_list)} batches into a single dataset.")
    return merged_dataset

def select_n_samples(dataset, n):
    '''
    Function to subset dataset by taking n samples for each unique class 
    '''
    log_message(f"Selecting {n} samples per class from the dataset.")
    selected = []
    counts = Counter(dataset['label'])
    current = 0
    for i in range(len(counts.keys())):
        for j in range(current, current+n):
            selected.append(j)
        current_label = dataset[current]["label"] 
        current += counts[current_label]
    return dataset.select(selected)