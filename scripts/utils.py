import re
import os
import matplotlib.pyplot as plt
from datetime import datetime
from collections import Counter
import itertools
import numpy as np

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

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues, figsize=(10, 8), is_norm=True):
        """
        This function plots a confusion matrix.

        Parameters:
            cm (array-like): Confusion matrix as returned by sklearn.metrics.confusion_matrix.
            classes (list): List of class names, e.g., ['Class 0', 'Class 1'].
            title (str): Title for the plot.
            cmap (matplotlib colormap): Colormap for the plot.
        """
        # Create a figure with a specified size
        cm_plot = plt.figure(figsize=figsize)
        
        
        # Display the confusion matrix as an image with a colormap
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        # Define tick marks and labels for the classes on the axes
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90)
        plt.yticks(tick_marks, classes)
        
        if is_norm:
            fmt = '.3f'
        else:
            fmt = '.0f'
        # Add text annotations to the plot indicating the values in the cells
        thresh = cm.max() / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
        # Label the axes
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        # Ensure the plot layout is tight
        plt.tight_layout()

        return cm_plot


    