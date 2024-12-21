'''
This file is an adaptation of the code present into the notebook "bird-species-by-sound-detection".
Specifically the first part which processes the audio data and puts it into an hugging face dataset to be feed to the LAM. 
'''

# Import basic libraries
import pandas as pd  # Pandas for data manipulation
import gc  # Garbage collection module
import re  # Regular expressions for text processing
import numpy as np  # NumPy for numerical operations

# Import json for saving files
import json

# Import Path from pathlib for working with file paths
from pathlib import Path

# Import class_weight calculation function from scikit-learn
from sklearn.utils.class_weight import compute_class_weight

# Import the Hugging Face Transformers library
import transformers

# Import torchaudio for audio processing with PyTorch
import torchaudio

# Import various classes and modules from Hugging Face Transformers and Datasets
from datasets import Dataset, Image, ClassLabel  # Import custom 'Dataset', 'ClassLabel', and 'Image' classes

# Import tqdm to follow progression of processing
from tqdm import tqdm

############################################################################################################

# Define the resampling rate in Hertz (Hz) for audio data
RATE_HZ = 16000
# Define the maximum audio interval length to consider in seconds
MAX_SECONDS = 10
# Calculate the maximum audio interval length in samples by multiplying the rate and seconds
MAX_LENGTH = RATE_HZ * MAX_SECONDS
# Define the minimum number of records per label required for the dataset
MIN_RECORDS_PER_LABEL = 25

print("Starting to extract files paths and labels")
# Initialize empty lists to store file paths and corresponding labels.
file_list = []  # To store file paths
label_list = []  # To store labels
# Iterate through all the .mp3 files in the specified directory and its subdirectories.
for file in tqdm(Path('raw_data/audio_files/').glob('*/*.mp3')):
  # Extract the label from the file path by splitting the path and retrieving the second-to-last part.
  # The label is assumed to be the second-to-last part, separated by '/' and '_' characters.
  label = str(file).split('/')[-2].split('_')[0]
  # Append the current file path to the file_list and its corresponding label to the label_list.
  file_list.append(file)
  label_list.append(label)
# Create an empty DataFrame to organize the data.
df = pd.DataFrame()
# Create two columns in the DataFrame: 'file' to store file paths and 'label' to store labels.
df['file'] = file_list
df['label'] = label_list
print("LOG --- Created dataframe of file paths and labels")
print(f"LOG --- Dataframe contains {len(df)} entries")

# Calculate label counts
label_counts = df['label'].value_counts()
# Identify undersampled labels
undersampled_labels = label_counts[label_counts < MIN_RECORDS_PER_LABEL].index
# Remove rows with undersampled labels
df = df[~df['label'].isin(undersampled_labels)]
print(f"LOG --- Dataframe contains {len(df)} entries after filtering for labels having more than {MIN_RECORDS_PER_LABEL} examples")

# Create a list of unique labels
labels_list = sorted(list(df['label'].unique()))

# Initialize empty dictionaries to map labels to IDs and vice versa
label2id, id2label = dict(), dict()

# Iterate over the unique labels and assign each label an ID, and vice versa
for i, label in enumerate(labels_list):
    label2id[label] = i  # Map the label to its corresponding ID
    id2label[i] = label  # Map the ID to its corresponding label

# Creating classlabels to match labels to IDs
ClassLabels = ClassLabel(num_classes=len(labels_list), names=labels_list)

#This  function Split files by chunks with == MAX_LENGTH size
def split_audio(file):
    '''
    This function takes a audio files, divides it into segments of specified MAX_LENGTH.
    Each segment is resampled to the specified RATE_HZ and stored into a DataFrame which is returned.

    INPUT
    file: path to the audiofile to be splitted (must readable by torchaudio)

    OUTPUT
    df_segments: a single column dataframe containing the resampled segments of the audio file
    '''
    try:
        # Load the audio file using torchaudio and get its sample rate.
        audio, rate = torchaudio.load(str(file))

        # Calculate the number of segments based on the MAX_LENGTH
        num_segments = (len(audio[0]) // MAX_LENGTH)  # Floor division to get segments

        # Create an empty list to store segmented audio data
        segmented_audio = []

        # Split the audio into segments
        for i in range(num_segments):
            start = i * MAX_LENGTH
            end = min((i + 1) * MAX_LENGTH, len(audio[0]))
            segment = audio[0][start:end]

            # Create a transformation to resample the audio to a specified sample rate (RATE_HZ).
            transform = torchaudio.transforms.Resample(rate, RATE_HZ)
            segment = transform(segment).squeeze(0).numpy().reshape(-1)

            segmented_audio.append(segment)

        # Create a DataFrame from the segmented audio
        df_segments = pd.DataFrame({'audio': segmented_audio})

        return df_segments

    except Exception as e:
        # If an exception occurs (e.g., file not found), return nothing
        print(f"Error processing file: {e}")
        return None

# process and save the dataset in batches to avoid memory issues
batch_size = 100

for i in (range(len(df) // batch_size + 1)):

    batch_start = i * batch_size
    batch_end = batch_start + batch_size

    if batch_end > len(df):
        batch_end = len(df)

    current_batch = df.iloc[batch_start:batch_end]

    print(f"LOG --- Starting the processing of the audio files for the examples {batch_start} - {batch_end}")
    df_list = []
    for input_file, input_label in tqdm(zip(current_batch['file'].values, current_batch['label'].values), total = len(current_batch)):
        resulting_df = split_audio(input_file)
        if resulting_df is not None:
            resulting_df['label'] = input_label
            df_list.append(resulting_df)
    current_batch = pd.concat(df_list, axis=0)
    print(f"LOG --- Finished sound splitting and processing for batch {i}")

    # Selecting rows in the DataFrame where the 'audio' column is not null (contains non-missing values).
    current_batch = current_batch[~current_batch['audio'].isnull()]
    print(f"LOG --- After removing invalid entries this batch contaings {len(current_batch)} samples")

    del df_list
    gc.collect()

    # Create a dataset from the Pandas DataFrame 'df'
    dataset = Dataset.from_pandas(current_batch)

    del current_batch
    gc.collect()

    # Mapping labels to IDs
    def map_label2id(example):
        example['label'] = ClassLabels.str2int(example['label'])
        return example

    dataset = dataset.map(map_label2id, batched=True)

    # Casting label column to ClassLabel Object
    dataset = dataset.cast_column('label', ClassLabels)
    ("LOG --- Successfully mapped each label to a unique number identifier")

    # Save dataset batch
    dataset.save_to_disk(f"processed_data/batch_{i}")

    del dataset
    gc.collect()
    
# Save label2id and id2label
with open("processed_data/label_mappings.json", "w") as f:
    json.dump({"label2id": label2id, "id2label": id2label}, f)
