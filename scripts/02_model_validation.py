# %%
# basic libraries
import numpy as np
from random import sample
import json 
from datetime import datetime


# To access the model on hugging face
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor, TrainingArguments, Trainer

# To play audio interactively
from IPython.display import Audio 

# Import performance metrics
import evaluate

# To load the dataset
from utils import load_and_merge_batches

# %% [markdown]
# Load the model from the hugging face repository

# %%
model_name = "dima806/bird_sounds_classification"

# Load the feature extractor and model
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)

# %% [markdown]
# Load and merge batches into one dataset

# %%
batches_folder = "../processed_data"
dataset = load_and_merge_batches(batches_folder)

# %% [markdown]
# Load dictionary of the labels

# %%
with open('../processed_data/label_mappings.json', 'r') as file:
    label_mappings = json.load(file)

label2id = label_mappings["label2id"]
id2label = label_mappings["id2label"]
del label_mappings

# %% [markdown]
# Given that we do not intend to train the model, we going to use the test set.

# %%
test_split = 0.1
dataset = dataset.train_test_split(test_size=test_split, shuffle=True, stratify_by_column="label")

# %% [markdown]
# Process the dataset

# %%
def preprocess_function(batch):
    # Extract audio features from the input batch using the feature_extractor
    inputs = feature_extractor(batch['audio'], sampling_rate=Hz_rate)
    
    # Extract and store only the 'input_values' component from the extracted features
    inputs['input_values'] = inputs['input_values'][0]
    
    return inputs

dataset['test']= dataset['test'].map(preprocess_function, remove_columns="audio", batched=False)

# %% [markdown]
# Define accuracy metrics

# %%
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions = eval_pred.predictions
    predictions = np.exp(predictions) / np.exp(predictions).sum(axis=1, keepdims=True)  # Softmax
    label_ids = eval_pred.label_ids
    acc_score = accuracy.compute(predictions=predictions.argmax(axis=1), references=label_ids)['accuracy']
    return {"accuracy": acc_score}

# %%
# Define TrainingArguments for evaluation
training_args = TrainingArguments(
    output_dir=model_name,  # Directory to save the model (not really used in evaluation)
    per_device_eval_batch_size=16,  # Evaluation batch size
    evaluation_strategy='epoch',  # Evaluate at the end of each epoch (for eval only)
    save_strategy='no',  # No saving during evaluation
    load_best_model_at_end=False,  # No training, so no best model to load
    logging_steps=1,  # Log every step
    report_to="none",  # No need to report to mlflow during evaluation
)

# Create Trainer object (without training setup)
trainer = Trainer(
    model=model,  # The trained model you want to evaluate
    args=training_args,  # TrainingArguments (used for evaluation configuration)
    eval_dataset=dataset['test'],  # Use the test dataset for evaluation
    tokenizer=feature_extractor,  # Tokenizer (if needed)
    compute_metrics=compute_metrics,  # Define the metric function for evaluation
)

# %%
# Evaluate the model
eval_results = trainer.evaluate()

# Print the evaluation results
print(eval_results)

# %% [markdown]
# Log of the accuracies
# 
# Evaluation n.1: \
# {'eval_loss': 1.0484373569488525, 'eval_model_preparation_time': 0.0044, 'eval_accuracy': 0.7383673469387755, 'eval_runtime': 1150.2315, 'eval_samples_per_second': 2.13, 'eval_steps_per_second': 0.134}
# 
# Evaluation n.2: \
# {'eval_loss': 0.531235933303833, 'eval_accuracy': 0.8922448979591837, 'eval_runtime': 1284.9757, 'eval_samples_per_second': 1.907, 'eval_steps_per_second': 0.12}
# 
# 

# %%
log_file = "validation.log"

# Function to log a message manually
def log_message(message, level="INFO"):
    with open(log_file, "a") as file:  # Open in append mode
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{timestamp} - {level} - {message}\n"
        file.write(log_entry)

log_message(f"Evalutation_results:\n {eval_results}")


