import argparse
from  copy import deepcopy
import sys
from utils import load_and_merge_batches, log_message, select_n_samples

import torch
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score

def prune_model(model, layers_to_keep):
    """
    Prune the model by removing specified layers.
    Args:
        model: The model to be pruned.
        layers_to_remove: List of indices of layers to be removed.
    Returns:
        model: The pruned model.
    """
    # Access the encoder layers
    pruned_model  = deepcopy(model)
    encoder_layers = pruned_model.wav2vec2.encoder.layers
    pruned_layers = torch.nn.ModuleList([encoder_layers[i] for i in layers_to_keep])
    # Update the model's encoder layers
    pruned_model.wav2vec2.encoder.layers = pruned_layers
    # Update the config to reflect the new number of layers
    pruned_model.config.num_hidden_layers = len(pruned_layers)

    return pruned_model

def validate_model(model, feature_extractor, dataset, output_dir):
    # Define training arguments
    training_args = TrainingArguments(
        seed=42,
        output_dir=f"{output_dir}/tmp",
        per_device_eval_batch_size=4,
    )

    # Define Trainer object to handle training and testing
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=dataset,
        processing_class=feature_extractor,
    )

    # Get the predicted and true labels
    outputs = trainer.predict(dataset)
    y_true = outputs.label_ids
    y_pred = outputs.predictions.argmax(1)
    
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    return accuracy, f1

def main(data_dir, OG_model_dir, output_dir):

    # Redirect all output to a log file (easier to keep track)
    log_file = open(f'pruning_results.txt', 'w')
    sys.stdout = log_file
    
    # Load dataset by merging the batches
    dataset = load_and_merge_batches(data_dir)
    dataset = select_n_samples(dataset, 50)
    dataset = dataset.train_test_split(test_size=0.1, shuffle=True, stratify_by_column="label", seed=42)

    # Load model and feature extractor from the hugging face hub
    model = AutoModelForAudioClassification.from_pretrained(OG_model_dir)
    feature_extractor = AutoFeatureExtractor.from_pretrained(OG_model_dir)

    def preprocess_function(example):
        inputs = feature_extractor(example['audio'], sampling_rate=16000, padding=True)
        return inputs
    dataset["test"] = dataset["test"].map(preprocess_function, remove_columns="audio", batched=True, batch_size=32)

    n_of_layers = len(model.wav2vec2.encoder.layers) 
    prunable_layers = list(range(1,n_of_layers)) 

    for pruning_method in ["forward", "backward"]:
        log_message(f"\nPruning_method: {pruning_method}")
        print(f"{'Layers':<35} {'Accuracy':<10} {'F1 Score':<10} {'n_parameters':<10}")
        print("-" * 75)

        for i in range(1, n_of_layers-1):
            if pruning_method == "forward":
                keep = [0] + prunable_layers[i:]
            else:
                keep = [0] + prunable_layers[:-i]
            pruned_model = prune_model(model, keep) 
            n_parameters = pruned_model.num_parameters(only_trainable=True) / 1e6
            acc, f1 = validate_model(pruned_model,feature_extractor, dataset["test"], output_dir)
            print(f"{str(keep):<35} {acc:<10.5f} {f1:<10.5f} {n_parameters:<10.5f}")

    middle = prunable_layers[len(prunable_layers) // 2]
    log_message(f"\nPruning_method: middle")
    print(f"{'Layers':<35} {'Accuracy':<10} {'F1 Score':<10} {'n_parameters':<10}")
    print("-" * 75)
    for i in range(1, len(prunable_layers) // 2):
        keep = [0] + deepcopy(prunable_layers)
        to_remove = list(range(middle - i, middle + 1 + i)) 
        for layer in to_remove: keep.remove(layer)
        pruned_model = prune_model(model, keep)
        n_parameters = pruned_model.num_parameters(only_trainable=True) / 1e6
        acc, f1 = validate_model(pruned_model,feature_extractor, dataset["test"], output_dir)
        print(f"{str(keep):<35} {acc:<10.5f} {f1:<10.5f} {n_parameters:<10.5f}")

    log_message(f"\nPruning_method: single")
    print(f"{'Layers':<35} {'Accuracy':<10} {'F1 Score':<10}")
    for layer in prunable_layers:
        keep = [0] + prunable_layers
        keep.remove(layer)
        pruned_model = prune_model(model, keep) 
        n_parameters = pruned_model.num_parameters(only_trainable=True) / 1e6
        acc, f1 = validate_model(pruned_model,feature_extractor, dataset["test"], output_dir)
        print(f"{str(keep):<35} {acc:<10.3f} {f1:<10.5f} {n_parameters:<10.5f}")
    
    log_message("FINISHED!!!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="processed_data", help="Directory containing processed data batches")
    parser.add_argument("--OG_model_dir", default="20241119_141957/model", help="Directory of the model to prune")
    parser.add_argument("--output_dir", default="./pruning", help="Directory to save the results")

    args = parser.parse_args()
    main(args.data_dir, args.OG_model_dir, args.output_dir)