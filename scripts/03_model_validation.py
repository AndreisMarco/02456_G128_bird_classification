import os
import argparse
import json
import numpy as np
from datetime import datetime
from utils import load_and_merge_batches, plot_confusion_matrix, log_message, select_n_samples

from transformers import AutoModelForAudioClassification, AutoFeatureExtractor, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

def main(data_dir, mapping_json, model_dir, output_dir):

    # If the output folder does not exist, create it
    if not os.path.exists(output_dir):
        os.makedirs(f"{output_dir}")

    # Load dataset by merging the batches
    dataset = load_and_merge_batches(data_dir)
    
    # FOR TESTING!!! Work only on a balanced subset of the dataset
    dataset = select_n_samples(dataset, 50)

    num_classes = len(set(dataset["label"]))
    log_message(f"Number of classes in the dataset: {num_classes}")

    # Load label2id dictionary
    with open(mapping_json, "r") as json_file:
        mappings = json.load(json_file)

    label2id = mappings['label2id']
    del mappings
    class_labels = label2id.keys()

    # Load model and feature extractor from the hugging face hub
    model = AutoModelForAudioClassification.from_pretrained(model_dir, num_labels=num_classes)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_dir)

    dataset = dataset.train_test_split(test_size=0.1, shuffle=True, stratify_by_column="label", seed=42)

    def preprocess_function(example):
        inputs = feature_extractor(example['audio'], sampling_rate=16000, padding=True)
        return inputs
    dataset["test"] = dataset["test"].map(preprocess_function, remove_columns="audio", batched=True, batch_size=32)

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
        eval_dataset=dataset["test"],
        processing_class=feature_extractor,
    )

    # Get model prediction on test data
    outputs = trainer.predict(dataset['test'])
     # Print the metrics obtained from the prediction outputs.
    print(outputs.metrics)

    # Save predictions
    np.save(f"{output_dir}/predictions.npy", outputs.predictions)
    # Save label IDs
    np.save(f"{output_dir}/label_ids.npy", outputs.label_ids)

    predictions = np.load(f"{output_dir}/predictions.npy")
    y_true = np.load(f"{output_dir}/label_ids.npy")

    # Predict the labels by selecting the class with the highest probability
    y_pred = predictions.argmax(1)

    # Calculate accuracy and F1 score
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')

    # Display accuracy and F1 score
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Calculate confusion matrix and save
    cm = confusion_matrix(y_true, y_pred)
    cm_plot = plot_confusion_matrix(cm, class_labels, figsize=(18, 16), is_norm=False)
    cm_plot.savefig(f"{output_dir}/confusion_matrix.pdf", format="pdf")

    # Calculate classification report and save it into txt
    report = classification_report(y_true, y_pred, target_names=class_labels, digits=4)
    with open(f"{output_dir}/classification_report.txt", "w") as f:
        f.write("Classification report:\n")
        f.write(report)

    # Create a file containing general model info
    n_parameters = model.num_parameters(only_trainable=True) / 1e6
    with open(f"{output_dir}/model_infos.txt", "w") as f:
        f.write(f"Number of parameters (in millions): {n_parameters}")
        f.write("Model architecture:")
        f.write(str(model))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="processed_data", help="Directory containing processed data batches")
    parser.add_argument("--mapping_json", default="processed_data/label_mappings.json", help="File containing the mapping from label2id and viceversa ")
    parser.add_argument("--model_dir", default="facebook/wav2vec2-base-960h", help="Directory of the model to evaluate")
    parser.add_argument("--output_dir", default="./validation_results", help="Directory to save the results of the analysis")
    
    args = parser.parse_args()
    main(args.data_dir, args.mapping_json, args.model_dir, args.output_dir)