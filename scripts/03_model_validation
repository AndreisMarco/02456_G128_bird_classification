import os
import argparse
import itertools
import json
import numpy as np
from utils import load_and_merge_batches, log_message, select_n_samples
import matplotlib.pyplot as plt

from datasets import Dataset, concatenate_datasets
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

def main(data_dir, mapping_json, model_dir, output_dir):

    # If the output folder does not exist, create it
    if not os.path.exists(output_dir):
        os.makedirs(f"{output_dir}")

    # Load dataset by merging the batches
    dataset = load_and_merge_batches(data_dir)
    
    # FOR TESTING!!! Work only on a balanced subset of the dataset
    #dataset = select_n_samples(dataset, 50)

    num_classes = len(set(dataset["label"]))
    log_message(f"Number of classes in the dataset: {num_classes}")

    # Load label2id and viceversa dictionaries
    with open(mapping_json, "r") as json_file:
        mappings = json.load(json_file)

    id2label = mappings['id2label']
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

    # get model prediction on test data
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

    # Define a function to plot a confusion matrix
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
        fig = plt.figure(figsize=figsize)
        
        
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
        # Display the plot
        plt.show()
        fig.savefig(f"{output_dir}/confusion_matrix.pdf", format="pdf")


    # Calculate accuracy and F1 score
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')

    # Display accuracy and F1 score
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Get the confusion matrix if there are a relatively small number of labels
    if num_classes <= 120:
        # Compute the confusion matrix
        cm = confusion_matrix(y_true, y_pred) # normalize='true'

        # Plot the confusion matrix using the defined function
        plot_confusion_matrix(cm, class_labels, figsize=(18, 16), is_norm=False)

    # Calculate classification report and save it into txt
    report = classification_report(y_true, y_pred, target_names=class_labels, digits=4)
    with open(f"{output_dir}/classification_report.txt", "w") as f:
        f.write("Classification report:\n")
        f.write(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="processed_data", help="Directory containing processed data batches")
    parser.add_argument("--mapping_json", default="processed_data/label_mappings.json", help="File containing the mapping from label2id and viceversa ")
    parser.add_argument("--model_dir", default="og_training_settings_20241117_173251/model", help="Directory of the model to evaluate")
    parser.add_argument("--output_dir", default="./validation_results", help="Directory to save the results of the analysis")
    
    args = parser.parse_args()
    main(args.data_dir, args.mapping_json, args.model_dir, args.output_dir)