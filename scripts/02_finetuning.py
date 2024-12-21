import sys
import os
import argparse
import numpy as np
from datetime import datetime

from utils import log_message, load_and_merge_batches, select_n_samples

from sklearn.utils.class_weight import compute_class_weight
import torch
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor, TrainingArguments, Trainer
import evaluate

class WeightedTrainer(Trainer):
    '''
    Modified Trainer class that uses a vector of weights to compute
    CrossEntropyLoss for the imbalanced dataset
    '''
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Calculate loss using class weitghs 
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss

def main(data_dir, base_model, output_dir):
    
    # If the output folder does not exist, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # The output folder for each training run is exclusive by using the start time
    output_dir = f'{output_dir}/{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    # create the output folder for the specific run
    os.makedirs(output_dir)
    # and the folder to hold the tensorboard readable logs 
    logs_dir = f'{output_dir}/tensorboard_logs'

    # Redirect all output to a log file (easier to keep track)
    log_file = open(f'{output_dir}/log.txt', 'w')
    sys.stdout = log_file

    # Check if the GPU is available 
    gpu_available = torch.cuda.is_available()
    log_message(f"Is GPU available? {gpu_available}")
    # Log GPU availability
    if gpu_available:
        current_device = torch.cuda.current_device()
        log_message(f"Current GPU Device: {torch.cuda.get_device_name(current_device)}")
    else:
        log_message("Using CPU instead of GPU.")

    # If available work on GPU 
    device = torch.device('cuda' if gpu_available else 'cpu')
    
    # Load dataset by merging the batches
    log_message(f"Dataset source: {data_dir}")
    dataset = load_and_merge_batches(data_dir)

    # FOR TESTING!!! Work only on a balanced subset of the dataset
    # dataset = select_n_samples(dataset, 2)

    num_classes = len(set(dataset["label"]))
    log_message(f"Number of classes in the dataset: {num_classes}")

    # Load model and feature extractor from the hugging face hub
    model = AutoModelForAudioClassification.from_pretrained(base_model, num_labels=num_classes)
    feature_extractor = AutoFeatureExtractor.from_pretrained(base_model)
    log_message(f"Base model source: {base_model}")
    
    # Preprocess all the examples with the feature extractor
    def preprocess_function(example):
        inputs = feature_extractor(example['audio'], sampling_rate=16000, padding=True)
        return inputs
    dataset = dataset.map(preprocess_function, remove_columns="audio", batched=True, batch_size=32)
    log_message("Preprocessed dataset with feature extractor.")

    # Split dataset into train and test (set seed for reproducibility)
    dataset = dataset.train_test_split(test_size=0.1, shuffle=True, stratify_by_column="label", seed=42)
    log_message("Split dataset into training and testing.")

    # Compute class weights and store in a dict
    class_weights = compute_class_weight('balanced', classes=np.unique(dataset['train']['label']), y=dataset['train']['label'])
    class_weights = {class_id: weight for class_id, weight in zip(np.unique(dataset['train']['label']), class_weights)}
    log_message(f"Computed class weights: {class_weights}")
    # Convert weights to Tensor
    class_weight_tensor = torch.tensor(list(class_weights.values()), dtype=torch.float32).to(device)
    log_message(f"Class weights tensor moved to device: {device}")

    # Use accuracy as performace metric
    accuracy = evaluate.load("accuracy")
    log_message("Loaded evaluation metric: accuracy.")

    def compute_metrics(eval_pred):
        # Extract the model's predictions from eval_pred.
        predictions = eval_pred.predictions
        # Apply the softmax function to convert prediction scores into probabilities.
        predictions = np.exp(predictions) / np.exp(predictions).sum(axis=1, keepdims=True)
        # Extract the true label IDs from eval_pred.
        label_ids = eval_pred.label_ids
        # Calculate accuracy using the loaded accuracy metric by comparing predicted classes
        # (argmax of probabilities) with the true label IDs.
        acc_score = accuracy.compute(predictions=predictions.argmax(axis=1), references=label_ids)['accuracy']
        # Return the computed accuracy as a dictionary with a key "accuracy."
        return {
            "accuracy": acc_score
        }

    # Define training arguments
    training_args = TrainingArguments(
        seed=42,
        learning_rate=3e-6,  
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=30,
        warmup_steps=50,  
        weight_decay=0.02,
        gradient_checkpointing=True,

        eval_strategy="epoch",  
        logging_dir=logs_dir,
        logging_steps=500,
        logging_first_step=True,
        
        save_strategy="epoch",
        output_dir=f"{output_dir}/model/",
        load_best_model_at_end=True,
        save_total_limit=1,
    )

    # Define Trainer object to handle training and testing
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=feature_extractor,
        class_weights=class_weight_tensor,
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    log_message("Starting training.")
    print("#")
    trainer.train()
    print("#")
    log_message("Training completed.")

    # Save trained model
    trainer.save_model(f"{output_dir}/model/")
    log_message(f"Model saved to {output_dir}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="processed_data", help="Directory containing processed data batches")
    parser.add_argument("--base_model", default="facebook/wav2vec2-base-960h", help="Pre-trained model to fine-tune")
    parser.add_argument("--output_dir", default="./runs", help="Directory to save trained model")
    
    args = parser.parse_args()
    main(args.data_dir, args.base_model, args.output_dir)
