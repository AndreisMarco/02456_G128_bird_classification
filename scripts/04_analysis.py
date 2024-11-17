import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from  tqdm import tqdm

import torch
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from utils import load_and_merge_batches, log_message, select_n_samples

from convexity import graph_convexity
from similarity_metrics import cka, mutual_knn, cosine_similarity

device = "cuda" if torch.cuda.is_available() else "cpu"

#####################################################################################################################################################

def extract_features(model, processor, dataset, output_dir, num_layers=13, num_features=768):
    
    model.to(device)
    #initialize hidden states tensor to avoid itereative concatenation which is very slow
    hidden_states = torch.zeros((len(dataset), num_layers, num_features))

    # iterate over the dataset
    for i, example in enumerate(tqdm(dataset)):
        #preprocess the data
        inputs = processor(example["audio"], return_tensors="pt", padding=True, sampling_rate=16000).input_values.to(device)

        # perform inference on inputs and retreive the hidden states
        with torch.no_grad():
            outputs = model(inputs, output_hidden_states=True, return_dict=True)

        # average the hidden states over the time axis for each layer j
        for j, hidden_state in enumerate(outputs.hidden_states):
            hs_mean = torch.mean(hidden_state, dim=1)
            hidden_states[i, j] = hs_mean

        #optional saving after 100 steps
        # depending on the size of your dataset this process can time out so better save intermediate results
        if i>0 and i%100 == 0:
            np.save(os.path.join(output_dir, f"/data/hidden_states.npy"), hidden_states.cpu().numpy())

    np.save(os.path.join(output_dir, f"/data/hidden_states.npy"), hidden_states.cpu().numpy())

#####################################################################################################################################################

def main(data_dir, model_name, output_dir):
    
    # If the output folder does not exist, create it
    if not os.path.exists(output_dir):
        os.makedirs(f"{output_dir}/data")

    # Load dataset by merging the batches
    dataset = load_and_merge_batches(data_dir)

    # FOR TESTING!!! Work only on a balanced subset of the dataset
    #dataset = select_n_samples(dataset, 50)

    num_classes = len(set(dataset["label"]))
    log_message(f"Number of classes in the dataset: {num_classes}")

    # Load model and feature extractor from the hugging face hub
    model = AutoModelForAudioClassification.from_pretrained(model_name, num_labels=num_classes)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

    # Split dataset into train and test
    dataset = dataset.train_test_split(test_size=0.1, shuffle=True, stratify_by_column="label", seed=42)
    log_message("Split dataset into training and testing.")

    # # extract features from test set
    # extract_features(model, feature_extractor, dataset["test"], output_dir=output_dir)
    # np.save(f"{output_dir}/data/labels.npy", dataset["test"]["label"])

    hidden_states = np.load(f"{output_dir}/data/hidden_states.npy")
    labels = np.load(f"{output_dir}/data/labels.npy")

    # perform convexity analysis
    convexity,_ = graph_convexity(hidden_states, labels, num_neighbours=10)
    # plot convexity curve
    convexity_plot = plt.figure(figsize=(10, 6))
    plt.plot([x[0] for x in convexity])
    plt.xlabel("Layers")
    plt.ylabel("Convexity")
    plt.title("Convexity of sounds across Layers")
    convexity_plot.savefig(f"{output_dir}/convexity.pdf", format="pdf")

    num_layers = hidden_states.shape[1]
    num_samples = hidden_states.shape[0]
    hidden_states = torch.tensor(hidden_states)
    
    # perform CKA analysis
    cka_matrix = np.zeros((num_layers, num_layers))
    for i in range(num_layers):
        for j in range(num_layers):
            cka_matrix[i, j] = cka(hidden_states[:, i, :], hidden_states[:, j, :])

    # perform mutual KNN analysis
    knn_matrix = np.zeros((num_layers, num_layers))
    for i in range(num_layers):
        for j in range(num_layers):
            knn_matrix[i, j] = mutual_knn(hidden_states[:, i, :], hidden_states[:, j, :], topk=5)

    # Compute cosine similarity
    cosine_matrix = np.zeros((num_layers, num_layers))
    for i in range(num_layers):
        for j in range(num_layers):
            cosine_matrix[i, j] = cosine_similarity(hidden_states[:, i, :], hidden_states[:, j, :]).trace()/num_samples

    # CKA Similarity Plot
    CKA_similarity_plot = plt.figure(figsize=(10, 6))
    sns.heatmap(cka_matrix, cmap="viridis")
    plt.title("CKA Similarity Matrix")
    plt.xlabel("Layers")
    plt.ylabel("Layers")
    CKA_similarity_plot.savefig(f"{output_dir}/CKA_similarity.pdf", format="pdf")

    # Mutual kNN Plot
    mutual_kNN_plot = plt.figure(figsize=(10, 6))
    sns.heatmap(knn_matrix, cmap="viridis")
    plt.title("Mutual kNN Similarity Matrix ") 
    plt.xlabel("Layers")
    plt.ylabel("Layers")
    mutual_kNN_plot.savefig(f"{output_dir}/mutual_kNN_plot.pdf", format="pdf")

    # Cosine Similarity Plot
    cosine_similarity_plot = plt.figure(figsize=(10, 6))
    sns.heatmap(cosine_matrix, cmap="viridis")
    plt.title("Cosine Similarity Matrix")  
    plt.xlabel("Layers")
    plt.ylabel("Layers")
    cosine_similarity_plot.savefig(f"{output_dir}/cosine_similarity.pdf", format="pdf")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="processed_data", help="Directory containing processed data batches")
    parser.add_argument("--model_dir", default="facebook/wav2vec2-base-960h", help="Directory of the model to evaluate")
    parser.add_argument("--output_dir", default="./analysis_results", help="Directory to save the results of the analysis")
    
    args = parser.parse_args()
    main(args.data_dir, args.model_name, args.output_dir)