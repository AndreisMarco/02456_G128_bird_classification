import sys
import torch
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor, Trainer, TrainingArguments

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
    encoder_layers = model.wav2vec2.encoder.layers
    pruned_layers = torch.nn.ModuleList([encoder_layers[i] for i in layers_to_keep])
    # Update the model's encoder layers
    model.wav2vec2.encoder.layers = pruned_layers
    # Update the config to reflect the new number of layers
    model.config.num_hidden_layers = len(pruned_layers)
    return model

print("Welcome! This is a simple pruning tool for hugging face models")

check = True
while check:
    model_directory = input("Insert directory of the original model:")

    try:
        model = AutoModelForAudioClassification.from_pretrained(model_directory, num_labels=50)
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_directory)
        check = False
    except:
        print("The directory is not valid")

number_of_layers = len(model.wav2vec2.encoder.layers)
print(f"The model has {number_of_layers} layers")

available_layers = list(range(number_of_layers))

check = True
while check:
    print(f"\nCurrent layers: {available_layers}")
    selected = input("Use '-<layer_number>' to remove a layer or '+<layer_number> to add a layer' or 'ok' to confirm:\n")

    if selected[0] == "+":
        available_layers = available_layers + [int(selected[1:])]
    elif selected[0] == "-":
        available_layers.remove(int(selected[1:]))
    elif selected[:2] == "ok":
        break
    available_layers = sorted(list(set(available_layers)))

try:
    model = prune_model(model, available_layers)
    n_parameters = model.num_parameters(only_trainable=True) / 1e6
except: 
    print("The selected layers are uncompatible with the selected model")


print("----------------------------------------------------------------")
print(model)
print("----------------------------------------------------------------")
print(f"This is the final model architecture. It contains {n_parameters} millions parameters.")

while True:
    save_dir = input("Where do you want to save the pruned model? Use 'q' to quit without saving.\n")
    
    if save_dir == "q":
        sys.exit()
    
    try: 
        model.save_pretrained(f"./{save_dir}")
        feature_extractor.save_pretrained(f"./{save_dir}")
        print("Model correctly saved")
        break
    except:
        print("The directory is not valid")




