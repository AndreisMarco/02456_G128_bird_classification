# -*- coding: utf-8 -*-
"""scripts/06_Audio classification with CNN.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/13MY9NVHrsoaYgsL5JAM26RBt5nBGm8vD

# 1. Set up environment
"""

!pip install datasets evaluate --quiet

"""## 1.1 Import libraries"""

# setting up Drive and path for data loading and saving
import os
from google.colab import drive

# for data processing
import numpy as np
import re
from datasets import Dataset, concatenate_datasets
from transformers import AutoFeatureExtractor

# for model training and evaluation
import torch
from torch import nn
import evaluate
from torch.utils.data import DataLoader
from datetime import datetime

# for visualisation
import seaborn as sns
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# enforcing reproducibility - from NLP - lab 2 notebook
import random
import numpy as np

def enforce_reproducibility(seed=42):
    # Sets seed manually for both CPU and CUDA
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For atomic operations there is currently
    # no simple way to enforce determinism, as
    # the order of parallel operations is not known.
    # CUDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # System based
    random.seed(seed)
    np.random.seed(seed)

enforce_reproducibility()

"""## 1.2 Prepare data"""

# from Marco's code

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
    print(f"Loading and merging batches from folder: {batch_folder}")
    batch_paths = [f for f in os.listdir(batch_folder) if os.path.isdir(os.path.join(batch_folder, f))]
    batch_paths = sort_numerically(batch_paths)
    datasets_list = []

    for batch_dir in batch_paths:
        batch_path = os.path.join(batch_folder, batch_dir)
        dataset = Dataset.load_from_disk(batch_path)
        datasets_list.append(dataset)

    merged_dataset = concatenate_datasets(datasets_list)
    print(f"Merged {len(datasets_list)} batches into a single dataset.")
    return merged_dataset

dataset = load_and_merge_batches("processed_data")

# inspect structure
dataset = dataset.remove_columns('__index_level_0__')

num_classes = len(set(dataset['label']))

# re-assign classes - takes 3 min
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(dataset["label"])  # Fit on all labels in training set

def remap_labels(batch, label_column='label'):
    batch[label_column] = label_encoder.transform(batch[label_column])
    return batch

dataset = dataset.map(remap_labels, batched=True)
all_classes = np.unique(dataset['label'])
print(f"Renamed classes: {all_classes}")

"""### 1.2.1 Feature Extraction"""

model_dir = 'facebook/wav2vec2-base-960h'
feature_extractor = AutoFeatureExtractor.from_pretrained(model_dir)

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Compute class weights
all_classes = np.unique(dataset['label'])  # Assuming 'label' is your target column

"""### 1.2.2 Split Dataset"""

# split dataset into train, val and test - from Marco's code
dataset = dataset.train_test_split(test_size=0.1, shuffle=True, stratify_by_column="label", seed=42)
print("Split dataset into training and testing.")

"""### 1.2.3 Compute class weights"""

class_weights = compute_class_weight('balanced', classes=all_classes, y=dataset['train']['label'])

# Handle missing classes
all_class_weights = {}
for class_id in all_classes:
  if class_id in class_weights:
      all_class_weights[class_id] = class_weights[class_id]
  else:
      all_class_weights[class_id] = 1.0  # or any default weight you prefer
# Convert to Tensor
class_weight_tensor = torch.tensor(list(all_class_weights.values()), dtype=torch.float32).to(device)

"""### 1.2.4 Dataset to Loader"""

MAX_LENGTH = 160000  # Use the longest sequence length in your dataset

def collate_fn(batch):
    inputs = []
    for item in batch:
        input_values = item['audio'].clone().detach()

        # running the line below didn't work properly, I had to manually pad
        # padded_inputs = pad_sequence(inputs, batch_first=True).unsqueeze(1)

        if len(input_values) < MAX_LENGTH: # pad
            padded = torch.cat((input_values, torch.zeros(MAX_LENGTH - len(input_values))))
        else: # truncate
            padded = input_values[:MAX_LENGTH]
        inputs.append(padded)
    labels = torch.tensor([item['label'] for item in batch])
    return torch.stack(inputs).unsqueeze(1), labels

# convert dataset col to tensors
dataset['train'].set_format(type='torch', columns=['audio', 'label'])
dataset['test'].set_format(type='torch', columns=['audio', 'label'])

"""# 2. Model Training

## 2.1 Initiate CNN
"""

class AudioCNN(nn.Module):
    def __init__(self, num_classes):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        # self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(2, 2)
        self.fc1 = nn.Linear(64 * 40000, 256)  # input size should be 160000, which is confirmed wile the model was trained
        self.bn_fc1 = nn.BatchNorm1d(256) # batchnorm didn't really work
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        # debug step:
        # print("Shape after conv1:", x.shape)

        x = self.pool(self.relu(self.conv2(x)))
        # print("Shape after conv2:", x.shape)

        # x = self.pool(self.relu(self.conv3(x)))
        # print("Shape after conv3:", x.shape)

        x = x.view(x.size(0), -1)  # Flatten the tensor
        #print("Shape after flattening:", x.shape)

        x = self.dropout(self.relu(self.fc1(x)))
        # x = self.dropout(self.relu(self.bn_fc1(self.fc1(x))))
        # print("Shape after fc1:", x.shape)

        x = self.fc2(x)  # Second fully connected layer (output)
        # print("Shape after fc2:", x.shape)
        return x

cnn = AudioCNN(num_classes).to(device)
print(cnn)

"""## 2.2 Load Loss, Optimizer and Performance Metric"""

loss_fct = nn.CrossEntropyLoss(weight=class_weight_tensor)
optimizer = torch.optim.Adam(cnn.parameters(), lr=3e-3, weight_decay=1e-3)

# load performace metric: accuracy
accuracy = evaluate.load("accuracy")

"""## 2.3 Model Training"""

# directory for model saving
current_time = datetime.now().strftime("%m-%d_%H-%M-%S")
save_dir = f"checkpoints_{current_time}"
os.makedirs(save_dir, exist_ok=True)

num_epochs = 100
batch_size = 16

# load data
train_loader = torch.utils.data.DataLoader(dataset['train'], batch_size=batch_size, collate_fn=collate_fn)
test_loader = torch.utils.data.DataLoader(dataset['test'], batch_size=batch_size, collate_fn=collate_fn)

# init metric containers
train_iter, train_losses, train_accs = [], [], []
test_iter, test_losses, test_accs = [], [], []

best_val_loss = float('inf')

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# train and evaluate
for epoch in range(num_epochs):
  cnn.train()
  step = 0

  if epoch % 5 == 0:
    scheduler.step()

  for inputs, labels in train_loader:
    step += 1

    inputs, labels = inputs.to(device), labels.to(device)
    # print(f"Shape of inputs before passing to model: {inputs.shape}")
    # print(f"Unique values in labels: {torch.unique(labels)}")

    optimizer.zero_grad()
    outputs = cnn(inputs)

    loss = loss_fct(outputs, labels)
    loss.backward()
    optimizer.step()

    if step % 344 == 0:
      train_iter.append(step + epoch*len(train_loader))
      train_losses.append(loss.item())
      train_acc = accuracy.compute(predictions=outputs.argmax(axis=1), references=labels)['accuracy']
      train_accs.append(train_acc)
      print(f"Epoch [{epoch+1}/{num_epochs}], Step [{step}/{len(train_loader)}], Train loss: {loss.item():.4f}, Train accuracy: {train_acc:.4f}")

      # calculate, append and display evaluation reports
      cnn.eval()
      test_loss, test_acc = 0, 0
      with torch.no_grad():
        for inputs, labels in test_loader:
          inputs, labels = inputs.to(device), labels.to(device)
          labels = torch.clamp(labels, 0, num_classes - 1).long()
          outputs = cnn(inputs)
          test_loss += loss_fct(outputs, labels).item()
          test_acc += accuracy.compute(predictions=outputs.argmax(axis=1), references=labels)['accuracy']

        # append to reports
        test_iter.append(step + epoch*len(train_loader))
        test_losses.append(test_loss / len(test_loader))
        test_accs.append(test_acc / len(test_loader))

        # display reports
        print(f"Test Loss: {test_loss / len(test_loader):.4f}, Test Accuracy: {test_acc / len(test_loader):.4f}")

  # At the end of each epoch
  avg_test_loss = test_loss / len(test_loader)
  if avg_test_loss <= best_val_loss:
      best_val_loss = avg_test_loss
      checkpoint_path = os.path.join(save_dir, "best_model.pth")
      torch.save({
          'epoch': epoch + 1,
          'model_state_dict': cnn.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'loss': loss.item(),
      }, checkpoint_path)
      print(f"New best model saved to {checkpoint_path}")

print('')
print(f'Final training loss: {str(train_losses[-1])} accuracy: {str(train_accs[-1])}')
print(f'Final validation loss: {str(test_losses[-1])} accuracy: {str(test_accs[-1])}')


# Save the final model - with datetime_id

final_model_path =  os.path.join(save_dir, f"/final_model.pth")
torch.save(cnn.state_dict(), final_model_path)
print(f"Final model saved to {final_model_path}")

"""## 2.4 Plot Performance"""

# plots of final loss and accuracy of training and validation data
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(train_iter, train_losses, label='Train Loss')
plt.plot(test_iter, test_losses, label='Test Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_iter, train_accs, label='Train Accuracy')
plt.plot(test_iter, test_accs, label='Test Accuracy')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.legend()

plt.suptitle('Training and Validation Loss and Accuracy')

plt.savefig('CNN_performance.png')