import os

# for data processing
import numpy as np
from utils import load_and_merge_batches, select_n_samples
from sklearn.utils.class_weight import compute_class_weight

# for model training and evaluation
import torch
from torch import nn
import evaluate
from datetime import datetime
import matplotlib.pyplot as plt

# enforcing reproducibility - from NLP - lab 2 notebook
import random
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

# set to work on GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# load dataset 
dataset = load_and_merge_batches("processed_data")
num_classes = len(set(dataset['label']))

# FOR TESTING!!! Work only on a balanced subset of the dataset
dataset = select_n_samples(dataset, 50)

# split dataset
dataset = dataset.train_test_split(test_size=0.1, shuffle=True, stratify_by_column="label", seed=42)

# compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(dataset['train']['label']), y=dataset['train']['label'])
class_weights = {class_id: weight for class_id, weight in zip(np.unique(dataset['train']['label']), class_weights)}
class_weight_tensor = torch.tensor(list(class_weights.values()), dtype=torch.float32).to(device)

 # Use the longest sequence length in your dataset
MAX_LENGTH = 58050

# Define CNN 
class AudioCNN(nn.Module):
    def __init__(self, num_classes):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(5)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool1d(5)
        self.fc1 = nn.Linear(32 * 2322, 128)  
        #self.bn_fc1 = nn.BatchNorm1d(256) 
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        # debug step:
        # print("Shape after conv1:", x.shape)

        x = self.pool2(self.relu(self.conv2(x)))
        # print("Shape after conv2:", x.shape)

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
trainable_params = sum(p.numel() for p in cnn.parameters() if p.requires_grad)
print(f"Model has {trainable_params} trainable parameters.")


"""## 2.2 Load Loss, Optimizer and Performance Metric"""

loss_fct = nn.CrossEntropyLoss(weight=class_weight_tensor)
optimizer = torch.optim.Adam(cnn.parameters(), lr=3e-3, weight_decay=1e-3)

# load performace metric
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

"""## 2.3 Model Training"""

# directory for model saving
current_time = datetime.now().strftime("%m-%d_%H-%M-%S")
save_dir = f"./CNN_training_{current_time}"
os.makedirs(save_dir, exist_ok=True)

num_epochs = 1
batch_size = 4

# load data
# Define padding function
def collate_fn(batch):
    inputs = []
    for item in batch:
        input_values = item['audio'].clone().detach()

        # manually pad or truncate to desired lenght
        if len(input_values) < MAX_LENGTH: 
            padded = torch.cat((input_values, torch.zeros(MAX_LENGTH - len(input_values))))
        else: 
            padded = input_values[:MAX_LENGTH]
        inputs.append(padded)

    labels = torch.tensor([item['label'] for item in batch])
    return torch.stack(inputs).unsqueeze(1), labels

# convert dataset columns to tensors
dataset['train'].set_format(type='torch', columns=['audio', 'label'])
dataset['test'].set_format(type='torch', columns=['audio', 'label'])

train_loader = torch.utils.data.DataLoader(dataset['train'], batch_size=batch_size, collate_fn=collate_fn)
test_loader = torch.utils.data.DataLoader(dataset['test'], batch_size=batch_size, collate_fn=collate_fn)

# init metric containers
train_iter, train_losses, train_accs = [], [], []
test_iter, test_losses, test_accs = [], [], []

best_val_loss = float('inf')
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Load accuracy and F1 metrics
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

# Train and evaluate with F1-score logging
for epoch in range(num_epochs):
    cnn.train()
    step = 0

    if epoch % 5 == 0:
        scheduler.step()

    # Training Loop
    for inputs, labels in train_loader:
        step += 1

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = cnn(inputs)

        loss = loss_fct(outputs, labels)
        loss.backward()
        optimizer.step()

    # At the end of the epoch, log training metrics
    all_train_preds, all_train_labels = [], []
    cnn.eval()
    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = cnn(inputs)
            preds = outputs.argmax(axis=1)

            all_train_preds.extend(preds.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())

        # Compute training metrics using evaluate
        train_accuracy = accuracy.compute(predictions=all_train_preds, references=all_train_labels)['accuracy']
        train_f1 = f1.compute(predictions=all_train_preds, references=all_train_labels, average='weighted')['f1']

    print(f"Epoch [{epoch+1}/{num_epochs}] Training: Loss: {loss.item():.4f}, Accuracy: {train_accuracy:.4f}, F1-Score: {train_f1:.4f}")

    # Testing Loop
    all_test_preds, all_test_labels = [], []
    test_loss = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = cnn(inputs)
            preds = outputs.argmax(axis=1)

            all_test_preds.extend(preds.cpu().numpy())
            all_test_labels.extend(labels.cpu().numpy())

            test_loss += loss_fct(outputs, labels).item()

        # Compute testing metrics using evaluate
        test_accuracy = accuracy.compute(predictions=all_test_preds, references=all_test_labels)['accuracy']
        test_f1 = f1.compute(predictions=all_test_preds, references=all_test_labels, average='weighted')['f1']
        test_loss /= len(test_loader)

    # Log test metrics
    print(f"Epoch [{epoch+1}/{num_epochs}] Testing: Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, F1-Score: {test_f1:.4f}")

    # Save the best model
    if test_loss <= best_val_loss:
        best_val_loss = test_loss
        checkpoint_path = os.path.join(save_dir, "best_model.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': cnn.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, checkpoint_path)
        print(f"New best model saved to {checkpoint_path}")

# Final log
print(f'Final training loss: {loss.item():.4f}, Accuracy: {train_accuracy:.4f}, F1-Score: {train_f1:.4f}')
print(f'Final validation loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, F1-Score: {test_f1:.4f}')

# Save the final model - with datetime_id
final_model_path =  os.path.join(save_dir, f"final_model.pth")
torch.save(cnn.state_dict(), final_model_path)
print(f"Final model saved to {final_model_path}")