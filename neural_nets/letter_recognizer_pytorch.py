import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from pathlib import Path
import random

csv_path = Path(__file__).parent.parent / "data" / "A_Z Handwritten Data.csv"
print(csv_path.resolve())
print(csv_path.exists())

data = pd.read_csv(csv_path)

data = torch.tensor(data.values, dtype=torch.float32) # array of letters, in number form

letters = {
    0 : 'A', 1 : 'B', 2 : 'C', 3 : 'D', 4 : 'E', 5 : 'F', 6 : 'G', 7 : 'H', 8 : 'I',
    9 : 'J', 10 : 'K', 11 : 'L', 12 : 'M', 13 : 'N', 14 : 'O', 15 : 'P', 16 : 'Q', 17 : 'R',
    18 : 'S', 19 : 'T', 20 : 'U', 21 : 'V', 22 : 'W', 23 : 'X', 24 : 'Y', 25 : 'Z'
}

def convert_to_letter(num):
    return letters.get(num, '?')  # Returns '?' if num isn't in the dictionary

examples, features = data.shape
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Shuffle the rows of the tensor
indices = torch.randperm(data.size(0))  # generates a random permutation of row indices (tensor[132, 523, 23, 15512, 1506, ... etc])

# Split the full dataset into features and labels
X = data[:, 1:] / 255  # normalize pixel values, since its a 2D array, need to select all rows, and from column 1 to end (since 1 is labels)
Y = data[:, 0].long()  # class labels (first column)

'''
Use stratified sampling to split 10,000 dev, rest for training
Basically this takes all the data (X and Y), then splits it and ensures that the distribution
is even. Sklearn is very powerful for modifying data to work with machine learning.

Most of this is done in stratify, it ensures that the labels have even spread, since just shuffling
the data like before isn't a guaranteed way to make an even distribution.
'''
X_train_np, X_dev_np, Y_train_np, Y_dev_np = train_test_split(
    X.numpy(), Y.numpy(),
    test_size=10000,
    stratify=Y.numpy(),
    random_state=42
)

# Convert back to tensors and move to device
X_train = torch.tensor(X_train_np, dtype=torch.float32).to(device)
Y_train = torch.tensor(Y_train_np, dtype=torch.long).to(device)
X_dev = torch.tensor(X_dev_np, dtype=torch.float32).to(device)
Y_dev = torch.tensor(Y_dev_np, dtype=torch.long).to(device)

# Create train loader
train_dataset = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# Debug: Show class distribution in dev set
unique, counts = torch.unique(Y_dev.cpu(), return_counts=True)

'''
nn.Linear is basically the whole layer. It generates random biases and weights by default.
784 is the amount of inputs, and 128 is the amount of outputs we want. 
Fpr example, fc1 has 128 neurons with (784*128) weight connections, and 128 biases.
nn.Linear also generates the dot product of everything automatically, so no need to do
that manually in forward_prop

F is what we imported in the beginning, it has a lot of activation functions we'll need, but 
the difference between using F.relu from nn.ReLU is that nn.ReLU applies the activation, then
stores it as a layer, whereas F.relu simply applies the activation to a given layer.
'''
class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)  # input to hidden1
        self.fc2 = nn.Linear(128, 64)   # hidden1 to hidden2
        self.fc3 = nn.Linear(64, 26)    # hidden2 to output
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def gradient_descent(epochs):
    letter_model = model().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(letter_model.parameters(), lr=0.001)

    for epoch in range(epochs):
        total_correct = 0
        total = 0
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            # Forward pass
            Y_pred = letter_model(X_batch)
            loss = loss_fn(Y_pred, Y_batch)

            # Accuracy tracking
            preds = torch.argmax(Y_pred, dim=1)
            total_correct += (preds == Y_batch).sum().item()
            total += Y_batch.size(0)

            # Backward + optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        accuracy = total_correct / total
        if epoch % 10 == 0:
            letter_model.eval()
            with torch.inference_mode():
                dev_preds = letter_model(X_dev)
                dev_preds = torch.argmax(dev_preds, dim=1)
                dev_correct = (dev_preds == Y_dev).sum().item()
                dev_acc = dev_correct / Y_dev.size(0)
            letter_model.train()
            
            print(f"Epoch: {epoch} | Train Accuracy: {accuracy:.4f} | Dev Accuracy: {dev_acc:.4f}")
    return letter_model

MODEL_PATH = Path("../models")
MODEL_NAME = "pytorch_letter_recognizer.pth"
full_path = MODEL_PATH / MODEL_NAME

if full_path.exists():
    print(f"Model found! Loading model..")
    loaded_model = model().to(device)
    loaded_model.load_state_dict(torch.load(full_path))
else:
    print(f"No model detected, training a new model..")
    trained_model = gradient_descent(epochs=20)
    print(f"Saving new model to: {MODEL_PATH}")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)  # Ensure folder exists
    torch.save(trained_model.state_dict(), full_path)

# Plot image
idx = random.randint(0, X_dev.size(0) - 1)

# Get image and label
img = X_dev[idx].cpu().reshape(28, 28)
true_label = convert_to_letter(Y_dev[idx].item())

# Use trained or loaded model
model_to_use = trained_model if 'trained_model' in locals() else loaded_model
model_to_use.eval()

# Predict
with torch.inference_mode():
    output = model_to_use(X_dev[idx].unsqueeze(0))  # Add batch dimension
    pred = torch.argmax(output, dim=1).item()
    predicted_letter = convert_to_letter(pred)

# Plot image
plt.imshow(img, cmap='gray')
plt.title(f"Predicted: {predicted_letter} | Actual: {true_label}")
plt.axis('off')
plt.savefig("prediction_torch.png")
print("Saved plot as prediction_torch.png")
