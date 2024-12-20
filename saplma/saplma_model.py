import numpy as np
import torch.nn as nn
import torch
from sklearn.metrics import accuracy_score
import os
import pandas as pd

# SAPLMA Classifier from https://arxiv.org/abs/2304.13734
class SaplmaClassifier(nn.Module):
    def __init__(self, input_size, dropout=0.2):
        super(SaplmaClassifier, self).__init__()

        self.network = nn.Sequential(

            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)



def train_classifier_saplma(classifier, train_loader, optimizer, criterion, epochs=30, device="cpu"):
    classifier.train()
    for epoch in range(epochs):
        for batch in train_loader:
            X_batch, y_batch = batch
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = classifier(X_batch)

            loss = criterion(outputs.flatten(), y_batch)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

def evaluate_classifier_saplma(classifier, test_loader, layer, device="cpu"):
    classifier.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in test_loader:
            X_batch, y_batch = batch
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = classifier(X_batch)
            y_true.append(y_batch)
            y_pred.append(outputs)

    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)
    create_csv_for_ece(y_true, y_pred, layer)

    y_pred = y_pred > 0.5
    y_pred = y_pred.flatten()
    correct_predictions = (y_pred == y_true).sum()

    accuracy = correct_predictions / len(y_true)

    print(f"Accuracy: {accuracy}")
    return accuracy


def create_csv_for_ece(y_true, y_pred, layer):
    data_folder = os.path.join(os.path.dirname(__file__), 'data')
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)

    dict = {'prediction': y_pred, 'label': y_true}

    df_chatgpt4 = pd.DataFrame(dict)
    df_chatgpt4.to_csv(os.path.join(data_folder, f'saplma_ece_{layer}.csv'), index=False)




