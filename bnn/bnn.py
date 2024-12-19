from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator
import torch.nn as nn
import torch
from sklearn.metrics import accuracy_score
import os
import numpy as np
import pandas as pd


@variational_estimator
class BayesianSAPLMA(nn.Module):
    def __init__(self, input_size, dropout=0.2):
        super(BayesianSAPLMA, self).__init__()

        self.network = nn.Sequential(
            BayesianLinear(input_size, 256),  
            nn.ReLU(),
            nn.Dropout(dropout),
            BayesianLinear(256, 128),        
            nn.ReLU(),
            nn.Dropout(dropout),
            BayesianLinear(128, 64),         
            nn.ReLU(),
            nn.Dropout(dropout),
            BayesianLinear(64, 1),           
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)
def train_classifier_bnn(classifier, train_loader, optimizer, criterion, epochs=5, device="cuda"):
    classifier.train()


    for epoch in range(epochs):
        for batch in train_loader:
            X_batch, y_batch = batch
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()           
            loss = classifier.sample_elbo(
                inputs=X_batch,
                labels=y_batch,
                criterion=lambda preds, targets: criterion(preds.squeeze(), targets),
                sample_nbr=20,
            )
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")


def evaluate_classifier_bnn(classifier, test_loader, criterion,layer, device="cuda", num_samples=10):
    classifier.eval()
    y_true = []
    y_pred = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in test_loader:
            X_batch, y_batch = batch
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Generate multiple samples for Bayesian predictions
            outputs = [classifier(X_batch) for _ in range(num_samples)]
            outputs = torch.stack(outputs)  # Shape: (num_samples, batch_size, output_dim)
            pred_mean = outputs.mean(dim=0).flatten()  # Mean across samples

            # Compute loss
            loss = criterion(pred_mean, y_batch)
            total_loss += loss.item()

            # Collect predictions and true labels
            y_true.append(y_batch)
            y_pred.append(pred_mean)

    # Concatenate all predictions and true labels
    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)

    # Create CSV for ECE
    create_csv_for_ece(y_true, y_pred,layer)

    # Calculate accuracy
    y_pred_binary = y_pred > 0.5  # Threshold at 0.5
    y_pred_binary = y_pred_binary.flatten()
    correct_predictions = (y_pred_binary == y_true).sum()

    accuracy = correct_predictions / len(y_true)

    print(f"Average Loss: {total_loss / len(test_loader)}, Accuracy: {accuracy}")
    return total_loss / len(test_loader), accuracy



def create_csv_for_ece(y_true, y_pred, layer):


    data_folder = os.path.join(os.path.dirname(__file__), 'data')
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    # Convert tensors to numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    # Ensure arrays are 1D
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)

    # Prepare data for CSV
    data_dict = {'prediction': y_pred, 'label': y_true}

    # Create DataFrame and save
    df_chatgpt4 = pd.DataFrame(data_dict)
    df_chatgpt4.to_csv(os.path.join(data_folder, f'bnn_ece_{layer}.csv'), index=False)

