import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from sklearn.model_selection import train_test_split
import json


def split_data(hidden_states, labels, test_size=0.2, random_state=42):
    hidden_states = [json.loads(question_vector) for question_vector in hidden_states]
    hidden_states = np.array(hidden_states, dtype=np.float32)

    labels = np.array(labels, dtype=np.float32)

    print(labels.shape)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        hidden_states, labels, test_size=test_size, random_state=random_state,shuffle=False
    )
    # Convert to PyTorch tensors
    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )



