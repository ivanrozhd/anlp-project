import torch
from saplma.saplma_model import SaplmaClassifier, train_classifier_saplma, evaluate_classifier_saplma
from bnn.bnn import BayesianSAPLMA, train_classifier_bnn, evaluate_classifier_bnn
from dataset_scripts.load_data import extract_hidden_states_with_labels
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main( hidden_states_file, labels_file, arc, layer):

    if layer not in (1, 16, 32):
        print("Invalid layer")
        return "Invalid layer"
    # Load data
    hidden_states, labels = extract_hidden_states_with_labels(hidden_states_file,labels_file, layer=layer)
    input_size = 4096 #len(hidden_states[0])

    if arc == "bnn":
        classifier = BayesianSAPLMA(input_size).to(device)

    elif arc == "saplma":
        classifier = SaplmaClassifier(input_size).to(device)

    else:
        print("Invalid architecture")
        return "Invalid architecture"


    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.00001)

    X_train, X_test, y_train, y_test = split_data(hidden_states, labels)
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=512)

    if arc == "bnn":
        train_classifier_bnn(classifier, train_loader, optimizer, criterion, epochs=5, device=device)
        test_loss, test_accuracy = evaluate_classifier_bnn(classifier, test_loader, criterion, layer, device=device)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    elif arc == "saplma":
        train_classifier_saplma(classifier, train_loader, optimizer, criterion, epochs=85, device=device)
        evaluate_classifier_saplma(classifier, test_loader, layer, device=device)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_states_path", type=str, required=True)
    parser.add_argument("--labels_file", type=str, required=True)
    parser.add_argument("--arc", type=str, required=True)
    parser.add_argument("--layer", type=int, required=True)
    args = parser.parse_args()
    print(f"Using device: {device}")
    hidden_states = args.hidden_states_path
    labels = args.labels_file
    arc = args.arc
    layer = args.layer

    main(hidden_states, labels, arc, layer)