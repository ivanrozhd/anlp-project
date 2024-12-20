import pandas as pd
import numpy as np
import argparse
from dataset_scripts.load_data import create_logits_samples


# ECE calculation based on the code https://towardsdatascience.com/expected-calibration-error-ece-a-step-by-step-visual-explanation-with-python-code-c3e9aa12937d


def prepare_data(file_path):
    # Load the CSV file
    data = pd.read_csv(file_path)

    # Prepare samples and true labels
    predictions = data['prediction'].values.reshape(-1, 1)  # Reshape to 2D array
    labels = data['label'].values.astype(int)  # Ensure labels are integers

    # Add a second column for the complementary probability (1 - prediction)
    samples = np.hstack((predictions, 1 - predictions))
    return samples, labels

def expected_calibration_error(samples, true_labels, M=5):
    # uniform binning approach with M number of bins
    bin_boundaries = np.linspace(0, 1, M + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # get max probability per sample i
    confidences = np.max(samples, axis=1)
    # get predictions from confidences (positional in this case)
    predicted_label = np.argmax(samples, axis=1)

    # get a boolean list of correct/false predictions
    accuracies = predicted_label==true_labels

    ece = np.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # determine if sample is in bin m (between bin lower & upper)
        in_bin = np.logical_and(confidences > bin_lower.item(), confidences <= bin_upper.item())
        # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
        prob_in_bin = in_bin.mean()

        if prob_in_bin.item() > 0:
            # get the accuracy of bin m: acc(Bm)
            accuracy_in_bin = accuracies[in_bin].mean()
            # get the average confidence of bin m: conf(Bm)
            avg_confidence_in_bin = confidences[in_bin].mean()
            # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin
    return ece



def main(data):
    samples, labels = prepare_data(data)
    ece = expected_calibration_error(samples, labels)
    print(f"ECE: {ece.item():.4f}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_data", type=str, required=True)
    args = parser.parse_args()
    file_path = args.file_data
    main(file_path)