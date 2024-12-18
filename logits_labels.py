from dataset_scripts.load_data import create_logits_samples
import argparse

# One method to extract essential data from both datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_data", type=str, required=True)
    parser.add_argument("--logits_data", type=str, required=True)
    args = parser.parse_args()
    create_logits_samples(args.label_data, args.logits_data)
