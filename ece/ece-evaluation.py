













if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--saplma-data", type=str, required=True)
    parser.add_argument("--bnn-data", type=str, required=True)
    parser.add_argument("--logits-data", type=str, required=True)
    args = parser.parse_args()
    main(args.saplma-data, args.bnn-data, args.logits-data)