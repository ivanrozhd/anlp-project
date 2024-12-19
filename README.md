# anlp-deep-learning-project
This is the repository of the course project of Advanced Natural Language Processing and Deep Learning (Autumn 2024) at IT University of Copenhagen.

# Group Name and Members
Group 8 bilibili: Ivan Rozhdestvenskii and Levente András Wallis 

Following is the structure of the project and steps to run the code:

bnn - contains the code for Bayesian Neural Network and data folder with evluation results for ECE calculation 
saplma - contains the code for SAPLMA and data folder with evluation results for ECE calculation
data - contains the data files
dataset_scripts - contains the scripts for dataset creation based on the dataset provided (natural_questions_sample.csv)
ece - contains the code for ECE calculation

# Create three datasets: chatgpt4_evaluation.csv, dataset_hidden_layers.csv and dataset_logits.csv
chatgpt4_evaluation.csv - contains the questions, short answers (gt) and responses from the pretrained model - for evaluation purposes
dataset_hidden_layers.csv - contains the questions, short answers (gt), hidden layers representations for training purposes - specifically layers 1, 16, 32
dataset_logits.csv - contains questions, short_answers, responses, logits_of_answers

Example command:
```shell
python dataset_creation.py  --token gt_dasdasdasdasdsafrgwr --model meta-llama/Llama-3.1-8B-Instruct --filepath data/natural_questions_sample.csv
```

evaluated.csv - a csv file with labels for the responses from chatgpt4 (1 - correct, 0 - incorrect)

# Train a SAPLMA/BayseianSAPLMA model for classification of the questions whether the model is able to answer them
Example command:
-hidden_state_path - csv file with hidden states
-labels_file - csv file with labels evaluated by chatgpt4
-arc - architecture of the model (saplma or bnn)
-layer - layer of the model (1, 16, 32)
```shell
python classifier_train.py  --hidden_states_path dataset_hidden_layers.csv --labels_file evaluated.csv --arc saplma/bnn --layer 1/16/32

```

# Help method to concatenate logits with labels

```shell
python logits_label.py  --label_data evaluated.csv  --logits_data dataset_logits.csv 

```

# Calculate ECE for the model

returns the estimated calibration error for the model

Example command:
```shell
python ece-evaluation.py  --file_data bnn_ece_1.csv 

```


