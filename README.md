# ITU_ANLP_Final
This is the repository of the course project of Advanced Natural Language Processing and Deep Learning (Autumn 2024) at IT University of Copenhagen.

# Group Name and Members
Group 8 bilibili: Ivan Rozhdestvenskii, Levente András Wallis and Shiling Deng (Lost during the project)


# Create three datasets: chatgpt4_evaluation.csv, dataset_hidden_layers.csv and dataset_logits.csv
chatgpt4_evaluation.csv - contains the questions, short answers (gt) and responses from the pretrained model - for evaluation purposes
dataset_hidden_layers.csv - contains the questions, short answers (gt), hidden layers representations for training purposes - specifically layers 1, 16, 32
dataset_logits.csv - contains questions, short_answers, responses, logits_of_answers

Example command:
```shell
python dataset_creation.py  --token gt_dasdasdasdasdsafrgwr --model meta-llama/Llama-3.1-8B-Instruct --filepath data/natural_questions_sample.csv
```


# Train a SAPLMA/BayseianSAPLMA model for classification of the questions whether the model is able to answer them
Example command:
-hidden_state_path - csv file with hidden states
-labels_file - csv file with labels evaluated by chatgpt4
-arc - architecture of the model (saplma or bnn)
-layer - layer of the model (1, 16, 32)
```shell
python classifier_train.py  --hidden_states_path dataset_training.csv --labels_file Evaluation_of_Responses.csv --arc saplma/bnn --layer 1/16/32

```

# Help method to concatenate logits with labels

```shell
python logits_label.py  --label_data data/Short_Answer_Evaluation_Results.csv  --logits_data dataset_scripts/data/dataset_logits.csv 

```

# Calculate ECE for the model

returns the estimated calibration error for the model

Example command:
```shell
python ece-evaluation.py  --file_data bnn/data/bnn_ece_1.csv 

```


