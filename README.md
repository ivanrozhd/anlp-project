# ANLP Deep Learning Project

This is the repository for the course project of **Advanced Natural Language Processing and Deep Learning** (Autumn 2024) at IT University of Copenhagen.

---

## Group Name and Members

**Group 8 - bilibili**  
- Ivan Rozhdestvenskii  
- Levente András Wallis  

---

## Repository Structure

- **`bnn/`**  
  Contains the code for the **Bayesian Neural Network (BNN)** implementation.  
  Includes a `data/` folder with evaluation results used for **Expected Calibration Error (ECE)** calculation.

- **`saplma/`**  
  Contains the code for the **SAPLMA model** implementation.  
  Includes a `data/` folder with evaluation results used for **Expected Calibration Error (ECE)** calculation.

- **`data/`**  
  Contains all the relevant data files required for model training and evaluation.

- **`dataset_scripts/`**  
  Contains the scripts for **dataset creation** based on the provided dataset file (`natural_questions_sample.csv`).

- **`ece/`**  
  Contains the code for **Expected Calibration Error (ECE)** calculation.

---

## Datasets



---

## Example Commands

### Dataset Creation

Example command:
```shell
python dataset_creation.py  --token gt_dasdasdasdasdsafrgwr --model meta-llama/Llama-3.1-8B-Instruct --filepath data/natural_questions_sample.csv
```

## Output

### 1. **`chatgpt4_evaluation.csv`**
   - **Contains**:
     - Questions  
     - Short answers (ground truth)  
     - Responses from the pretrained model  
   - **Purpose**: Evaluation of model responses.

### 2. **`dataset_hidden_layers.csv`**
   - **Contains**:
     - Questions  
     - Short answers (ground truth)  
     - Hidden layer representations (specifically layers 1, 16, and 32)  
   - **Purpose**: Used for training purposes.

### 3. **`dataset_logits.csv`**
   - **Contains**:
     - Questions  
     - Short answers (ground truth)  
     - Model responses  
     - Logits of the answers  
   - **Purpose**: Used for detailed evaluation and analysis of model confidence.


### ChatGPT4 Evaluation
evaluated.csv - a csv file with labels for the responses from chatgpt4 (1 - correct, 0 - incorrect), whether the responses of our `meta-llama/Llama-3.1-8B-Instruct` are correct or incorrect

### Train a SAPLMA/BayseianSAPLMA model for classification of the questions whether the model is able to answer them
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


