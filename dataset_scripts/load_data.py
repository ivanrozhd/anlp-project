import json
import os
import csv
import pandas as pd


def load_and_prepare_data(file_path):  # assume it is csv data # Create empty lists to store the extracted data
    ids = []
    questions = []
    short_answers = []

    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)

        # Iterate over the rows and append each relevant field to the lists
        for row in reader:
            ids.append(row['ID'])
            questions.append(row['question'])
            short_answers.append(row['short_answers'])

    return ids, questions, short_answers


def load_and_prepare_data(file_path):  # assume it is csv data # Create empty lists to store the extracted data

    questions = []
    short_answers = []

    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)

        # Iterate over the rows and append each relevant field to the lists
        for row in reader:
            questions.append(row['question'])
            short_answers.append(row['short_answers'])

    return questions, short_answers


def create_csv(questions, short_answers, responses, hidden_states_1, hidden_states_16, hidden_states_32, logits_of_answers):

    data_folder = os.path.join(os.path.dirname(__file__), 'data')

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    # dictionary of lists
    dict = {'questions': questions, 'short_answers': short_answers, 'responses': responses}

    df_chatgpt4 = pd.DataFrame(dict)
    df_chatgpt4.to_csv(os.path.join(data_folder, 'data/chatgpt4_evaluation.csv'))

    dict[f'hidden_state_{1}'] = hidden_states_1
    dict[f'hidden_state_{16}'] = hidden_states_16
    dict[f'hidden_state_{32}'] = hidden_states_32
    df = pd.DataFrame(dict)

    # saving the dataframe
    df.to_csv(os.path.join(data_folder, 'data/dataset_hidden_layers.csv'))

    dict1 = {'questions': questions, 'short_answers': short_answers, 'responses': responses,
             'logits_of_answers': logits_of_answers}
    df_logits = pd.DataFrame(dict1)
    df_logits.to_csv(os.path.join(data_folder, 'data/dataset_logits.csv'))


def extract_hidden_states_with_labels(hidden_states_file, labels_file, layer):
    labels = []
    with open(labels_file, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            labels.append(row['matches'])

    hidden_states = []

    with open(hidden_states_file, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            hidden_states.append(row[f'hidden_state_{layer}'])

    return hidden_states, labels


def create_logits_samples(matches_file_path, logits_file_path):
    logits_data = pd.read_csv(logits_file_path)
    matches_data = pd.read_csv(matches_file_path)

    if not logits_data.index.equals(matches_data.index):
        logits_data.reset_index(drop=True, inplace=True)
        matches_data.reset_index(drop=True, inplace=True)

    # Combine the two datasets
    combined_data = pd.DataFrame({
        'prediction': logits_data['logits_of_answers'],
        'label': matches_data['matches']
    })

    # Extract the last 20% of rows
    last_20_percent = combined_data.iloc[-int(0.2 * len(combined_data)):]

    # Optionally save the results to a new CSV file
    last_20_percent.to_csv("data/logits_ece.csv", index=False)

    return last_20_percent

