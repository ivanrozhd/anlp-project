import os
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import torch
import argparse
from load_data import load_and_prepare_data, create_csv


device = "cuda:0" if torch.cuda.is_available() else "cpu"

def initialize_model(model_name, token):

    os.environ["HUGGING_FACE_TOKEN"] = token
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        eos_token_id = tokenizer.eos_token_id,
        pad_token_id = tokenizer.pad_token_id
    )
    return tokenizer, model


def generate_response(tokenizer, model, question, max_new_tokens=6, layer_step=5):
    set_seed(18)
    # Tokenize the input question and feed them into the model
    inputs = tokenizer(question, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        output_scores=True,
        output_hidden_states=True,
        pad_token_id=tokenizer.eos_token_id
    )

    # Extract hidden states from specific layers
    hidden_states_1 = []
    hidden_states_16 = []
    hidden_states_32 = []


    hidden_state_1 = outputs.hidden_states[1][1].mean(dim=1)
    hidden_state_16 = outputs.hidden_states[1][16].mean(dim=1)
    hidden_state_32 = outputs.hidden_states[1][32].mean(dim=1)


    hidden_states_1.append(hidden_state_1.squeeze().tolist())
    hidden_states_16.append(hidden_state_16.squeeze().tolist())
    hidden_states_32.append(hidden_state_32.squeeze().tolist())


    # Decode the generated sequence
    generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    generated_response = generated_text.split(question, 1)[-1].strip()

    return generated_response, outputs, hidden_states_1, hidden_states_16, hidden_states_32


def main(access_token, model_name,file_path):

    questions, references = load_and_prepare_data(file_path)

    # Initialize models
    tokenizer, model = initialize_model(model_name, access_token)
    length_set = set()
    for i in references:
        length_set.add(len(tokenizer.tokenize(i)))
    print(length_set, max(length_set))
    # Generate and evaluate responses
    responses = []
    hidden_states_1 = []
    hidden_states_16 = []
    hidden_states_32 = []
    logits_of_answers = []

    for i, question in enumerate(questions):

        # Generate a response
        response, outputs, hidden_state_1, hidden_state_16, hidden_state_32  = generate_response(tokenizer, model, question)
        responses.append(response)
        hidden_states_1.append(hidden_state_1)
        hidden_states_16.append(hidden_state_16)
        hidden_states_32.append(hidden_state_32)


        # logits extraction
        question_length = len(tokenizer(question)['input_ids'])
        logits = torch.nn.functional.softmax(torch.cat(outputs.scores,0), dim=1)
        average_probability = 0

        for j in range(len(outputs[0][0][question_length:])):
            average_probability += logits[j, outputs[0][0][question_length:][j]]
        average_probability /= 6
        logits_of_answers.append(average_probability.item())

        # Print the question
        print(f"Question {i+1}: {question} {response}; GT: {references[i]}")




    # Create a CSV file with the data
    create_csv(questions, references, responses, hidden_states_1, hidden_states_16, hidden_states_32, logits_of_answers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--filepath", type=str, required=True)
    args = parser.parse_args()
    main(args.token, args.model, args.filepath)
