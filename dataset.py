import torch
from tokenizer import Tokenizer

def create_dataset(file_path, sequence_length=32):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    tokenizer = Tokenizer()
    tokenizer.build_vocabulary([text])

    token_ids = tokenizer.encode(text)

    inputs = []
    targets = []


    for i in range(len(token_ids) - sequence_length):
        input_sequence = token_ids[i:i + sequence_length]
        target_sequence = token_ids[i + 1:i + sequence_length + 1]

        inputs.append(input_sequence)
        targets.append( target_sequence)

    inputs = torch.tensor(inputs, dtype=torch.long)

    targets = torch.tensor(targets, dtype=torch.long)

    return (inputs, targets, tokenizer)
