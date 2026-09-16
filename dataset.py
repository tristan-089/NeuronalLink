import torch
from tokenizer import Tokenizer

# Charger le model
with open("data/conversations.txt", "r", encoding="utf-8") as file:
    text = file.read()

tokenizer = Tokenizer()
tokenizer.build_vocabulary([text])

tokens = tokenizer.tokenize(text)
token_ids = tokenizer.encode(text)

print(f"Nombre de tokens: {len(tokens)}")
print(f"Taille du Vocabulaire: {len(tokenizer.token_to_id)}")

print()
print("Premiers Tokens: ")
print(tokens[:20])

print()
print("Premiers IDs: ")
print(token_ids[:20])

inputs = []
targets = []

sequence_length = 8

for i in range(len(token_ids) - sequence_length):
    input_sequence = token_ids[i:i+sequence_length]
    target_sequence = token_ids[i + 1:i + sequence_length + 1]

    inputs.append(input_sequence)
    targets.append(target_sequence)

inputs = torch.tensor(inputs, dtype=torch.long)
targets = torch.tensor(targets, dtype=torch.long)

print()
print("Taille des Inputs: ", inputs.shape)
print("Taille des Targets: ", targets.shape)

