import torch
import torch.nn as nn

from dataset import create_dataset
from model import MiniGPT


# CONFIGURATION
DATASET_PATH = "data/conversations.txt"
EPOCHS = 100
SEQUENCE_LENGTH = 32
EMBEDDING_DIM = 128
NUM_LAYERS = 4
LEARNING_RATE = 0.0003
MODEL_PATH = "mini_gpt.pth"

# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device utilisé :", device)

inputs, targets, tokenizer = create_dataset(DATASET_PATH, sequence_length=SEQUENCE_LENGTH)

print("Nombre d'exemples :", len(inputs))
print("Taille du vocabulaire :", len(tokenizer.token_to_id))

# MODELE
model = MiniGPT(
    vocab_size=len(tokenizer.token_to_id), embedding_dim=EMBEDDING_DIM, num_layers=NUM_LAYERS, max_sequence_length=SEQUENCE_LENGTH)

model = model.to(device)

# LOSS
criterion = nn.CrossEntropyLoss()

# OPTIMIZER
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# ENTRAINEMENT
model.train()


for epoch in range(EPOCHS):
    total_loss = 0
    permutation = torch.randperm(inputs.size(0))

    for index in permutation:
        input_ids = inputs[index]
        target_ids = targets[index]

        input_ids = input_ids.unsqueeze(0)
        target_ids = target_ids.unsqueeze(0)

        #GPU/CPU
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        optimizer.zero_grad()

        logits = model(input_ids)

        # Calcul de la Loss
        loss = criterion(logits.reshape(-1,logits.size(-1)), target_ids.reshape(-1))
        loss.backward()

        optimizer.step()
        total_loss += loss.item()

    average_loss = (total_loss / len(inputs))

    print(f"Epoch {epoch + 1:03d}/{EPOCHS} "f"| Loss: {average_loss:.4f}")

# SAUVEGARDE
torch.save(model.state_dict(), MODEL_PATH)

print()
print(f"Modèle sauvegardé dans : {MODEL_PATH}")
