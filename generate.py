import torch
from tokenizer import Tokenizer
from model import MiniGPT

MODEL_PATH = "mini_gpt.pth"

checkpoint = torch.load(MODEL_PATH, map_location="cpu")

tokenizer = Tokenizer()
tokenizer.token_to_id = checkpoint["token_to_id"]
tokenizer.rebuild_id_to_token()

model = MiniGPT(
    vocab_size=len(tokenizer.token_to_id),
    embedding_dim=checkpoint["embedding_dim"],
    num_layers=checkpoint["num_layers"],
    max_sequence_length=checkpoint["sequence_length"]
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

prompt = input("Vous : ")

input_ids = tokenizer.encode(prompt)
input_ids = torch.tensor([input_ids], dtype=torch.long)

with torch.no_grad():
    output_ids = model.generate(input_ids, max_new_tokens=30)

response = tokenizer.decode(output_ids[0].tolist())
print("IA :", response)