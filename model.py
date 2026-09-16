import torch
import torch.nn as nn
import math

from tokenizer import Tokenizer

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, max_sequence_length=128):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)

        self.position_embedding = nn.Embedding(max_sequence_length, embedding_dim)

    def forward(self, input_ids):
        # Nombre de tokens dans la séquence
        sequence_length = input_ids.shape[1]

        token_embeddings = self.token_embedding(input_ids)
        positions = torch.arange(sequence_length, device=input_ids.device)

        position_embeddings = self.position_embedding(positions)

        embeddings = (token_embeddings + position_embeddings)
        return embeddings

class SelfAttention(nn.Module):
    def __init__(self, embedding_dim, max_sequence_length=128):
        super().__init__()

        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)

        mask = torch.tril(torch.ones(max_sequence_length, max_sequence_length))

        self.register_buffer("mask", mask)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        scores = Q @ K.transpose(-2, -1)
        dimension = K.shape[-1]
        scores = scores / math.sqrt(dimension)

        sequence_length = x.shape[1]
        mask = self.mask[:sequence_length, :sequence_length]
        scores = scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = torch.softmax( scores, dim=-1)
        output = attention_weights @ V
        return output

