import torch
import torch.nn as nn
import math

from tokenizer import Tokenizer

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, num_layers=4, max_sequence_length=128, dropout=0.1):
        super().__init__()

        self.max_sequence_length = max_sequence_length
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_sequence_length, embedding_dim)
        self.blocks = nn.ModuleList([TransformerBlock(embedding_dim=embedding_dim, max_sequence_length=max_sequence_length,
            dropout=dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(embedding_dim)

        self.output_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, input_ids):
        batch_size, sequence_length = input_ids.shape

        # Vérification
        if sequence_length > self.max_sequence_length:
            raise ValueError("La séquence est trop longue")

        token_embeddings = self.token_embedding(input_ids)
        positions = torch.arange(sequence_length, device=input_ids.device)
        position_embeddings = self.position_embedding(positions)

        x = (token_embeddings + position_embeddings)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.output_layer(x)
        return logits

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

class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, max_sequence_length=128, dropout=0.1):
        super().__init__()
        self.attention = SelfAttention(embedding_dim, max_sequence_length)
        self.norm1 = nn.LayerNorm(embedding_dim)

        # Feed Forward
        self.feed_forward = nn.Sequential(nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.Dropout(dropout)
        )

        self.norm2 = nn.LayerNorm(embedding_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attention_output = self.attention(x)

        x = x + self.dropout(attention_output)

        # Normalisation
        x = self.norm1(x)
        feed_forward_output = self.feed_forward(x)

        x = x + feed_forward_output
        x = self.norm2(x)
        return x
