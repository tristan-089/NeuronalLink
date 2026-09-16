from pathlib import Path

import torch
from torch.utils.data import Dataset

from tokenizer import Tokenizer


class SlidingWindowDataset(Dataset):
    """Fenêtres glissantes (x[t:t+n], x[t+1:t+n+1]) d'une suite de tokens."""

    def __init__(self, token_ids, sequence_length):
        if sequence_length < 1:
            raise ValueError("sequence_length doit être supérieur ou égal à 1.")

        self.token_ids = torch.as_tensor(token_ids, dtype=torch.long).contiguous()
        self.sequence_length = sequence_length
        self.number_of_examples = self.token_ids.numel() - sequence_length

        if self.number_of_examples <= 0:
            raise ValueError(
                f"Le corpus contient {self.token_ids.numel()} tokens, ce qui est insuffisant "
                f"pour une séquence de longueur {sequence_length}."
            )

    def __len__(self):
        return self.number_of_examples

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)

        start = index
        end = start + self.sequence_length
        # Ce sont des vues du seul vecteur 1D : pas de copie pour les millions de fenêtres.
        return self.token_ids[start:end], self.token_ids[start + 1 : end + 1]


def create_dataset(file_path, sequence_length=32):
    """Retourne ``(dataset, tokenizer)`` pour le fichier de conversations.

    Le fichier doit contenir les marqueurs textuels <USER>, <ASSISTANT> et
    <EOS> si l'on souhaite que le modèle apprenne explicitement ces rôles.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset introuvable : {path.resolve()}")

    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("Le fichier de conversations est vide.")

    tokenizer = Tokenizer()
    tokenizer.build_vocabulary([text])
    token_ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    dataset = SlidingWindowDataset(token_ids, sequence_length)

    print(f"Tokens dans le corpus : {token_ids.numel():,}")
    return dataset, tokenizer
