"""Entraînement du Mini-GPT avec le dataset mémoire-efficace du projet."""

from pathlib import Path
import os
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import create_dataset
from model import MiniGPT


# Configuration
DATASET_PATH = Path("data/conversations.txt")
MODEL_PATH = Path("mini_gpt.pth")
EPOCHS = 10  # Commencer par 1 à 10 epochs ; 100 serait très long avec 6,4 M d'exemples.
SEQUENCE_LENGTH = 32
EMBEDDING_DIM = 128
NUM_LAYERS = 4
LEARNING_RATE = 3e-4
BATCH_SIZE = 64  # Bon point de départ pour une GTX 1660 Super (6 Go).
NUM_WORKERS = 0  # Ne pas dupliquer le très gros dataset en mémoire.
RESUME = True
SEED = 42


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device utilisé : {device} ({torch.cuda.get_device_name(0)})")
        return device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Device utilisé : {device}")
        return device

    device = torch.device("cpu")
    print("Device utilisé : cpu")
    return device


def save_checkpoint(path, model, optimizer, tokenizer, epoch, config):
    """Sauvegarde atomique : un arrêt pendant l'écriture ne détruit pas le checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "token_to_id": tokenizer.token_to_id,
        **config,
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, path)


def move_optimizer_state_to_device(optimizer, device):
    """Nécessaire si un checkpoint CPU est repris sur CUDA/MPS."""
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def load_checkpoint_if_possible(path, model, optimizer, tokenizer, device, config):
    if not RESUME or not path.exists():
        return 0

    # weights_only=False est intentionnel : le checkpoint contient aussi le vocabulaire et l'optimiseur.
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # Compatibilité avec PyTorch plus ancien.
        checkpoint = torch.load(path, map_location=device)

    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"{path} n'est pas un checkpoint Mini-GPT valide.")

    for key, expected_value in config.items():
        saved_value = checkpoint.get(key)
        if saved_value is not None and saved_value != expected_value:
            raise ValueError(
                f"Le checkpoint utilise {key}={saved_value}, mais train.py utilise "
                f"{expected_value}. Garde la même configuration ou renomme/supprime "
                f"{path} pour repartir de zéro."
            )

    saved_vocab = checkpoint.get("token_to_id")
    if saved_vocab is not None and saved_vocab != tokenizer.token_to_id:
        raise ValueError(
            "Le vocabulaire du dataset a changé depuis le checkpoint. "
            "On ne peut pas reprendre cet entraînement sans risquer d'associer les mauvais mots aux IDs."
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    if "optimizer_state_dict" not in checkpoint:
        print("Ancien checkpoint chargé : poids restaurés, mais reprise à l'epoch 1 (optimiseur absent).")
        return 0

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    move_optimizer_state_to_device(optimizer, device)
    start_epoch = int(checkpoint.get("epoch", 0))
    print(f"Checkpoint chargé : reprise après l'epoch {start_epoch}.")
    return start_epoch


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = choose_device()
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Dataset introuvable : {DATASET_PATH.resolve()}")

    dataset, tokenizer = create_dataset(
        str(DATASET_PATH), sequence_length=SEQUENCE_LENGTH
    )

    print(f"Nombre d'exemples : {len(dataset):,}")
    print(f"Taille du vocabulaire : {len(tokenizer.token_to_id):,}")
    print(f"Batch size : {BATCH_SIZE}")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    model = MiniGPT(
        vocab_size=len(tokenizer.token_to_id),
        embedding_dim=EMBEDDING_DIM,
        num_layers=NUM_LAYERS,
        max_sequence_length=SEQUENCE_LENGTH,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    config = {
        "embedding_dim": EMBEDDING_DIM,
        "num_layers": NUM_LAYERS,
        "sequence_length": SEQUENCE_LENGTH,
    }
    start_epoch = load_checkpoint_if_possible(
        MODEL_PATH, model, optimizer, tokenizer, device, config
    )

    if start_epoch >= EPOCHS:
        print(f"Entraînement déjà terminé : checkpoint à l'epoch {start_epoch}/{EPOCHS}.")
        return

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        loss_sum = 0.0
        example_count = 0
        progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", unit="batch")

        for input_ids, target_ids in progress:
            input_ids = input_ids.to(device, non_blocking=(device.type == "cuda"))
            target_ids = target_ids.to(device, non_blocking=(device.type == "cuda"))

            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids)
            loss = criterion(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size = input_ids.size(0)
            loss_sum += loss.item() * batch_size
            example_count += batch_size
            progress.set_postfix(loss=f"{loss.item():.4f}")

        average_loss = loss_sum / example_count
        print(f"Epoch {epoch + 1:03d}/{EPOCHS} | Loss moyenne : {average_loss:.4f}")
        save_checkpoint(MODEL_PATH, model, optimizer, tokenizer, epoch + 1, config)
        print(f"Checkpoint sauvegardé : {MODEL_PATH}")


if __name__ == "__main__":
    main()