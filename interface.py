from __future__ import annotations

from pathlib import Path
import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext

import torch

from model import MiniGPT
from tokenizer import Tokenizer


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "mini_gpt.pth"

# Pour une première version, une génération déterministe est plus facile à
# comprendre et à reproduire. Passez TEMPERATURE à 0.8 pour activer le sampling.
TEMPERATURE = 0.0
TOP_K = 20
MAX_NEW_TOKENS = 60


def choose_device() -> torch.device:
    """Utilise la GTX/CUDA quand PyTorch la détecte, sinon le CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path: Path, device: torch.device) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint introuvable : {path}\n\n"
            "Place interface.py dans le dossier qui contient mini_gpt.pth."
        )

    # map_location permet aussi de charger un checkpoint créé sur un autre PC.
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # Compatibilité avec les anciennes versions de PyTorch.
        checkpoint = torch.load(path, map_location=device)

    required_keys = {
        "model_state_dict",
        "token_to_id",
        "embedding_dim",
        "num_layers",
        "sequence_length",
    }
    missing = required_keys.difference(checkpoint)
    if missing:
        raise KeyError(
            "Le checkpoint ne correspond pas au format de train.py. "
            f"Clés manquantes : {', '.join(sorted(missing))}"
        )
    return checkpoint


class ChatEngine:
    """Chargement du modèle et génération, séparés de l'interface graphique."""

    def __init__(self) -> None:
        self.device = choose_device()
        checkpoint = load_checkpoint(MODEL_PATH, self.device)

        self.tokenizer = Tokenizer()
        self.tokenizer.load_vocabulary(checkpoint["token_to_id"])

        self.model = MiniGPT(
            vocab_size=len(self.tokenizer.token_to_id),
            embedding_dim=checkpoint["embedding_dim"],
            num_layers=checkpoint["num_layers"],
            max_sequence_length=checkpoint["sequence_length"],
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.max_sequence_length = checkpoint["sequence_length"]
        self.history: list[tuple[str, str]] = []

        self.user_id = self._special_id("<USER>")
        self.assistant_id = self._special_id("<ASSISTANT>")
        self.eos_id = self._special_id("<EOS>")
        self.stop_ids = {
            self.eos_id,
            self.user_id,
            self.assistant_id,
            self._special_id("<PAD>"),
        }

    def _special_id(self, token: str) -> int:
        try:
            return self.tokenizer.token_to_id[token]
        except KeyError as error:
            raise KeyError(f"Le vocabulaire ne contient pas le marqueur {token}.") from error

    @property
    def device_label(self) -> str:
        if self.device.type == "cuda":
            return f"CUDA — {torch.cuda.get_device_name(0)}"
        return "CPU"

    def clear_history(self) -> None:
        self.history.clear()

    def _build_prompt(self, user_message: str) -> list[int]:
        """Construit ``<USER> ... <ASSISTANT>`` dans la limite du contexte.

        Les anciens tours sont ajoutés du plus récent au plus ancien. Ainsi,
        lorsque la conversation devient longue, on conserve les derniers tours
        plutôt que de dépasser ``max_sequence_length``.
        """
        # Il faut toujours garder les deux marqueurs qui indiquent au modèle qui
        # parle et qu'il doit commencer une réponse.
        user_tokens = self.tokenizer.encode(user_message)
        room_for_user = max(0, self.max_sequence_length - 2)
        user_tokens = user_tokens[-room_for_user:]
        prompt = [self.user_id, *user_tokens, self.assistant_id]

        for old_user, old_assistant in reversed(self.history):
            turn = [
                self.user_id,
                *self.tokenizer.encode(old_user),
                self.assistant_id,
                *self.tokenizer.encode(old_assistant),
                self.eos_id,
            ]
            if len(turn) + len(prompt) > self.max_sequence_length:
                break
            prompt = turn + prompt

        return prompt

    @staticmethod
    def _sample_next_token(logits: torch.Tensor) -> int:
        if TEMPERATURE <= 0:
            return int(torch.argmax(logits, dim=-1).item())

        logits = logits / TEMPERATURE
        k = min(TOP_K, logits.size(-1))
        values, indices = torch.topk(logits, k=k, dim=-1)
        probabilities = torch.softmax(values, dim=-1)
        selected = torch.multinomial(probabilities, num_samples=1)
        return int(indices.gather(-1, selected).item())

    @torch.inference_mode()
    def reply(self, user_message: str) -> str:
        prompt_ids = self._build_prompt(user_message)
        all_ids = list(prompt_ids)
        generated_ids: list[int] = []

        for _ in range(MAX_NEW_TOKENS):
            # Le modèle ne connaît que des positions jusqu'à sequence_length.
            context_ids = all_ids[-self.max_sequence_length :]
            input_ids = torch.tensor([context_ids], dtype=torch.long, device=self.device)
            logits = self.model(input_ids)[:, -1, :]
            next_id = self._sample_next_token(logits)

            if next_id in self.stop_ids:
                break
            generated_ids.append(next_id)
            all_ids.append(next_id)

        # Point important : on décode uniquement generated_ids, jamais prompt_ids.
        # C'est ce qui évite d'afficher la question de l'utilisateur comme réponse.
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        if not response:
            response = "Je n'ai pas réussi à générer une réponse. Réessaie avec un autre message."

        self.history.append((user_message, response))
        return response


class MiniGPTApp(tk.Tk):
    def __init__(self, engine: ChatEngine) -> None:
        super().__init__()
        self.engine = engine
        self.title("Mini-GPT")
        self.geometry("760x560")
        self.minsize(540, 400)

        self.status = tk.StringVar(value=f"Modèle chargé sur : {engine.device_label}")
        self._build_widgets()
        self._append("Mini-GPT", "Bonjour ! Écris un message pour commencer.")

    def _build_widgets(self) -> None:
        main = tk.Frame(self, padx=12, pady=12)
        main.pack(fill="both", expand=True)

        tk.Label(main, textvariable=self.status, anchor="w").pack(fill="x", pady=(0, 8))

        self.chat = scrolledtext.ScrolledText(
            main, wrap="word", state="disabled", font=("Arial", 12), padx=10, pady=10
        )
        self.chat.tag_configure("speaker", font=("Arial", 12, "bold"))
        self.chat.pack(fill="both", expand=True)

        input_frame = tk.Frame(main, pady=10)
        input_frame.pack(fill="x")
        self.input = tk.Text(input_frame, height=3, wrap="word", font=("Arial", 12))
        self.input.pack(side="left", fill="both", expand=True, padx=(0, 8))
        self.input.bind("<Return>", self._on_return)

        buttons = tk.Frame(input_frame)
        buttons.pack(side="right", fill="y")
        self.send_button = tk.Button(buttons, text="Envoyer", command=self.send, width=12)
        self.send_button.pack(fill="x")
        tk.Button(buttons, text="Nouveau chat", command=self.new_chat, width=12).pack(
            fill="x", pady=(6, 0)
        )

    def _append(self, speaker: str, message: str) -> None:
        self.chat.configure(state="normal")
        self.chat.insert("end", f"{speaker} : ", "speaker")
        self.chat.insert("end", f"{message}\n\n")
        self.chat.configure(state="disabled")
        self.chat.see("end")

    def _on_return(self, event: tk.Event) -> str | None:
        # Entrée envoie ; Maj+Entrée ajoute une nouvelle ligne.
        if event.state & 0x0001:
            return None
        self.send()
        return "break"

    def send(self) -> None:
        message = self.input.get("1.0", "end-1c").strip()
        if not message or self.send_button["state"] == "disabled":
            return

        self.input.delete("1.0", "end")
        self._append("Vous", message)
        self.status.set("Mini-GPT génère une réponse…")
        self.send_button.configure(state="disabled")

        # La génération reste hors du thread Tkinter afin que la fenêtre ne gèle pas.
        threading.Thread(target=self._generate_in_background, args=(message,), daemon=True).start()

    def _generate_in_background(self, message: str) -> None:
        try:
            response = self.engine.reply(message)
        except Exception as error:  # L'erreur est affichée proprement dans la fenêtre.
            self.after(0, self._show_generation_error, str(error))
            return
        self.after(0, self._show_response, response)

    def _show_response(self, response: str) -> None:
        self._append("Mini-GPT", response)
        self.status.set(f"Prêt — modèle sur : {self.engine.device_label}")
        self.send_button.configure(state="normal")
        self.input.focus_set()

    def _show_generation_error(self, error: str) -> None:
        self.status.set("Erreur pendant la génération")
        self.send_button.configure(state="normal")
        messagebox.showerror("Mini-GPT", error)

    def new_chat(self) -> None:
        self.engine.clear_history()
        self.chat.configure(state="normal")
        self.chat.delete("1.0", "end")
        self.chat.configure(state="disabled")
        self._append("Mini-GPT", "Nouvelle conversation. Bonjour !")
        self.input.focus_set()


def main() -> None:
    try:
        engine = ChatEngine()
    except Exception as error:
        # Cette erreur survient avant que la fenêtre principale puisse exister.
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Impossible de lancer Mini-GPT", str(error))
        root.destroy()
        raise SystemExit(1) from error

    app = MiniGPTApp(engine)
    app.mainloop()


if __name__ == "__main__":
    main()