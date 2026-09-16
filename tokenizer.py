import re

class Tokenizer:
    SPECIAL_TOKENS = ("<PAD>", "<UNK>", "<BOS>", "<EOS>", "<USER>", "<ASSISTANT>")
    # Les marqueurs doivent être testés avant la ponctuation ; sinon <EOS> devient <, eos, >.
    TOKEN_PATTERN = re.compile(
        r"<(?:PAD|UNK|BOS|EOS|USER|ASSISTANT)>|\w+|[^\w\s]",
        flags=re.IGNORECASE,
    )

    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}

    def tokenize(self, text):
        """Transforme un texte en tokens, en normalisant les mots en minuscules."""
        if not isinstance(text, str):
            raise TypeError("text doit être une chaîne de caractères.")

        tokens = self.TOKEN_PATTERN.findall(text)
        return [token.upper() if token.startswith("<") else token.lower() for token in tokens]

    def rebuild_id_to_token(self):
        """Reconstruit la table inverse après le chargement d'un checkpoint."""
        self.id_to_token = {token_id: token for token, token_id in self.token_to_id.items()}

    def load_vocabulary(self, token_to_id):
        """Charge le vocabulaire sauvegardé par train.py."""
        if not isinstance(token_to_id, dict):
            raise TypeError("token_to_id doit être un dictionnaire.")
        self.token_to_id = dict(token_to_id)
        self.rebuild_id_to_token()

    def build_vocabulary(self, texts):
        """Crée un vocabulaire déterministe à partir d'un itérable de textes."""
        all_tokens = []
        for text in texts:
            all_tokens.extend(self.tokenize(text))

        # Exclut les marqueurs déjà présents dans les données pour ne pas les dupliquer.
        ordinary_tokens = sorted(set(all_tokens).difference(self.SPECIAL_TOKENS))
        vocabulary = list(self.SPECIAL_TOKENS) + ordinary_tokens

        # Important si build_vocabulary est rappelée avec un autre corpus.
        self.token_to_id = {token: token_id for token_id, token in enumerate(vocabulary)}
        self.rebuild_id_to_token()

    def encode(self, text):
        """Transforme un texte en identifiants ; les tokens inconnus deviennent <UNK>."""
        if "<UNK>" not in self.token_to_id:
            raise RuntimeError("Le vocabulaire doit être créé ou chargé avant encode().")
        unknown_id = self.token_to_id["<UNK>"]
        return [self.token_to_id.get(token, unknown_id) for token in self.tokenize(text)]

    def decode(self, ids, skip_special_tokens=False):
        """Transforme des identifiants en texte lisible."""
        if not self.id_to_token:
            raise RuntimeError("Le vocabulaire doit être créé ou chargé avant decode().")

        tokens = []
        for token_id in ids:
            try:
                token = self.id_to_token[int(token_id)]
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"ID de token invalide : {token_id!r}") from error
            if not (skip_special_tokens and token in self.SPECIAL_TOKENS):
                tokens.append(token)

        text = " ".join(tokens)
        # Aspect plus naturel, sans prétendre reconstruire exactement les espaces d'origine.
        text = re.sub(r"\s+([,.;:!?%)\]\}])", r"\1", text)
        text = re.sub(r"([([\{])\s+", r"\1", text)
        return text