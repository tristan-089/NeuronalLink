import re

class Tokenizer:
    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}

    def tokenize(self, text):
        """
        Transforme une phrase en liste de token
        """
        tokens = re.findall(r"\w+|[^\w\s]", text.lower())
        return tokens

    def build_vocabulary(self, texts):
        """
        Créer le vocabulaire à partir de plusieurs textes
        """
        all_tokens = []

        for text in texts:
            tokens = self.tokenize(text)
            all_tokens.extend(tokens)

        unique_tokens = sorted(set(all_tokens))

        special_tokens = [
            "<PAD>",
            "<UNK>",
            "<BOS>",
            "<EOS>"
        ]

        vocabulary = special_tokens + unique_tokens

        # token --> nombre
        for i, token in enumerate(vocabulary):
            self.token_to_id[token] = i

        # token --> nombre
        for token, i in self.token_to_id.items():
            self.id_to_token[i] = token

    def encode(self, text):
        """
        Transforme le texte en nombres.
        """

        tokens = self.tokenize(text)

        ids = []

        for token in tokens:
            if token in self.token_to_id:
                ids.append(self.token_to_id[token])
            else:
                ids.append(self.token_to_id["<UNK>"])

        return ids

    def decode(self, ids):
        """
        Transforme des nombres en texte.
        """

        tokens = []

        for id in ids:
            tokens.append(self.id_to_token[id])

        return " ".join(tokens)
