# Modified training_store.py
import torch
import torch.nn.functional as F


class TrainingDataStore:
    def __init__(self, encode_fn=None, decode_fn=None):
        self.embeddings = []
        self.texts = []
        self.encode_fn = encode_fn  # Function to encode text to tokens
        self.decode_fn = decode_fn  # Function to decode tokens to text

    def add_example(self, text, embedding):
        # Store the text directly - it's already in text form from your training loop
        self.embeddings.append(embedding.detach().cpu())
        self.texts.append(text)

    def find_nearest_neighbors(self, query_embedding, k=3):
        if not self.embeddings:
            return []

        similarities = F.cosine_similarity(
            query_embedding.unsqueeze(0),
            torch.stack(self.embeddings),
            dim=1
        )
        _, indices = similarities.topk(k)

        # Return the text examples
        return [self.texts[i] for i in indices]