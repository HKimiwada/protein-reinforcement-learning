import torch
import math
import random
import numpy as np
from torch.nn import functional as F

class SpiderSilkUtils:
    def __init__(self, esmc_model, tokenizer):
        """
        Utility class for sequence operations (no decision making)
        """
        self.model = esmc_model
        self.tokenizer = tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        # Get special token IDs
        self.mask_token_id = getattr(tokenizer, "mask_token_id",
                                     tokenizer.convert_tokens_to_ids("<mask>"))

        # Valid amino acids
        self.amino_acids = "ACDEFGHIKLMNPQRSTVWY"

        # Get amino acid token IDs
        self.aa_token_ids = {
            aa: tokenizer.convert_tokens_to_ids(aa)
            for aa in self.amino_acids
            if tokenizer.convert_tokens_to_ids(aa) != tokenizer.unk_token_id
        }

    def calculate_perplexity(self, sequence: str) -> float:
        """Calculate ESM-C perplexity for sequence"""
        self.model.eval()

        inputs = self.tokenizer(
            sequence, return_tensors="pt", truncation=True, padding=True, return_attention_mask=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"], return_dict=True)
            loss = outputs.loss.item()

        return math.exp(loss)

    def get_sequence_embedding(self, sequence: str):
        """Get ESM-C embedding for sequence"""
        self.model.eval()

        inputs = self.tokenizer(
            sequence, return_tensors="pt", truncation=True, padding=True, return_attention_mask=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, return_dict=True)
            # Use mean pooling of last hidden state
            embeddings = outputs.last_hidden_state.mean(dim=1)

        return embeddings.squeeze(0)  # Remove batch dimension

    def get_amino_acid_suggestions(self, sequence, position, top_k=5):
        """Get ESM-C suggestions for amino acid at position"""
        sequence_list = list(sequence)
        original_aa = sequence_list[position]
        sequence_list[position] = '<mask>'
        masked_sequence = ''.join(sequence_list)

        inputs = self.tokenizer(masked_sequence, return_tensors='pt', padding=True, truncation=True, return_attention_mask=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

            input_ids = inputs['input_ids'][0]
            try:
                mask_position = (input_ids == self.mask_token_id).nonzero(as_tuple=True)[0][0]
                mask_logits = logits[0, mask_position, :]
                probabilities = F.softmax(mask_logits, dim=-1)
            except:
                return []

        aa_suggestions = []
        for aa, token_id in self.aa_token_ids.items():
            if aa != original_aa:
                prob = probabilities[token_id].item()
                aa_suggestions.append((aa, prob))

        aa_suggestions.sort(key=lambda x: x[1], reverse=True)
        return aa_suggestions[:top_k]

    def validate_edit(self, old_sequence, new_sequence):
        """Check if edit satisfies constraints"""
        # Perplexity constraint
        new_perplexity = self.calculate_perplexity(new_sequence)
        if new_perplexity > 20:
            return False, f"Perplexity too high: {new_perplexity:.3f}"

        # Length constraint
        length_ratio = len(new_sequence) / len(old_sequence)
        if not (0.8 <= length_ratio <= 1.2):
            return False, f"Length ratio out of bounds: {length_ratio:.3f}"

        # Basic motif check
        if 'GPG' not in new_sequence or 'AAA' not in new_sequence:
            return False, "Essential motifs missing"

        return True, "Valid"