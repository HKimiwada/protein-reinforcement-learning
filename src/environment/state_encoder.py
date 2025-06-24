import torch

class SequenceStateEncoder:
    def __init__(self, utils, max_seq_length=800):
        self.utils = utils
        self.max_seq_length = max_seq_length
        self.state_dim = 1152 + 10 # 1152: esm_c embedding dimensions, 10 is the context features added to tracking history

    def encode_state(self, sequence, edit_history, step_count, original_sequence):
        """Create comprehensive state representation"""

        # 1. Get ESM-C sequence embedding
        seq_embedding = self.utils.get_sequence_embedding(sequence)

        # 2. Calculate context features
        context_features = self._calculate_context_features(
            sequence, edit_history, step_count, original_sequence
        ).to(seq_embedding.device)

        # 3. Combine features
        state = torch.cat([seq_embedding, context_features], dim=-1)

        return state

    def _calculate_context_features(self, sequence, edit_history, step_count, original_sequence):
        """Calculate context features"""

        # Edit type counts (last 10 edits)
        recent_edits = edit_history[-10:] if len(edit_history) >= 10 else edit_history
        sub_count = sum(1 for e in recent_edits if e.get('type') == 'substitution')
        ins_count = sum(1 for e in recent_edits if e.get('type') == 'insertion')
        del_count = sum(1 for e in recent_edits if e.get('type') == 'deletion')

        # Normalize by number of recent edits
        n_recent = len(recent_edits) if recent_edits else 1
        sub_ratio = sub_count / n_recent
        ins_ratio = ins_count / n_recent
        del_ratio = del_count / n_recent

        # Progress features
        total_improvement = sum(e.get('toughness_improvement', 0) for e in edit_history)

        # Sequence properties
        length_ratio = len(sequence) / len(original_sequence)
        perplexity = self.utils.calculate_perplexity(sequence)
        perplexity_normalized = min(perplexity / 3.0, 1.0)

        # Step progress
        step_progress = min(step_count / 50.0, 1.0)

        # Glycine content (important for spider silk)
        gly_content = sequence.count('G') / len(sequence)

        context = torch.tensor([
            sub_ratio, ins_ratio, del_ratio,
            total_improvement,
            length_ratio,
            perplexity_normalized,
            step_progress,
            gly_content,
            len(edit_history) / 50.0,  # Total edits normalized
            1.0 if step_count == 0 else 0.0  # First step flag
        ], dtype=torch.float32)

        return context