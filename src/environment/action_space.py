import torch

class SequenceActionSpace:
    def __init__(self):
        self.edit_types = ['substitution', 'insertion', 'deletion', 'stop']
        self.amino_acids = list('ACDEFGHIKLMNPQRSTVWY')
        self.max_sequence_length = 1000

    def sample_action(self, action_probs, sequence_length):
        """Sample action from policy output probabilities"""

        device = action_probs['edit_type'].device

        # 1) Sample edit type as a tensor
        edit_type_dist = torch.distributions.Categorical(action_probs['edit_type'])
        edit_type_idx = edit_type_dist.sample()         # tensor([...], device=...)
        edit_type = self.edit_types[edit_type_idx.item()]

        # Log-prob for edit_type
        logp = edit_type_dist.log_prob(edit_type_idx)

        if edit_type == 'stop':
            return {
                'type': 'stop',
                'log_prob': logp
            }

        # 2) Sample position
        valid_positions = sequence_length + (1 if edit_type == 'insertion' else 0)
        pos_logits      = action_probs['position'][:valid_positions]
        pos_probs       = pos_logits / pos_logits.sum(dim=0, keepdim=True)
        position_dist   = torch.distributions.Categorical(pos_probs)
        position_idx    = position_dist.sample()
        logp           += position_dist.log_prob(position_idx)

        # 3) (Optional) Sample amino acid
        amino_acid = None
        if edit_type in ('substitution','insertion'):
            aa_dist   = torch.distributions.Categorical(action_probs['amino_acid'])
            aa_idx    = aa_dist.sample()
            amino_acid= self.amino_acids[aa_idx.item()]
            logp     += aa_dist.log_prob(aa_idx)

        return {
            'type':       edit_type,
            'position':   position_idx.item(),
            'amino_acid': amino_acid,
            'log_prob':   logp
        }
