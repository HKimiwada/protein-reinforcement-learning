import torch
import torch.nn as nn
import torch.nn.functional as F
from src.environment.action_space import SequenceActionSpace

class SequenceEditPolicy(nn.Module):
    def __init__(self, state_dim=1162, hidden_dim=512, max_seq_length=1000, num_amino_acids=20):
        super().__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.max_seq_length = max_seq_length
        self.default_seq_length = 500  # Add default

        # Input processing
        self.input_projection = nn.Linear(state_dim, hidden_dim)

        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Action heads
        self.edit_type_head = nn.Linear(hidden_dim, 4)  # sub/ins/del/stop
        self.position_head = nn.Linear(hidden_dim, max_seq_length)
        self.amino_acid_head = nn.Linear(hidden_dim, num_amino_acids)

        # Value head for actor-critic
        self.value_head = nn.Linear(hidden_dim, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def safe_probs(self, p: torch.Tensor, eps=1e-6) -> torch.Tensor:
        """Ensure probabilities are valid"""
        p = torch.nan_to_num(p, nan=1.0/p.size(-1))
        p = p.clamp(min=eps)
        return p / p.sum(dim=-1, keepdim=True)

    def forward(self, state):
        """Forward pass through policy network"""
        # Process state
        x = F.relu(self.input_projection(state))
        features = self.backbone(x)

        # Generate action logits
        edit_type_logits = self.edit_type_head(features)
        position_logits = self.position_head(features)
        amino_acid_logits = self.amino_acid_head(features)

        # Value estimate
        value = self.value_head(features)

        probs_et = self.safe_probs(F.softmax(edit_type_logits, dim=-1))
        probs_pos = self.safe_probs(F.softmax(position_logits, dim=-1))
        probs_aa = self.safe_probs(F.softmax(amino_acid_logits, dim=-1))

        return {
            'edit_type': probs_et,
            'position': probs_pos,
            'amino_acid': probs_aa,
            'value': value,
            'logits': {
                'edit_type': edit_type_logits,
                'position': position_logits,
                'amino_acid': amino_acid_logits
            }
        }

    def get_action(self, state, deterministic=False, sequence_length=None):
        """Get action from policy (for inference) - FIXED VERSION"""
        # Ensure state is on correct device
        device = next(self.parameters()).device
        state = state.to(device)
        
        # Use default sequence length if not provided
        if sequence_length is None:
            sequence_length = self.default_seq_length
        
        with torch.no_grad():
            # Handle both single state and batch
            if state.dim() == 1:
                output = self.forward(state.unsqueeze(0))
                squeeze_output = True
            else:
                output = self.forward(state)
                squeeze_output = False

            if deterministic:
                # 🚀 FIXED: Greedy action selection with proper bounds checking
                if squeeze_output:
                    edit_type_probs = output['edit_type'].squeeze(0)
                    position_probs = output['position'].squeeze(0)
                    aa_probs = output['amino_acid'].squeeze(0)
                else:
                    edit_type_probs = output['edit_type'][0]
                    position_probs = output['position'][0]
                    aa_probs = output['amino_acid'][0]
                
                # Get edit type
                edit_type_idx = edit_type_probs.argmax().item()
                edit_type = ['substitution', 'insertion', 'deletion', 'stop'][edit_type_idx]
                
                # Handle stop action
                if edit_type == 'stop':
                    return {
                        'type': 'stop',
                        'position': 0,
                        'amino_acid': None,
                        'log_prob': torch.tensor(0.0, device=device)
                    }
                
                # 🚀 FIXED: Constrain position selection to valid range
                if edit_type == 'insertion':
                    valid_positions = sequence_length + 1  # Can insert at end
                else:
                    valid_positions = sequence_length  # Must be within sequence
                
                # Only consider valid positions for argmax
                valid_position_probs = position_probs[:valid_positions]
                position_idx = valid_position_probs.argmax().item()
                
                # Get amino acid
                aa_idx = aa_probs.argmax().item()
                amino_acid = list('ACDEFGHIKLMNPQRSTVWY')[aa_idx] if edit_type in ['substitution', 'insertion'] else None

                action = {
                    'type': edit_type,
                    'position': position_idx,
                    'amino_acid': amino_acid,
                    'log_prob': torch.tensor(0.0, device=device)
                }

            else:
                # Sample from distributions - this should already work correctly
                action_space = SequenceActionSpace()
                
                # Prepare output for action_space (expects no batch dim)
                if squeeze_output:
                    sample_output = {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v 
                                for k, v in output.items()}
                else:
                    sample_output = {k: v[0] if isinstance(v, torch.Tensor) else v 
                                for k, v in output.items()}
                
                action = action_space.sample_action(sample_output, sequence_length)

            return action