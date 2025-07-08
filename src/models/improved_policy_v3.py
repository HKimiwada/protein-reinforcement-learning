# src/models/improved_policy_v2.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.environment.action_space import SequenceActionSpace

class ImprovedSequenceEditPolicyV3(nn.Module):
    """Improved policy with better consistency and learning + prevents early stop"""
    
    def __init__(self, state_dim=1162, hidden_dim=1024, max_seq_length=1000, num_amino_acids=20):
        super().__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.max_seq_length = max_seq_length
        self.default_seq_length = 500
        self.min_episode_length = 8  # Must match environment

        # Much deeper network with residual connections
        self.input_projection = nn.Linear(state_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        # Residual blocks for better learning
        self.residual_blocks = nn.ModuleList([
            self._make_residual_block(hidden_dim) for _ in range(4)
        ])

        # Attention mechanism for sequence understanding
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim//4, 
            num_heads=8, 
            dropout=0.1,
            batch_first=True
        )
        self.attention_proj = nn.Linear(hidden_dim, hidden_dim//4)
        self.attention_out = nn.Linear(hidden_dim//4, hidden_dim)

        # Separate value and policy streams
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.ReLU(),
            nn.Linear(hidden_dim//4, 1)
        )

        self.policy_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Action heads with better initialization
        self.edit_type_head = nn.Linear(hidden_dim, 4)
        self.position_head = nn.Linear(hidden_dim, max_seq_length)
        self.amino_acid_head = nn.Linear(hidden_dim, num_amino_acids)

        self._init_weights()

    def _make_residual_block(self, dim):
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )

    def _init_weights(self):
        """Better weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.out_features == 1:  # Value head
                    nn.init.xavier_uniform_(m.weight, gain=0.01)
                else:
                    nn.init.xavier_uniform_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def forward(self, state, step_count=None, cumulative_improvement=None):
        """Forward pass with step-count and improvement-aware processing"""
        batch_size = state.size(0) if state.dim() > 1 else 1
        if state.dim() == 1:
            state = state.unsqueeze(0)

        # Input processing with normalization
        x = F.relu(self.input_projection(state))
        x = self.input_norm(x)

        # Residual blocks for deeper representation
        for block in self.residual_blocks:
            residual = x
            x = block(x) + residual  # Skip connection

        # Attention mechanism for sequence understanding
        attn_input = self.attention_proj(x).unsqueeze(1)  # [batch, 1, dim//4]
        attn_out, _ = self.attention(attn_input, attn_input, attn_input)
        attn_out = self.attention_out(attn_out.squeeze(1))  # [batch, hidden_dim]
        
        # Combine attention with main stream
        x = x + attn_out

        # Separate value and policy processing
        value = self.value_stream(x)
        policy_features = self.policy_stream(x)

        # Action logits
        edit_type_logits = self.edit_type_head(policy_features)
        position_logits = self.position_head(policy_features)
        amino_acid_logits = self.amino_acid_head(policy_features)

        # 🚀 ENHANCED: Selective stop action masking based on improvement
        if step_count is not None and step_count < self.min_episode_length:
            # Always mask stop before minimum length
            edit_type_logits[:, 3] = -1e6
        elif (step_count is not None and cumulative_improvement is not None and 
              cumulative_improvement <= 0.001):
            # Mask stop action if no meaningful improvement yet
            edit_type_logits[:, 3] = -1e6
        
        # Better probability computation with temperature scaling
        edit_type_probs = F.softmax(edit_type_logits / 0.8, dim=-1)  # Lower temperature
        position_probs = F.softmax(position_logits / 1.2, dim=-1)   # Higher temperature
        amino_acid_probs = F.softmax(amino_acid_logits / 1.0, dim=-1)

        # Ensure valid probabilities
        edit_type_probs = self.safe_probs(edit_type_probs)
        position_probs = self.safe_probs(position_probs)
        amino_acid_probs = self.safe_probs(amino_acid_probs)

        result = {
            'edit_type': edit_type_probs,
            'position': position_probs,
            'amino_acid': amino_acid_probs,
            'value': value,
            'logits': {
                'edit_type': edit_type_logits,
                'position': position_logits,
                'amino_acid': amino_acid_logits
            }
        }

        # Remove batch dimension if input was 1D
        if batch_size == 1:
            for key in ['edit_type', 'position', 'amino_acid', 'value']:
                if key in result:
                    result[key] = result[key].squeeze(0)

        return result

    def safe_probs(self, p: torch.Tensor, eps=1e-8) -> torch.Tensor:
        """Ensure probabilities are valid with better numerical stability"""
        p = torch.nan_to_num(p, nan=1.0/p.size(-1))
        p = torch.clamp(p, min=eps, max=1.0-eps)
        return p / p.sum(dim=-1, keepdim=True)

    def get_action(self, state, deterministic=False, sequence_length=None, step_count=None, cumulative_improvement=None):
        """FIXED action selection with early stop prevention and cumulative improvement awareness"""
        device = next(self.parameters()).device
        state = state.to(device)
        
        if sequence_length is None:
            sequence_length = self.default_seq_length
        
        if step_count is None:
            step_count = 0
            
        if cumulative_improvement is None:
            cumulative_improvement = 0.0
        
        # 🚨 CRITICAL: Ensure sequence_length is reasonable
        sequence_length = max(10, min(sequence_length, 800))  # Clamp to reasonable bounds
        
        with torch.no_grad():
            if state.dim() == 1:
                output = self.forward(state.unsqueeze(0), step_count, cumulative_improvement)
                squeeze_output = True
            else:
                output = self.forward(state, step_count, cumulative_improvement)
                squeeze_output = False

            if deterministic:
                if squeeze_output:
                    edit_type_probs = output['edit_type']
                    position_probs = output['position']
                    aa_probs = output['amino_acid']
                else:
                    edit_type_probs = output['edit_type'][0]
                    position_probs = output['position'][0]
                    aa_probs = output['amino_acid'][0]
                
                # Get edit type (stop should be very unlikely if step_count < min_episode_length or no improvement)
                edit_type_idx = edit_type_probs.argmax().item()
                edit_type = ['substitution', 'insertion', 'deletion', 'stop'][edit_type_idx]
                
                # 🚀 ADDITIONAL SAFETY: Force non-stop action based on step count and improvement
                if edit_type == 'stop' and (step_count < self.min_episode_length or cumulative_improvement <= 0.01):
                    print(f"🛑 POLICY OVERRIDE: Preventing stop at step {step_count} (improvement: {cumulative_improvement:.6f}), using substitution")
                    edit_type = 'substitution'
                    edit_type_idx = 0
                
                if edit_type == 'stop':
                    return {
                        'type': 'stop',
                        'position': 0,
                        'amino_acid': None,
                        'log_prob': torch.tensor(0.0, device=device)
                    }
                
                # Position selection with proper bounds
                if edit_type == 'insertion':
                    max_valid_pos = sequence_length  # Can insert at end
                else:  # substitution or deletion
                    max_valid_pos = sequence_length - 1  # Must be within sequence
                
                # Only consider valid positions
                if max_valid_pos >= 0:
                    valid_position_probs = position_probs[:max_valid_pos + 1]
                    if len(valid_position_probs) > 0:
                        position_idx = valid_position_probs.argmax().item()
                    else:
                        position_idx = 0
                else:
                    position_idx = 0
                
                # Bounds validation
                if edit_type == 'insertion' and position_idx > sequence_length:
                    position_idx = sequence_length
                elif edit_type in ['substitution', 'deletion'] and position_idx >= sequence_length:
                    position_idx = max(0, sequence_length - 1)
                
                aa_idx = aa_probs.argmax().item()
                amino_acid = list('ACDEFGHIKLMNPQRSTVWY')[aa_idx] if edit_type in ['substitution', 'insertion'] else None

                return {
                    'type': edit_type,
                    'position': position_idx,
                    'amino_acid': amino_acid,
                    'log_prob': torch.tensor(0.0, device=device)
                }

            else:
                # Stochastic action with early stop prevention
                action_space = SequenceActionSpace()
                
                if squeeze_output:
                    sample_output = {k: v for k, v in output.items() if isinstance(v, torch.Tensor)}
                else:
                    sample_output = {k: v[0] for k, v in output.items() if isinstance(v, torch.Tensor)}
                
                # Get action from action space with bounds
                action = action_space.sample_action(sample_output, sequence_length)
                
                # 🚀 ADDITIONAL SAFETY: Override stop actions based on step count and improvement
                if action['type'] == 'stop' and (step_count < self.min_episode_length or cumulative_improvement <= 0.001):
                    print(f"🛑 STOCHASTIC OVERRIDE: Preventing stop at step {step_count} (improvement: {cumulative_improvement:.6f}), using substitution")
                    action['type'] = 'substitution'
                    action['position'] = min(step_count, sequence_length - 1)
                    action['amino_acid'] = 'A'
                
                # Final bounds validation
                if action['type'] != 'stop':
                    pos = action.get('position', 0)
                    if action['type'] == 'insertion' and pos > sequence_length:
                        action['position'] = sequence_length
                    elif action['type'] in ['substitution', 'deletion'] and pos >= sequence_length:
                        action['position'] = max(0, sequence_length - 1)
                
                return action