import math
import torch
import numpy as np
import re
import logging
from typing import Tuple, Dict, Any, List

logger = logging.getLogger(__name__)

class SpiderSilkRewardFunction:
    """
    Spider silk reward function with comprehensive numerical stability protection
    and anti-exploitation measures
    """
    
    def __init__(self, silkomegpt_model, silkomegpt_tokenizer, esmc_model, max_episodes=2000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.silkomegpt = silkomegpt_model.to(self.device)
        self.tokenizer = silkomegpt_tokenizer
        self.esmc = esmc_model.to(self.device)
        self.max_episodes = max_episodes
        
        # MINIMAL FIX: Only fix tokenizer warnings without changing behavior
        if not hasattr(self.tokenizer, 'model_max_length') or self.tokenizer.model_max_length > 1000000:
            self.tokenizer.model_max_length = 1024
        
        if self.tokenizer.pad_token is None or self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            if hasattr(self.tokenizer, 'unk_token') and self.tokenizer.unk_token:
                self.tokenizer.pad_token = self.tokenizer.unk_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '<PAD>'})
                self.silkomegpt.resize_token_embeddings(len(self.tokenizer))
        
        # Safety constants
        self.min_toughness = 0.001
        self.max_toughness = 1.0
        self.min_std = 0.0001
        self.max_std = 0.5
        self.default_toughness = 0.1
        self.default_std = 0.05
        
        # Anti-exploitation constants
        self.min_edits_for_success = 2  # Require at least 2 edits for early termination
        self.real_improvement_threshold = 0.8  # 80% of target must be real improvement
        
        # Error counters for monitoring
        self.nan_count = 0
        self.prediction_errors = 0
        self.perplexity_errors = 0
        self.exploitation_attempts = 0  # Track exploitation attempts

    def _validate_number(self, value: float, name: str, fallback: float = 0.0) -> float:
        """Validate a number and return fallback if invalid"""
        if np.isnan(value) or np.isinf(value):
            logger.warning(f"Invalid {name}: {value}, using fallback {fallback}")
            self.nan_count += 1
            return fallback
        return value

    def _safe_math_operation(self, operation, *args, fallback: float = 0.0, operation_name: str = "operation"):
        """Safely perform mathematical operations with fallback"""
        try:
            result = operation(*args)
            if np.isnan(result) or np.isinf(result):
                logger.warning(f"Invalid result from {operation_name}: {result}, using fallback")
                return fallback
            return result
        except Exception as e:
            logger.warning(f"Error in {operation_name}: {e}, using fallback")
            return fallback

    def get_adaptive_weights(self, episode_number):
        """Get adaptive weights with safety checks"""
        frac = min(1.0, max(0.0, episode_number / self.max_episodes))
        
        weights = {
            'toughness': self._validate_number(0.4 + 0.3 * frac, "toughness_weight", 0.4),
            'realism': self._validate_number(0.5 - 0.3 * frac, "realism_weight", 0.5),
            'exploration': self._validate_number(0.1 * (1.0 - frac), "exploration_weight", 0.1),
            'efficiency': self._validate_number(0.0 + 0.05 * frac, "efficiency_weight", 0.0)
        }
        
        # Ensure weights sum to approximately 1.0
        total_weight = sum(weights.values())
        if total_weight > 0:
            for key in weights:
                weights[key] /= total_weight
        else:
            # Fallback to equal weights
            weights = {k: 0.25 for k in weights.keys()}
        
        return weights

    def get_total_improvement(self, edit_history):
        """Calculate total improvement with safety checks"""
        if not edit_history:
            return 0.0
        
        total = 0.0
        for edit in edit_history:
            improvement = edit.get('toughness_improvement', 0.0)
            improvement = self._validate_number(improvement, "toughness_improvement", 0.0)
            total += improvement
        
        return self._validate_number(total, "total_improvement", 0.0)

    def calculate_perplexity(self, sequence: str) -> float:
        """
        Compute the MLM perplexity with comprehensive error handling
        """
        try:
            # Input validation
            if not sequence or not isinstance(sequence, str):
                logger.warning("Invalid sequence for perplexity calculation")
                self.perplexity_errors += 1
                return 3.0  # Default high perplexity
            
            # Remove invalid characters and ensure minimum length
            valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
            cleaned_sequence = ''.join([aa for aa in sequence if aa in valid_aas])
            
            if len(cleaned_sequence) < 3:
                logger.warning(f"Sequence too short after cleaning: {len(cleaned_sequence)}")
                self.perplexity_errors += 1
                return 3.0
            
            # Set model to eval
            self.esmc.eval()

            # Tokenize with error handling
            try:
                inputs = self.esmc.tokenizer(
                    cleaned_sequence,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    return_attention_mask=True,
                    max_length=1024  # Reasonable limit
                )
            except Exception as e:
                logger.warning(f"Tokenization error: {e}")
                self.perplexity_errors += 1
                return 3.0

            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                try:
                    outputs = self.esmc(
                        **inputs,
                        labels=inputs["input_ids"],
                        return_dict=True
                    )
                    loss = outputs.loss.item()
                    
                    # Validate loss
                    loss = self._validate_number(loss, "perplexity_loss", 1.1)  # ln(3) ≈ 1.1
                    
                    # Compute perplexity with bounds
                    perplexity = math.exp(min(loss, 10.0))  # Cap to prevent overflow
                    perplexity = max(1.0, min(perplexity, 100.0))  # Reasonable bounds
                    
                    return perplexity
                    
                except Exception as e:
                    logger.warning(f"Model forward pass error: {e}")
                    self.perplexity_errors += 1
                    return 3.0

        except Exception as e:
            logger.error(f"Critical error in perplexity calculation: {e}")
            self.perplexity_errors += 1
            return 3.0

    def predict_toughness(self, sequence: str) -> Tuple[float, float]:
        """
        Predict toughness with comprehensive error handling and fallbacks
        """
        try:
            # Input validation
            if not sequence or not isinstance(sequence, str):
                logger.warning("Invalid sequence for toughness prediction")
                self.prediction_errors += 1
                return self.default_toughness, self.default_std
            
            # Clean sequence
            valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
            cleaned_sequence = ''.join([aa for aa in sequence if aa in valid_aas])
            
            if len(cleaned_sequence) < 5:
                logger.warning(f"Sequence too short for toughness prediction: {len(cleaned_sequence)}")
                self.prediction_errors += 1
                return self.default_toughness, self.default_std

            # Device setup
            self.silkomegpt.to(self.device)

            # Build prompt
            prompt = f"CalculateSilkContent<{cleaned_sequence}>"
            
            try:
                # MINIMAL FIX: Use proper tokenization but keep same behavior
                inputs = self.tokenizer(
                    prompt,
                    add_special_tokens=False,
                    return_tensors="pt",
                    max_length=1024,
                    truncation=True,
                    padding=False,  # Keep original behavior - no padding
                    return_attention_mask=False  # Keep original behavior - no attention mask
                )
                tokens = inputs['input_ids'].to(self.device)
                
            except Exception as e:
                logger.warning(f"Tokenization error in toughness prediction: {e}")
                self.prediction_errors += 1
                return self.default_toughness, self.default_std

            # KEEP ORIGINAL GENERATION: Don't change what was working
            try:
                with torch.no_grad():
                    output_ids = self.silkomegpt.generate(
                        inputs=tokens,  # Keep original parameter name
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.pad_token_id,
                        max_length=min(tokens.shape[1] + 50, 1024),
                        do_sample=False,  # Keep original settings
                        temperature=1.0,
                        top_p=1.0,
                        num_return_sequences=1
                    )[0]

                decoded = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                
            except Exception as e:
                logger.warning(f"Generation error: {e}")
                self.prediction_errors += 1
                return self.default_toughness, self.default_std

            # KEEP ORIGINAL PARSING: Don't change what was working
            try:
                # Primary parsing strategy
                m = re.search(r"\[([^\]]+)\]", decoded)
                if m:
                    parts = [p.strip() for p in m.group(1).split(",")]
                    nums = []
                    for p in parts:
                        try:
                            num = float(p)
                            if np.isfinite(num):
                                nums.append(num)
                        except (ValueError, TypeError):
                            continue
                    
                    if len(nums) >= 1:
                        norm_toughness = nums[0]
                        norm_std = nums[1] if len(nums) > 1 else 0.1
                    else:
                        raise ValueError("No valid numbers found")
                else:
                    # Fallback parsing strategies
                    numbers = re.findall(r'-?\d+\.?\d*', decoded)
                    if len(numbers) >= 1:
                        norm_toughness = float(numbers[0])
                        norm_std = float(numbers[1]) if len(numbers) > 1 else 0.1
                    else:
                        raise ValueError("No numbers found in output")

            except Exception as e:
                logger.warning(f"Parsing error: {e}, decoded: {decoded[:100]}")
                self.prediction_errors += 1
                return self.default_toughness, self.default_std

            # Validate normalized values
            norm_toughness = self._validate_number(norm_toughness, "norm_toughness", 0.5)
            norm_std = self._validate_number(norm_std, "norm_std", 0.1)
            
            # Clamp normalized values to reasonable range
            norm_toughness = max(0.0, min(1.0, norm_toughness))
            norm_std = max(0.0, min(1.0, norm_std))

            # Denormalize with safety checks
            def safe_denormalize(x: float, lo: float, hi: float, name: str) -> float:
                try:
                    result = x * (hi - lo) + lo
                    return self._validate_number(result, f"denorm_{name}", (lo + hi) / 2)
                except Exception as e:
                    logger.warning(f"Denormalization error for {name}: {e}")
                    return (lo + hi) / 2

            predicted_toughness = safe_denormalize(norm_toughness, 0.005, 0.39, "toughness")
            predicted_stddev = safe_denormalize(norm_std, 0.001, 0.136, "stddev")

            # Final bounds checking
            predicted_toughness = max(self.min_toughness, min(self.max_toughness, predicted_toughness))
            predicted_stddev = max(self.min_std, min(self.max_std, predicted_stddev))

            return predicted_toughness, predicted_stddev

        except Exception as e:
            logger.error(f"Critical error in toughness prediction: {e}")
            self.prediction_errors += 1
            return self.default_toughness, self.default_std

    def _validate_actual_improvement(self, new_seq: str, original_seq: str, target_improvement: float) -> Tuple[bool, float]:
        try:
            current_tough, _ = self.predict_toughness(new_seq)
            original_tough, _ = self.predict_toughness(original_seq)
            actual_improvement = current_tough - original_tough
            
            # Only require that actual improvement meets target - don't punish efficiency!
            required_improvement = target_improvement * self.real_improvement_threshold
            is_valid = actual_improvement >= required_improvement
            
            logger.info(f"✅ Real improvement: {actual_improvement:.4f} vs required {required_improvement:.4f}")
            
            return is_valid, actual_improvement
        except Exception as e:
            logger.error(f"Error in improvement validation: {e}")
            return False, 0.0

    def calculate_reward(self,
                     old_seq,
                     new_seq,
                     edit_history,
                     original_seq,
                     episode_number,
                     target_improvement=0.02):
        """
        Calculate reward with comprehensive error handling and fixed anti-exploitation
        """
        
        # ✅ Initialize ALL variables at the start to prevent reference errors
        total_imp = 0.0
        edit_count = 0
        early_stop_penalty = 0.0
        r_tough = 0.0
        r_real = -0.1
        r_explore = 0.0
        r_eff = -0.01
        actual_improvement = 0.0
        
        try:
            # Input validation
            if not all(isinstance(seq, str) for seq in [old_seq, new_seq, original_seq]):
                logger.warning("Invalid sequence types in reward calculation")
                return {'total': -0.1, 'done': False, 'components': {}}

            # Safe edit count calculation
            edit_count = len(edit_history) if edit_history else 0
            
            # Safe total improvement calculation
            try:
                total_imp = self.get_total_improvement(edit_history)
                total_imp = self._validate_number(total_imp, "total_improvement", 0.0)
            except Exception as e:
                logger.warning(f"Error getting total improvement: {e}")
                total_imp = 0.0

            # Early stopping penalty - only for zero edits
            if edit_count == 0:
                early_stop_penalty = -0.2  # Small penalty for no exploration
            else:
                early_stop_penalty = 0.0

            # ✅ FIXED: Early termination check with PROPER anti-exploitation
            if total_imp >= target_improvement:
                try:
                    # Validate this is REAL improvement, not exploitation
                    is_valid, actual_improvement = self._validate_actual_improvement(
                        new_seq, original_seq, target_improvement
                    )
                    
                    # ✅ REMOVED edit count requirement - efficiency should be rewarded!
                    if is_valid:
                        logger.info(f"🎉 SUCCESS! Actual improvement: {actual_improvement:.4f} in {edit_count} edits")
                        return {
                            'total': 5.0,
                            'done': True,
                            'components': {
                                'toughness': 5.0,
                                'realism': 0.0,
                                'exploration': 0.0,
                                'efficiency': 0.0
                            }
                        }
                    else:
                        # Only penalize if actual improvement is significantly less than claimed
                        if actual_improvement < target_improvement * 0.5:  # Less than 50% of target
                            self.exploitation_attempts += 1
                            logger.warning(f"🚨 Exploitation detected! Total_imp: {total_imp:.4f}, "
                                        f"Actual: {actual_improvement:.4f}, Edits: {edit_count}")
                            return {
                                'total': -1.0,  # Reduced penalty
                                'done': False,
                                'components': {
                                    'toughness': -1.0,
                                    'realism': 0.0,
                                    'exploration': 0.0,
                                    'efficiency': 0.0
                                }
                            }
                        else:
                            # Close to target but not quite there - just continue normally
                            logger.info(f"Close to target: {actual_improvement:.4f}, continuing...")
                            
                except Exception as e:
                    logger.warning(f"Error in early termination check: {e}")
                    # Continue with normal reward calculation

            # Get adaptive weights
            weights = self.get_adaptive_weights(episode_number)

            # Component rewards with error handling
            try:
                r_tough = self.toughness_reward(old_seq, new_seq)
                r_tough = self._validate_number(r_tough, "toughness_reward", 0.0)
            except Exception as e:
                logger.warning(f"Error in toughness reward: {e}")
                r_tough = 0.0

            try:
                r_real = self.realism_reward(new_seq, original_seq)
                r_real = self._validate_number(r_real, "realism_reward", -0.1)
            except Exception as e:
                logger.warning(f"Error in realism reward: {e}")
                r_real = -0.1

            try:
                r_explore = self.exploration_reward(edit_history, episode_number)
                r_explore = self._validate_number(r_explore, "exploration_reward", 0.0)
            except Exception as e:
                logger.warning(f"Error in exploration reward: {e}")
                r_explore = 0.0

            try:
                r_eff = self.efficiency_reward(edit_count, total_imp, target_improvement)
                r_eff = self._validate_number(r_eff, "efficiency_reward", -0.01)
            except Exception as e:
                logger.warning(f"Error in efficiency reward: {e}")
                r_eff = -0.01

            # Validate component rewards
            components = [r_tough, r_real, r_explore, r_eff]
            if any(np.isnan(x) or np.isinf(x) for x in components):
                logger.warning(f"Invalid reward components: {components}")
                return {'total': -0.1, 'done': False, 'components': {}}

            # Weighted sum with safety
            total = (weights['toughness'] * r_tough +
                    weights['realism'] * r_real +
                    weights['exploration'] * r_explore +
                    weights['efficiency'] * r_eff +
                    early_stop_penalty)

            total = self._validate_number(total, "total_reward", -0.1)
            
            # Clip to safe range
            total = float(np.clip(total, -5.0, 5.0))

            # Debug logging every 50 episodes instead of 100
            if episode_number % 50 == 0:
                logger.info(f"Episode {episode_number}: total_imp={total_imp:.4f}, "
                        f"actual_imp={actual_improvement:.4f}, edit_count={edit_count}, reward={total:.3f}")

            return {
                'total': total,
                'done': False,
                'components': {
                    'toughness': r_tough,
                    'realism': r_real,
                    'exploration': r_explore,
                    'efficiency': r_eff,
                    'early_stop_penalty': early_stop_penalty
                },
                'weights': weights
            }

        except Exception as e:
            logger.error(f"Critical error in reward calculation: {e}")
            # Return safe fallback with initialized variables
            return {
                'total': -0.1, 
                'done': False, 
                'components': {
                    'toughness': r_tough,
                    'realism': r_real,
                    'exploration': r_explore,
                    'efficiency': r_eff,
                    'early_stop_penalty': early_stop_penalty
                }
            }

    def toughness_reward(self, old_sequence, new_sequence):
        """
        Calculate toughness reward with error handling and tighter bounds
        """
        try:
            # Get toughness predictions
            old_t, old_s = self.predict_toughness(old_sequence)
            new_t, new_s = self.predict_toughness(new_sequence)

            # Validate predictions
            old_t = self._validate_number(old_t, "old_toughness", self.default_toughness)
            old_s = self._validate_number(old_s, "old_std", self.default_std)
            new_t = self._validate_number(new_t, "new_toughness", self.default_toughness)
            new_s = self._validate_number(new_s, "new_std", self.default_std)

            # Ensure positive standard deviations
            old_s = max(self.min_std, old_s)
            new_s = max(self.min_std, new_s)

            # Calculate improvements
            raw_imp = new_t - old_t
            penalty = 0.5 * (new_s - old_s)
            adj_imp = raw_imp - penalty

            # Validate intermediate calculations
            raw_imp = self._validate_number(raw_imp, "raw_improvement", 0.0)
            penalty = self._validate_number(penalty, "penalty", 0.0)
            adj_imp = self._validate_number(adj_imp, "adjusted_improvement", 0.0)

            # Calculate significance threshold safely
            combined_s = self._safe_math_operation(
                lambda: math.sqrt(old_s**2 + new_s**2),
                fallback=self.default_std,
                operation_name="combined_std"
            )
            
            sig_thr = 1.96 * combined_s

            # Calculate reward based on significance with tighter scaling
            if adj_imp > sig_thr:
                reward = adj_imp * 2.0  # Scale up significant improvements
            elif adj_imp > 0:
                reward = 0.1 * adj_imp  # Reduce reward for marginal improvements
            else:
                reward = 1.0 * adj_imp  # Reduce penalty for decreases

            # Apply log transformation for positive rewards (more conservative)
            if reward > 0:
                reward = self._safe_math_operation(
                    lambda: math.log1p(reward * 5) / 5,  # More conservative scaling
                    fallback=reward * 0.05,
                    operation_name="log_transform"
                )

            # Final validation and tighter clipping
            reward = self._validate_number(reward, "final_toughness_reward", 0.0)
            reward = max(-1.0, min(1.0, reward))  # Tighter bounds

            return reward

        except Exception as e:
            logger.error(f"Critical error in toughness reward calculation: {e}")
            return 0.0

    def realism_reward(self, seq, original_seq):
        """
        Calculate realism reward with error handling
        """
        try:
            total_reward = 0.0

            # Perplexity penalty
            try:
                thresh, alpha = 2.5, 1.0
                ppl = self.calculate_perplexity(seq)
                ppl = self._validate_number(ppl, "perplexity", 3.0)
                ppl_penalty = -alpha * max(0.0, ppl - thresh)
                ppl_penalty = self._validate_number(ppl_penalty, "perplexity_penalty", -0.5)
                total_reward += ppl_penalty
            except Exception as e:
                logger.warning(f"Error in perplexity penalty: {e}")
                total_reward += -0.5

            # Length penalty
            try:
                if len(original_seq) > 0:
                    lr = len(seq) / len(original_seq)
                    lr = self._validate_number(lr, "length_ratio", 1.0)
                    len_pen = -0.5 * (lr - 1.0)**2
                    len_pen = self._validate_number(len_pen, "length_penalty", 0.0)
                    total_reward += len_pen
            except Exception as e:
                logger.warning(f"Error in length penalty: {e}")

            # Motif penalty
            try:
                motif_pen = self._motif_penalty(seq)
                motif_pen = self._validate_number(motif_pen, "motif_penalty", 0.0)
                total_reward += motif_pen
            except Exception as e:
                logger.warning(f"Error in motif penalty: {e}")

            # Composition bonus
            try:
                comp_pen = self._composition_bonus(seq)
                comp_pen = self._validate_number(comp_pen, "composition_bonus", 0.0)
                total_reward += comp_pen
            except Exception as e:
                logger.warning(f"Error in composition bonus: {e}")

            # Final validation
            total_reward = self._validate_number(total_reward, "total_realism_reward", -0.1)
            total_reward = max(-3.0, min(1.0, total_reward))  # Reasonable bounds

            return total_reward

        except Exception as e:
            logger.error(f"Critical error in realism reward calculation: {e}")
            return -0.1

    def exploration_reward(self, edit_history, episode_number):
        """
        Calculate exploration reward with error handling and increased incentives
        """
        try:
            if not edit_history:
                return -0.1  # Small penalty for no exploration

            # Diversity bonus (increased)
            recent = edit_history[-10:] if len(edit_history) >= 10 else edit_history
            types = {e.get('type', '<unk>') for e in recent}
            div_bonus = 0.05 * len(types)  # Increased from 0.02
            div_bonus = self._validate_number(div_bonus, "diversity_bonus", 0.0)

            # Position spread bonus (increased)
            positions = [e.get('position', 0) for e in recent]
            if positions:
                spread = max(positions) - min(positions)
                spread = self._validate_number(spread, "position_spread", 0.0)
                total_len = len(edit_history) if edit_history else 1
                pos_bonus = 0.2 if spread > 0.3 * total_len else 0.0  # Increased from 0.1
            else:
                pos_bonus = 0.0

            # Edit count bonus (new)
            edit_count_bonus = min(0.1, len(edit_history) * 0.02)  # Bonus for making edits

            # Combine bonuses
            base_bonus = div_bonus + pos_bonus + edit_count_bonus
            base_bonus = self._validate_number(base_bonus, "base_exploration_bonus", 0.0)

            # Apply exploration weight
            try:
                w_explore = self.get_adaptive_weights(episode_number)['exploration']
                if w_explore > 0:
                    final_bonus = base_bonus * (w_explore / 0.1)
                else:
                    final_bonus = base_bonus * 0.5  # Still give some exploration reward
            except Exception as e:
                logger.warning(f"Error getting exploration weight: {e}")
                final_bonus = base_bonus

            # Final validation
            final_bonus = self._validate_number(final_bonus, "final_exploration_reward", 0.0)
            final_bonus = max(-0.1, min(0.8, final_bonus))  # Increased upper bound

            return final_bonus

        except Exception as e:
            logger.error(f"Critical error in exploration reward calculation: {e}")
            return 0.0

    def efficiency_reward(self, edit_count, total_improvement, target_improvement):
        """
        Calculate efficiency reward with error handling and balanced incentives
        """
        try:
            # Reduced step penalty to encourage exploration
            step_pen = -0.005 * edit_count  # Reduced from -0.01
            step_pen = self._validate_number(step_pen, "step_penalty", -0.005)

            # Efficiency bonus
            if total_improvement >= target_improvement:
                eff_bonus = 0.5 * max(0, 1.0 - edit_count / 50)
                eff_bonus = self._validate_number(eff_bonus, "efficiency_bonus", 0.0)
            else:
                eff_bonus = 0.0

            # Terminal bonus
            if total_improvement >= 0.8 * target_improvement:
                term_bonus = 0.2
            else:
                term_bonus = 0.0

            # Combine all components
            total_reward = step_pen + eff_bonus + term_bonus
            total_reward = self._validate_number(total_reward, "total_efficiency_reward", -0.005)
            
            # Reasonable bounds
            total_reward = max(-1.0, min(1.0, total_reward))

            return total_reward

        except Exception as e:
            logger.error(f"Critical error in efficiency reward calculation: {e}")
            return -0.005

    def _motif_penalty(self, seq):
        """Calculate motif penalty with safety checks"""
        try:
            pen = 0.0
            if seq.count('GPG') < 2:
                pen -= 0.2  # Reduced penalty
            if 'AAAA' not in seq:
                pen -= 0.2  # Reduced penalty
            return self._validate_number(pen, "motif_penalty", 0.0)
        except Exception as e:
            logger.warning(f"Error in motif penalty: {e}")
            return 0.0

    def _composition_bonus(self, seq):
        """Calculate composition bonus with safety checks"""
        try:
            if not seq:
                return -0.1  # Reduced penalty

            bonus = 0.0
            
            # Glycine content
            try:
                g_frac = seq.count('G') / len(seq)
                g_frac = self._validate_number(g_frac, "glycine_fraction", 0.2)
                if 0.15 <= g_frac <= 0.40:
                    bonus += 0.05
                else:
                    bonus -= 0.1  # Reduced penalty
            except Exception as e:
                logger.warning(f"Error in glycine composition: {e}")
                bonus -= 0.05

            # Check for amino acid dominance
            try:
                for aa in 'ACDEFGHIKLMNPQRSTVWY':
                    aa_frac = seq.count(aa) / len(seq)
                    aa_frac = self._validate_number(aa_frac, f"{aa}_fraction", 0.0)
                    if aa_frac > 0.6:
                        bonus -= 0.3  # Reduced penalty
                        break
            except Exception as e:
                logger.warning(f"Error in amino acid dominance check: {e}")

            return self._validate_number(bonus, "composition_bonus", 0.0)

        except Exception as e:
            logger.error(f"Critical error in composition bonus: {e}")
            return 0.0

    def get_error_stats(self) -> Dict[str, int]:
        """Get error statistics for monitoring"""
        return {
            'nan_count': self.nan_count,
            'prediction_errors': self.prediction_errors,
            'perplexity_errors': self.perplexity_errors,
            'exploitation_attempts': self.exploitation_attempts
        }