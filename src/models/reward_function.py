import math
import torch
import numpy as np
import re
from typing import Tuple

class SpiderSilkRewardFunction:
    def __init__(self, silkomegpt_model, silkomegpt_tokenizer, esmc_model, max_episodes=2000):
        self.device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.silkomegpt   = silkomegpt_model.to(self.device)
        self.tokenizer    = silkomegpt_tokenizer
        self.esmc         = esmc_model.to(self.device)
        self.max_episodes = max_episodes


    def get_adaptive_weights(self, episode_number):
        frac = min(1.0, episode_number / self.max_episodes)
        return {
            'toughness':   0.4 + 0.3 * frac,       # 0.4→0.7
            'realism':     0.5 - 0.3 * frac,       # 0.5→0.2
            'exploration': 0.1 * (1.0 - frac),     # 0.1→0.0
            'efficiency':  0.0 + 0.05 * frac       # 0.0→0.05
        }

    def get_total_improvement(self, edit_history):
        return sum(edit.get('toughness_improvement', 0.0)
                   for edit in edit_history)

    def calculate_perplexity(self, sequence: str) -> float:
        """
        Compute the MLM perplexity of a single protein sequence.
        """
        # Set model to eval
        self.esmc.eval()

        # Tokenize & move inputs to correct device
        inputs = self.esmc.tokenizer(
            sequence,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        # Move all inputs to the same device as model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            # For an MLM model, passing labels=input_ids returns outputs.loss
            outputs = self.esmc(
                **inputs,
                labels=inputs["input_ids"],
                return_dict=True
            )
            loss = outputs.loss.item()

        # Perplexity is exp(average negative log likelihood)
        return math.exp(loss)

    def predict_toughness(self, sequence: str) -> Tuple[float, float]:
        """
        Queries SilkomeGPT to predict a normalized [toughness, stddev] for `sequence`,
        then denormalizes both back to the original scales.

        Returns:
            (predicted_toughness, predicted_stddev)
            or (nan, nan) if parsing fails.
        """
        # 1) Device setup
        if self.device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.silkomegpt.to(self.device)

        # 2) Build prompt and tokenize
        prompt = f"CalculateSilkContent<{sequence}>"
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(self.device)

        # 3) Generate
        output_ids = self.silkomegpt.generate(
            inputs         = tokens,
            eos_token_id   = self.tokenizer.eos_token_id,
            pad_token_id   = self.tokenizer.pad_token_id,
            max_length     = tokens.shape[1] + 50,
            do_sample      = False
        )[0]

        decoded = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        # 4) Extract the two numbers inside [...]
        m = re.search(r"\[([^\]]+)\]", decoded)
        if not m:
            return np.nan, np.nan

        parts = [p.strip() for p in m.group(1).split(",")]
        nums = []
        for p in parts:
            try:
                nums.append(float(p))
            except ValueError:
                pass

        if not nums:
            return np.nan, np.nan

        norm_toughness = nums[0]
        norm_std       = nums[1] if len(nums) > 1 else np.nan

        # 5) Denormalize each to its own original range
        def denormalize(x: float, lo: float, hi: float) -> float:
            return x * (hi - lo) + lo

        predicted_toughness = denormalize(norm_toughness, lo=0.005, hi=0.39)
        predicted_stddev    = denormalize(norm_std,       lo=0.001, hi=0.136) \
                            if not np.isnan(norm_std) else np.nan

        return predicted_toughness, predicted_stddev

    def calculate_reward(self,
                         old_seq,
                         new_seq,
                         edit_history,
                         original_seq,
                         episode_number,
                         target_improvement=0.15):

        # 1) Early termination
        total_imp = self.get_total_improvement(edit_history)
        if total_imp >= target_improvement:
            return {'total':  5.0,
                    'done':   True,
                    'components': {}}

        # 2) Annealed weights
        weights = self.get_adaptive_weights(episode_number)

        # 3) Component rewards
        r_tough   = self.toughness_reward(old_seq, new_seq)
        r_real    = self.realism_reward(new_seq, original_seq)
        r_explore = self.exploration_reward(edit_history, episode_number)
        r_eff     = self.efficiency_reward(len(edit_history),
                                           total_imp,
                                           target_improvement)

        # 4) Weighted sum & clip
        total = (weights['toughness']   * r_tough +
                 weights['realism']     * r_real +
                 weights['exploration'] * r_explore +
                 weights['efficiency']  * r_eff)
        total = float(np.clip(total, -5.0, 5.0))

        return {
            'total':      total,
            'done':       False,
            'components': {
                'toughness':   r_tough,
                'realism':     r_real,
                'exploration': r_explore,
                'efficiency':  r_eff
            },
            'weights': weights
        }

    def toughness_reward(self, old_sequence, new_sequence):
        old_t, old_s = self.predict_toughness(old_sequence)
        new_t, new_s = self.predict_toughness(new_sequence)

        raw_imp = new_t - old_t
        penalty = 0.5 * (new_s - old_s)
        adj_imp = raw_imp - penalty

        combined_s  = math.sqrt(old_s**2 + new_s**2)
        sig_thr     = 1.96 * combined_s

        if adj_imp > sig_thr:
            reward = adj_imp
        elif adj_imp > 0:
            reward = 0.3 * adj_imp
        else:
            reward = 2.0 * adj_imp

        if reward > 0:
            reward = math.log1p(reward * 10) / 2

        return reward

    def realism_reward(self, seq, original_seq):
        # continuous perplexity penalty
        thresh, alpha = 2.5, 1.0
        ppl = self.calculate_perplexity(seq)
        ppl_penalty = -alpha * max(0.0, ppl - thresh)

        # quadratic length penalty
        lr = len(seq) / len(original_seq)
        len_pen = -0.5 * (lr - 1.0)**2

        # motif + composition
        motif_pen = self._motif_penalty(seq)
        comp_pen  = self._composition_bonus(seq)

        return ppl_penalty + len_pen + motif_pen + comp_pen

    def exploration_reward(self, edit_history, episode_number):
        recent = edit_history[-10:]
        # fallback to '<unk>' if no 'type'
        types = { e.get('type','<unk>') for e in recent }
        div_bonus = 0.02 * len(types)

        # fallback to position 0 if missing
        positions = [ e.get('position',0) for e in recent ]
        spread = max(positions) - min(positions) if positions else 0
        pos_bonus = 0.1 if spread > 0.3 * (len(edit_history) or 1) else 0.0

        base_bonus = div_bonus + pos_bonus
        w_explore = self.get_adaptive_weights(episode_number)['exploration']
        return base_bonus * (w_explore / 0.1)


    def efficiency_reward(self, edit_count, total_improvement, target_improvement):
        step_pen = -0.01 * edit_count
        if total_improvement >= target_improvement:
            eff_bonus = 0.5 * max(0, 1.0 - edit_count / 50)
        else:
            eff_bonus = 0.0
        term_bonus = 0.2 if total_improvement >= 0.8 * target_improvement else 0.0
        return step_pen + eff_bonus + term_bonus

    # — helper stubs you can fill in —
    def _motif_penalty(self, seq):
        pen = 0.0
        if seq.count('GPG') < 2: pen -= 0.3
        if 'AAAA' not in seq:    pen -= 0.3
        return pen

    def _composition_bonus(self, seq):
        bonus = 0.0
        g_frac = seq.count('G') / len(seq)
        bonus += 0.05 if 0.15 <= g_frac <= 0.40 else -0.2
        for aa in 'ACDEFGHIKLMNPQRSTVWY':
            if seq.count(aa)/len(seq) > 0.6:
                bonus -= 0.5
        return bonus