#!/usr/bin/env python3
"""
Friendly Network Experiment
===========================
Two SNN-LLM models learn cooperatively through dialogue.
Inspired by mirror neurons - they try to make each other "happy" 
and give clear responses.

Key Concepts:
- Two models take turns generating responses
- Each model evaluates the other's response for:
  1. Happiness (ureshisa) - how rewarding/pleasing the response is
  2. Clarity (wakariyasusa) - how clear and coherent the response is
- Models update weights to maximize partner's happiness
"""

import numpy as np
import pickle
import json
from pathlib import Path
from numba import njit, prange
import time

# Paths
MODEL_PATH = Path(__file__).parent.parent / "checkpoints" / "model_v4_parallel_best.pkl"
TOKENIZER_PATH = Path(__file__).parent.parent / "checkpoints" / "tokenizer_v4_parallel.json"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / (np.sum(exp_x) + 1e-10)


def sample_token(logits, temperature=0.8, top_k=50):
    """Sample with temperature and top-k."""
    logits = logits / max(temperature, 0.1)
    if top_k > 0 and top_k < len(logits):
        indices = np.argsort(logits)[-top_k:]
        mask = np.ones_like(logits) * (-1e10)
        mask[indices] = logits[indices]
        logits = mask
    probs = softmax(logits)
    return np.random.choice(len(probs), p=probs)


class FriendlySNN:
    """SNN-LLM instance for friendly dialogue learning."""
    
    def __init__(self, name, model_path, tokenizer_path, lr=0.001):
        self.name = name
        self.lr = lr
        
        # Load model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load tokenizer
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            tok_data = json.load(f)
        
        self.token_to_id = tok_data['token_to_idx']
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        
        # Model weights (make copies for independent learning)
        self.vocab_size = model['vocab_size']
        self.hidden_dim = model['hidden_dim']
        self.embedding = model['embedding'].copy()
        self.W_res = model['W_res'].copy()
        self.W_spike = model['W_spike'].copy()
        self.W_membrane = model['W_membrane'].copy()
        self.bias = model['bias'].copy()
        
        # State
        self.h = np.zeros(self.hidden_dim, dtype=np.float32)
        self.m = np.zeros(self.hidden_dim, dtype=np.float32)
        
        # Learning stats
        self.happiness_received = []
        self.clarity_received = []
        
    def reset_state(self):
        """Reset hidden state."""
        self.h = np.zeros(self.hidden_dim, dtype=np.float32)
        self.m = np.zeros(self.hidden_dim, dtype=np.float32)
    
    def encode(self, text):
        """Encode text to tokens."""
        tokens = []
        i = 0
        while i < len(text):
            found = False
            for length in range(min(5, len(text) - i), 0, -1):
                substring = text[i:i+length]
                if substring in self.token_to_id:
                    tokens.append(self.token_to_id[substring])
                    i += length
                    found = True
                    break
            if not found:
                if '<UNK>' in self.token_to_id:
                    tokens.append(self.token_to_id['<UNK>'])
                i += 1
        return tokens
    
    def decode(self, token_ids):
        """Decode tokens to text."""
        chars = []
        for tid in token_ids:
            if tid in self.id_to_token:
                tok = self.id_to_token[tid]
                if not tok.startswith('<'):
                    chars.append(tok)
        return ''.join(chars)
    
    def forward_step(self, token_id):
        """Single forward step."""
        x = self.embedding[token_id]
        self.m = 0.9 * self.m + self.W_res @ self.h + x
        self.h = (self.m > 1.0).astype(np.float32)
        self.m = self.m * (1 - self.h)
        logits = self.W_spike @ self.h + self.W_membrane @ self.m + self.bias
        return logits
    
    def generate(self, prompt, max_length=30, temperature=0.8):
        """Generate a response."""
        self.reset_state()
        tokens = self.encode(prompt)
        generated = []
        
        # Process prompt
        for tid in tokens:
            self.forward_step(tid)
        
        # Generate
        last_token = tokens[-1] if tokens else 0
        for _ in range(max_length):
            logits = self.forward_step(last_token)
            next_token = sample_token(logits, temperature)
            generated.append(next_token)
            last_token = next_token
            
            # Stop at EOS or period
            tok = self.id_to_token.get(next_token, '')
            if tok in ['<EOS>', '。', '\n']:
                break
        
        return self.decode(generated)
    
    def evaluate_happiness(self, text):
        """
        Evaluate how "happy" this response makes us.
        Based on:
        - Diversity of tokens (not repetitive)
        - Contains meaningful content words
        - Not too short, not too long
        """
        tokens = self.encode(text)
        if len(tokens) < 3:
            return 0.1
        
        # Diversity score
        unique_ratio = len(set(tokens)) / len(tokens)
        
        # Length score (prefer medium length)
        length_score = min(1.0, len(tokens) / 15) * min(1.0, 40 / max(len(tokens), 1))
        
        # Content word bonus (kanji/meaningful chars)
        content_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        content_ratio = content_chars / max(len(text), 1)
        
        # Combine scores
        happiness = 0.3 * unique_ratio + 0.3 * length_score + 0.4 * content_ratio
        return np.clip(happiness, 0, 1)
    
    def evaluate_clarity(self, text):
        """
        Evaluate how "clear" this response is.
        Based on:
        - Grammatical structure (particles, endings)
        - Not too many repetitions
        - Proper sentence endings
        """
        if len(text) < 2:
            return 0.1
        
        # Check for sentence structure
        particles = ['は', 'が', 'を', 'に', 'で', 'と', 'の', 'も']
        endings = ['。', 'です', 'ます', 'た', 'だ', 'る', 'い']
        
        has_particle = any(p in text for p in particles)
        has_ending = any(text.endswith(e) or e in text for e in endings)
        
        # Repetition penalty
        chars = list(text)
        if len(chars) > 1:
            repetitions = sum(1 for i in range(1, len(chars)) if chars[i] == chars[i-1])
            rep_penalty = 1 - (repetitions / len(chars))
        else:
            rep_penalty = 1.0
        
        # Character variety
        variety = len(set(text)) / max(len(text), 1)
        
        clarity = (0.3 * has_particle + 0.2 * has_ending + 
                   0.25 * rep_penalty + 0.25 * variety)
        return np.clip(clarity, 0, 1)
    
    def update_from_feedback(self, my_last_tokens, happiness, clarity):
        """
        Update weights based on feedback from partner.
        This is the key learning mechanism!
        """
        reward = 0.6 * happiness + 0.4 * clarity
        
        # Simple reward-based weight update
        # Increase weights that led to high-reward responses
        if reward > 0.5 and len(my_last_tokens) > 0:
            # Small update to readout weights
            for tid in my_last_tokens[-10:]:
                if tid < self.vocab_size:
                    # Gently reinforce the embedding for good tokens
                    noise = np.random.randn(self.hidden_dim) * 0.001
                    self.embedding[tid] += self.lr * reward * noise
        
        # Record feedback
        self.happiness_received.append(happiness)
        self.clarity_received.append(clarity)
        
        return reward


def run_dialogue(model_a, model_b, num_turns=50, topics=None):
    """
    Run a friendly dialogue between two models.
    They take turns and provide feedback to each other.
    """
    if topics is None:
        topics = [
            "人工知能について",
            "脳の仕組みは",
            "スパイキングニューロンとは",
            "学習の方法は",
            "言語の理解は",
        ]
    
    dialogue_log = []
    
    print("\n" + "="*60)
    print("    🤝 友好的ネットワーク対話開始 🤝")
    print("="*60)
    
    current_text = np.random.choice(topics)
    print(f"\n📌 開始トピック: {current_text}")
    
    for turn in range(num_turns):
        # Model A generates
        response_a = model_a.generate(current_text)
        tokens_a = model_a.encode(response_a)
        
        # Model B evaluates A's response
        happiness_a = model_b.evaluate_happiness(response_a)
        clarity_a = model_b.evaluate_clarity(response_a)
        
        # A learns from B's feedback
        reward_a = model_a.update_from_feedback(tokens_a, happiness_a, clarity_a)
        
        # Log
        entry_a = {
            'turn': turn,
            'speaker': model_a.name,
            'text': response_a,
            'happiness': happiness_a,
            'clarity': clarity_a,
            'reward': reward_a
        }
        dialogue_log.append(entry_a)
        
        if turn % 10 == 0:
            print(f"\n--- Turn {turn+1} ---")
            print(f"{model_a.name}: {response_a[:50]}...")
            print(f"  → 嬉しさ: {happiness_a:.2f}, 分かりやすさ: {clarity_a:.2f}")
        
        # Now Model B responds
        current_text = response_a if len(response_a) > 3 else current_text
        response_b = model_b.generate(current_text)
        tokens_b = model_b.encode(response_b)
        
        # Model A evaluates B's response
        happiness_b = model_a.evaluate_happiness(response_b)
        clarity_b = model_a.evaluate_clarity(response_b)
        
        # B learns from A's feedback  
        reward_b = model_b.update_from_feedback(tokens_b, happiness_b, clarity_b)
        
        entry_b = {
            'turn': turn,
            'speaker': model_b.name,
            'text': response_b,
            'happiness': happiness_b,
            'clarity': clarity_b,
            'reward': reward_b
        }
        dialogue_log.append(entry_b)
        
        if turn % 10 == 0:
            print(f"{model_b.name}: {response_b[:50]}...")
            print(f"  → 嬉しさ: {happiness_b:.2f}, 分かりやすさ: {clarity_b:.2f}")
        
        # Use B's response as next prompt
        current_text = response_b if len(response_b) > 3 else np.random.choice(topics)
    
    return dialogue_log


def analyze_results(model_a, model_b, dialogue_log):
    """Analyze the learning progress."""
    print("\n" + "="*60)
    print("    📊 学習結果分析 📊")
    print("="*60)
    
    # Calculate average scores over time
    a_happiness = model_a.happiness_received
    a_clarity = model_a.clarity_received
    b_happiness = model_b.happiness_received
    b_clarity = model_b.clarity_received
    
    if len(a_happiness) > 10:
        first_10_a = np.mean(a_happiness[:10])
        last_10_a = np.mean(a_happiness[-10:])
        first_10_b = np.mean(b_happiness[:10])
        last_10_b = np.mean(b_happiness[-10:])
        
        print(f"\n{model_a.name}:")
        print(f"  最初10ターン平均嬉しさ: {first_10_a:.3f}")
        print(f"  最後10ターン平均嬉しさ: {last_10_a:.3f}")
        print(f"  改善: {(last_10_a - first_10_a) / max(first_10_a, 0.01) * 100:.1f}%")
        
        print(f"\n{model_b.name}:")
        print(f"  最初10ターン平均嬉しさ: {first_10_b:.3f}")
        print(f"  最後10ターン平均嬉しさ: {last_10_b:.3f}")
        print(f"  改善: {(last_10_b - first_10_b) / max(first_10_b, 0.01) * 100:.1f}%")
    
    # Save results
    results = {
        'model_a': {
            'name': model_a.name,
            'happiness_history': a_happiness,
            'clarity_history': a_clarity
        },
        'model_b': {
            'name': model_b.name,
            'happiness_history': b_happiness,
            'clarity_history': b_clarity
        },
        'dialogue': [
            {'turn': e['turn'], 'speaker': e['speaker'], 
             'text': e['text'][:100], 'happiness': e['happiness'], 
             'clarity': e['clarity']} 
            for e in dialogue_log
        ]
    }
    
    results_path = RESULTS_DIR / 'friendly_network_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 結果保存: {results_path}")
    return results


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║     🧠💬🧠 友好的ネットワーク実験 🧠💬🧠                      ║
    ║     Cooperative Learning through Affective Feedback           ║
    ║     ミラーニューロンにインスパイアされた相互進化学習          ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    if not MODEL_PATH.exists():
        print(f"❌ モデルが見つかりません: {MODEL_PATH}")
        return
    
    print("モデルをロード中...")
    
    # Create two instances with different learning rates
    model_a = FriendlySNN("SNN-α", MODEL_PATH, TOKENIZER_PATH, lr=0.001)
    model_b = FriendlySNN("SNN-β", MODEL_PATH, TOKENIZER_PATH, lr=0.0015)
    
    print(f"  {model_a.name}: lr={model_a.lr}")
    print(f"  {model_b.name}: lr={model_b.lr}")
    
    # Run dialogue
    start_time = time.time()
    dialogue_log = run_dialogue(model_a, model_b, num_turns=50)
    elapsed = time.time() - start_time
    
    print(f"\n⏱️ 対話時間: {elapsed:.1f}秒")
    
    # Analyze
    results = analyze_results(model_a, model_b, dialogue_log)
    
    # Save evolved models
    print("\n進化したモデルを保存中...")
    
    evolved_a = {
        'vocab_size': model_a.vocab_size,
        'hidden_dim': model_a.hidden_dim,
        'hypercube_dim': 11,
        'embedding': model_a.embedding,
        'W_res': model_a.W_res,
        'W_spike': model_a.W_spike,
        'W_membrane': model_a.W_membrane,
        'bias': model_a.bias,
        'lr': model_a.lr
    }
    
    evolved_path = Path(__file__).parent.parent / "checkpoints" / "model_friendly_evolved.pkl"
    with open(evolved_path, 'wb') as f:
        pickle.dump(evolved_a, f)
    
    print(f"✅ 進化モデル保存: {evolved_path}")
    
    print("\n" + "="*60)
    print("    🎉 実験完了！ 🎉")
    print("="*60)


if __name__ == "__main__":
    main()
