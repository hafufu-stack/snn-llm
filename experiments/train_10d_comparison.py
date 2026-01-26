"""
Lightweight Comparison Experiment: 10D vs 11D
=============================================

Quick experiment to compare 10D hypercube with the main 11D experiment.
Runs in parallel with the main training.

Author: ろーる
Date: 2026-01-22
"""

import numpy as np
import numba
from numba import njit, prange, config
import os
import sys
import time
import json
import pickle
from collections import Counter

# Use fewer threads to not interfere with main training
config.NUMBA_NUM_THREADS = 8
print(f"Comparison experiment using {config.NUMBA_NUM_THREADS} threads")


@njit(parallel=True, fastmath=True, cache=True)
def create_hypercube_mask(dim, size):
    n = 2 ** dim
    mask = np.zeros((size, size), dtype=np.float32)
    for i in prange(size):
        for j in range(size):
            i_mod = i % n
            j_mod = j % n
            xor = i_mod ^ j_mod
            bit_count = 0
            while xor > 0:
                bit_count += xor & 1
                xor >>= 1
            if bit_count == 1:
                mask[i, j] = 1.0
    return mask


@njit(parallel=True, fastmath=True, cache=True)
def reservoir_step(membrane, spike_rate, embedding, W_res):
    hidden_dim = membrane.shape[0]
    new_membrane = np.zeros_like(membrane)
    new_spike_rate = np.zeros_like(spike_rate)
    
    for i in prange(hidden_dim):
        h_rec = 0.0
        for j in range(hidden_dim):
            h_rec += np.tanh(membrane[j]) * W_res[i, j]
        new_membrane[i] = 0.7 * membrane[i] + 0.3 * (embedding[i] + h_rec)
        if new_membrane[i] > 0.5:
            spike = 1.0
            new_membrane[i] *= 0.3
        else:
            spike = 0.0
        new_spike_rate[i] = 0.8 * spike_rate[i] + 0.2 * spike
    return new_membrane, new_spike_rate


@njit(fastmath=True, cache=True)
def softmax(logits):
    max_val = np.max(logits)
    exp_logits = np.exp(logits - max_val)
    return exp_logits / np.sum(exp_logits)


class JapaneseTokenizer:
    def __init__(self, vocab_size=3000):
        self.vocab_size = vocab_size
        self.token_to_idx = {}
        self.idx_to_token = {}
        self.actual_vocab_size = 0
    
    def build_vocab(self, text, min_freq=2):
        char_counts = Counter(text)
        self.token_to_idx = {'<PAD>': 0, '<UNK>': 1}
        idx = 2
        for char, count in char_counts.most_common():
            if count >= min_freq and char not in self.token_to_idx:
                self.token_to_idx[char] = idx
                idx += 1
                if idx >= self.vocab_size:
                    break
        self.idx_to_token = {i: t for t, i in self.token_to_idx.items()}
        self.actual_vocab_size = len(self.token_to_idx)
    
    def encode(self, text):
        return [self.token_to_idx.get(c, 1) for c in text]
    
    def decode(self, indices):
        return ''.join(self.idx_to_token.get(int(i), '?') for i in indices)


class LightweightSNNLM:
    def __init__(self, vocab_size, hidden_dim=512, hypercube_dim=10, seed=42):
        np.random.seed(seed)
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.hypercube_dim = hypercube_dim
        
        self.embedding = np.random.randn(vocab_size, hidden_dim).astype(np.float32) * 0.05
        
        mask = create_hypercube_mask(hypercube_dim, hidden_dim)
        W_res = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.15
        self.W_res = (W_res * mask).astype(np.float32)
        
        sample_size = min(256, hidden_dim)
        eig = np.linalg.eigvals(self.W_res[:sample_size, :sample_size])
        self.W_res *= 1.2 / (np.max(np.abs(eig)) + 0.01)
        self.W_res = self.W_res.astype(np.float32)
        
        self.W_out = np.random.randn(vocab_size, hidden_dim).astype(np.float32) * 0.01
        self.b_out = np.zeros(vocab_size, dtype=np.float32)
        
        self.membrane = np.zeros(hidden_dim, dtype=np.float32)
        self.spike_rate = np.zeros(hidden_dim, dtype=np.float32)
        self.lr = 0.01
    
    def forward(self, token_idx):
        emb = self.embedding[token_idx]
        self.membrane, self.spike_rate = reservoir_step(
            self.membrane, self.spike_rate, emb, self.W_res
        )
        return self.W_out @ self.spike_rate + self.b_out
    
    def train_step(self, input_idx, target_idx):
        logits = self.forward(input_idx)
        probs = softmax(logits)
        loss = -np.log(probs[target_idx] + 1e-10)
        grad = probs.copy()
        grad[target_idx] -= 1
        self.W_out -= self.lr * np.outer(grad, self.spike_rate)
        self.b_out -= self.lr * grad
        return loss
    
    def reset(self):
        self.membrane.fill(0)
        self.spike_rate.fill(0)
    
    def evaluate(self, tokens, max_eval=3000):
        self.reset()
        tokens = tokens[:max_eval]
        total_loss = 0
        for i in range(len(tokens) - 1):
            logits = self.forward(tokens[i])
            probs = softmax(logits)
            total_loss += -np.log(probs[tokens[i+1]] + 1e-10)
        return np.exp(total_loss / (len(tokens) - 1))
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'vocab_size': self.vocab_size,
                'hidden_dim': self.hidden_dim,
                'hypercube_dim': self.hypercube_dim,
                'embedding': self.embedding,
                'W_res': self.W_res,
                'W_out': self.W_out,
                'b_out': self.b_out,
            }, f)


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║   🔬 Comparison Experiment: 10D Hypercube (Lightweight)       ║
    ║   Running in parallel with main 11D experiment                ║
    ║   8 threads, 512 hidden, 20 epochs                            ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    start_time = time.time()
    
    config = {
        'hidden_dim': 512,
        'hypercube_dim': 10,  # Compare with 11D
        'vocab_size': 2000,
        'epochs': 20,
    }
    
    print(f"Config: {config}")
    
    # Load corpus
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    for name in ["japanese_corpus_full.txt", "japanese_corpus_large.txt", "japanese_corpus.txt"]:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            break
    
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()[:100000]  # Use subset for speed
    
    print(f"Corpus: {len(text):,} chars (subset)")
    
    tokenizer = JapaneseTokenizer(vocab_size=config['vocab_size'])
    tokenizer.build_vocab(text)
    tokens = np.array(tokenizer.encode(text), dtype=np.int32)
    print(f"Tokens: {len(tokens):,}")
    
    model = LightweightSNNLM(
        vocab_size=tokenizer.actual_vocab_size,
        hidden_dim=config['hidden_dim'],
        hypercube_dim=config['hypercube_dim']
    )
    
    # Warmup
    for i in range(50):
        model.train_step(tokens[i], tokens[i+1])
    model.reset()
    
    print("\nTraining...")
    history = []
    best_ppl = float('inf')
    
    for epoch in range(config['epochs']):
        model.reset()
        t0 = time.time()
        total_loss = 0
        n = len(tokens) - 1
        
        for i in range(n):
            total_loss += model.train_step(tokens[i], tokens[i+1])
        
        avg_loss = total_loss / n
        model.reset()
        ppl = model.evaluate(tokens[:3000])
        elapsed = time.time() - t0
        
        print(f"  Epoch {epoch+1:>2}/{config['epochs']}: Loss={avg_loss:.4f}, PPL={ppl:.2f}, Time={elapsed:.0f}s")
        
        history.append({'epoch': epoch+1, 'loss': float(avg_loss), 'ppl': float(ppl)})
        
        if ppl < best_ppl:
            best_ppl = ppl
            model.save("checkpoints/model_10d_comparison_best.pkl")
    
    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/10d_comparison_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    total_time = time.time() - start_time
    
    print(f"""
    ╔═══════════════════════════════════════════════════════════════╗
    ║   10D Comparison Complete!                                    ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║   Best PPL: {best_ppl:>8.2f}                                        ║
    ║   Time:     {total_time/60:>8.1f} min                               ║
    ║   Dim:              10D                                       ║
    ╚═══════════════════════════════════════════════════════════════╝
    
    Compare with main 11D experiment to see the difference!
    """)


if __name__ == "__main__":
    main()
