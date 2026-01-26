"""
Japanese SNN-LLM v4 Parallel Training
======================================

CPU-parallelized version using Numba.
Optimized for Ryzen AI 9 HX 375 (24 threads).

Features:
- Numba JIT compilation
- Parallel processing across 24 threads
- Vectorized operations
- 5-10x faster than naive Python

Usage: python experiments/train_japanese_v4_parallel.py

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

# Set threads
config.NUMBA_NUM_THREADS = 24
print(f"Numba version: {numba.__version__}")
print(f"Using {config.NUMBA_NUM_THREADS} threads")


# ==============================================================================
# Numba-accelerated Functions
# ==============================================================================

@njit(parallel=True, fastmath=True, cache=True)
def reservoir_step_parallel(membrane, spike_rate, embedding, W_res, batch_size):
    """Parallel reservoir update for batch of tokens"""
    hidden_dim = membrane.shape[0]
    
    new_membrane = np.zeros_like(membrane)
    new_spike_rate = np.zeros_like(spike_rate)
    
    # Parallel recurrent computation
    for i in prange(hidden_dim):
        h_rec = 0.0
        for j in range(hidden_dim):
            h_rec += np.tanh(membrane[j]) * W_res[i, j]
        
        new_membrane[i] = 0.7 * membrane[i] + 0.3 * (embedding[i] + h_rec)
        
        # Spiking
        if new_membrane[i] > 0.5:
            spike = 1.0
            new_membrane[i] *= 0.3
        else:
            spike = 0.0
        
        new_spike_rate[i] = 0.8 * spike_rate[i] + 0.2 * spike
    
    return new_membrane, new_spike_rate


@njit(parallel=True, fastmath=True, cache=True)
def compute_logits_parallel(spike_rate, membrane, W_spike, W_membrane, bias):
    """Parallel hybrid readout computation"""
    vocab_size = W_spike.shape[0]
    logits = np.zeros(vocab_size, dtype=np.float32)
    
    for i in prange(vocab_size):
        s = bias[i]
        for j in range(spike_rate.shape[0]):
            s += W_spike[i, j] * spike_rate[j]
            s += W_membrane[i, j] * membrane[j]
        logits[i] = s
    
    return logits


@njit(fastmath=True, cache=True)
def softmax(logits):
    """Numerically stable softmax"""
    max_val = np.max(logits)
    exp_logits = np.exp(logits - max_val)
    return exp_logits / np.sum(exp_logits)


@njit(parallel=True, fastmath=True, cache=True)
def update_weights_parallel(W_spike, W_membrane, bias, grad, spike_rate, membrane, lr):
    """Parallel weight update"""
    vocab_size = W_spike.shape[0]
    hidden_dim = W_spike.shape[1]
    
    for i in prange(vocab_size):
        bias[i] -= lr * grad[i]
        for j in range(hidden_dim):
            W_spike[i, j] -= lr * grad[i] * spike_rate[j]
            W_membrane[i, j] -= lr * 0.5 * grad[i] * membrane[j]


@njit(parallel=True, fastmath=True, cache=True)
def create_hypercube_mask(dim, size):
    """Create 11D hypercube adjacency mask"""
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


# ==============================================================================
# Tokenizer
# ==============================================================================

class JapaneseTokenizer:
    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.token_to_idx = {}
        self.idx_to_token = {}
        self.actual_vocab_size = 0
    
    def build_vocab(self, text, min_freq=2):
        char_counts = Counter(text)
        self.token_to_idx = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
        
        idx = len(self.token_to_idx)
        for char, count in char_counts.most_common():
            if count >= min_freq and char not in self.token_to_idx:
                self.token_to_idx[char] = idx
                idx += 1
                if idx >= self.vocab_size // 2:
                    break
        
        bigram_counts = Counter()
        for i in range(len(text) - 1):
            bigram = text[i:i+2]
            bigram_counts[bigram] += 1
        
        for bigram, count in bigram_counts.most_common():
            if count >= min_freq * 2 and bigram not in self.token_to_idx:
                self.token_to_idx[bigram] = idx
                idx += 1
                if idx >= self.vocab_size:
                    break
        
        self.idx_to_token = {i: t for t, i in self.token_to_idx.items()}
        self.actual_vocab_size = len(self.token_to_idx)
    
    def encode(self, text):
        tokens = []
        i = 0
        while i < len(text):
            matched = False
            for length in [2, 1]:
                if i + length <= len(text):
                    substr = text[i:i+length]
                    if substr in self.token_to_idx:
                        tokens.append(self.token_to_idx[substr])
                        i += length
                        matched = True
                        break
            if not matched:
                tokens.append(self.token_to_idx.get('<UNK>', 1))
                i += 1
        return tokens
    
    def decode(self, indices):
        return ''.join(self.idx_to_token.get(int(i), '<UNK>') for i in indices)
    
    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'token_to_idx': self.token_to_idx,
                'vocab_size': self.actual_vocab_size
            }, f, ensure_ascii=False, indent=2)


# ==============================================================================
# SNN-LM v4 Parallel
# ==============================================================================

class SNNLM_v4_Parallel:
    """
    Numba-parallelized SNN Language Model v4
    - 11D Hypercube topology
    - Hybrid readout
    - 24-thread parallel processing
    """
    
    def __init__(self, vocab_size, hidden_dim=2048, hypercube_dim=11, seed=42):
        np.random.seed(seed)
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.hypercube_dim = hypercube_dim
        
        # Embedding
        self.embedding = np.random.randn(vocab_size, hidden_dim).astype(np.float32) * 0.05
        
        # 11D Hypercube reservoir
        print("  Creating 11D hypercube mask (parallel)...")
        mask = create_hypercube_mask(hypercube_dim, hidden_dim)
        
        W_res = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.15
        self.W_res = (W_res * mask).astype(np.float32)
        
        # Scale spectral radius
        sample_size = min(256, hidden_dim)
        eig = np.linalg.eigvals(self.W_res[:sample_size, :sample_size])
        self.W_res *= 1.2 / (np.max(np.abs(eig)) + 0.01)
        self.W_res = self.W_res.astype(np.float32)
        
        self.mask = mask
        
        # Hybrid readout
        self.W_spike = np.random.randn(vocab_size, hidden_dim).astype(np.float32) * 0.01
        self.W_membrane = np.random.randn(vocab_size, hidden_dim).astype(np.float32) * 0.005
        self.bias = np.zeros(vocab_size, dtype=np.float32)
        
        # State
        self.membrane = np.zeros(hidden_dim, dtype=np.float32)
        self.spike_rate = np.zeros(hidden_dim, dtype=np.float32)
        
        # Learning
        self.lr = 0.01
    
    def forward(self, token_idx):
        emb = self.embedding[token_idx]
        
        # Parallel reservoir step
        self.membrane, self.spike_rate = reservoir_step_parallel(
            self.membrane, self.spike_rate, emb, self.W_res, 1
        )
        
        # Parallel logits computation
        logits = compute_logits_parallel(
            self.spike_rate, self.membrane,
            self.W_spike, self.W_membrane, self.bias
        )
        
        return logits
    
    def train_step(self, input_idx, target_idx):
        logits = self.forward(input_idx)
        probs = softmax(logits)
        
        # Cross-entropy loss
        loss = -np.log(probs[target_idx] + 1e-10)
        
        # Gradient
        grad = probs.copy()
        grad[target_idx] -= 1
        
        # Parallel weight update
        update_weights_parallel(
            self.W_spike, self.W_membrane, self.bias,
            grad, self.spike_rate, self.membrane, self.lr
        )
        
        return loss
    
    def reset(self):
        self.membrane = np.zeros(self.hidden_dim, dtype=np.float32)
        self.spike_rate = np.zeros(self.hidden_dim, dtype=np.float32)
    
    def evaluate(self, tokens, max_eval=5000):
        self.reset()
        tokens = tokens[:max_eval]
        total_loss = 0
        
        for i in range(len(tokens) - 1):
            logits = self.forward(tokens[i])
            probs = softmax(logits)
            total_loss += -np.log(probs[tokens[i+1]] + 1e-10)
        
        return np.exp(total_loss / (len(tokens) - 1))
    
    def generate(self, start_tokens, max_len=100, temperature=0.7):
        self.reset()
        generated = list(start_tokens)
        
        for idx in start_tokens:
            self.forward(idx)
        
        for _ in range(max_len):
            logits = self.forward(generated[-1])
            log_p = logits / temperature
            probs = softmax(log_p)
            next_idx = np.random.choice(len(probs), p=probs)
            generated.append(next_idx)
        
        return generated
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'vocab_size': self.vocab_size,
                'hidden_dim': self.hidden_dim,
                'hypercube_dim': self.hypercube_dim,
                'embedding': self.embedding,
                'W_res': self.W_res,
                'W_spike': self.W_spike,
                'W_membrane': self.W_membrane,
                'bias': self.bias,
                'lr': self.lr,
            }, f)
    
    def get_stats(self):
        return {
            'vocab_size': self.vocab_size,
            'hidden_dim': self.hidden_dim,
            'hypercube_dim': self.hypercube_dim,
            'total_params': (
                self.embedding.size +
                self.W_res.size +
                self.W_spike.size +
                self.W_membrane.size +
                self.bias.size
            ),
            'reservoir_connections': int(self.mask.sum())
        }


# ==============================================================================
# Main
# ==============================================================================

def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║   ⚡ Japanese SNN-LLM v4 Parallel: 24-Thread Edition ⚡       ║
    ║   Numba JIT + 11D Hypercube + Ryzen AI 9 HX 375               ║
    ║   Estimated: 1-2 hours (5-10x faster than naive)              ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    start_time = time.time()
    
    # Configuration
    config = {
        'hidden_dim': 2048,
        'hypercube_dim': 11,
        'vocab_size': 4000,
        'epochs': 100,
        'lr': 0.01,
    }
    
    print("Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print(f"  threads: {numba.config.NUMBA_NUM_THREADS}")
    print("-" * 60)
    
    # Load corpus
    print("\n[1/4] Loading corpus...")
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    
    for corpus_name in ["japanese_corpus_full.txt", "japanese_corpus_large.txt", "japanese_corpus.txt"]:
        corpus_path = os.path.join(data_dir, corpus_name)
        if os.path.exists(corpus_path):
            break
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"  Corpus: {os.path.basename(corpus_path)}")
    print(f"  Characters: {len(text):,}")
    
    # Tokenizer
    print("\n[2/4] Building tokenizer...")
    tokenizer = JapaneseTokenizer(vocab_size=config['vocab_size'])
    tokenizer.build_vocab(text)
    print(f"  Vocabulary: {tokenizer.actual_vocab_size}")
    
    tokens = np.array(tokenizer.encode(text), dtype=np.int32)
    print(f"  Tokens: {len(tokens):,}")
    
    # Model
    print("\n[3/4] Creating parallel model...")
    model = SNNLM_v4_Parallel(
        vocab_size=tokenizer.actual_vocab_size,
        hidden_dim=config['hidden_dim'],
        hypercube_dim=config['hypercube_dim']
    )
    
    stats = model.get_stats()
    print(f"  Parameters: {stats['total_params']:,}")
    print(f"  Hypercube: {config['hypercube_dim']}D (OPTIMAL)")
    
    # Warmup JIT
    print("\n  Warming up JIT compilation...")
    for i in range(100):
        model.train_step(tokens[i], tokens[i+1])
    model.reset()
    print("  JIT warmup complete!")
    
    # Directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Training
    print("\n[4/4] Training (24-thread parallel)...")
    print("=" * 60)
    
    history = []
    best_ppl = float('inf')
    
    for epoch in range(config['epochs']):
        model.reset()
        t0 = time.time()
        total_loss = 0
        n_tokens = len(tokens) - 1
        
        # Progress
        log_interval = n_tokens // 5
        
        for i in range(n_tokens):
            loss = model.train_step(tokens[i], tokens[i + 1])
            total_loss += loss
            
            if (i + 1) % log_interval == 0:
                progress = (i + 1) / n_tokens * 100
                tokens_per_sec = (i + 1) / (time.time() - t0)
                print(f"    Epoch {epoch+1}: {progress:.0f}% ({tokens_per_sec:.0f} tok/s)")
        
        avg_loss = total_loss / n_tokens
        
        # Evaluate
        model.reset()
        ppl = model.evaluate(tokens[:5000])
        
        elapsed = time.time() - t0
        tokens_per_sec = n_tokens / elapsed
        
        # LR decay every 20 epochs
        if (epoch + 1) % 20 == 0:
            model.lr *= 0.9
            print(f"    LR decayed to {model.lr:.5f}")
        
        print(f"  Epoch {epoch+1:>3}/{config['epochs']}: Loss={avg_loss:.4f}, PPL={ppl:.2f}, Time={elapsed:.0f}s, Speed={tokens_per_sec:.0f} tok/s")
        
        history.append({
            'epoch': epoch + 1,
            'loss': float(avg_loss),
            'ppl': float(ppl),
            'time': elapsed,
            'tokens_per_sec': tokens_per_sec
        })
        
        # Save checkpoints
        if (epoch + 1) % 20 == 0:
            model.save(f"checkpoints/model_v4_parallel_epoch{epoch+1}.pkl")
        
        if ppl < best_ppl:
            best_ppl = ppl
            model.save("checkpoints/model_v4_parallel_best.pkl")
            print(f"    → Best model saved! (PPL={ppl:.2f})")
    
    # Save final
    model.save("checkpoints/model_v4_parallel_final.pkl")
    tokenizer.save("checkpoints/tokenizer_v4_parallel.json")
    
    with open("results/v4_parallel_history.json", 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    
    # Generation
    print("\n" + "=" * 60)
    print("GENERATION SAMPLES")
    print("=" * 60)
    
    for prompt in ["脳は", "人工知能", "スパイキング", "言語モデル"]:
        prompt_tokens = tokenizer.encode(prompt)
        gen_tokens = model.generate(prompt_tokens, max_len=50)
        gen_text = tokenizer.decode(gen_tokens)
        print(f"\n'{prompt}' → '{gen_text[:80]}...'")
    
    # Summary
    total_time = time.time() - start_time
    avg_speed = sum(h['tokens_per_sec'] for h in history) / len(history)
    
    print(f"""
    ╔═══════════════════════════════════════════════════════════════╗
    ║            ⚡ Parallel Training Complete! ⚡                  ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║   Best Perplexity: {best_ppl:>8.2f}                                 ║
    ║   Total Time:      {total_time/60:>8.1f} min                        ║
    ║   Avg Speed:       {avg_speed:>8.0f} tokens/sec                  ║
    ║   Parameters:      {stats['total_params']:>8,}                      ║
    ║   Threads:                 {numba.config.NUMBA_NUM_THREADS}                             ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║   Model: checkpoints/model_v4_parallel_best.pkl               ║
    ║   History: results/v4_parallel_history.json                   ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
