"""
Japanese SNN-LLM v4 Training
============================

Enhanced model with:
1. 11D Hypercube (optimal dimension)
2. Larger corpus (589K chars)
3. Hybrid readout (spike + state)
4. 100 epochs for better convergence

Usage: python experiments/train_japanese_v4.py

Author: ろーる
Date: 2026-01-22
"""

import numpy as np
import os
import sys
import time
import json
import pickle
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hypercube import create_hypercube_mask


# ==============================================================================
# Tokenizer (same as before)
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
        return ''.join(self.idx_to_token.get(i, '<UNK>') for i in indices)
    
    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'token_to_idx': self.token_to_idx,
                'vocab_size': self.actual_vocab_size
            }, f, ensure_ascii=False, indent=2)


# ==============================================================================
# SNN-LM v4: 11D Hypercube + Hybrid Readout
# ==============================================================================

class SNNLM_v4:
    """
    SNN Language Model v4
    - 11D Hypercube (optimal dimension from experiments)
    - Hybrid readout: spike count + membrane state
    - Larger hidden dimension
    """
    
    def __init__(self, vocab_size, hidden_dim=2048, hypercube_dim=11, seed=42):
        np.random.seed(seed)
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.hypercube_dim = hypercube_dim
        
        # Embedding
        self.embedding = np.random.randn(vocab_size, hidden_dim).astype(np.float32) * 0.05
        
        # 11D Hypercube reservoir
        mask = create_hypercube_mask(hypercube_dim)
        
        # Resize mask
        target_size = hidden_dim
        orig_size = mask.shape[0]
        if orig_size != target_size:
            new_mask = np.zeros((target_size, target_size), dtype=np.float32)
            for i in range(target_size):
                for j in range(target_size):
                    new_mask[i, j] = mask[i % orig_size, j % orig_size]
            mask = new_mask
        
        W_res = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.15
        self.W_res = (W_res * mask).astype(np.float32)
        
        # Scale spectral radius for chaotic edge
        try:
            sample_size = min(512, hidden_dim)
            eig = np.linalg.eigvals(self.W_res[:sample_size, :sample_size])
            self.W_res *= 1.2 / (np.max(np.abs(eig)) + 0.01)
        except:
            pass
        
        self.mask = mask
        
        # Hybrid readout: spike weight + membrane weight
        self.W_spike = np.random.randn(vocab_size, hidden_dim).astype(np.float32) * 0.01
        self.W_membrane = np.random.randn(vocab_size, hidden_dim).astype(np.float32) * 0.005
        self.b_out = np.zeros(vocab_size, dtype=np.float32)
        
        # State
        self.membrane = np.zeros(hidden_dim, dtype=np.float32)
        self.spike_rate = np.zeros(hidden_dim, dtype=np.float32)
        
        # Learning
        self.lr = 0.01
        self.lr_decay = 0.95
    
    def forward(self, token_idx):
        emb = self.embedding[token_idx]
        
        # Reservoir dynamics
        h_rec = np.tanh(self.W_res @ self.membrane)
        self.membrane = 0.7 * self.membrane + 0.3 * (emb + h_rec)
        
        # Spiking threshold (LIF-like)
        spikes = (self.membrane > 0.5).astype(np.float32)
        self.membrane = np.where(spikes > 0, self.membrane * 0.3, self.membrane)
        
        # Update spike rate (exponential moving average)
        self.spike_rate = 0.8 * self.spike_rate + 0.2 * spikes
        
        # Hybrid readout: combine spike and membrane information
        logits = (self.W_spike @ self.spike_rate + 
                  self.W_membrane @ self.membrane + 
                  self.b_out)
        
        return logits
    
    def train_step(self, input_idx, target_idx):
        logits = self.forward(input_idx)
        
        # Softmax
        logits_max = np.max(logits)
        exp_l = np.exp(logits - logits_max)
        probs = exp_l / (np.sum(exp_l) + 1e-10)
        
        # Cross-entropy loss
        loss = -np.log(probs[target_idx] + 1e-10)
        
        # Gradient
        grad = probs.copy()
        grad[target_idx] -= 1
        
        # Update both readout weights
        self.W_spike -= self.lr * np.outer(grad, self.spike_rate)
        self.W_membrane -= self.lr * 0.5 * np.outer(grad, self.membrane)
        self.b_out -= self.lr * grad
        
        return loss
    
    def decay_lr(self):
        self.lr *= self.lr_decay
    
    def reset(self):
        self.membrane = np.zeros(self.hidden_dim, dtype=np.float32)
        self.spike_rate = np.zeros(self.hidden_dim, dtype=np.float32)
    
    def generate(self, start_indices, max_len=100, temperature=0.7):
        self.reset()
        generated = list(start_indices)
        
        for idx in start_indices:
            self.forward(idx)
        
        for _ in range(max_len):
            logits = self.forward(generated[-1])
            log_p = logits / temperature
            exp_l = np.exp(log_p - np.max(log_p))
            probs = exp_l / np.sum(exp_l)
            next_idx = np.random.choice(len(probs), p=probs)
            generated.append(next_idx)
        
        return generated
    
    def evaluate(self, indices, max_eval=5000):
        self.reset()
        indices = indices[:max_eval]
        total_loss = 0
        for i in range(len(indices) - 1):
            logits = self.forward(indices[i])
            exp_l = np.exp(logits - np.max(logits))
            probs = exp_l / (np.sum(exp_l) + 1e-10)
            total_loss += -np.log(probs[indices[i+1]] + 1e-10)
        return np.exp(total_loss / (len(indices) - 1))
    
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
                'b_out': self.b_out,
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
                self.b_out.size
            ),
            'reservoir_connections': int(self.mask.sum())
        }


# ==============================================================================
# Main Training
# ==============================================================================

def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║      🧠 Japanese SNN-LLM v4: 11D Hypercube Edition 🧠        ║
    ║      Optimal dimension + Hybrid readout + Full corpus         ║
    ║      Estimated: 4-5 hours (100 epochs)                        ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    start_time = time.time()
    
    # Configuration
    config = {
        'hidden_dim': 2048,      # Larger
        'hypercube_dim': 11,     # Optimal!
        'vocab_size': 4000,      # Larger vocab
        'epochs': 100,           # More epochs
        'lr': 0.01,
    }
    
    print("Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print("-" * 60)
    
    # Load corpus (use full corpus with synthetic data)
    print("\n[1/4] Loading full Japanese corpus...")
    corpus_path = os.path.join(os.path.dirname(__file__), "..", "data", "japanese_corpus_full.txt")
    
    if not os.path.exists(corpus_path):
        # Fallback to large corpus
        corpus_path = os.path.join(os.path.dirname(__file__), "..", "data", "japanese_corpus_large.txt")
    
    if not os.path.exists(corpus_path):
        # Fallback to original
        corpus_path = os.path.join(os.path.dirname(__file__), "..", "data", "japanese_corpus.txt")
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"  Corpus: {corpus_path}")
    print(f"  Total characters: {len(text):,}")
    
    # Build tokenizer
    print("\n[2/4] Building tokenizer...")
    tokenizer = JapaneseTokenizer(vocab_size=config['vocab_size'])
    tokenizer.build_vocab(text)
    print(f"  Vocabulary size: {tokenizer.actual_vocab_size}")
    
    # Tokenize
    tokens = np.array(tokenizer.encode(text), dtype=np.int32)
    print(f"  Total tokens: {len(tokens):,}")
    print(f"  Compression ratio: {len(text) / len(tokens):.2f}x")
    
    # Create model
    print("\n[3/4] Creating v4 model (11D Hypercube + Hybrid Readout)...")
    model = SNNLM_v4(
        vocab_size=tokenizer.actual_vocab_size,
        hidden_dim=config['hidden_dim'],
        hypercube_dim=config['hypercube_dim']
    )
    
    stats = model.get_stats()
    print(f"  Parameters: {stats['total_params']:,}")
    print(f"  Reservoir connections: {stats['reservoir_connections']:,}")
    print(f"  Hypercube dimension: {stats['hypercube_dim']}D (OPTIMAL)")
    
    # Directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Training
    print("\n[4/4] Training...")
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
                print(f"    Epoch {epoch+1}: {progress:.0f}%")
        
        avg_loss = total_loss / n_tokens
        
        # Evaluate
        model.reset()
        ppl = model.evaluate(tokens[:5000])
        
        elapsed = time.time() - t0
        
        # LR decay every 20 epochs
        if (epoch + 1) % 20 == 0:
            model.decay_lr()
            print(f"    LR decayed to {model.lr:.5f}")
        
        print(f"  Epoch {epoch+1:>3}/{config['epochs']}: Loss={avg_loss:.4f}, PPL={ppl:.2f}, Time={elapsed:.0f}s")
        
        history.append({
            'epoch': epoch + 1,
            'loss': float(avg_loss),
            'ppl': float(ppl),
            'time': elapsed
        })
        
        # Save checkpoints
        if (epoch + 1) % 20 == 0:
            model.save(f"checkpoints/model_v4_epoch{epoch+1}.pkl")
        
        if ppl < best_ppl:
            best_ppl = ppl
            model.save("checkpoints/model_v4_best.pkl")
            print(f"    → Best model saved! (PPL={ppl:.2f})")
    
    # Save final
    model.save("checkpoints/model_v4_final.pkl")
    tokenizer.save("checkpoints/tokenizer_v4.json")
    
    with open("results/v4_history.json", 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    
    # Generation samples
    print("\n" + "=" * 60)
    print("GENERATION SAMPLES")
    print("=" * 60)
    
    prompts = ["脳は", "人工知能", "スパイキング", "言語モデル"]
    for prompt in prompts:
        prompt_tokens = tokenizer.encode(prompt)
        gen_tokens = model.generate(prompt_tokens, max_len=80, temperature=0.7)
        gen_text = tokenizer.decode(gen_tokens)
        print(f"\n'{prompt}' →")
        print(f"  '{gen_text[:100]}...'")
    
    # Summary
    total_time = time.time() - start_time
    
    print(f"""
    ╔═══════════════════════════════════════════════════════════════╗
    ║            🎉 SNN-LLM v4 訓練完了！🎉                         ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║   Best Perplexity: {best_ppl:>8.2f}                                 ║
    ║   総訓練時間:      {total_time/60:>8.1f} 分                         ║
    ║   コーパスサイズ:  {len(text):>8,} chars                        ║
    ║   パラメータ数:    {stats['total_params']:>8,}                      ║
    ║   Hypercube次元:           {config['hypercube_dim']}D (OPTIMAL)              ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║   モデル: checkpoints/model_v4_best.pkl                       ║
    ║   履歴:   results/v4_history.json                             ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
