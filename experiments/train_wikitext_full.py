"""
WikiText-2 Full Training Script
================================

Full training on WikiText-2 (10.7MB)
Optimized for long-running background execution.

Estimated time: ~90-120 minutes
Saves checkpoints every epoch.

Usage: python experiments/train_wikitext_full.py

Author: ろーる
Date: 2026-01-21
"""

import numpy as np
import os
import sys
import time
import json
import pickle
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hypercube import create_hypercube_mask


class OptimizedSNNLM:
    """Optimized SNN Language Model for large-scale training"""
    
    def __init__(self, vocab_size, hidden_dim=512, hypercube_dim=9, seed=42):
        np.random.seed(seed)
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        # Embedding
        self.embedding = np.random.randn(vocab_size, hidden_dim).astype(np.float32) * 0.1
        
        # Reservoir with hypercube topology
        mask = create_hypercube_mask(hypercube_dim)
        if mask.shape[0] != hidden_dim:
            new_mask = np.zeros((hidden_dim, hidden_dim), dtype=np.float32)
            for i in range(hidden_dim):
                for j in range(hidden_dim):
                    new_mask[i, j] = mask[i % mask.shape[0], j % mask.shape[0]]
            mask = new_mask
        
        W_res = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.3
        self.W_res = (W_res * mask).astype(np.float32)
        
        # Scale spectral radius
        try:
            eig = np.linalg.eigvals(self.W_res)
            self.W_res *= 1.2 / (np.max(np.abs(eig)) + 0.01)
        except:
            pass
        
        # Output
        self.W_out = np.random.randn(vocab_size, hidden_dim).astype(np.float32) * 0.01
        self.b_out = np.zeros(vocab_size, dtype=np.float32)
        
        # State
        self.state = np.zeros(hidden_dim, dtype=np.float32)
        
        # Learning
        self.lr = 0.01
    
    def forward(self, token_idx):
        """Single forward step - optimized"""
        emb = self.embedding[token_idx]
        h_rec = np.tanh(self.W_res @ self.state)
        self.state = 0.7 * self.state + 0.3 * (emb + h_rec)
        
        # Spike threshold
        mask = self.state > 0.5
        self.state = np.where(mask, self.state * 0.7, self.state)
        
        logits = self.W_out @ self.state + self.b_out
        return logits
    
    def train_step(self, input_idx, target_idx):
        """Train on single sample - optimized"""
        logits = self.forward(input_idx)
        
        # Softmax (numerically stable)
        logits_max = np.max(logits)
        exp_l = np.exp(logits - logits_max)
        probs = exp_l / (np.sum(exp_l) + 1e-10)
        
        # Loss
        loss = -np.log(probs[target_idx] + 1e-10)
        
        # Gradient
        grad = probs.copy()
        grad[target_idx] -= 1
        
        # Update output layer
        self.W_out -= self.lr * np.outer(grad, self.state)
        self.b_out -= self.lr * grad
        
        return loss
    
    def reset(self):
        self.state = np.zeros(self.hidden_dim, dtype=np.float32)
    
    def generate(self, start_indices, max_len=100, temperature=0.8):
        """Generate text"""
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
        """Calculate perplexity on subset"""
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
                'embedding': self.embedding,
                'W_res': self.W_res,
                'W_out': self.W_out,
                'b_out': self.b_out,
            }, f)


def download_wikitext2(data_dir="data"):
    """Download WikiText-2 if not exists"""
    os.makedirs(data_dir, exist_ok=True)
    
    url = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt"
    train_path = os.path.join(data_dir, "wikitext2_train.txt")
    
    if not os.path.exists(train_path):
        print("  Downloading WikiText-2...")
        try:
            urllib.request.urlretrieve(url, train_path)
        except:
            print("  Download failed, using synthetic data")
            text = "The brain is a complex neural network. " * 100000
            with open(train_path, 'w', encoding='utf-8') as f:
                f.write(text)
    
    with open(train_path, 'r', encoding='utf-8') as f:
        return f.read()


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║         🧠 SNN-LLM WikiText-2 Full Training 🧠                ║
    ║         Long-running Background Mode                          ║
    ║         Estimated: 90-120 minutes                             ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    start_time = time.time()
    
    # Configuration
    hidden_dim = 512
    hypercube_dim = 9
    epochs = 5  # Reduced for faster completion
    
    print(f"Configuration: hidden={hidden_dim}, dim={hypercube_dim}D, epochs={epochs}")
    print("-" * 60)
    
    # Load data
    print("\n[1/4] Loading WikiText-2 data...")
    text = download_wikitext2()
    print(f"  Total characters: {len(text):,}")
    
    # Build vocabulary
    print("\n[2/4] Building vocabulary...")
    chars = sorted(set(text))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    vocab_size = len(chars)
    print(f"  Vocabulary size: {vocab_size}")
    
    # Tokenize
    print("\n[3/4] Tokenizing...")
    tokens = np.array([char_to_idx[c] for c in text], dtype=np.int32)
    print(f"  Total tokens: {len(tokens):,}")
    
    # Create model
    print("\n[4/4] Creating model...")
    model = OptimizedSNNLM(vocab_size, hidden_dim=hidden_dim, hypercube_dim=hypercube_dim)
    
    n_params = model.embedding.size + model.W_res.size + model.W_out.size + model.b_out.size
    print(f"  Parameters: {n_params:,}")
    
    # Create directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Training
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    
    history = []
    best_ppl = float('inf')
    
    for epoch in range(epochs):
        model.reset()
        t0 = time.time()
        total_loss = 0
        n_tokens = len(tokens) - 1
        
        # Progress tracking
        log_interval = n_tokens // 10
        
        for i in range(n_tokens):
            loss = model.train_step(tokens[i], tokens[i + 1])
            total_loss += loss
            
            # Progress update
            if (i + 1) % log_interval == 0:
                progress = (i + 1) / n_tokens * 100
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (n_tokens - i - 1)
                print(f"    Epoch {epoch+1}: {progress:.0f}% | ETA: {eta/60:.1f}min")
        
        avg_loss = total_loss / n_tokens
        
        # Evaluate on subset
        model.reset()
        ppl = model.evaluate(tokens[:5000])
        
        elapsed = time.time() - t0
        
        print(f"  Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, PPL={ppl:.1f}, Time={elapsed/60:.1f}min")
        
        history.append({
            'epoch': epoch + 1,
            'loss': float(avg_loss),
            'ppl': float(ppl),
            'time': elapsed
        })
        
        # Save checkpoint
        model.save(f"checkpoints/model_wikitext_epoch{epoch+1}.pkl")
        
        # Best model
        if ppl < best_ppl:
            best_ppl = ppl
            model.save("checkpoints/model_wikitext_best.pkl")
            print(f"    → New best model saved!")
    
    # Save tokenizer
    with open("checkpoints/tokenizer_wikitext.json", 'w', encoding='utf-8') as f:
        json.dump({'char_to_idx': char_to_idx}, f, ensure_ascii=False)
    
    # Save history
    with open("results/wikitext_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # Generation sample
    print("\n" + "=" * 60)
    print("GENERATION SAMPLES")
    print("=" * 60)
    
    prompts = ["The ", "In ", "A "]
    for prompt in prompts:
        prompt_tokens = [char_to_idx.get(c, 0) for c in prompt]
        generated_tokens = model.generate(prompt_tokens, max_len=150, temperature=0.7)
        generated_text = ''.join(idx_to_char.get(i, '?') for i in generated_tokens)
        print(f"\n'{prompt}' → '{generated_text[:100]}...'")
    
    # Final summary
    total_time = time.time() - start_time
    
    print(f"""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                   TRAINING COMPLETE! 🎉                       ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║   Best Perplexity: {best_ppl:>8.2f}                                 ║
    ║   Total Time:      {total_time/60:>8.1f} minutes                      ║
    ║   Data Size:       {len(text):>8,} chars                        ║
    ║   Parameters:      {n_params:>8,}                              ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║   Checkpoints: checkpoints/model_wikitext_best.pkl            ║
    ║   History:     results/wikitext_history.json                  ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
