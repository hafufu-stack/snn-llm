"""
Simple SNN-LLM Training (Sequential, Fast Demo)
================================================

A simpler, faster training script for quick testing.
Uses smaller data and sequential processing.

Usage: python experiments/train_simple.py

Author: ろーる
Date: 2026-01-21
"""

import numpy as np
import os
import sys
import time
import json
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hypercube import create_hypercube_mask


class SimpleSNNLM:
    """Simple SNN Language Model for fast training"""
    
    def __init__(self, vocab_size, hidden_dim=256, hypercube_dim=8, seed=42):
        np.random.seed(seed)
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        # Embedding
        self.embedding = np.random.randn(vocab_size, hidden_dim) * 0.1
        
        # Reservoir with hypercube topology
        mask = create_hypercube_mask(hypercube_dim)
        if mask.shape[0] != hidden_dim:
            # Resize
            new_mask = np.zeros((hidden_dim, hidden_dim))
            for i in range(hidden_dim):
                for j in range(hidden_dim):
                    new_mask[i, j] = mask[i % mask.shape[0], j % mask.shape[0]]
            mask = new_mask
        
        W_res = np.random.randn(hidden_dim, hidden_dim) * 0.3
        self.W_res = W_res * mask
        
        # Scale spectral radius
        try:
            eig = np.linalg.eigvals(self.W_res)
            self.W_res *= 1.2 / (np.max(np.abs(eig)) + 0.01)
        except:
            pass
        
        # Output
        self.W_out = np.random.randn(vocab_size, hidden_dim) * 0.01
        self.b_out = np.zeros(vocab_size)
        
        # State
        self.state = np.zeros(hidden_dim)
        
        # Learning
        self.lr = 0.02
    
    def forward(self, token_idx):
        """Single forward step"""
        emb = self.embedding[token_idx]
        h_rec = np.tanh(self.W_res @ self.state)
        self.state = 0.7 * self.state + 0.3 * (emb + h_rec)
        
        # Spike threshold
        spikes = (self.state > 0.5).astype(float)
        self.state *= (1 - spikes * 0.3)
        
        logits = self.W_out @ self.state + self.b_out
        return logits
    
    def train_step(self, input_idx, target_idx):
        """Train on single sample"""
        logits = self.forward(input_idx)
        
        # Softmax
        exp_l = np.exp(logits - np.max(logits))
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
        self.state = np.zeros(self.hidden_dim)
    
    def generate(self, start_indices, max_len=50, temperature=0.8):
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
    
    def evaluate(self, indices):
        """Calculate perplexity"""
        self.reset()
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


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║         🧠 SNN-LLM Simple Training 🧠                         ║
    ║         Fast Demo Mode                                        ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Sample text
    text = """
    The brain is the most complex organ in the human body. It controls thought, 
    memory, emotion, touch, motor skills, vision, breathing, temperature, hunger 
    and every process that regulates our body. The brain consists of about 86 
    billion neurons. These neurons communicate through synapses, forming complex 
    networks. Spiking Neural Networks attempt to mimic this biological efficiency.
    Unlike traditional artificial neural networks, SNNs use temporal coding to 
    represent information. This allows for massive information capacity increases 
    while reducing energy consumption compared to von Neumann architectures.
    The future of computing lies in brain-inspired systems that can learn and 
    adapt like the human brain. Language models are a key application area where
    energy efficiency matters. Running large language models requires significant
    computational resources. However, a brain-like approach could enable local
    processing on edge devices with minimal power consumption.
    """ * 20  # Repeat for more data
    
    print(f"Training data: {len(text):,} characters")
    
    # Build vocabulary
    chars = sorted(set(text))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    vocab_size = len(chars)
    
    print(f"Vocabulary size: {vocab_size}")
    
    # Tokenize
    tokens = [char_to_idx[c] for c in text]
    
    # Create model
    hidden_dim = 256
    model = SimpleSNNLM(vocab_size, hidden_dim=hidden_dim, hypercube_dim=8)
    
    n_params = model.embedding.size + model.W_res.size + model.W_out.size + model.b_out.size
    print(f"Parameters: {n_params:,}")
    print("-" * 60)
    
    # Training
    epochs = 10
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    history = []
    
    for epoch in range(epochs):
        model.reset()
        t0 = time.time()
        total_loss = 0
        
        for i in range(len(tokens) - 1):
            loss = model.train_step(tokens[i], tokens[i + 1])
            total_loss += loss
        
        avg_loss = total_loss / (len(tokens) - 1)
        
        # Evaluate
        model.reset()
        ppl = model.evaluate(tokens[:500])
        
        elapsed = time.time() - t0
        
        print(f"  Epoch {epoch+1:>2}/{epochs}: Loss={avg_loss:.4f}, PPL={ppl:.1f}, Time={elapsed:.1f}s")
        
        history.append({'epoch': epoch+1, 'loss': avg_loss, 'ppl': ppl, 'time': elapsed})
    
    # Save
    model.save("checkpoints/model_simple.pkl")
    
    with open("results/simple_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # Generate sample
    print("\n" + "=" * 60)
    print("GENERATION")
    print("=" * 60)
    
    prompt = "The brain "
    prompt_tokens = [char_to_idx.get(c, 0) for c in prompt]
    generated_tokens = model.generate(prompt_tokens, max_len=100, temperature=0.7)
    generated_text = ''.join(idx_to_char.get(i, '?') for i in generated_tokens)
    
    print(f"\nPrompt: '{prompt}'")
    print(f"Generated: '{generated_text[:150]}...'")
    
    print(f"""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                   Training Complete! 🎉                       ║
    ║   Model saved: checkpoints/model_simple.pkl                   ║
    ║   History: results/simple_history.json                        ║
    ╚═══════════════════════════════════════════════════════════════╝
    
    Final Perplexity: {history[-1]['ppl']:.2f}
    Total Time: {sum(h['time'] for h in history):.1f}s
    """)


if __name__ == "__main__":
    main()
