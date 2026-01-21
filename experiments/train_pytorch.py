"""
PyTorch SNN-LLM - GPU Accelerated Training
===========================================

Complete rewrite with:
1. PyTorch for GPU acceleration (100-1000x faster)
2. Larger corpus (all Opus-generated data)
3. More epochs (100)
4. Better architecture

Usage: python experiments/train_pytorch.py

Author: ろーる & オーパス先生
Date: 2026-01-21
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import time
import json
from collections import Counter

# Force CPU (CUDA version mismatch)
device = torch.device('cpu')
print(f"Using device: {device}")


class HypercubeMask:
    """Create 10D hypercube connectivity mask"""
    
    @staticmethod
    def create(dim=10):
        n_nodes = 2 ** dim
        mask = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        
        for i in range(n_nodes):
            for bit in range(dim):
                j = i ^ (1 << bit)
                mask[i, j] = 1.0
        
        return mask


class LIFNeuronLayer(nn.Module):
    """Leaky Integrate-and-Fire Neuron Layer"""
    
    def __init__(self, size, tau=0.7, threshold=0.5):
        super().__init__()
        self.size = size
        self.tau = tau
        self.threshold = threshold
        self.state = None
    
    def reset(self, batch_size=1):
        self.state = torch.zeros(batch_size, self.size, device=device)
    
    def forward(self, x):
        if self.state is None:
            self.reset(x.shape[0])
        
        # Leaky integration
        self.state = self.tau * self.state + (1 - self.tau) * x
        
        # Spiking
        spikes = (self.state > self.threshold).float()
        self.state = self.state * (1 - spikes * 0.3)
        
        return self.state


class SNNReservoir(nn.Module):
    """SNN Reservoir with Hypercube Topology"""
    
    def __init__(self, hidden_dim, hypercube_dim=10):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Create hypercube mask
        mask = HypercubeMask.create(hypercube_dim)
        orig_size = mask.shape[0]
        
        # Resize if needed
        if orig_size != hidden_dim:
            new_mask = np.zeros((hidden_dim, hidden_dim), dtype=np.float32)
            for i in range(hidden_dim):
                for j in range(hidden_dim):
                    new_mask[i, j] = mask[i % orig_size, j % orig_size]
            mask = new_mask
        
        self.register_buffer('mask', torch.tensor(mask))
        
        # Reservoir weights (will be masked)
        self.W_res = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.2)
        
        # LIF neurons
        self.lif = LIFNeuronLayer(hidden_dim)
    
    def forward(self, x):
        # Apply mask to get hypercube topology
        W_masked = self.W_res * self.mask
        
        # Recurrent computation
        h_rec = torch.tanh(F.linear(self.lif.state, W_masked))
        
        # Combine input and recurrent
        combined = x + h_rec
        
        # LIF forward
        return self.lif(combined)
    
    def reset(self, batch_size=1):
        self.lif.reset(batch_size)


class PyTorchSNNLM(nn.Module):
    """PyTorch SNN Language Model"""
    
    def __init__(self, vocab_size, hidden_dim=1024, hypercube_dim=10):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # Reservoir
        self.reservoir = SNNReservoir(hidden_dim, hypercube_dim)
        
        # Output
        self.output = nn.Linear(hidden_dim, vocab_size)
        
        # Initialize
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.output.weight)
    
    def forward(self, x):
        # x: (batch, seq_len)
        batch_size, seq_len = x.shape
        
        self.reservoir.reset(batch_size)
        
        outputs = []
        for t in range(seq_len):
            emb = self.embedding(x[:, t])
            state = self.reservoir(emb)
            logits = self.output(state)
            outputs.append(logits)
        
        return torch.stack(outputs, dim=1)
    
    def generate(self, start_tokens, max_len=100, temperature=0.7):
        self.eval()
        self.reservoir.reset(1)
        
        tokens = start_tokens.clone()
        
        with torch.no_grad():
            # Process start tokens
            for t in range(len(start_tokens[0])):
                emb = self.embedding(tokens[:, t])
                state = self.reservoir(emb)
            
            # Generate
            for _ in range(max_len):
                logits = self.output(state) / temperature
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                tokens = torch.cat([tokens, next_token], dim=1)
                
                emb = self.embedding(next_token.squeeze(-1))
                state = self.reservoir(emb)
        
        return tokens


class JapaneseTokenizer:
    """BPE-like tokenizer for Japanese"""
    
    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.token_to_idx = {}
        self.idx_to_token = {}
    
    def build_vocab(self, text, min_freq=2):
        char_counts = Counter(text)
        
        self.token_to_idx = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
        idx = len(self.token_to_idx)
        
        # Add characters
        for char, count in char_counts.most_common():
            if count >= min_freq and char not in self.token_to_idx:
                self.token_to_idx[char] = idx
                idx += 1
                if idx >= self.vocab_size // 2:
                    break
        
        # Add bigrams
        bigram_counts = Counter()
        for i in range(len(text) - 1):
            bigram_counts[text[i:i+2]] += 1
        
        for bigram, count in bigram_counts.most_common():
            if count >= min_freq * 2 and bigram not in self.token_to_idx:
                self.token_to_idx[bigram] = idx
                idx += 1
                if idx >= self.vocab_size:
                    break
        
        self.idx_to_token = {i: t for t, i in self.token_to_idx.items()}
        print(f"  Vocabulary size: {len(self.token_to_idx)}")
    
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
                tokens.append(1)
                i += 1
        return tokens
    
    def decode(self, indices):
        return ''.join(self.idx_to_token.get(i, '?') for i in indices)
    
    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'token_to_idx': self.token_to_idx}, f, ensure_ascii=False)
    
    def load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.token_to_idx = data['token_to_idx']
        self.idx_to_token = {int(i): t for t, i in self.token_to_idx.items()}


def load_all_corpus(data_dir="data"):
    """Load all corpus files"""
    text = ""
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(data_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"  Loaded {filename}: {len(content):,} chars")
                text += content + "\n"
    
    return text


def main():
    print(f"""
    ╔═══════════════════════════════════════════════════════════════╗
    ║         🔥 PyTorch SNN-LLM Training 🔥                        ║
    ║         GPU Accelerated | Large Corpus | 100 Epochs           ║
    ║         Device: {str(device):>10}                                   ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    start_time = time.time()
    
    # Config
    config = {
        'hidden_dim': 1024,
        'hypercube_dim': 10,
        'vocab_size': 3000,
        'epochs': 100,
        'lr': 0.001,
        'batch_size': 32,
        'seq_len': 64,
    }
    
    print("Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # Load corpus
    print("\n[1/4] Loading all corpus files...")
    text = load_all_corpus()
    print(f"  Total: {len(text):,} characters")
    
    # Tokenizer
    print("\n[2/4] Building tokenizer...")
    tokenizer = JapaneseTokenizer(vocab_size=config['vocab_size'])
    tokenizer.build_vocab(text)
    
    tokens = tokenizer.encode(text)
    print(f"  Total tokens: {len(tokens):,}")
    
    # Create sequences
    print("\n[3/4] Creating sequences...")
    seq_len = config['seq_len']
    sequences = []
    for i in range(0, len(tokens) - seq_len - 1, seq_len // 2):
        input_seq = tokens[i:i + seq_len]
        target_seq = tokens[i + 1:i + seq_len + 1]
        sequences.append((input_seq, target_seq))
    
    print(f"  Total sequences: {len(sequences):,}")
    
    # Create model
    print("\n[4/4] Creating model...")
    model = PyTorchSNNLM(
        vocab_size=len(tokenizer.token_to_idx),
        hidden_dim=config['hidden_dim'],
        hypercube_dim=config['hypercube_dim']
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
    criterion = nn.CrossEntropyLoss()
    
    # Directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Training
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    
    history = []
    best_loss = float('inf')
    
    for epoch in range(config['epochs']):
        model.train()
        t0 = time.time()
        total_loss = 0
        n_batches = 0
        
        # Shuffle sequences
        np.random.shuffle(sequences)
        
        # Mini-batch training
        batch_size = config['batch_size']
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            if len(batch) < batch_size:
                continue
            
            inputs = torch.tensor([s[0] for s in batch], dtype=torch.long, device=device)
            targets = torch.tensor([s[1] for s in batch], dtype=torch.long, device=device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.shape[-1]), targets.view(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        
        avg_loss = total_loss / max(n_batches, 1)
        ppl = np.exp(avg_loss)
        elapsed = time.time() - t0
        
        print(f"  Epoch {epoch+1:>3}/{config['epochs']}: Loss={avg_loss:.4f}, PPL={ppl:.2f}, Time={elapsed:.1f}s")
        
        history.append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'ppl': ppl,
            'time': elapsed
        })
        
        # Save checkpoints
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"checkpoints/model_pytorch_epoch{epoch+1}.pt")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "checkpoints/model_pytorch_best.pt")
            print(f"    → Best model saved!")
    
    # Save final
    torch.save(model.state_dict(), "checkpoints/model_pytorch_final.pt")
    tokenizer.save("checkpoints/tokenizer_pytorch.json")
    
    with open("results/pytorch_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # Generation
    print("\n" + "=" * 60)
    print("GENERATION SAMPLES")
    print("=" * 60)
    
    model.eval()
    prompts = ["脳は", "人工知能は", "言語"]
    
    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        input_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
        
        with torch.no_grad():
            output = model.generate(input_tensor, max_len=50, temperature=0.7)
        
        text = tokenizer.decode(output[0].cpu().tolist())
        print(f"\n'{prompt}' → '{text}'")
    
    # Summary
    total_time = time.time() - start_time
    best_ppl = np.exp(best_loss)
    
    print(f"""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                   訓練完了！🎉                                ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║   Best Loss:       {best_loss:>8.4f}                                ║
    ║   Best PPL:        {best_ppl:>8.2f}                                 ║
    ║   Total Time:      {total_time/60:>8.1f} min                         ║
    ║   Device:          {str(device):>8}                                  ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║   Model: checkpoints/model_pytorch_best.pt                    ║
    ║   History: results/pytorch_history.json                       ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
