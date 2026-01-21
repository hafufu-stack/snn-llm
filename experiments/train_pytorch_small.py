"""
PyTorch SNN-LLM - Small Version (Opus Corpus Only)
===================================================
Parallel training with smaller corpus for faster results
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import json
from collections import Counter

device = torch.device('cpu')
print(f"Using device: {device}")


class HypercubeMask:
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
        self.state = self.tau * self.state + (1 - self.tau) * x
        spikes = (self.state > self.threshold).float()
        self.state = self.state * (1 - spikes * 0.3)
        return self.state


class SNNReservoir(nn.Module):
    def __init__(self, hidden_dim, hypercube_dim=10):
        super().__init__()
        self.hidden_dim = hidden_dim
        mask = HypercubeMask.create(hypercube_dim)
        orig_size = mask.shape[0]
        if orig_size != hidden_dim:
            new_mask = np.zeros((hidden_dim, hidden_dim), dtype=np.float32)
            for i in range(hidden_dim):
                for j in range(hidden_dim):
                    new_mask[i, j] = mask[i % orig_size, j % orig_size]
            mask = new_mask
        self.register_buffer('mask', torch.tensor(mask))
        self.W_res = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.2)
        self.lif = LIFNeuronLayer(hidden_dim)
    
    def forward(self, x):
        W_masked = self.W_res * self.mask
        h_rec = torch.tanh(F.linear(self.lif.state, W_masked))
        combined = x + h_rec
        return self.lif(combined)
    
    def reset(self, batch_size=1):
        self.lif.reset(batch_size)


class PyTorchSNNLM(nn.Module):
    def __init__(self, vocab_size, hidden_dim=512, hypercube_dim=9):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.reservoir = SNNReservoir(hidden_dim, hypercube_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.output.weight)
    
    def forward(self, x):
        batch_size, seq_len = x.shape
        self.reservoir.reset(batch_size)
        outputs = []
        for t in range(seq_len):
            emb = self.embedding(x[:, t])
            state = self.reservoir(emb)
            logits = self.output(state)
            outputs.append(logits)
        return torch.stack(outputs, dim=1)


class JapaneseTokenizer:
    def __init__(self, vocab_size=2000):
        self.vocab_size = vocab_size
        self.token_to_idx = {}
        self.idx_to_token = {}
    
    def build_vocab(self, text, min_freq=2):
        char_counts = Counter(text)
        self.token_to_idx = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
        idx = len(self.token_to_idx)
        for char, count in char_counts.most_common():
            if count >= min_freq and char not in self.token_to_idx:
                self.token_to_idx[char] = idx
                idx += 1
                if idx >= self.vocab_size:
                    break
        self.idx_to_token = {i: t for t, i in self.token_to_idx.items()}
        print(f"  Vocabulary size: {len(self.token_to_idx)}")
    
    def encode(self, text):
        return [self.token_to_idx.get(c, 1) for c in text]
    
    def decode(self, indices):
        return ''.join(self.idx_to_token.get(i, '?') for i in indices)
    
    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'token_to_idx': self.token_to_idx}, f, ensure_ascii=False)


def load_opus_corpus():
    """Load only Opus-generated corpus (small, clean)"""
    text = ""
    opus_files = ['opus_corpus.txt', 'opus_corpus_part2.txt', 'opus_corpus_part3.txt']
    for filename in opus_files:
        filepath = os.path.join("data", filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"  Loaded {filename}: {len(content):,} chars")
                text += content + "\n"
    return text


def main():
    print(f"""
    ╔═══════════════════════════════════════════════════════════════╗
    ║      🔥 PyTorch SNN-LLM (Small/Fast Version) 🔥               ║
    ║      Opus Corpus Only | 100 Epochs | ~2 hours                 ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    start_time = time.time()
    
    config = {
        'hidden_dim': 512,
        'hypercube_dim': 9,
        'vocab_size': 1000,
        'epochs': 100,
        'lr': 0.002,
        'batch_size': 16,
        'seq_len': 32,
    }
    
    print("Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    print("\n[1/4] Loading Opus corpus...")
    text = load_opus_corpus()
    print(f"  Total: {len(text):,} characters")
    
    print("\n[2/4] Building tokenizer...")
    tokenizer = JapaneseTokenizer(vocab_size=config['vocab_size'])
    tokenizer.build_vocab(text, min_freq=1)
    tokens = tokenizer.encode(text)
    print(f"  Total tokens: {len(tokens):,}")
    
    print("\n[3/4] Creating sequences...")
    seq_len = config['seq_len']
    sequences = []
    for i in range(0, len(tokens) - seq_len - 1, seq_len // 2):
        input_seq = tokens[i:i + seq_len]
        target_seq = tokens[i + 1:i + seq_len + 1]
        sequences.append((input_seq, target_seq))
    print(f"  Total sequences: {len(sequences):,}")
    
    print("\n[4/4] Creating model...")
    model = PyTorchSNNLM(
        vocab_size=len(tokenizer.token_to_idx),
        hidden_dim=config['hidden_dim'],
        hypercube_dim=config['hypercube_dim']
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
    criterion = nn.CrossEntropyLoss()
    
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    print("\n" + "=" * 60)
    print("TRAINING (Small Version)")
    print("=" * 60)
    
    history = []
    best_loss = float('inf')
    
    for epoch in range(config['epochs']):
        model.train()
        t0 = time.time()
        total_loss = 0
        n_batches = 0
        
        np.random.shuffle(sequences)
        
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
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:>3}/{config['epochs']}: Loss={avg_loss:.4f}, PPL={ppl:.2f}, Time={elapsed:.1f}s")
        
        history.append({'epoch': epoch + 1, 'loss': avg_loss, 'ppl': ppl})
        
        if (epoch + 1) % 20 == 0:
            torch.save(model.state_dict(), f"checkpoints/model_small_epoch{epoch+1}.pt")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "checkpoints/model_small_best.pt")
    
    torch.save(model.state_dict(), "checkpoints/model_small_final.pt")
    tokenizer.save("checkpoints/tokenizer_small.json")
    
    with open("results/small_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    total_time = time.time() - start_time
    best_ppl = np.exp(best_loss)
    
    print(f"""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                   訓練完了！🎉 (Small)                        ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║   Best Loss:       {best_loss:>8.4f}                                ║
    ║   Best PPL:        {best_ppl:>8.2f}                                 ║
    ║   Total Time:      {total_time/60:>8.1f} min                         ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║   Model: checkpoints/model_small_best.pt                      ║
    ║   History: results/small_history.json                         ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
