"""
Generation Test for Small Model
================================
Test the trained small PyTorch model's generation quality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os

print('='*60)
print('Small Model Generation Test')
print('='*60)

device = torch.device('cpu')

class HypercubeMask:
    @staticmethod
    def create(dim=9):
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
        return self.state

class SNNReservoir(nn.Module):
    def __init__(self, hidden_dim, hypercube_dim=9):
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
    
    def generate(self, start_tokens, tokenizer, max_len=100, temperature=0.8):
        self.eval()
        self.reservoir.reset(1)
        
        generated = list(start_tokens)
        
        with torch.no_grad():
            # Process start tokens
            for token in start_tokens:
                x = torch.tensor([[token]], dtype=torch.long, device=device)
                emb = self.embedding(x[:, 0])
                state = self.reservoir(emb)
            
            # Generate new tokens
            for _ in range(max_len):
                logits = self.output(state) / temperature
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs[0], 1).item()
                generated.append(next_token)
                
                x = torch.tensor([[next_token]], dtype=torch.long, device=device)
                emb = self.embedding(x[:, 0])
                state = self.reservoir(emb)
        
        return generated


# Load tokenizer
print("\n[1/3] Loading tokenizer...")
with open('checkpoints/tokenizer_small.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
token_to_idx = data['token_to_idx']
idx_to_token = {int(v): k for k, v in token_to_idx.items()}
vocab_size = len(token_to_idx)
print(f"  Vocab size: {vocab_size}")

# Load model
print("\n[2/3] Loading model...")
model = PyTorchSNNLM(vocab_size=vocab_size, hidden_dim=512, hypercube_dim=9)
model.load_state_dict(torch.load('checkpoints/model_small_best.pt', map_location=device))
model.eval()
print("  Model loaded!")

# Encode function
def encode(text):
    return [token_to_idx.get(c, 1) for c in text]

def decode(indices):
    return ''.join(idx_to_token.get(i, '?') for i in indices)

# Generate samples
print("\n[3/3] Generating samples...")
print("="*60)

prompts = [
    "脳は",
    "人工知能",
    "言語",
    "スパイキング",
    "11次元",
]

for prompt in prompts:
    print(f"\nPrompt: 「{prompt}」")
    start_tokens = encode(prompt)
    
    for temp in [0.5, 0.8, 1.0]:
        generated = model.generate(start_tokens, None, max_len=50, temperature=temp)
        text = decode(generated)
        print(f"  T={temp}: {text}")

print("\n" + "="*60)
print("Generation test complete!")
print("="*60)
