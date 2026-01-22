"""
Temporal Coding Comparison Experiment
=====================================
Compare: Rate Coding vs Temporal Coding for SNN-LLM
"""

import numpy as np
import time

print('='*60)
print('Temporal vs Rate Coding Comparison')
print('='*60)

def create_hypercube_mask(dim):
    n = 2 ** dim
    mask = np.zeros((n, n))
    for i in range(n):
        for bit in range(dim):
            j = i ^ (1 << bit)
            mask[i, j] = 1.0
    return mask


class RateCodingSNN:
    """Rate Coding: Information in spike frequency"""
    
    def __init__(self, vocab_size, hidden_dim, hypercube_dim):
        np.random.seed(42)
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        self.embedding = np.random.randn(vocab_size, hidden_dim).astype(np.float32) * 0.1
        mask = create_hypercube_mask(hypercube_dim)
        if mask.shape[0] != hidden_dim:
            new_mask = np.zeros((hidden_dim, hidden_dim), dtype=np.float32)
            for i in range(hidden_dim):
                for j in range(hidden_dim):
                    new_mask[i, j] = mask[i % mask.shape[0], j % mask.shape[0]]
            mask = new_mask
        W = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.3
        self.W_res = W * mask
        self.W_out = np.random.randn(vocab_size, hidden_dim).astype(np.float32) * 0.01
        self.b_out = np.zeros(vocab_size, dtype=np.float32)
        self.state = np.zeros(hidden_dim, dtype=np.float32)
        self.lr = 0.02
        self.spike_count = 0
        self.total_steps = 0
    
    def train_step(self, inp, tgt):
        emb = self.embedding[inp]
        h = np.tanh(self.W_res @ self.state)
        self.state = 0.7 * self.state + 0.3 * (emb + h)
        
        # Rate coding: use firing rate (continuous)
        rate = np.clip(self.state, 0, 1)
        self.spike_count += np.sum(rate > 0.5)
        self.total_steps += 1
        
        logits = self.W_out @ rate + self.b_out
        exp_l = np.exp(logits - np.max(logits))
        probs = exp_l / (np.sum(exp_l) + 1e-10)
        loss = -np.log(probs[tgt] + 1e-10)
        grad = probs.copy()
        grad[tgt] -= 1
        self.W_out -= self.lr * np.outer(grad, rate)
        self.b_out -= self.lr * grad
        return loss
    
    def reset(self):
        self.state = np.zeros(self.hidden_dim, dtype=np.float32)


class TemporalCodingSNN:
    """Temporal Coding: Information in spike timing"""
    
    def __init__(self, vocab_size, hidden_dim, hypercube_dim):
        np.random.seed(42)
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        self.embedding = np.random.randn(vocab_size, hidden_dim).astype(np.float32) * 0.1
        mask = create_hypercube_mask(hypercube_dim)
        if mask.shape[0] != hidden_dim:
            new_mask = np.zeros((hidden_dim, hidden_dim), dtype=np.float32)
            for i in range(hidden_dim):
                for j in range(hidden_dim):
                    new_mask[i, j] = mask[i % mask.shape[0], j % mask.shape[0]]
            mask = new_mask
        W = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.3
        self.W_res = W * mask
        self.W_out = np.random.randn(vocab_size, hidden_dim).astype(np.float32) * 0.01
        self.b_out = np.zeros(vocab_size, dtype=np.float32)
        self.membrane = np.zeros(hidden_dim, dtype=np.float32)
        self.spike_times = np.zeros(hidden_dim, dtype=np.float32)
        self.time_step = 0
        self.lr = 0.02
        self.spike_count = 0
        self.total_steps = 0
        self.threshold = 0.5
    
    def train_step(self, inp, tgt):
        self.time_step += 1
        emb = self.embedding[inp]
        h = np.tanh(self.W_res @ self.membrane)
        self.membrane = 0.7 * self.membrane + 0.3 * (emb + h)
        
        # Temporal coding: spike when crossing threshold, record timing
        spikes = (self.membrane > self.threshold).astype(np.float32)
        self.spike_count += np.sum(spikes)
        self.total_steps += 1
        
        # Update spike times
        newly_spiked = (spikes > 0) & (self.spike_times == 0)
        self.spike_times = np.where(newly_spiked, self.time_step, self.spike_times)
        
        # Use both spike presence and timing for output
        timing_weight = 1.0 / (1.0 + 0.1 * (self.time_step - self.spike_times))
        combined = spikes * timing_weight + 0.5 * self.membrane
        
        # Reset after spike
        self.membrane = self.membrane * (1 - spikes * 0.3)
        
        logits = self.W_out @ combined + self.b_out
        exp_l = np.exp(logits - np.max(logits))
        probs = exp_l / (np.sum(exp_l) + 1e-10)
        loss = -np.log(probs[tgt] + 1e-10)
        grad = probs.copy()
        grad[tgt] -= 1
        self.W_out -= self.lr * np.outer(grad, combined)
        self.b_out -= self.lr * grad
        return loss
    
    def reset(self):
        self.membrane = np.zeros(self.hidden_dim, dtype=np.float32)
        self.spike_times = np.zeros(self.hidden_dim, dtype=np.float32)
        self.time_step = 0


# Test corpus
text = '''脳は人間の身体の中で最も複雑な器官です。思考、記憶、感情、運動制御など、すべての認知機能を担っています。
ニューロンは電気信号を通じて情報を伝達します。スパイキングニューラルネットワークは、この生物学的効率性を模倣しようとしています。''' * 100

chars = sorted(set(text))
c2i = {c: i for i, c in enumerate(chars)}
tokens = [c2i[c] for c in text]
vocab = len(chars)

print(f"Corpus: {len(text)} chars, {vocab} unique")

# Test Rate Coding
print("\n[1/2] Rate Coding SNN...")
rate_model = RateCodingSNN(vocab, 512, 9)
t0 = time.time()
for epoch in range(10):
    rate_model.reset()
    total_loss = 0
    for i in range(min(5000, len(tokens)-1)):
        loss = rate_model.train_step(tokens[i], tokens[i+1])
        total_loss += loss
    ppl = np.exp(total_loss / min(5000, len(tokens)-1))
    print(f"  Epoch {epoch+1}: PPL={ppl:.2f}")

rate_ppl = ppl
rate_sparsity = 1 - (rate_model.spike_count / (rate_model.total_steps * 512))
rate_time = time.time() - t0

# Test Temporal Coding
print("\n[2/2] Temporal Coding SNN...")
temp_model = TemporalCodingSNN(vocab, 512, 9)
t0 = time.time()
for epoch in range(10):
    temp_model.reset()
    total_loss = 0
    for i in range(min(5000, len(tokens)-1)):
        loss = temp_model.train_step(tokens[i], tokens[i+1])
        total_loss += loss
    ppl = np.exp(total_loss / min(5000, len(tokens)-1))
    print(f"  Epoch {epoch+1}: PPL={ppl:.2f}")

temp_ppl = ppl
temp_sparsity = 1 - (temp_model.spike_count / (temp_model.total_steps * 512))
temp_time = time.time() - t0

# Results
print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"\n{'Metric':<20} {'Rate Coding':<15} {'Temporal Coding':<15}")
print("-"*50)
print(f"{'PPL':<20} {rate_ppl:<15.2f} {temp_ppl:<15.2f}")
print(f"{'Sparsity':<20} {rate_sparsity*100:<14.1f}% {temp_sparsity*100:<14.1f}%")
print(f"{'Time':<20} {rate_time:<14.1f}s {temp_time:<14.1f}s")

if temp_ppl < rate_ppl:
    diff = (rate_ppl - temp_ppl) / rate_ppl * 100
    print(f"\n✅ Temporal Coding is {diff:.1f}% better!")
else:
    diff = (temp_ppl - rate_ppl) / temp_ppl * 100
    print(f"\n⚠️ Rate Coding is {diff:.1f}% better")

print("""
Implications:
- Temporal coding uses spike timing for richer information
- Rate coding is simpler but may lose temporal information
- Brain uses both methods depending on context
""")
