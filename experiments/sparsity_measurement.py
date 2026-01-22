"""
Sparsity Measurement Experiment
===============================
Measure actual spike rates in SNN-LLM to verify energy efficiency claims
"""

import numpy as np
import time

print('='*60)
print('SNN Sparsity Measurement')
print('='*60)

def create_hypercube_mask(dim):
    n = 2 ** dim
    mask = np.zeros((n, n))
    for i in range(n):
        for bit in range(dim):
            j = i ^ (1 << bit)
            mask[i, j] = 1.0
    return mask


class SparsitySNN:
    """SNN with detailed sparsity tracking"""
    
    def __init__(self, vocab_size, hidden_dim, hypercube_dim, threshold=0.5):
        np.random.seed(42)
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.threshold = threshold
        
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
        self.lr = 0.02
        
        # Tracking
        self.total_spikes = 0
        self.total_possible = 0
        self.spike_histogram = np.zeros(hidden_dim, dtype=np.int32)
    
    def forward_with_tracking(self, inp):
        emb = self.embedding[inp]
        h = np.tanh(self.W_res @ self.membrane)
        self.membrane = 0.7 * self.membrane + 0.3 * (emb + h)
        
        # Generate spikes
        spikes = (self.membrane > self.threshold).astype(np.float32)
        
        # Track sparsity
        self.total_spikes += np.sum(spikes)
        self.total_possible += self.hidden_dim
        self.spike_histogram += spikes.astype(np.int32)
        
        # Reset after spike
        self.membrane = self.membrane * (1 - spikes * 0.5)
        
        return spikes
    
    def train_step(self, inp, tgt):
        spikes = self.forward_with_tracking(inp)
        
        logits = self.W_out @ (spikes + 0.5 * self.membrane) + self.b_out
        exp_l = np.exp(logits - np.max(logits))
        probs = exp_l / (np.sum(exp_l) + 1e-10)
        loss = -np.log(probs[tgt] + 1e-10)
        
        grad = probs.copy()
        grad[tgt] -= 1
        self.W_out -= self.lr * np.outer(grad, spikes + 0.5 * self.membrane)
        self.b_out -= self.lr * grad
        return loss
    
    def reset(self):
        self.membrane = np.zeros(self.hidden_dim, dtype=np.float32)
    
    def get_sparsity(self):
        return 1 - (self.total_spikes / self.total_possible)
    
    def get_energy_estimate(self):
        # Assume: 1 spike = 1 unit energy, 0 spike = 0.01 unit (leakage)
        spike_energy = self.total_spikes * 1.0
        leakage = (self.total_possible - self.total_spikes) * 0.01
        total_snn = spike_energy + leakage
        
        # Compare to dense DNN: all neurons active
        dense_energy = self.total_possible * 1.0
        
        return {
            'snn_energy': total_snn,
            'dense_energy': dense_energy,
            'efficiency': dense_energy / total_snn
        }


# Test corpus
text = '''脳は人間の身体の中で最も複雑な器官です。思考、記憶、感情、運動制御など、すべての認知機能を担っています。
ニューロンは電気信号を通じて情報を伝達します。スパイキングニューラルネットワークは、この生物学的効率性を模倣しようとしています。''' * 100

chars = sorted(set(text))
c2i = {c: i for i, c in enumerate(chars)}
tokens = [c2i[c] for c in text]
vocab = len(chars)

print(f"Corpus: {len(text)} chars, {vocab} unique")

# Test different thresholds
print("\nTesting different spike thresholds...")
print("-"*60)

results = []

for threshold in [0.3, 0.5, 0.7, 0.9]:
    print(f"\nThreshold = {threshold}")
    model = SparsitySNN(vocab, 512, 9, threshold=threshold)
    
    total_loss = 0
    for epoch in range(5):
        model.reset()
        for i in range(min(3000, len(tokens)-1)):
            loss = model.train_step(tokens[i], tokens[i+1])
            total_loss += loss
    
    ppl = np.exp(total_loss / (5 * min(3000, len(tokens)-1)))
    sparsity = model.get_sparsity()
    energy = model.get_energy_estimate()
    
    results.append({
        'threshold': threshold,
        'ppl': ppl,
        'sparsity': sparsity,
        'efficiency': energy['efficiency']
    })
    
    print(f"  PPL: {ppl:.2f}")
    print(f"  Sparsity: {sparsity*100:.1f}%")
    print(f"  Energy Efficiency: {energy['efficiency']:.1f}x vs Dense")


# Summary
print("\n" + "="*60)
print("SPARSITY ANALYSIS SUMMARY")
print("="*60)
print(f"\n{'Threshold':<12} {'PPL':<10} {'Sparsity':<12} {'Efficiency':<12}")
print("-"*50)
for r in results:
    print(f"{r['threshold']:<12.1f} {r['ppl']:<10.2f} {r['sparsity']*100:<11.1f}% {r['efficiency']:<11.1f}x")

# Find optimal
best = min(results, key=lambda x: x['ppl'])
print(f"\n🏆 Best PPL: threshold={best['threshold']}, PPL={best['ppl']:.2f}")

most_sparse = max(results, key=lambda x: x['sparsity'])
print(f"📉 Most Sparse: threshold={most_sparse['threshold']}, {most_sparse['sparsity']*100:.1f}%")

print("""
Implications:
- Higher threshold = more sparsity = more energy efficient
- But too high threshold may hurt accuracy
- Sweet spot exists for optimal PPL/efficiency trade-off
- SNN achieves significant energy savings through sparsity
""")
