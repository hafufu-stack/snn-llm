"""
Noise Resilience Test
=====================
Test SNN-LLM's robustness against weight noise
"""

import numpy as np
import time

print('='*60)
print('SNN-LLM Noise Resilience Experiment')
print('='*60)

def create_hypercube_mask(dim):
    n = 2 ** dim
    mask = np.zeros((n, n))
    for i in range(n):
        for bit in range(dim):
            j = i ^ (1 << bit)
            mask[i, j] = 1.0
    return mask

class SNNModel:
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
        
        # Store original weights for noise test
        self.W_res_orig = self.W_res.copy()
        self.W_out_orig = self.W_out.copy()
    
    def train_step(self, inp, tgt):
        emb = self.embedding[inp]
        h = np.tanh(self.W_res @ self.state)
        self.state = 0.7 * self.state + 0.3 * (emb + h)
        logits = self.W_out @ self.state + self.b_out
        exp_l = np.exp(logits - np.max(logits))
        probs = exp_l / (np.sum(exp_l) + 1e-10)
        loss = -np.log(probs[tgt] + 1e-10)
        grad = probs.copy()
        grad[tgt] -= 1
        self.W_out -= self.lr * np.outer(grad, self.state)
        self.b_out -= self.lr * grad
        return loss
    
    def add_noise(self, noise_level):
        """Add Gaussian noise to weights"""
        noise_res = np.random.randn(*self.W_res.shape) * noise_level
        noise_out = np.random.randn(*self.W_out.shape) * noise_level
        self.W_res = self.W_res_orig + noise_res
        self.W_out = self.W_out_orig + noise_out
    
    def reset_weights(self):
        """Reset to original weights"""
        self.W_res = self.W_res_orig.copy()
        self.W_out = self.W_out_orig.copy()
    
    def evaluate(self, tokens, n_steps=1000):
        """Evaluate perplexity without training"""
        self.state = np.zeros(self.hidden_dim, dtype=np.float32)
        total_loss = 0
        for i in range(min(n_steps, len(tokens)-1)):
            emb = self.embedding[tokens[i]]
            h = np.tanh(self.W_res @ self.state)
            self.state = 0.7 * self.state + 0.3 * (emb + h)
            logits = self.W_out @ self.state + self.b_out
            exp_l = np.exp(logits - np.max(logits))
            probs = exp_l / (np.sum(exp_l) + 1e-10)
            loss = -np.log(probs[tokens[i+1]] + 1e-10)
            total_loss += loss
        return np.exp(total_loss / min(n_steps, len(tokens)-1))
    
    def reset(self):
        self.state = np.zeros(self.hidden_dim, dtype=np.float32)


# Test data
text = '''脳は人間の身体の中で最も複雑な器官です。思考、記憶、感情、運動制御など、すべての認知機能を担っています。
ニューロンは電気信号を通じて情報を伝達します。スパイキングニューラルネットワークは、この生物学的効率性を模倣しようとしています。''' * 100

chars = sorted(set(text))
c2i = {c: i for i, c in enumerate(chars)}
tokens = [c2i[c] for c in text]
vocab = len(chars)

print(f"Corpus: {len(text)} chars, {vocab} unique")

# Train model first
print("\n[1/2] Training baseline model...")
model = SNNModel(vocab, 512, 9)  # 9D for speed
model.reset()
for i in range(min(3000, len(tokens)-1)):
    model.train_step(tokens[i], tokens[i+1])

# Store trained weights
model.W_res_orig = model.W_res.copy()
model.W_out_orig = model.W_out.copy()

# Baseline PPL
baseline_ppl = model.evaluate(tokens)
print(f"  Baseline PPL: {baseline_ppl:.2f}")

# Test noise resilience
print("\n[2/2] Testing noise resilience...")
print("-"*60)

noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
results = []

for noise in noise_levels:
    model.reset_weights()
    if noise > 0:
        model.add_noise(noise)
    
    ppl = model.evaluate(tokens)
    degradation = ((ppl - baseline_ppl) / baseline_ppl) * 100
    
    results.append({
        'noise': noise,
        'ppl': ppl,
        'degradation': degradation
    })
    
    status = "✅" if degradation < 20 else "⚠️" if degradation < 50 else "❌"
    print(f"  Noise {noise*100:>5.1f}%: PPL={ppl:>7.2f} ({degradation:>+6.1f}%) {status}")

# Summary
print("\n" + "="*60)
print("NOISE RESILIENCE SUMMARY")
print("="*60)

max_tolerable = max([r['noise'] for r in results if r['degradation'] < 20], default=0)
print(f"Maximum tolerable noise (<20% degradation): {max_tolerable*100:.1f}%")

if max_tolerable >= 0.1:
    print("\n✅ SNN-LLM shows STRONG noise resilience!")
    print("   This supports the hypothesis that SNNs are robust like biological brains.")
elif max_tolerable >= 0.05:
    print("\n⚠️ SNN-LLM shows MODERATE noise resilience.")
else:
    print("\n❌ SNN-LLM is sensitive to noise.")

print("""
Implications:
- High noise tolerance = suitable for edge/IoT deployment
- Robust to hardware imperfections (analog computing)
- Brain-like fault tolerance

Next Steps:
- Test on larger models
- Compare with traditional DNNs
- Deploy on neuromorphic hardware
""")
