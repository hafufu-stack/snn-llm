"""
Hyperparameter Search Experiment
================================
Find optimal learning rate and dimension for SNN-LLM
"""

import numpy as np
import time
import itertools

print('='*60)
print('Hyperparameter Search for SNN-LLM')
print('='*60)

def create_hypercube_mask(dim):
    n = 2 ** dim
    mask = np.zeros((n, n))
    for i in range(n):
        for bit in range(dim):
            j = i ^ (1 << bit)
            mask[i, j] = 1.0
    return mask


class SNNLM:
    def __init__(self, vocab_size, hidden_dim, hypercube_dim, lr):
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
        self.lr = lr
    
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
    
    def reset(self):
        self.state = np.zeros(self.hidden_dim, dtype=np.float32)


# Test corpus
text = '''脳は人間の身体の中で最も複雑な器官です。思考、記憶、感情、運動制御など、すべての認知機能を担っています。
ニューロンは電気信号を通じて情報を伝達します。スパイキングニューラルネットワークは、この生物学的効率性を模倣しようとしています。
言語モデルは文脈から次の単語を予測することを学習します。SNNは時間的符号化を使用して情報を処理します。''' * 50

chars = sorted(set(text))
c2i = {c: i for i, c in enumerate(chars)}
tokens = [c2i[c] for c in text]
vocab = len(chars)

print(f"Corpus: {len(text)} chars, {vocab} unique")

# Hyperparameter grid
learning_rates = [0.005, 0.01, 0.02, 0.05]
dimensions = [8, 9, 10]

results = []
total = len(learning_rates) * len(dimensions)
current = 0

print(f"\nSearching {total} configurations...")
print("-"*60)

for lr, dim in itertools.product(learning_rates, dimensions):
    current += 1
    hidden = 2 ** dim
    
    model = SNNLM(vocab, hidden, dim, lr)
    
    t0 = time.time()
    total_loss = 0
    n_steps = min(3000, len(tokens)-1)
    
    # Train for 5 epochs
    for epoch in range(5):
        model.reset()
        epoch_loss = 0
        for i in range(n_steps):
            loss = model.train_step(tokens[i], tokens[i+1])
            epoch_loss += loss
        total_loss = epoch_loss
    
    avg_loss = total_loss / n_steps
    ppl = np.exp(avg_loss)
    elapsed = time.time() - t0
    
    results.append({
        'lr': lr,
        'dim': dim,
        'hidden': hidden,
        'ppl': ppl,
        'time': elapsed
    })
    
    print(f"  [{current}/{total}] lr={lr}, dim={dim}D: PPL={ppl:.2f}, Time={elapsed:.1f}s")


# Find best
print("\n" + "="*60)
print("HYPERPARAMETER SEARCH RESULTS")
print("="*60)

# Sort by PPL
sorted_results = sorted(results, key=lambda x: x['ppl'])

print(f"\n{'Rank':<6} {'LR':<8} {'Dim':<6} {'Hidden':<8} {'PPL':<10}")
print("-"*45)
for i, r in enumerate(sorted_results[:5]):
    print(f"{i+1:<6} {r['lr']:<8} {r['dim']}D{'':<3} {r['hidden']:<8} {r['ppl']:<10.2f}")

best = sorted_results[0]
print(f"\n🏆 Best Configuration:")
print(f"   Learning Rate: {best['lr']}")
print(f"   Dimension: {best['dim']}D")
print(f"   Hidden Size: {best['hidden']}")
print(f"   PPL: {best['ppl']:.2f}")

# LR analysis
print("\n" + "-"*40)
print("Learning Rate Analysis (averaged across dims):")
for lr in learning_rates:
    avg_ppl = np.mean([r['ppl'] for r in results if r['lr'] == lr])
    print(f"  LR={lr}: avg PPL={avg_ppl:.2f}")

# Dimension analysis
print("\n" + "-"*40)
print("Dimension Analysis (averaged across LRs):")
for dim in dimensions:
    avg_ppl = np.mean([r['ppl'] for r in results if r['dim'] == dim])
    print(f"  {dim}D: avg PPL={avg_ppl:.2f}")

print("""
Recommendations:
- Use the best LR and dimension found above
- Consider learning rate scheduling
- Deeper architectures may need lower LR
""")
