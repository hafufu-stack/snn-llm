"""
Hypercube Dimension Comparison Experiment
==========================================
Test 5D to 11D hypercube performance
"""

import numpy as np
import time

print('='*60)
print('Hypercube Dimension Comparison Experiment')
print('='*60)

def create_hypercube_mask(dim):
    n = 2 ** dim
    mask = np.zeros((n, n))
    for i in range(n):
        for bit in range(dim):
            j = i ^ (1 << bit)
            mask[i, j] = 1.0
    return mask

class QuickSNNLM:
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
        self.connections = int(mask.sum())
    
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

# Test data (Japanese)
text = '''脳は人間の身体の中で最も複雑な器官です。思考、記憶、感情、運動制御など、すべての認知機能を担っています。
ニューロンは電気信号を通じて情報を伝達します。スパイキングニューラルネットワークは、この生物学的効率性を模倣しようとしています。
言語モデルは文脈から次の単語を予測することを学習します。SNNは時間的符号化を使用して情報を処理します。
日本語は独特な言語です。ひらがな、カタカナ、漢字という三種類の文字体系を使用します。
人工知能は人間の知能を模倣しようとする技術分野です。機械学習、深層学習、強化学習など、様々なアプローチがあります。''' * 50

chars = sorted(set(text))
c2i = {c: i for i, c in enumerate(chars)}
tokens = [c2i[c] for c in text]
vocab = len(chars)

print(f"\nCorpus: {len(text)} chars, {vocab} unique chars")

results = []

for dim in [5, 6, 7, 8, 9, 10, 11]:
    hidden = 2 ** dim
    print(f'\nTesting {dim}D Hypercube (hidden={hidden})...')
    
    model = QuickSNNLM(vocab, hidden, dim)
    
    t0 = time.time()
    model.reset()
    total_loss = 0
    n_steps = min(5000, len(tokens)-1)
    for i in range(n_steps):
        loss = model.train_step(tokens[i], tokens[i+1])
        total_loss += loss
    
    avg_loss = total_loss / n_steps
    ppl = np.exp(avg_loss)
    elapsed = time.time() - t0
    
    results.append({
        'dim': dim,
        'hidden': hidden,
        'connections': model.connections,
        'ppl': ppl,
        'time': elapsed
    })
    
    print(f'  Dim={dim}, Hidden={hidden}, Connections={model.connections:,}, PPL={ppl:.2f}, Time={elapsed:.1f}s')

print('\n' + '='*60)
print('RESULTS SUMMARY')
print('='*60)
print(f"{'Dim':>4} | {'Hidden':>6} | {'Connections':>12} | {'PPL':>8} | {'Time':>6}")
print('-'*50)
for r in results:
    print(f"{r['dim']:>4}D | {r['hidden']:>6} | {r['connections']:>12,} | {r['ppl']:>8.2f} | {r['time']:>5.1f}s")

# Find optimal
best = min(results, key=lambda x: x['ppl'])
print(f"\n🏆 Best: {best['dim']}D (PPL={best['ppl']:.2f})")

# Efficiency metric
print('\n' + '='*60)
print('EFFICIENCY ANALYSIS (PPL per 1K Connections)')
print('='*60)
for r in results:
    efficiency = r['ppl'] / (r['connections'] / 1000)
    print(f"{r['dim']}D: PPL/Conn = {efficiency:.4f}")
