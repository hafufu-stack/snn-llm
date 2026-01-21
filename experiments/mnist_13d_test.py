"""
MNIST 13D+ Test - Does higher dimension collapse?
"""
import numpy as np
import time

print('='*60)
print('MNIST 13D+ Test (Does it collapse?)')
print('='*60)

def create_hypercube_mask(dim):
    n = 2 ** dim
    mask = np.zeros((n, n))
    for i in range(n):
        for bit in range(dim):
            j = i ^ (1 << bit)
            mask[i, j] = 1.0
    return mask

# Load MNIST
from sklearn.datasets import fetch_openml
print('Loading MNIST...')
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist.data[:10000].astype(np.float32) / 255.0
y = mnist.target[:10000].astype(np.int32)

class SNNClassifier:
    def __init__(self, in_dim, n_cls, hidden, hc_dim):
        np.random.seed(42)
        self.hidden = hidden
        mask = create_hypercube_mask(hc_dim)
        if mask.shape[0] != hidden:
            new_mask = np.zeros((hidden, hidden), dtype=np.float32)
            for i in range(hidden):
                for j in range(hidden):
                    new_mask[i,j] = mask[i % mask.shape[0], j % mask.shape[0]]
            mask = new_mask
        self.W_in = np.random.randn(hidden, in_dim).astype(np.float32) * 0.01
        W = np.random.randn(hidden, hidden).astype(np.float32) * 0.2
        self.W_res = W * mask
        self.W_out = np.random.randn(n_cls, hidden).astype(np.float32) * 0.01
        self.b_out = np.zeros(n_cls, dtype=np.float32)
        self.state = np.zeros(hidden, dtype=np.float32)
        self.lr = 0.001
        
    def train_step(self, x, y):
        h = np.tanh(self.W_in @ x)
        rec = np.tanh(self.W_res @ self.state)
        self.state = 0.8 * self.state + 0.2 * (h + rec)
        logits = self.W_out @ self.state + self.b_out
        exp_l = np.exp(logits - np.max(logits))
        probs = exp_l / (np.sum(exp_l) + 1e-10)
        loss = -np.log(probs[y] + 1e-10)
        grad = probs.copy()
        grad[y] -= 1
        self.W_out -= self.lr * np.outer(grad, self.state)
        self.b_out -= self.lr * grad
        return loss
        
    def predict(self, x):
        h = np.tanh(self.W_in @ x)
        rec = np.tanh(self.W_res @ self.state)
        self.state = 0.8 * self.state + 0.2 * (h + rec)
        logits = self.W_out @ self.state + self.b_out
        return np.argmax(logits)
        
    def reset(self):
        self.state = np.zeros(self.hidden, dtype=np.float32)

results = []
# Test 11D, 12D, 13D, 14D (with reasonable hidden sizes)
for dim in [11, 12, 13, 14]:
    # Use actual 2^dim but cap at 4096 for memory
    hidden = min(2 ** dim, 4096)
    print(f'\n{dim}D Hypercube (hidden={hidden})...')
    
    model = SNNClassifier(784, 10, hidden, dim)
    
    t0 = time.time()
    for epoch in range(2):
        model.reset()
        for i in range(3000):
            model.train_step(X[i], y[i])
    
    # Test
    model.reset()
    correct = 0
    for i in range(3000, 5000):
        if model.predict(X[i]) == y[i]:
            correct += 1
    
    acc = correct / 2000 * 100
    elapsed = time.time() - t0
    results.append({'dim': dim, 'hidden': hidden, 'acc': acc, 'time': elapsed})
    print(f'  Accuracy: {acc:.1f}%, Time: {elapsed:.1f}s')

print('\n' + '='*60)
print('RESULTS:')
print('='*60)
for r in results:
    marker = ' <-- COLLAPSE?' if r['acc'] < 15 else ''
    print(f"  {r['dim']}D: {r['acc']:.1f}%{marker}")
