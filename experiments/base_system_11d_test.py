"""
Base Number System vs 11D Brain Experiment
===========================================
Hypothesis: 11D brain is optimized for 10-digit (base-10) system
           11+ base numbers should be harder to process

Test: Fix 11D hypercube, vary number of classes (bases)
"""

import numpy as np
import time

print('='*60)
print('Base Number System vs 11D Brain Experiment')
print('='*60)
print()
print("Hypothesis: 11D structure is optimized for 10 classes (base-10)")
print("           11+ classes should show degraded performance")
print()

def create_hypercube_mask(dim):
    n = 2 ** dim
    mask = np.zeros((n, n))
    for i in range(n):
        for bit in range(dim):
            j = i ^ (1 << bit)
            mask[i, j] = 1.0
    return mask


class SNN11DClassifier:
    """11D Hypercube SNN for classification"""
    
    def __init__(self, input_dim, n_classes, hidden_dim=2048):
        np.random.seed(42)
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.W_in = np.random.randn(hidden_dim, input_dim).astype(np.float32) * 0.1
        
        # 11D Hypercube reservoir
        mask = create_hypercube_mask(11)  # Fixed 11D
        W = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.2
        self.W_res = W * mask
        
        # Output layer
        self.W_out = np.random.randn(n_classes, hidden_dim).astype(np.float32) * 0.01
        self.b_out = np.zeros(n_classes, dtype=np.float32)
        
        self.state = np.zeros(hidden_dim, dtype=np.float32)
        self.lr = 0.01
    
    def forward(self, x):
        h = np.tanh(self.W_in @ x)
        rec = np.tanh(self.W_res @ self.state)
        self.state = 0.7 * self.state + 0.3 * (h + rec)
        logits = self.W_out @ self.state + self.b_out
        return logits
    
    def train_step(self, x, y):
        logits = self.forward(x)
        exp_l = np.exp(logits - np.max(logits))
        probs = exp_l / (np.sum(exp_l) + 1e-10)
        loss = -np.log(probs[y] + 1e-10)
        
        # Backward
        grad = probs.copy()
        grad[y] -= 1
        self.W_out -= self.lr * np.outer(grad, self.state)
        self.b_out -= self.lr * grad
        
        return loss
    
    def reset(self):
        self.state = np.zeros(self.hidden_dim, dtype=np.float32)


# Generate synthetic data for different base systems
def generate_data(n_classes, n_samples=2000, input_dim=100):
    np.random.seed(42)
    X = np.random.randn(n_samples, input_dim).astype(np.float32)
    y = np.random.randint(0, n_classes, n_samples)
    return X, y


# Test different base number systems with FIXED 11D brain
print("="*60)
print("Testing different base systems with FIXED 11D hypercube")
print("="*60)

base_systems = [
    (8, "8進数 (Octal)"),
    (10, "10進数 (Decimal) - Human Standard"),
    (11, "11進数"),
    (12, "12進数 (Duodecimal)"),
    (16, "16進数 (Hexadecimal)"),
    (20, "20進数 (Vigesimal - Maya)"),
]

results = []

for n_classes, name in base_systems:
    print(f"\n--- Testing {name} ({n_classes} classes) ---")
    
    X, y = generate_data(n_classes, n_samples=3000)
    
    model = SNN11DClassifier(input_dim=100, n_classes=n_classes)
    
    # Training
    train_losses = []
    for epoch in range(5):
        model.reset()
        epoch_loss = 0
        for i in range(2000):
            loss = model.train_step(X[i], y[i])
            epoch_loss += loss
        train_losses.append(epoch_loss / 2000)
    
    # Evaluation
    model.reset()
    correct = 0
    total_loss = 0
    for i in range(2000, 3000):
        logits = model.forward(X[i])
        pred = np.argmax(logits)
        if pred == y[i]:
            correct += 1
        exp_l = np.exp(logits - np.max(logits))
        probs = exp_l / (np.sum(exp_l) + 1e-10)
        total_loss += -np.log(probs[y[i]] + 1e-10)
    
    accuracy = correct / 1000 * 100
    final_loss = total_loss / 1000
    
    # Normalize by chance level
    chance = 100 / n_classes
    relative_accuracy = accuracy / chance  # 1.0 = chance level
    
    results.append({
        'base': n_classes,
        'name': name,
        'accuracy': accuracy,
        'chance': chance,
        'relative': relative_accuracy,
        'loss': final_loss
    })
    
    print(f"  Accuracy: {accuracy:.1f}% (chance: {chance:.1f}%)")
    print(f"  Relative to chance: {relative_accuracy:.2f}x")
    print(f"  Loss: {final_loss:.3f}")


# Summary
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
print()
print(f"{'Base':>6} | {'Accuracy':>10} | {'Chance':>8} | {'Relative':>10} | {'Loss':>8}")
print("-"*55)

best_relative = max(results, key=lambda x: x['relative'])

for r in results:
    marker = " 🏆" if r['base'] == best_relative['base'] else ""
    marker2 = " ★" if r['base'] == 10 else ""
    print(f"{r['base']:>6} | {r['accuracy']:>9.1f}% | {r['chance']:>7.1f}% | {r['relative']:>9.2f}x | {r['loss']:>8.3f}{marker}{marker2}")


# Analysis
print("\n" + "="*60)
print("HYPOTHESIS TEST")
print("="*60)

base10 = next(r for r in results if r['base'] == 10)
base11_plus = [r for r in results if r['base'] >= 11]

print(f"\nBase-10 (Decimal) performance:")
print(f"  Relative accuracy: {base10['relative']:.2f}x chance")
print(f"  Loss: {base10['loss']:.3f}")

better_than_10 = [r for r in base11_plus if r['relative'] > base10['relative']]
worse_than_10 = [r for r in base11_plus if r['relative'] < base10['relative']]

print(f"\nBases 11+ that perform BETTER than base-10: {len(better_than_10)}")
for r in better_than_10:
    print(f"  - Base-{r['base']}: {r['relative']:.2f}x")

print(f"\nBases 11+ that perform WORSE than base-10: {len(worse_than_10)}")
for r in worse_than_10:
    print(f"  - Base-{r['base']}: {r['relative']:.2f}x")

if len(worse_than_10) > len(better_than_10):
    print("\n✅ HYPOTHESIS SUPPORTED!")
    print("   Base-10 shows relatively GOOD performance on 11D brain")
    print("   Higher bases (11+) tend to perform WORSE")
    print()
    print("   This suggests the brain's 11D structure may indeed")
    print("   be optimized for 10-digit processing!")
else:
    print("\n⚠️ HYPOTHESIS INCONCLUSIVE")
    print("   More investigation needed")

print("\n" + "="*60)
print("IMPLICATIONS")
print("="*60)
print("""
If hypothesis is supported:
1. Human brain's 11D structure → optimal for 10 categories
2. 10進数 is not arbitrary but reflects brain architecture
3. Languages/cultures using other bases may have efficiency trade-offs

For SNN-LLM:
1. 11D hypercube is optimal for standard categorical processing
2. Token vocabulary should be designed with this in mind
3. Japanese (with ~3000 kanji) may benefit from hierarchical encoding
""")
