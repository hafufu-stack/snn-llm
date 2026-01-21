"""
Japanese vs English PPL Comparison
==================================
Compare SNN-LLM performance on Japanese vs English text
Hypothesis: Japanese may be more suitable for SNN processing
"""

import numpy as np
import time

print('='*60)
print('Japanese vs English SNN-LLM Comparison')
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


def train_and_eval(text, name, epochs=5):
    print(f"\n--- {name} ---")
    print(f"  Text length: {len(text)} chars")
    
    chars = sorted(set(text))
    c2i = {c: i for i, c in enumerate(chars)}
    tokens = [c2i[c] for c in text]
    vocab = len(chars)
    
    print(f"  Vocabulary: {vocab} unique chars")
    
    model = SNNLM(vocab, 512, 9)
    
    results = []
    for epoch in range(epochs):
        model.reset()
        total_loss = 0
        n_steps = min(5000, len(tokens)-1)
        for i in range(n_steps):
            loss = model.train_step(tokens[i], tokens[i+1])
            total_loss += loss
        
        avg_loss = total_loss / n_steps
        ppl = np.exp(avg_loss)
        results.append(ppl)
        print(f"  Epoch {epoch+1}: PPL={ppl:.2f}")
    
    return results[-1]


# Japanese corpus
japanese_text = """
脳は人間の身体の中で最も複雑な器官です。思考、記憶、感情、運動制御など、すべての認知機能を担っています。
ニューロンは電気信号を通じて情報を伝達します。スパイキングニューラルネットワークは、この生物学的効率性を模倣しようとしています。
言語モデルは文脈から次の単語を予測することを学習します。SNNは時間的符号化を使用して情報を処理します。
日本語は独特な言語です。ひらがな、カタカナ、漢字という三種類の文字体系を使用します。
人工知能は人間の知能を模倣しようとする技術分野です。機械学習、深層学習、強化学習など、様々なアプローチがあります。
脳の構造は11次元であることが発見されました。これは人間が10進数を使用する理由かもしれません。
スパイキングニューラルネットワークは、従来のニューラルネットワークと比較して、エネルギー効率が大幅に優れています。
""" * 50

# English corpus (similar content)
english_text = """
The brain is the most complex organ in the human body. It handles all cognitive functions including thinking, memory, emotions, and motor control.
Neurons transmit information through electrical signals. Spiking neural networks attempt to mimic this biological efficiency.
Language models learn to predict the next word from context. SNNs use temporal coding to process information.
Japanese is a unique language. It uses three types of writing systems: hiragana, katakana, and kanji.
Artificial intelligence is a technical field that attempts to mimic human intelligence. There are various approaches including machine learning, deep learning, and reinforcement learning.
It was discovered that the brain's structure is 11-dimensional. This may be why humans use the decimal system.
Spiking neural networks are significantly more energy-efficient compared to traditional neural networks.
""" * 50

print("\n[1/2] Training on Japanese corpus...")
ppl_ja = train_and_eval(japanese_text, "Japanese", epochs=10)

print("\n[2/2] Training on English corpus...")
ppl_en = train_and_eval(english_text, "English", epochs=10)

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"\nJapanese PPL: {ppl_ja:.2f}")
print(f"English PPL:  {ppl_en:.2f}")

diff = ((ppl_en - ppl_ja) / ppl_en) * 100
if ppl_ja < ppl_en:
    print(f"\n✅ Japanese is {diff:.1f}% BETTER for SNN!")
    print("   This supports the hypothesis that SNNs may be better suited for Japanese.")
else:
    print(f"\n❌ English is {-diff:.1f}% better for SNN")
    print("   More investigation needed.")

print("\n" + "="*60)
print("ANALYSIS")
print("="*60)
print("""
Possible reasons for difference:
1. Character density: Japanese kanji carry more meaning per character
2. Consistent syllable structure: Japanese has regular CV patterns
3. Less ambiguity: Japanese grammar is more regular
4. Information density: Fewer characters needed for same meaning

Implications for SNN-LLM:
- If Japanese is better, focus on Japanese-first development
- Leverage temporal coding for Japanese's high information density
- May explain why the brain evolved 11D structure
""")
