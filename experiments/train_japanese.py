"""
Japanese SNN-LLM Full Training
==============================

Complete Japanese language model with:
1. More epochs (50)
2. Word/subword-level tokenization (BPE-like)
3. Larger model (1024 hidden, 10D hypercube)

Estimated time: 3-4 hours

Usage: python experiments/train_japanese.py

Author: ろーる
Date: 2026-01-21
"""

import numpy as np
import os
import sys
import time
import json
import pickle
import urllib.request
import re
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hypercube import create_hypercube_mask


# ==============================================================================
# Japanese Text Processing
# ==============================================================================

def download_japanese_corpus(data_dir="data"):
    """Download or create Japanese corpus"""
    os.makedirs(data_dir, exist_ok=True)
    
    corpus_path = os.path.join(data_dir, "japanese_corpus.txt")
    
    if os.path.exists(corpus_path):
        print(f"  Japanese corpus exists: {corpus_path}")
    else:
        print("  Creating Japanese corpus...")
        
        # Sample Japanese text (脳科学・AI関連)
        japanese_text = """
脳は人間の身体の中で最も複雑な器官です。思考、記憶、感情、運動、視覚、呼吸、体温調節など、あらゆる機能を制御しています。
人間の脳は約860億個のニューロンで構成されています。これらのニューロンはシナプスを通じて相互に通信し、複雑なネットワークを形成しています。
スパイキングニューラルネットワークは、この生物学的効率性を模倣しようとしています。従来の人工ニューラルネットワークとは異なり、SNNは時間的符号化を使用して情報を表現します。
これにより、フォン・ノイマン型アーキテクチャと比較してエネルギー消費を削減しながら、膨大な情報容量を実現できます。
コンピューティングの未来は、人間の脳のように学習し適応できる脳型システムにあります。
言語モデルはエネルギー効率が重要なアプリケーション分野です。大規模な言語モデルの実行には膨大な計算リソースが必要です。
しかし、脳型アプローチにより、最小限の電力消費でエッジデバイス上でのローカル処理が可能になります。
人工知能は急速に進歩を続けています。深層学習はコンピュータサイエンスの多くの分野に革命をもたらしました。
ニューロモーフィックコンピューティングは、脳のアーキテクチャを模倣することを目指しています。
脳の構造には11次元のクリークが存在することがブルーブレインプロジェクトによって発見されました。
この高次元トポロジーにより、最小限のエネルギー消費で効率的な情報処理が可能になります。
自然言語処理は人工知能の重要な分野です。機械翻訳、テキスト生成、感情分析など、多くのアプリケーションがあります。
日本語は複雑な言語です。ひらがな、カタカナ、漢字の3種類の文字体系を使用します。
文法構造も英語とは大きく異なり、主語-目的語-動詞の語順を持ちます。
言語モデルは文脈を理解し、次の単語や文字を予測することを学習します。
十分なデータで訓練されたモデルは、文法的に正しく意味のある文章を生成できます。
未来のAIシステムは、より省エネルギーで効率的になる必要があります。
スパイキングニューラルネットワークは、この目標を達成するための有望なアプローチです。
ハイパーキューブトポロジーは、効率的な情報伝播を可能にする構造です。
各ノードは限られた数の隣接ノードに接続されていますが、全体として高速な通信が可能です。
脳型コンピューティングは、次世代のAIシステムの基盤となる可能性があります。
研究者たちは、より効率的で強力なニューラルネットワークアーキテクチャを開発し続けています。
"""
        
        # Repeat to create larger corpus
        full_corpus = japanese_text * 100
        
        with open(corpus_path, 'w', encoding='utf-8') as f:
            f.write(full_corpus)
        
        print(f"  Created corpus: {len(full_corpus):,} chars")
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        return f.read()


# ==============================================================================
# BPE-like Tokenizer for Japanese
# ==============================================================================

class JapaneseTokenizer:
    """
    Simple subword tokenizer for Japanese
    Uses character bigrams and common patterns
    """
    
    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.token_to_idx = {}
        self.idx_to_token = {}
        self.actual_vocab_size = 0
    
    def build_vocab(self, text, min_freq=2):
        """Build vocabulary from text"""
        
        # Start with characters
        char_counts = Counter(text)
        
        # Add special tokens
        self.token_to_idx = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
        
        # Add single characters
        idx = len(self.token_to_idx)
        for char, count in char_counts.most_common():
            if count >= min_freq and char not in self.token_to_idx:
                self.token_to_idx[char] = idx
                idx += 1
                if idx >= self.vocab_size // 2:
                    break
        
        # Add common bigrams (simple BPE)
        bigram_counts = Counter()
        for i in range(len(text) - 1):
            bigram = text[i:i+2]
            bigram_counts[bigram] += 1
        
        for bigram, count in bigram_counts.most_common():
            if count >= min_freq * 2 and bigram not in self.token_to_idx:
                self.token_to_idx[bigram] = idx
                idx += 1
                if idx >= self.vocab_size:
                    break
        
        # Add common trigrams
        trigram_counts = Counter()
        for i in range(len(text) - 2):
            trigram = text[i:i+3]
            trigram_counts[trigram] += 1
        
        for trigram, count in trigram_counts.most_common():
            if count >= min_freq * 3 and trigram not in self.token_to_idx:
                self.token_to_idx[trigram] = idx
                idx += 1
                if idx >= self.vocab_size:
                    break
        
        self.idx_to_token = {i: t for t, i in self.token_to_idx.items()}
        self.actual_vocab_size = len(self.token_to_idx)
        
        print(f"  Vocabulary size: {self.actual_vocab_size}")
        print(f"  Sample tokens: {list(self.token_to_idx.keys())[10:20]}")
    
    def encode(self, text):
        """Encode text to token indices (greedy longest match)"""
        tokens = []
        i = 0
        while i < len(text):
            # Try longest match first
            matched = False
            for length in [3, 2, 1]:
                if i + length <= len(text):
                    substr = text[i:i+length]
                    if substr in self.token_to_idx:
                        tokens.append(self.token_to_idx[substr])
                        i += length
                        matched = True
                        break
            
            if not matched:
                tokens.append(self.token_to_idx.get('<UNK>', 1))
                i += 1
        
        return tokens
    
    def decode(self, indices):
        """Decode token indices to text"""
        return ''.join(self.idx_to_token.get(i, '<UNK>') for i in indices)
    
    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'token_to_idx': self.token_to_idx,
                'vocab_size': self.actual_vocab_size
            }, f, ensure_ascii=False, indent=2)
    
    def load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.token_to_idx = data['token_to_idx']
        self.idx_to_token = {int(i): t for t, i in self.token_to_idx.items()}
        self.actual_vocab_size = data['vocab_size']


# ==============================================================================
# Large SNN Language Model
# ==============================================================================

class LargeSNNLM:
    """
    Large SNN Language Model
    - 1024 hidden units (vs 512)
    - 10D hypercube (vs 9D)
    - BPE tokenization
    """
    
    def __init__(self, vocab_size, hidden_dim=1024, hypercube_dim=10, seed=42):
        np.random.seed(seed)
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.hypercube_dim = hypercube_dim
        
        # Embedding (larger)
        self.embedding = np.random.randn(vocab_size, hidden_dim).astype(np.float32) * 0.05
        
        # Reservoir with 10D hypercube
        mask = create_hypercube_mask(hypercube_dim)
        
        # Resize if needed
        target_size = hidden_dim
        orig_size = mask.shape[0]
        if orig_size != target_size:
            new_mask = np.zeros((target_size, target_size), dtype=np.float32)
            for i in range(target_size):
                for j in range(target_size):
                    new_mask[i, j] = mask[i % orig_size, j % orig_size]
            mask = new_mask
        
        W_res = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.2
        self.W_res = (W_res * mask).astype(np.float32)
        
        # Scale spectral radius
        try:
            eig = np.linalg.eigvals(self.W_res[:256, :256])  # Sample for speed
            self.W_res *= 1.3 / (np.max(np.abs(eig)) + 0.01)
        except:
            pass
        
        self.mask = mask
        
        # Output layer
        self.W_out = np.random.randn(vocab_size, hidden_dim).astype(np.float32) * 0.01
        self.b_out = np.zeros(vocab_size, dtype=np.float32)
        
        # State
        self.state = np.zeros(hidden_dim, dtype=np.float32)
        
        # Learning rate (with decay)
        self.lr = 0.01
        self.lr_decay = 0.95
    
    def forward(self, token_idx):
        emb = self.embedding[token_idx]
        h_rec = np.tanh(self.W_res @ self.state)
        self.state = 0.7 * self.state + 0.3 * (emb + h_rec)
        
        # Spiking threshold
        mask = self.state > 0.5
        self.state = np.where(mask, self.state * 0.7, self.state)
        
        return self.W_out @ self.state + self.b_out
    
    def train_step(self, input_idx, target_idx):
        logits = self.forward(input_idx)
        
        # Softmax
        logits_max = np.max(logits)
        exp_l = np.exp(logits - logits_max)
        probs = exp_l / (np.sum(exp_l) + 1e-10)
        
        # Loss
        loss = -np.log(probs[target_idx] + 1e-10)
        
        # Gradient
        grad = probs.copy()
        grad[target_idx] -= 1
        
        # Update
        self.W_out -= self.lr * np.outer(grad, self.state)
        self.b_out -= self.lr * grad
        
        return loss
    
    def decay_lr(self):
        self.lr *= self.lr_decay
    
    def reset(self):
        self.state = np.zeros(self.hidden_dim, dtype=np.float32)
    
    def generate(self, start_indices, max_len=100, temperature=0.7):
        self.reset()
        generated = list(start_indices)
        
        for idx in start_indices:
            self.forward(idx)
        
        for _ in range(max_len):
            logits = self.forward(generated[-1])
            log_p = logits / temperature
            exp_l = np.exp(log_p - np.max(log_p))
            probs = exp_l / np.sum(exp_l)
            next_idx = np.random.choice(len(probs), p=probs)
            generated.append(next_idx)
        
        return generated
    
    def evaluate(self, indices, max_eval=3000):
        self.reset()
        indices = indices[:max_eval]
        total_loss = 0
        for i in range(len(indices) - 1):
            logits = self.forward(indices[i])
            exp_l = np.exp(logits - np.max(logits))
            probs = exp_l / (np.sum(exp_l) + 1e-10)
            total_loss += -np.log(probs[indices[i+1]] + 1e-10)
        return np.exp(total_loss / (len(indices) - 1))
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'vocab_size': self.vocab_size,
                'hidden_dim': self.hidden_dim,
                'hypercube_dim': self.hypercube_dim,
                'embedding': self.embedding,
                'W_res': self.W_res,
                'W_out': self.W_out,
                'b_out': self.b_out,
                'lr': self.lr,
            }, f)


# ==============================================================================
# Main Training
# ==============================================================================

def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║         🇯🇵 Japanese SNN-LLM Full Training 🇯🇵               ║
    ║         50 epochs, BPE, 1024 hidden, 10D hypercube            ║
    ║         Estimated: 3-4 hours                                  ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    start_time = time.time()
    
    # Configuration
    config = {
        'hidden_dim': 1024,
        'hypercube_dim': 10,
        'vocab_size': 3000,
        'epochs': 50,
        'lr': 0.01,
    }
    
    print("Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print("-" * 60)
    
    # Load corpus
    print("\n[1/4] Loading Japanese corpus...")
    text = download_japanese_corpus()
    print(f"  Total characters: {len(text):,}")
    
    # Build tokenizer
    print("\n[2/4] Building BPE-like tokenizer...")
    tokenizer = JapaneseTokenizer(vocab_size=config['vocab_size'])
    tokenizer.build_vocab(text)
    
    # Tokenize
    tokens = np.array(tokenizer.encode(text), dtype=np.int32)
    print(f"  Total tokens: {len(tokens):,}")
    print(f"  Compression ratio: {len(text) / len(tokens):.2f}x")
    
    # Create model
    print("\n[3/4] Creating large model...")
    model = LargeSNNLM(
        vocab_size=tokenizer.actual_vocab_size,
        hidden_dim=config['hidden_dim'],
        hypercube_dim=config['hypercube_dim']
    )
    
    n_params = (model.embedding.size + model.W_res.size + 
                model.W_out.size + model.b_out.size)
    print(f"  Parameters: {n_params:,}")
    print(f"  Reservoir connections: {int(model.mask.sum()):,}")
    
    # Directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Training
    print("\n[4/4] Training...")
    print("=" * 60)
    
    history = []
    best_ppl = float('inf')
    
    for epoch in range(config['epochs']):
        model.reset()
        t0 = time.time()
        total_loss = 0
        n_tokens = len(tokens) - 1
        
        # Progress
        log_interval = n_tokens // 5
        
        for i in range(n_tokens):
            loss = model.train_step(tokens[i], tokens[i + 1])
            total_loss += loss
            
            if (i + 1) % log_interval == 0:
                progress = (i + 1) / n_tokens * 100
                print(f"    Epoch {epoch+1}: {progress:.0f}%")
        
        avg_loss = total_loss / n_tokens
        
        # Evaluate
        model.reset()
        ppl = model.evaluate(tokens[:3000])
        
        elapsed = time.time() - t0
        
        # LR decay every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.decay_lr()
            print(f"    LR decayed to {model.lr:.5f}")
        
        print(f"  Epoch {epoch+1:>2}/{config['epochs']}: Loss={avg_loss:.4f}, PPL={ppl:.1f}, Time={elapsed:.0f}s")
        
        history.append({
            'epoch': epoch + 1,
            'loss': float(avg_loss),
            'ppl': float(ppl),
            'time': elapsed
        })
        
        # Save checkpoints
        if (epoch + 1) % 10 == 0:
            model.save(f"checkpoints/model_japanese_epoch{epoch+1}.pkl")
        
        if ppl < best_ppl:
            best_ppl = ppl
            model.save("checkpoints/model_japanese_best.pkl")
            print(f"    → Best model saved!")
    
    # Save final
    model.save("checkpoints/model_japanese_final.pkl")
    tokenizer.save("checkpoints/tokenizer_japanese.json")
    
    with open("results/japanese_history.json", 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    
    # Generation samples
    print("\n" + "=" * 60)
    print("GENERATION SAMPLES")
    print("=" * 60)
    
    prompts = ["脳は", "人工知能は", "言語"]
    for prompt in prompts:
        prompt_tokens = tokenizer.encode(prompt)
        gen_tokens = model.generate(prompt_tokens, max_len=50, temperature=0.7)
        gen_text = tokenizer.decode(gen_tokens)
        print(f"\n'{prompt}' → '{gen_text}'")
    
    # Summary
    total_time = time.time() - start_time
    
    print(f"""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                   訓練完了！🎉                                ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║   Best Perplexity: {best_ppl:>8.2f}                                 ║
    ║   総訓練時間:      {total_time/60:>8.1f} 分                         ║
    ║   パラメータ数:    {n_params:>8,}                              ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║   モデル: checkpoints/model_japanese_best.pkl                 ║
    ║   履歴:   results/japanese_history.json                       ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
