#!/usr/bin/env python3
"""
交互学習 (Interleaved Transfer Learning)
単語↔文字を何回か切り替えて学習

イメージ:
  Round 1: 単語200ep → 文字200ep
  Round 2: 単語200ep → 文字200ep  
  Round 3: 単語200ep → 文字200ep
  ...

徐々に両方の表現を学ぶ
"""

import numpy as np
import pickle
import json
import time
from pathlib import Path
from numba import njit, prange, set_num_threads

# MeCab
import fugashi
tagger = fugashi.Tagger()
print("✅ fugashi (MeCab) ロード成功")

set_num_threads(24)

# Paths
CORPUS_PATH = Path(__file__).parent.parent / "data" / "japanese_corpus.txt"
CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"

# パラメータ
INPUT_SCALE = 10.0
LEAK = 0.99
THRESHOLD = 1.0
HIDDEN_DIM = 2048
LR = 0.005

# 交互学習設定
ROUNDS = 5          # 何回交互に切り替えるか
EPOCHS_PER_PHASE = 200  # 各フェーズのエポック数


@njit(cache=True)
def softmax_numba(x):
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / (np.sum(exp_x) + 1e-10)


@njit(parallel=True, cache=True)
def parallel_forward(embedding_scaled, W_res, W_spike, W_membrane, bias,
                     token_ids, hidden_dim, vocab_size, leak):
    n_tokens = len(token_ids)
    h = np.zeros(hidden_dim, dtype=np.float32)
    m = np.zeros(hidden_dim, dtype=np.float32)
    
    all_logits = np.zeros((n_tokens, vocab_size), dtype=np.float32)
    all_h = np.zeros((n_tokens, hidden_dim), dtype=np.float32)
    all_m = np.zeros((n_tokens, hidden_dim), dtype=np.float32)
    
    for t in range(n_tokens):
        tid = token_ids[t]
        if tid >= vocab_size:
            tid = 0
        x = embedding_scaled[tid]
        
        for i in prange(hidden_dim):
            m[i] = leak * m[i] + x[i]
            for j in range(hidden_dim):
                m[i] += W_res[i, j] * h[j]
        
        for i in prange(hidden_dim):
            if m[i] > 1.0:
                h[i] = 1.0
                m[i] = 0.0
            else:
                h[i] = 0.0
        
        all_h[t] = h.copy()
        all_m[t] = m.copy()
        
        for i in prange(vocab_size):
            all_logits[t, i] = bias[i]
            for j in range(hidden_dim):
                all_logits[t, i] += W_spike[i, j] * h[j]
                all_logits[t, i] += W_membrane[i, j] * m[j]
    
    return all_logits, all_h, all_m


@njit(parallel=True, cache=True)
def parallel_weight_update(W_spike, W_membrane, bias, errors, all_h, all_m, lr):
    n_tokens, hidden_dim = all_h.shape
    vocab_size = len(bias)
    
    for t in range(n_tokens):
        error = errors[t]
        h = all_h[t]
        m = all_m[t]
        
        for i in prange(vocab_size):
            bias[i] -= lr * error[i]
            for j in range(hidden_dim):
                W_spike[i, j] -= lr * error[i] * h[j]
                W_membrane[i, j] -= lr * error[i] * m[j]


def tokenize_words(text):
    words = []
    for word in tagger(text):
        if word.surface.strip():
            words.append(word.surface)
    return words


def tokenize_chars(text):
    return list(text.replace(' ', ''))


class InterleavedModel:
    """単語と文字の両方を扱えるモデル"""
    
    def __init__(self, word_vocab_size, char_vocab_size):
        self.hidden_dim = HIDDEN_DIM
        
        # 共有: W_res（リザバー接続）
        self.W_res = np.random.randn(HIDDEN_DIM, HIDDEN_DIM).astype(np.float32) * 0.01
        
        # 単語用
        self.word_vocab_size = word_vocab_size
        self.word_embedding = np.random.randn(word_vocab_size, HIDDEN_DIM).astype(np.float32) * 0.01
        self.word_W_spike = np.random.randn(word_vocab_size, HIDDEN_DIM).astype(np.float32) * 0.01
        self.word_W_membrane = np.random.randn(word_vocab_size, HIDDEN_DIM).astype(np.float32) * 0.01
        self.word_bias = np.zeros(word_vocab_size, dtype=np.float32)
        
        # 文字用
        self.char_vocab_size = char_vocab_size
        self.char_embedding = np.random.randn(char_vocab_size, HIDDEN_DIM).astype(np.float32) * 0.01
        self.char_W_spike = np.random.randn(char_vocab_size, HIDDEN_DIM).astype(np.float32) * 0.01
        self.char_W_membrane = np.random.randn(char_vocab_size, HIDDEN_DIM).astype(np.float32) * 0.01
        self.char_bias = np.zeros(char_vocab_size, dtype=np.float32)
    
    def train_phase(self, mode, train_data, epochs, token_to_idx, idx_to_token):
        """特定モードで学習"""
        if mode == 'word':
            vocab_size = self.word_vocab_size
            embedding_scaled = self.word_embedding * INPUT_SCALE
            W_spike = self.word_W_spike
            W_membrane = self.word_W_membrane
            bias = self.word_bias
        else:  # char
            vocab_size = self.char_vocab_size
            embedding_scaled = self.char_embedding * INPUT_SCALE
            W_spike = self.char_W_spike
            W_membrane = self.char_W_membrane
            bias = self.char_bias
        
        batch_size = 50
        losses = []
        
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            total_tokens = 0
            
            np.random.shuffle(train_data)
            
            for token_ids in train_data[:batch_size]:
                if len(token_ids) < 2:
                    continue
                
                n_seq = len(token_ids) - 1
                
                logits, all_h, all_m = parallel_forward(
                    embedding_scaled, self.W_res, W_spike, W_membrane, bias,
                    token_ids[:-1], self.hidden_dim, vocab_size, LEAK
                )
                
                errors = np.zeros((n_seq, vocab_size), dtype=np.float32)
                
                for t in range(n_seq):
                    probs = softmax_numba(logits[t])
                    target = token_ids[t + 1]
                    if target >= vocab_size:
                        target = 1
                    
                    loss = -np.log(probs[target] + 1e-10)
                    epoch_loss += loss
                    total_tokens += 1
                    
                    errors[t] = probs.copy()
                    errors[t, target] -= 1.0
                
                parallel_weight_update(W_spike, W_membrane, bias, errors, all_h, all_m, LR)
            
            avg_loss = epoch_loss / max(total_tokens, 1)
            losses.append(avg_loss)
        
        return losses[-1] if losses else 0.0
    
    def generate(self, mode, prompt_ids, idx_to_token, max_len=15):
        """テキスト生成"""
        if mode == 'word':
            vocab_size = self.word_vocab_size
            embedding_scaled = self.word_embedding * INPUT_SCALE
            W_spike = self.word_W_spike
            W_membrane = self.word_W_membrane
            bias = self.word_bias
        else:
            vocab_size = self.char_vocab_size
            embedding_scaled = self.char_embedding * INPUT_SCALE
            W_spike = self.char_W_spike
            W_membrane = self.char_W_membrane
            bias = self.char_bias
        
        h = np.zeros(self.hidden_dim, dtype=np.float32)
        m = np.zeros(self.hidden_dim, dtype=np.float32)
        
        generated = list(prompt_ids)
        for _ in range(max_len):
            tid = generated[-1] if generated[-1] < vocab_size else 0
            x = embedding_scaled[tid]
            m = LEAK * m + x + self.W_res @ h
            h = np.where(m > THRESHOLD, 1.0, 0.0).astype(np.float32)
            m = np.where(m > THRESHOLD, 0.0, m)
            
            logits = W_spike @ h + W_membrane @ m + bias
            probs = softmax_numba(logits / 0.8)
            try:
                next_token = np.random.choice(len(probs), p=probs)
            except:
                next_token = np.argmax(logits)
            generated.append(next_token)
        
        return "".join([idx_to_token.get(t, '?') for t in generated])


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║    🔄 交互学習 (Interleaved Transfer Learning)                ║
    ║                                                               ║
    ║    単語↔文字を交互に学習                                      ║
    ║    W_res（リザバー接続）を共有                                ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # コーパス読み込み
    print("コーパスをロード中...")
    with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
        corpus = [line.strip() for line in f if line.strip()]
    print(f"コーパス行数: {len(corpus)}")
    
    # 語彙構築（単語）
    print("\n単語語彙を構築中...")
    word_freq = {}
    word_tokenized = []
    for line in corpus:
        tokens = tokenize_words(line)
        word_tokenized.append(tokens)
        for t in tokens:
            word_freq[t] = word_freq.get(t, 0) + 1
    
    word_to_idx = {'<PAD>': 0, '<UNK>': 1, '<EOS>': 2}
    for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:1000]:
        word_to_idx[word] = len(word_to_idx)
    idx_to_word = {v: k for k, v in word_to_idx.items()}
    word_vocab_size = len(word_to_idx)
    print(f"単語語彙サイズ: {word_vocab_size}")
    
    word_train_data = []
    for tokens in word_tokenized:
        ids = np.array([word_to_idx.get(t, 1) for t in tokens], dtype=np.int64)
        if len(ids) >= 3:
            word_train_data.append(ids)
    
    # 語彙構築（文字）
    print("\n文字語彙を構築中...")
    char_freq = {}
    char_tokenized = []
    for line in corpus:
        tokens = tokenize_chars(line)
        char_tokenized.append(tokens)
        for t in tokens:
            char_freq[t] = char_freq.get(t, 0) + 1
    
    char_to_idx = {'<PAD>': 0, '<UNK>': 1, '<EOS>': 2}
    for char, freq in sorted(char_freq.items(), key=lambda x: x[1], reverse=True)[:1000]:
        char_to_idx[char] = len(char_to_idx)
    idx_to_char = {v: k for k, v in char_to_idx.items()}
    char_vocab_size = len(char_to_idx)
    print(f"文字語彙サイズ: {char_vocab_size}")
    
    char_train_data = []
    for tokens in char_tokenized:
        ids = np.array([char_to_idx.get(t, 1) for t in tokens], dtype=np.int64)
        if len(ids) >= 3:
            char_train_data.append(ids)
    
    # モデル初期化
    print("\nモデルを初期化中...")
    model = InterleavedModel(word_vocab_size, char_vocab_size)
    
    # JIT compile
    print("JITコンパイル中...")
    dummy_ids = np.array([0, 1, 2], dtype=np.int64)
    _ = parallel_forward(
        model.word_embedding * INPUT_SCALE, model.W_res,
        model.word_W_spike, model.word_W_membrane, model.word_bias,
        dummy_ids, HIDDEN_DIM, word_vocab_size, LEAK
    )
    _ = parallel_forward(
        model.char_embedding * INPUT_SCALE, model.W_res,
        model.char_W_spike, model.char_W_membrane, model.char_bias,
        dummy_ids, HIDDEN_DIM, char_vocab_size, LEAK
    )
    print("JIT完了!")
    
    # 交互学習
    print(f"\n{'='*60}")
    print(f"🔄 交互学習開始：{ROUNDS}ラウンド × {EPOCHS_PER_PHASE}ep/phase")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    for round_num in range(1, ROUNDS + 1):
        print(f"\n--- Round {round_num}/{ROUNDS} ---")
        
        # 単語学習
        word_loss = model.train_phase('word', word_train_data, EPOCHS_PER_PHASE, word_to_idx, idx_to_word)
        word_prompt = [word_to_idx.get(w, 1) for w in ["人工", "知能"]]
        word_gen = model.generate('word', word_prompt, idx_to_word)
        print(f"  単語: Loss={word_loss:.3f} | 生成: {word_gen[:40]}...")
        
        # 文字学習
        char_loss = model.train_phase('char', char_train_data, EPOCHS_PER_PHASE, char_to_idx, idx_to_char)
        char_prompt = [char_to_idx.get(c, 1) for c in "人工知能"]
        char_gen = model.generate('char', char_prompt, idx_to_char)
        print(f"  文字: Loss={char_loss:.3f} | 生成: {char_gen[:40]}...")
    
    total_time = time.time() - start_time
    
    # 保存
    save_data = {
        'W_res': model.W_res,
        'word_vocab_size': word_vocab_size,
        'word_embedding': model.word_embedding,
        'word_W_spike': model.word_W_spike,
        'word_W_membrane': model.word_W_membrane,
        'word_bias': model.word_bias,
        'word_to_idx': word_to_idx,
        'char_vocab_size': char_vocab_size,
        'char_embedding': model.char_embedding,
        'char_W_spike': model.char_W_spike,
        'char_W_membrane': model.char_W_membrane,
        'char_bias': model.char_bias,
        'char_to_idx': char_to_idx,
    }
    
    save_path = CHECKPOINT_DIR / 'model_interleaved.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(save_data, f)
    
    print(f"\n{'='*60}")
    print(f"📊 交互学習完了！")
    print(f"{'='*60}")
    print(f"総時間: {total_time/60:.1f}分")
    print(f"モデル: {save_path}")


if __name__ == "__main__":
    main()
