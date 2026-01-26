#!/usr/bin/env python3
"""
2段階学習 (Transfer Learning)
人間の言語習得を模倣

Phase 1: 単語レベルで意味構造を学習（赤ちゃんが話せるようになる）
Phase 2: 文字レベルで表記を学習（小学生がひらがな・漢字を習う）

重み転移: W_res（リザバー接続）を共有
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

# 最適パラメータ
INPUT_SCALE = 10.0
LEAK = 0.99
THRESHOLD = 1.0
HIDDEN_DIM = 2048
LR = 0.005


@njit(cache=True)
def softmax_numba(x):
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / (np.sum(exp_x) + 1e-10)


@njit(parallel=True, cache=True)
def parallel_forward(embedding_scaled, W_res, W_spike, W_membrane, bias,
                     token_ids, hidden_dim, vocab_size, leak):
    """並列フォワードパス"""
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
    """並列重み更新"""
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


def train_phase(corpus, tokenize_fn, epochs, vocab_size_max, phase_name, W_res_init=None):
    """汎用学習関数"""
    print(f"\n{'='*60}")
    print(f"🎓 {phase_name}")
    print(f"{'='*60}")
    
    # トークン化
    print("トークン化中...")
    word_freq = {}
    all_tokenized = []
    
    for line in corpus:
        tokens = tokenize_fn(line)
        all_tokenized.append(tokens)
        for t in tokens:
            word_freq[t] = word_freq.get(t, 0) + 1
    
    # 語彙構築
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    vocab_size = min(vocab_size_max, len(sorted_words) + 3)
    
    token_to_idx = {'<PAD>': 0, '<UNK>': 1, '<EOS>': 2}
    for word, freq in sorted_words[:vocab_size - 3]:
        token_to_idx[word] = len(token_to_idx)
    
    idx_to_token = {v: k for k, v in token_to_idx.items()}
    vocab_size = len(token_to_idx)
    print(f"語彙サイズ: {vocab_size}")
    
    # データ変換
    train_data = []
    for tokens in all_tokenized:
        ids = np.array([token_to_idx.get(t, 1) for t in tokens], dtype=np.int64)
        if len(ids) >= 3:
            train_data.append(ids)
    print(f"学習サンプル数: {len(train_data)}")
    
    # モデル初期化
    embedding = np.random.randn(vocab_size, HIDDEN_DIM).astype(np.float32) * 0.01
    
    if W_res_init is not None:
        print("🔄 W_res を前フェーズから転移！")
        W_res = W_res_init.copy()
    else:
        W_res = np.random.randn(HIDDEN_DIM, HIDDEN_DIM).astype(np.float32) * 0.01
    
    W_spike = np.random.randn(vocab_size, HIDDEN_DIM).astype(np.float32) * 0.01
    W_membrane = np.random.randn(vocab_size, HIDDEN_DIM).astype(np.float32) * 0.01
    bias = np.zeros(vocab_size, dtype=np.float32)
    
    embedding_scaled = embedding * INPUT_SCALE
    
    params = vocab_size * HIDDEN_DIM * 3 + HIDDEN_DIM * HIDDEN_DIM + vocab_size
    print(f"パラメータ数: {params:,}")
    
    # JIT compile
    print("JITコンパイル中...")
    dummy_ids = np.array([0, 1, 2], dtype=np.int64)
    _ = parallel_forward(embedding_scaled, W_res, W_spike, W_membrane, bias,
                         dummy_ids, HIDDEN_DIM, vocab_size, LEAK)
    print("JIT完了!")
    
    # 学習
    print(f"\n🧠 {epochs}エポック学習開始！")
    start_time = time.time()
    losses = []
    batch_size = 50
    
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        total_tokens = 0
        
        np.random.shuffle(train_data)
        
        for token_ids in train_data[:batch_size]:
            if len(token_ids) < 2:
                continue
            
            n_seq = len(token_ids) - 1
            
            logits, all_h, all_m = parallel_forward(
                embedding_scaled, W_res, W_spike, W_membrane, bias,
                token_ids[:-1], HIDDEN_DIM, vocab_size, LEAK
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
        
        if epoch % 100 == 0:
            elapsed = time.time() - start_time
            speed = epoch / elapsed
            remaining = (epochs - epoch) / speed
            
            # 生成テスト
            test_tokens = tokenize_fn("人工知能")
            prompt_ids = [token_to_idx.get(t, 1) for t in test_tokens]
            if len(prompt_ids) == 0:
                prompt_ids = [0]
            
            h = np.zeros(HIDDEN_DIM, dtype=np.float32)
            m = np.zeros(HIDDEN_DIM, dtype=np.float32)
            
            generated = list(prompt_ids)
            for _ in range(15):
                tid = generated[-1] if generated[-1] < vocab_size else 0
                x = embedding_scaled[tid]
                m = LEAK * m + x + W_res @ h
                h = np.where(m > THRESHOLD, 1.0, 0.0).astype(np.float32)
                m = np.where(m > THRESHOLD, 0.0, m)
                
                logits_gen = W_spike @ h + W_membrane @ m + bias
                probs = softmax_numba(logits_gen / 0.8)
                try:
                    next_token = np.random.choice(len(probs), p=probs)
                except:
                    next_token = np.argmax(logits_gen)
                generated.append(next_token)
            
            gen_text = "".join([idx_to_token.get(t, '?') for t in generated])
            
            print(f"Epoch {epoch:5d}/{epochs} | Loss: {avg_loss:.3f} | "
                  f"速度: {speed:.1f} ep/s | 残り: {remaining/60:.0f}分")
            print(f"  生成: {gen_text[:50]}...")
    
    total_time = time.time() - start_time
    print(f"\n✅ {phase_name} 完了！Loss: {losses[-1]:.3f}, 時間: {total_time/60:.1f}分")
    
    return {
        'W_res': W_res,
        'W_spike': W_spike,
        'W_membrane': W_membrane,
        'embedding': embedding,
        'bias': bias,
        'token_to_idx': token_to_idx,
        'vocab_size': vocab_size,
        'final_loss': losses[-1],
    }


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║    🎓 2段階学習 (Transfer Learning)                           ║
    ║                                                               ║
    ║    Phase 1: 単語レベル学習（意味構造）                        ║
    ║    Phase 2: 文字レベル学習（表記、W_res転移）                 ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # コーパス読み込み
    print("コーパスをロード中...")
    with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
        corpus = [line.strip() for line in f if line.strip()]
    print(f"コーパス行数: {len(corpus)}")
    
    # ============================================
    # Phase 1: 単語レベル学習
    # ============================================
    phase1_result = train_phase(
        corpus=corpus,
        tokenize_fn=tokenize_words,
        epochs=1000,
        vocab_size_max=4000,
        phase_name="Phase 1: 単語レベル学習",
        W_res_init=None
    )
    
    # Phase 1 保存
    phase1_path = CHECKPOINT_DIR / 'model_phase1_word.pkl'
    with open(phase1_path, 'wb') as f:
        pickle.dump(phase1_result, f)
    print(f"💾 Phase 1 保存: {phase1_path.name}")
    
    # ============================================
    # Phase 2: 文字レベル学習（W_res転移）
    # ============================================
    phase2_result = train_phase(
        corpus=corpus,
        tokenize_fn=tokenize_chars,
        epochs=1000,
        vocab_size_max=4000,
        phase_name="Phase 2: 文字レベル学習（W_res転移）",
        W_res_init=phase1_result['W_res']  # 🔑 転移！
    )
    
    # Phase 2 保存
    phase2_path = CHECKPOINT_DIR / 'model_phase2_char.pkl'
    with open(phase2_path, 'wb') as f:
        pickle.dump(phase2_result, f)
    print(f"💾 Phase 2 保存: {phase2_path.name}")
    
    # サマリー
    print("\n" + "=" * 60)
    print("📊 2段階学習完了！")
    print("=" * 60)
    print(f"Phase 1 (単語): Loss {phase1_result['final_loss']:.3f}")
    print(f"Phase 2 (文字): Loss {phase2_result['final_loss']:.3f}")


if __name__ == "__main__":
    main()
