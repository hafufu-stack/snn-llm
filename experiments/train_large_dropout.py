#!/usr/bin/env python3
"""
大規模コーパス + 強Dropout 学習
多様性向上実験

改善点:
1. 10000行のユニークなコーパス
2. Dropout 25%（過学習防止）
3. 単語単位学習
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
CORPUS_PATH = Path(__file__).parent.parent / "data" / "japanese_corpus_large.txt"
CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"
RESULTS_DIR = Path(__file__).parent.parent / "results"

# パラメータ
INPUT_SCALE = 10.0
LEAK = 0.99
THRESHOLD = 1.0
HIDDEN_DIM = 2048
LR = 0.005
DROPOUT_RATE = 0.25  # 強いDropout！
EPOCHS = 3000


@njit(cache=True)
def softmax_numba(x):
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / (np.sum(exp_x) + 1e-10)


@njit(cache=True)
def apply_dropout(x, rate, training=True):
    """Dropoutを適用（学習時のみ）"""
    if not training or rate <= 0:
        return x
    mask = np.random.random(len(x)) > rate
    return x * mask / (1 - rate)


@njit(parallel=True, cache=True)
def parallel_forward_dropout(embedding_scaled, W_res, W_spike, W_membrane, bias,
                             token_ids, hidden_dim, vocab_size, leak, dropout_rate, training):
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
        
        # Dropout適用
        if training and dropout_rate > 0:
            mask = np.random.random(hidden_dim) > dropout_rate
            h = h * mask.astype(np.float32) / np.float32(1 - dropout_rate)
        
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


def main():
    print(f"""
    ╔═══════════════════════════════════════════════════════════════╗
    ║    🚀 大規模コーパス + 強Dropout 学習                         ║
    ║                                                               ║
    ║    コーパス: 10000行ユニーク                                  ║
    ║    Dropout: {DROPOUT_RATE*100:.0f}%                                           ║
    ║    エポック: {EPOCHS}                                           ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # コーパス読み込み
    print("コーパスをロード中...")
    with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
        corpus = [line.strip() for line in f if line.strip()]
    print(f"コーパス行数: {len(corpus)}")
    
    # トークン化
    print("\n🔄 単語トークン化中...")
    word_freq = {}
    all_tokenized = []
    
    for line in corpus:
        tokens = tokenize_words(line)
        all_tokenized.append(tokens)
        for t in tokens:
            word_freq[t] = word_freq.get(t, 0) + 1
    
    # ユニークな単語数カウント
    print(f"✅ 総単語種類: {len(word_freq)}")
    
    # 上位3000単語のみ使用
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    word_to_idx = {'<PAD>': 0, '<UNK>': 1, '<EOS>': 2}
    for word, freq in sorted_words[:3000]:
        word_to_idx[word] = len(word_to_idx)
    
    idx_to_word = {v: k for k, v in word_to_idx.items()}
    vocab_size = len(word_to_idx)
    print(f"語彙サイズ: {vocab_size}")
    
    # 学習データ変換
    train_data = []
    for tokens in all_tokenized:
        ids = np.array([word_to_idx.get(t, 1) for t in tokens], dtype=np.int64)
        if len(ids) >= 3:
            train_data.append(ids)
    
    print(f"学習サンプル数: {len(train_data)}")
    
    # モデル初期化
    print("\nモデル初期化...")
    embedding = np.random.randn(vocab_size, HIDDEN_DIM).astype(np.float32) * 0.01
    W_res = np.random.randn(HIDDEN_DIM, HIDDEN_DIM).astype(np.float32) * 0.01
    W_spike = np.random.randn(vocab_size, HIDDEN_DIM).astype(np.float32) * 0.01
    W_membrane = np.random.randn(vocab_size, HIDDEN_DIM).astype(np.float32) * 0.01
    bias = np.zeros(vocab_size, dtype=np.float32)
    
    embedding_scaled = embedding * INPUT_SCALE
    
    params = vocab_size * HIDDEN_DIM * 3 + HIDDEN_DIM * HIDDEN_DIM + vocab_size
    print(f"パラメータ数: {params:,}")
    print(f"Dropout率: {DROPOUT_RATE*100:.0f}%")
    
    # JIT compile
    print("\nJITコンパイル中...")
    dummy_ids = np.array([0, 1, 2], dtype=np.int64)
    _ = parallel_forward_dropout(embedding_scaled, W_res, W_spike, W_membrane, bias,
                                 dummy_ids, HIDDEN_DIM, vocab_size, LEAK, DROPOUT_RATE, True)
    print("JIT完了!")
    
    # 学習
    print(f"\n{'='*60}")
    print(f"🧠 {EPOCHS}エポック学習開始！")
    print(f"{'='*60}")
    
    start_time = time.time()
    losses = []
    batch_size = 100  # より多く
    
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        total_tokens = 0
        
        np.random.shuffle(train_data)
        
        for token_ids in train_data[:batch_size]:
            if len(token_ids) < 2:
                continue
            
            n_seq = len(token_ids) - 1
            
            # 学習時はDropout有効
            logits, all_h, all_m = parallel_forward_dropout(
                embedding_scaled, W_res, W_spike, W_membrane, bias,
                token_ids[:-1], HIDDEN_DIM, vocab_size, LEAK, DROPOUT_RATE, True
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
        
        # 進捗表示
        if epoch % 100 == 0:
            elapsed = time.time() - start_time
            speed = epoch / elapsed
            remaining = (EPOCHS - epoch) / speed
            
            # 生成テスト（Dropoutなし）
            prompt_words = tokenize_words("人工知能")
            prompt_ids = [word_to_idx.get(w, 1) for w in prompt_words]
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
                probs = softmax_numba(logits_gen / 1.0)  # 高temperature
                try:
                    next_token = np.random.choice(len(probs), p=probs)
                except:
                    next_token = np.argmax(logits_gen)
                generated.append(next_token)
            
            gen_text = "".join([idx_to_word.get(t, '?') for t in generated])
            
            print(f"Epoch {epoch:5d}/{EPOCHS} | Loss: {avg_loss:.3f} | "
                  f"速度: {speed:.1f} ep/s | 残り: {remaining/60:.0f}分")
            print(f"  生成: {gen_text[:50]}...")
        
        # チェックポイント
        if epoch % 1000 == 0:
            cp_data = {
                'epoch': epoch,
                'vocab_size': vocab_size,
                'embedding': embedding,
                'W_res': W_res,
                'W_spike': W_spike,
                'W_membrane': W_membrane,
                'bias': bias,
                'word_to_idx': word_to_idx,
                'losses': losses,
                'dropout_rate': DROPOUT_RATE,
            }
            cp_path = CHECKPOINT_DIR / f'model_large_dropout_{epoch}epochs.pkl'
            with open(cp_path, 'wb') as f:
                pickle.dump(cp_data, f)
            print(f"  💾 チェックポイント: {cp_path.name}")
    
    # 最終保存
    total_time = time.time() - start_time
    
    final_data = {
        'epoch': EPOCHS,
        'vocab_size': vocab_size,
        'embedding': embedding,
        'W_res': W_res,
        'W_spike': W_spike,
        'W_membrane': W_membrane,
        'bias': bias,
        'word_to_idx': word_to_idx,
        'losses': losses,
        'dropout_rate': DROPOUT_RATE,
    }
    
    final_path = CHECKPOINT_DIR / 'model_large_dropout_final.pkl'
    with open(final_path, 'wb') as f:
        pickle.dump(final_data, f)
    
    # 結果保存
    results = {
        'epochs': EPOCHS,
        'final_loss': float(losses[-1]),
        'vocab_size': vocab_size,
        'corpus_size': len(train_data),
        'dropout_rate': DROPOUT_RATE,
        'training_time_hours': total_time / 3600,
    }
    with open(RESULTS_DIR / 'large_dropout_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"📊 学習完了！")
    print(f"{'='*60}")
    print(f"最終Loss: {losses[-1]:.3f}")
    print(f"総時間: {total_time/60:.1f}分")
    print(f"モデル: {final_path}")


if __name__ == "__main__":
    main()
