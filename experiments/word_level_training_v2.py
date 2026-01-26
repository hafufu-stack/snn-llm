#!/usr/bin/env python3
"""
単語単位SNN-LLM学習 v2 (高速版)
事前トークン化で並列処理を最大化

修正点:
- 全データを事前にトークン化（MeCab処理は1回だけ）
- 学習ループはNumba並列のみ
- 24スレッド並列処理
"""

import numpy as np
import pickle
import json
import time
from pathlib import Path
from numba import njit, prange, set_num_threads

# MeCab系
try:
    import fugashi
    tagger = fugashi.Tagger()
    print("✅ fugashi (MeCab) ロード成功")
except Exception as e:
    print(f"❌ fugashi ロード失敗: {e}")
    exit(1)

set_num_threads(24)

# Paths
CORPUS_PATH = Path(__file__).parent.parent / "data" / "japanese_corpus.txt"
CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"
RESULTS_DIR = Path(__file__).parent.parent / "results"

# 最適パラメータ
INPUT_SCALE = 10.0
LEAK = 0.99
THRESHOLD = 1.0

# モデルパラメータ
HIDDEN_DIM = 2048
VOCAB_SIZE = 8000
EPOCHS = 5000
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
    """単語単位でトークン化"""
    words = []
    for word in tagger(text):
        surface = word.surface
        if surface.strip():
            words.append(surface)
    return words


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║    🔤 単語単位 SNN-LLM 学習 v2 (高速版)                       ║
    ║                                                               ║
    ║    修正: 事前トークン化で並列処理最大化                        ║
    ║    最適化: scale=10, leak=0.99                               ║
    ║    並列: 24スレッド                                           ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # コーパス読み込み
    print("コーパスをロード中...")
    with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
        corpus_lines = [line.strip() for line in f if line.strip()]
    print(f"コーパス行数: {len(corpus_lines)}")
    
    # === 事前トークン化（ここでMeCab処理を全部やる）===
    print("\n🔄 事前トークン化中（MeCab処理）...")
    start_tokenize = time.time()
    
    word_freq = {}
    all_tokenized = []
    
    for i, line in enumerate(corpus_lines[:1000]):  # 最初の1000行
        words = tokenize_words(line)
        all_tokenized.append(words)
        for w in words:
            word_freq[w] = word_freq.get(w, 0) + 1
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/1000行完了...")
    
    tokenize_time = time.time() - start_tokenize
    print(f"✅ 事前トークン化完了！ ({tokenize_time:.1f}秒)")
    
    # 語彙構築
    print("\n語彙を構築中...")
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    word_to_idx = {'<PAD>': 0, '<UNK>': 1, '<EOS>': 2}
    
    for word, freq in sorted_words[:VOCAB_SIZE - 3]:
        word_to_idx[word] = len(word_to_idx)
    
    idx_to_word = {v: k for k, v in word_to_idx.items()}
    actual_vocab_size = len(word_to_idx)
    print(f"語彙サイズ: {actual_vocab_size}")
    
    # 全データをID列に変換
    print("\nデータをID列に変換中...")
    train_data = []
    for words in all_tokenized:
        ids = np.array([word_to_idx.get(w, 1) for w in words], dtype=np.int64)
        if len(ids) >= 3:
            train_data.append(ids)
    print(f"学習サンプル数: {len(train_data)}")
    
    # モデル初期化（または既存モデルをロード）
    print("\nモデルを初期化中...")
    embedding = np.random.randn(VOCAB_SIZE, HIDDEN_DIM).astype(np.float32) * 0.01
    W_res = np.random.randn(HIDDEN_DIM, HIDDEN_DIM).astype(np.float32) * 0.01
    W_spike = np.random.randn(VOCAB_SIZE, HIDDEN_DIM).astype(np.float32) * 0.01
    W_membrane = np.random.randn(VOCAB_SIZE, HIDDEN_DIM).astype(np.float32) * 0.01
    bias = np.zeros(VOCAB_SIZE, dtype=np.float32)
    
    embedding_scaled = embedding * INPUT_SCALE
    
    params = VOCAB_SIZE * HIDDEN_DIM * 3 + HIDDEN_DIM * HIDDEN_DIM + VOCAB_SIZE
    print(f"パラメータ数: {params:,}")
    
    # JIT compile
    print("\nJITコンパイル中...")
    dummy_ids = np.array([0, 1, 2], dtype=np.int64)
    _ = parallel_forward(embedding_scaled, W_res, W_spike, W_membrane, bias,
                         dummy_ids, HIDDEN_DIM, VOCAB_SIZE, LEAK)
    
    dummy_errors = np.zeros((2, VOCAB_SIZE), dtype=np.float32)
    dummy_h = np.zeros((2, HIDDEN_DIM), dtype=np.float32)
    dummy_m = np.zeros((2, HIDDEN_DIM), dtype=np.float32)
    parallel_weight_update(W_spike.copy(), W_membrane.copy(), bias.copy(),
                           dummy_errors, dummy_h, dummy_m, LR)
    print("JIT完了!")
    
    # 学習開始
    print("\n" + "=" * 60)
    print(f"🧠 {EPOCHS}エポック学習開始！(事前トークン化済み)")
    print("=" * 60)
    
    start_time = time.time()
    losses = []
    
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        total_tokens = 0
        
        np.random.shuffle(train_data)
        
        for token_ids in train_data[:100]:  # バッチ
            if len(token_ids) < 2:
                continue
            
            n_seq = len(token_ids) - 1
            
            logits, all_h, all_m = parallel_forward(
                embedding_scaled, W_res, W_spike, W_membrane, bias,
                token_ids[:-1], HIDDEN_DIM, VOCAB_SIZE, LEAK
            )
            
            errors = np.zeros((n_seq, VOCAB_SIZE), dtype=np.float32)
            
            for t in range(n_seq):
                probs = softmax_numba(logits[t])
                target = token_ids[t + 1]
                if target >= VOCAB_SIZE:
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
            remaining = (EPOCHS - epoch) / speed
            
            # 生成テスト
            prompt_words = ["人工"]
            prompt_ids = [word_to_idx.get(w, 1) for w in prompt_words]
            
            h = np.zeros(HIDDEN_DIM, dtype=np.float32)
            m = np.zeros(HIDDEN_DIM, dtype=np.float32)
            
            generated = list(prompt_ids)
            for _ in range(10):
                tid = generated[-1] if generated[-1] < VOCAB_SIZE else 0
                x = embedding_scaled[tid]
                m = LEAK * m + x + W_res @ h
                h = np.where(m > THRESHOLD, 1.0, 0.0).astype(np.float32)
                m = np.where(m > THRESHOLD, 0.0, m)
                
                logits_gen = W_spike @ h + W_membrane @ m + bias
                probs = softmax_numba(logits_gen / 0.7)
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
                'vocab_size': VOCAB_SIZE,
                'hidden_dim': HIDDEN_DIM,
                'embedding': embedding,
                'W_res': W_res,
                'W_spike': W_spike,
                'W_membrane': W_membrane,
                'bias': bias,
                'word_to_idx': word_to_idx,
                'input_scale': INPUT_SCALE,
                'leak': LEAK,
            }
            cp_path = CHECKPOINT_DIR / f'model_word_v2_{epoch}epochs.pkl'
            with open(cp_path, 'wb') as f:
                pickle.dump(cp_data, f)
            print(f"  💾 チェックポイント: {cp_path.name}")
    
    # 最終保存
    final_data = {
        'vocab_size': VOCAB_SIZE,
        'hidden_dim': HIDDEN_DIM,
        'embedding': embedding,
        'W_res': W_res,
        'W_spike': W_spike,
        'W_membrane': W_membrane,
        'bias': bias,
        'word_to_idx': word_to_idx,
        'input_scale': INPUT_SCALE,
        'leak': LEAK,
    }
    
    final_path = CHECKPOINT_DIR / 'model_word_v2_final.pkl'
    with open(final_path, 'wb') as f:
        pickle.dump(final_data, f)
    
    tokenizer_path = CHECKPOINT_DIR / 'tokenizer_word_v2.json'
    with open(tokenizer_path, 'w', encoding='utf-8') as f:
        json.dump({'word_to_idx': word_to_idx}, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 60)
    print("📊 学習完了!")
    print("=" * 60)
    print(f"最終Loss: {losses[-1]:.3f}")
    print(f"総時間: {(time.time() - start_time)/3600:.1f}時間")
    print(f"モデル: {final_path}")


if __name__ == "__main__":
    main()
