#!/usr/bin/env python3
"""
単語単位SNN-LLM学習 v3 (高速版 - 動的語彙サイズ)
モデルサイズを実際の語彙サイズに合わせて最適化

修正点:
- 語彙サイズを動的に設定（無駄な計算を削減）
- 事前トークン化で並列処理最大化
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
MAX_VOCAB_SIZE = 4000  # 最大語彙サイズ（実際は動的）
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
    ║    🔤 単語単位 SNN-LLM 学習 v3 (動的語彙サイズ版)             ║
    ║                                                               ║
    ║    修正: モデルサイズを実際の語彙に合わせて最適化             ║
    ║    最適化: scale=10, leak=0.99                               ║
    ║    並列: 24スレッド                                           ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # コーパス読み込み
    print("コーパスをロード中...")
    with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
        corpus_lines = [line.strip() for line in f if line.strip()]
    print(f"コーパス行数: {len(corpus_lines)}")
    
    # === 事前トークン化 ===
    print("\n🔄 事前トークン化中（MeCab処理）...")
    start_tokenize = time.time()
    
    word_freq = {}
    all_tokenized = []
    
    # 全コーパスを使用
    for i, line in enumerate(corpus_lines):
        words = tokenize_words(line)
        all_tokenized.append(words)
        for w in words:
            word_freq[w] = word_freq.get(w, 0) + 1
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(corpus_lines)}行完了...")
    
    tokenize_time = time.time() - start_tokenize
    print(f"✅ 事前トークン化完了！ ({tokenize_time:.1f}秒)")
    print(f"総単語種類: {len(word_freq)}")
    
    # 語彙構築（動的サイズ）
    print("\n語彙を構築中...")
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    # 実際の語彙サイズを決定（最大4000、または実際の単語数+3）
    actual_vocab_size = min(MAX_VOCAB_SIZE, len(sorted_words) + 3)
    
    word_to_idx = {'<PAD>': 0, '<UNK>': 1, '<EOS>': 2}
    
    for word, freq in sorted_words[:actual_vocab_size - 3]:
        word_to_idx[word] = len(word_to_idx)
    
    idx_to_word = {v: k for k, v in word_to_idx.items()}
    vocab_size = len(word_to_idx)
    print(f"語彙サイズ: {vocab_size}")
    
    # 全データをID列に変換
    print("\nデータをID列に変換中...")
    train_data = []
    for words in all_tokenized:
        ids = np.array([word_to_idx.get(w, 1) for w in words], dtype=np.int64)
        if len(ids) >= 3:
            train_data.append(ids)
    print(f"学習サンプル数: {len(train_data)}")
    
    # モデル初期化（動的サイズ）
    print(f"\nモデルを初期化中（語彙サイズ: {vocab_size}）...")
    embedding = np.random.randn(vocab_size, HIDDEN_DIM).astype(np.float32) * 0.01
    W_res = np.random.randn(HIDDEN_DIM, HIDDEN_DIM).astype(np.float32) * 0.01
    W_spike = np.random.randn(vocab_size, HIDDEN_DIM).astype(np.float32) * 0.01
    W_membrane = np.random.randn(vocab_size, HIDDEN_DIM).astype(np.float32) * 0.01
    bias = np.zeros(vocab_size, dtype=np.float32)
    
    embedding_scaled = embedding * INPUT_SCALE
    
    params = vocab_size * HIDDEN_DIM * 3 + HIDDEN_DIM * HIDDEN_DIM + vocab_size
    print(f"パラメータ数: {params:,}")
    
    # JIT compile
    print("\nJITコンパイル中...")
    dummy_ids = np.array([0, 1, 2], dtype=np.int64)
    _ = parallel_forward(embedding_scaled, W_res, W_spike, W_membrane, bias,
                         dummy_ids, HIDDEN_DIM, vocab_size, LEAK)
    
    dummy_errors = np.zeros((2, vocab_size), dtype=np.float32)
    dummy_h = np.zeros((2, HIDDEN_DIM), dtype=np.float32)
    dummy_m = np.zeros((2, HIDDEN_DIM), dtype=np.float32)
    parallel_weight_update(W_spike.copy(), W_membrane.copy(), bias.copy(),
                           dummy_errors, dummy_h, dummy_m, LR)
    print("JIT完了!")
    
    # 学習開始
    print("\n" + "=" * 60)
    print(f"🧠 {EPOCHS}エポック学習開始！")
    print("=" * 60)
    
    start_time = time.time()
    losses = []
    batch_size = 50  # 各エポックでのサンプル数
    
    for epoch in range(1, EPOCHS + 1):
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
            remaining = (EPOCHS - epoch) / speed
            
            # 生成テスト
            prompt_words = ["人工", "知能"]
            prompt_ids = [word_to_idx.get(w, 1) for w in prompt_words]
            
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
                probs = softmax_numba(logits_gen / 0.7)
                try:
                    next_token = np.random.choice(len(probs), p=probs)
                except:
                    next_token = np.argmax(logits_gen)
                generated.append(next_token)
            
            gen_text = "".join([idx_to_word.get(t, '?') for t in generated])
            
            print(f"Epoch {epoch:5d}/{EPOCHS} | Loss: {avg_loss:.3f} | "
                  f"速度: {speed:.1f} ep/s | 残り: {remaining/60:.0f}分")
            print(f"  生成: {gen_text[:60]}...")
        
        # チェックポイント
        if epoch % 1000 == 0:
            cp_data = {
                'vocab_size': vocab_size,
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
            cp_path = CHECKPOINT_DIR / f'model_word_v3_{epoch}epochs.pkl'
            with open(cp_path, 'wb') as f:
                pickle.dump(cp_data, f)
            print(f"  💾 チェックポイント: {cp_path.name}")
    
    # 最終保存
    final_data = {
        'vocab_size': vocab_size,
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
    
    final_path = CHECKPOINT_DIR / 'model_word_v3_final.pkl'
    with open(final_path, 'wb') as f:
        pickle.dump(final_data, f)
    
    tokenizer_path = CHECKPOINT_DIR / 'tokenizer_word_v3.json'
    with open(tokenizer_path, 'w', encoding='utf-8') as f:
        json.dump({'word_to_idx': word_to_idx}, f, ensure_ascii=False, indent=2)
    
    # 結果保存
    results = {
        'epochs': EPOCHS,
        'final_loss': float(losses[-1]),
        'vocab_size': vocab_size,
        'params': params,
        'training_time_sec': time.time() - start_time,
    }
    with open(RESULTS_DIR / 'word_v3_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 60)
    print("📊 学習完了!")
    print("=" * 60)
    print(f"最終Loss: {losses[-1]:.3f}")
    print(f"総時間: {(time.time() - start_time)/3600:.1f}時間")
    print(f"モデル: {final_path}")


if __name__ == "__main__":
    main()
