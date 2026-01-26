#!/usr/bin/env python3
"""
単語単位SNN-LLM学習
文字単位から単語単位に変更して汎化性能を向上

変更点:
- fugashi (MeCab) で形態素解析
- 単語単位のトークン化
- 最適化パラメータ (scale=10, leak=0.99) 適用
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
VOCAB_SIZE = 8000  # 単語なので増やす
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
        
        for i in prange(vocab_size):
            all_logits[t, i] = bias[i]
            for j in range(hidden_dim):
                all_logits[t, i] += W_spike[i, j] * h[j]
                all_logits[t, i] += W_membrane[i, j] * h[j]
    
    return all_logits, all_h


def tokenize_words(text):
    """単語単位でトークン化"""
    words = []
    for word in tagger(text):
        surface = word.surface
        if surface.strip():
            words.append(surface)
    return words


def build_vocab(corpus_lines, max_vocab=VOCAB_SIZE):
    """コーパスから語彙を構築"""
    word_freq = {}
    for line in corpus_lines:
        words = tokenize_words(line)
        for w in words:
            word_freq[w] = word_freq.get(w, 0) + 1
    
    # 頻度順にソート
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    # 特殊トークン
    word_to_idx = {'<PAD>': 0, '<UNK>': 1, '<EOS>': 2}
    
    for word, freq in sorted_words[:max_vocab - 3]:
        word_to_idx[word] = len(word_to_idx)
    
    return word_to_idx


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║    🔤 単語単位 SNN-LLM 学習                                   ║
    ║                                                               ║
    ║    変更: 文字単位 → 単語単位 (MeCab)                          ║
    ║    最適化: scale=10, leak=0.99                               ║
    ║    並列: 24スレッド                                           ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # コーパス読み込み
    print("コーパスをロード中...")
    with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
        corpus_lines = [line.strip() for line in f if line.strip()]
    print(f"コーパス行数: {len(corpus_lines)}")
    
    # 語彙構築
    print("\n語彙を構築中...")
    word_to_idx = build_vocab(corpus_lines, max_vocab=VOCAB_SIZE)
    idx_to_word = {v: k for k, v in word_to_idx.items()}
    actual_vocab_size = len(word_to_idx)
    print(f"語彙サイズ: {actual_vocab_size}")
    
    # サンプル表示
    sample_text = corpus_lines[0][:50]
    sample_words = tokenize_words(sample_text)
    print(f"サンプル: 「{sample_text}」")
    print(f"単語分割: {sample_words[:10]}...")
    
    # モデル初期化
    print("\nモデルを初期化中...")
    embedding = np.random.randn(VOCAB_SIZE, HIDDEN_DIM).astype(np.float32) * 0.01
    W_res = np.random.randn(HIDDEN_DIM, HIDDEN_DIM).astype(np.float32) * 0.01
    W_spike = np.random.randn(VOCAB_SIZE, HIDDEN_DIM).astype(np.float32) * 0.01
    W_membrane = np.random.randn(VOCAB_SIZE, HIDDEN_DIM).astype(np.float32) * 0.01
    bias = np.zeros(VOCAB_SIZE, dtype=np.float32)
    
    embedding_scaled = embedding * INPUT_SCALE
    
    print(f"パラメータ数: {(VOCAB_SIZE * HIDDEN_DIM * 3 + HIDDEN_DIM * HIDDEN_DIM + VOCAB_SIZE):,}")
    
    # JIT compile
    print("\nJITコンパイル中...")
    dummy_ids = np.array([0, 1, 2], dtype=np.int64)
    _ = parallel_forward(embedding_scaled, W_res, W_spike, W_membrane, bias,
                         dummy_ids, HIDDEN_DIM, VOCAB_SIZE, LEAK)
    print("JIT完了!")
    
    # データ準備
    def text_to_ids(text):
        words = tokenize_words(text)
        return np.array([word_to_idx.get(w, 1) for w in words], dtype=np.int64)
    
    # 学習データ準備（最初の1000行を使用）
    train_data = []
    for line in corpus_lines[:1000]:
        ids = text_to_ids(line)
        if len(ids) >= 3:
            train_data.append(ids)
    print(f"学習サンプル数: {len(train_data)}")
    
    # 学習開始
    print("\n" + "=" * 60)
    print(f"🧠 {EPOCHS}エポック学習開始！")
    print("=" * 60)
    
    start_time = time.time()
    losses = []
    
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        total_tokens = 0
        
        np.random.shuffle(train_data)
        
        for token_ids in train_data[:100]:  # バッチ制限
            if len(token_ids) < 2:
                continue
            
            logits, _ = parallel_forward(
                embedding_scaled, W_res, W_spike, W_membrane, bias,
                token_ids[:-1], HIDDEN_DIM, VOCAB_SIZE, LEAK
            )
            
            for t in range(len(token_ids) - 1):
                probs = softmax_numba(logits[t])
                target = token_ids[t + 1]
                if target >= VOCAB_SIZE:
                    target = 1
                
                loss = -np.log(probs[target] + 1e-10)
                epoch_loss += loss
                total_tokens += 1
                
                error = probs.copy()
                error[target] -= 1.0
                
                W_spike -= LR * np.outer(error, np.ones(HIDDEN_DIM, dtype=np.float32))
                W_membrane -= LR * np.outer(error, np.ones(HIDDEN_DIM, dtype=np.float32))
                bias -= LR * error
        
        avg_loss = epoch_loss / max(total_tokens, 1)
        losses.append(avg_loss)
        
        if epoch % 100 == 0:
            elapsed = time.time() - start_time
            speed = epoch / elapsed
            remaining = (EPOCHS - epoch) / speed
            
            # 生成テスト
            prompt = "人工"
            prompt_ids = text_to_ids(prompt)
            if len(prompt_ids) == 0:
                prompt_ids = np.array([0], dtype=np.int64)
            
            h = np.zeros(HIDDEN_DIM, dtype=np.float32)
            m = np.zeros(HIDDEN_DIM, dtype=np.float32)
            
            generated = list(prompt_ids)
            for _ in range(10):
                tid = generated[-1] if generated[-1] < VOCAB_SIZE else 0
                x = embedding_scaled[tid]
                m = LEAK * m + x + W_res @ h
                h = np.where(m > THRESHOLD, 1.0, 0.0).astype(np.float32)
                m = np.where(m > THRESHOLD, 0.0, m)
                
                logits_gen = W_spike @ h + W_membrane @ h + bias
                probs = softmax_numba(logits_gen / 0.7)
                try:
                    next_token = np.random.choice(len(probs), p=probs)
                except:
                    next_token = np.argmax(logits_gen)
                generated.append(next_token)
            
            gen_text = "".join([idx_to_word.get(t, '?') for t in generated])
            
            print(f"Epoch {epoch:5d}/{EPOCHS} | Loss: {avg_loss:.3f} | "
                  f"速度: {speed:.1f} ep/s | 残り: {remaining/60:.0f}分")
            print(f"  生成: {gen_text[:40]}...")
        
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
            cp_path = CHECKPOINT_DIR / f'model_word_level_{epoch}epochs.pkl'
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
    
    final_path = CHECKPOINT_DIR / 'model_word_level_final.pkl'
    with open(final_path, 'wb') as f:
        pickle.dump(final_data, f)
    
    # トークナイザー保存
    tokenizer_path = CHECKPOINT_DIR / 'tokenizer_word_level.json'
    with open(tokenizer_path, 'w', encoding='utf-8') as f:
        json.dump({'word_to_idx': word_to_idx}, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 60)
    print("📊 学習完了!")
    print("=" * 60)
    print(f"最終Loss: {losses[-1]:.3f}")
    print(f"総時間: {(time.time() - start_time)/3600:.1f}時間")
    print(f"モデル: {final_path}")
    print(f"トークナイザー: {tokenizer_path}")


if __name__ == "__main__":
    main()
