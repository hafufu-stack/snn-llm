#!/usr/bin/env python3
"""
階層型SNN-LLM (Hierarchical Model)
単語レベル → 文字レベルの階層構造

構造:
  Level 1: 単語予測（次の単語を予測）
  Level 2: 文字生成（予測した単語を文字に展開）

長時間学習用（10000エポック）
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
RESULTS_DIR = Path(__file__).parent.parent / "results"

# パラメータ
INPUT_SCALE = 10.0
LEAK = 0.99
THRESHOLD = 1.0
HIDDEN_DIM = 2048
LR_WORD = 0.005
LR_CHAR = 0.003  # 文字は遅めに

# 長時間学習
TOTAL_EPOCHS = 10000


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


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║    🏛️ 階層型 SNN-LLM (10000エポック長時間学習)               ║
    ║                                                               ║
    ║    Level 1: 単語予測モデル                                    ║
    ║    Level 2: 単語→文字展開（各単語の文字を学習）               ║
    ║                                                               ║
    ║    夜間バッチ処理用                                           ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # コーパス読み込み
    print("コーパスをロード中...")
    with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
        corpus = [line.strip() for line in f if line.strip()]
    print(f"コーパス行数: {len(corpus)}")
    
    # === Level 1: 単語モデル ===
    print("\n" + "=" * 60)
    print("📚 Level 1: 単語モデル構築")
    print("=" * 60)
    
    word_freq = {}
    word_tokenized = []
    for line in corpus:
        tokens = tokenize_words(line)
        word_tokenized.append(tokens)
        for t in tokens:
            word_freq[t] = word_freq.get(t, 0) + 1
    
    word_to_idx = {'<PAD>': 0, '<UNK>': 1, '<EOS>': 2}
    for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:2000]:
        word_to_idx[word] = len(word_to_idx)
    idx_to_word = {v: k for k, v in word_to_idx.items()}
    word_vocab_size = len(word_to_idx)
    print(f"単語語彙サイズ: {word_vocab_size}")
    
    word_train_data = []
    for tokens in word_tokenized:
        ids = np.array([word_to_idx.get(t, 1) for t in tokens], dtype=np.int64)
        if len(ids) >= 3:
            word_train_data.append(ids)
    print(f"単語学習サンプル数: {len(word_train_data)}")
    
    # === Level 2: 単語→文字展開辞書 ===
    print("\n" + "=" * 60)
    print("🔤 Level 2: 単語→文字展開辞書構築")
    print("=" * 60)
    
    # 各単語の文字列を学習データとして準備
    word_to_chars = {}
    char_set = set()
    for word in word_to_idx.keys():
        if word not in ['<PAD>', '<UNK>', '<EOS>']:
            chars = list(word)
            word_to_chars[word] = chars
            char_set.update(chars)
    
    char_to_idx = {'<PAD>': 0, '<UNK>': 1, '<EOS>': 2, '<BOW>': 3}  # BOW = Beginning of Word
    for char in sorted(char_set):
        char_to_idx[char] = len(char_to_idx)
    idx_to_char = {v: k for k, v in char_to_idx.items()}
    char_vocab_size = len(char_to_idx)
    print(f"文字語彙サイズ: {char_vocab_size}")
    print(f"単語→文字ペア数: {len(word_to_chars)}")
    
    # 文字展開学習データ（単語ごとの文字列）
    char_train_data = []
    for word, chars in word_to_chars.items():
        # <BOW> + 文字列 + <EOS>
        ids = [3]  # <BOW>
        ids.extend([char_to_idx.get(c, 1) for c in chars])
        ids.append(2)  # <EOS>
        char_train_data.append((word_to_idx.get(word, 1), np.array(ids, dtype=np.int64)))
    print(f"文字学習サンプル数: {len(char_train_data)}")
    
    # === モデル初期化 ===
    print("\n" + "=" * 60)
    print("🧠 モデル初期化")
    print("=" * 60)
    
    # Level 1: 単語モデル
    word_embedding = np.random.randn(word_vocab_size, HIDDEN_DIM).astype(np.float32) * 0.01
    word_W_res = np.random.randn(HIDDEN_DIM, HIDDEN_DIM).astype(np.float32) * 0.01
    word_W_spike = np.random.randn(word_vocab_size, HIDDEN_DIM).astype(np.float32) * 0.01
    word_W_membrane = np.random.randn(word_vocab_size, HIDDEN_DIM).astype(np.float32) * 0.01
    word_bias = np.zeros(word_vocab_size, dtype=np.float32)
    word_embedding_scaled = word_embedding * INPUT_SCALE
    
    word_params = word_vocab_size * HIDDEN_DIM * 3 + HIDDEN_DIM * HIDDEN_DIM + word_vocab_size
    print(f"単語モデルパラメータ: {word_params:,}")
    
    # Level 2: 文字展開モデル（単語の埋め込みを入力として使う）
    char_embedding = np.random.randn(char_vocab_size, HIDDEN_DIM).astype(np.float32) * 0.01
    char_W_res = np.random.randn(HIDDEN_DIM, HIDDEN_DIM).astype(np.float32) * 0.01
    char_W_spike = np.random.randn(char_vocab_size, HIDDEN_DIM).astype(np.float32) * 0.01
    char_W_membrane = np.random.randn(char_vocab_size, HIDDEN_DIM).astype(np.float32) * 0.01
    char_bias = np.zeros(char_vocab_size, dtype=np.float32)
    # 単語埋め込み→文字モデルへの接続
    word_to_char_proj = np.random.randn(HIDDEN_DIM, HIDDEN_DIM).astype(np.float32) * 0.01
    char_embedding_scaled = char_embedding * INPUT_SCALE
    
    char_params = char_vocab_size * HIDDEN_DIM * 3 + HIDDEN_DIM * HIDDEN_DIM * 2 + char_vocab_size
    print(f"文字モデルパラメータ: {char_params:,}")
    print(f"総パラメータ: {word_params + char_params:,}")
    
    # JIT compile
    print("\nJITコンパイル中...")
    dummy_ids = np.array([0, 1, 2], dtype=np.int64)
    _ = parallel_forward(word_embedding_scaled, word_W_res, word_W_spike, word_W_membrane, word_bias,
                         dummy_ids, HIDDEN_DIM, word_vocab_size, LEAK)
    _ = parallel_forward(char_embedding_scaled, char_W_res, char_W_spike, char_W_membrane, char_bias,
                         dummy_ids, HIDDEN_DIM, char_vocab_size, LEAK)
    print("JIT完了!")
    
    # === 階層学習 ===
    print("\n" + "=" * 60)
    print(f"🏛️ 階層学習開始: {TOTAL_EPOCHS}エポック")
    print("=" * 60)
    
    start_time = time.time()
    word_losses = []
    char_losses = []
    batch_size = 50
    
    for epoch in range(1, TOTAL_EPOCHS + 1):
        # --- Level 1: 単語学習 ---
        word_loss = 0.0
        word_tokens = 0
        
        np.random.shuffle(word_train_data)
        for token_ids in word_train_data[:batch_size]:
            if len(token_ids) < 2:
                continue
            
            n_seq = len(token_ids) - 1
            logits, all_h, all_m = parallel_forward(
                word_embedding_scaled, word_W_res, word_W_spike, word_W_membrane, word_bias,
                token_ids[:-1], HIDDEN_DIM, word_vocab_size, LEAK
            )
            
            errors = np.zeros((n_seq, word_vocab_size), dtype=np.float32)
            for t in range(n_seq):
                probs = softmax_numba(logits[t])
                target = token_ids[t + 1]
                if target >= word_vocab_size:
                    target = 1
                word_loss += -np.log(probs[target] + 1e-10)
                word_tokens += 1
                errors[t] = probs.copy()
                errors[t, target] -= 1.0
            
            parallel_weight_update(word_W_spike, word_W_membrane, word_bias, errors, all_h, all_m, LR_WORD)
        
        avg_word_loss = word_loss / max(word_tokens, 1)
        word_losses.append(avg_word_loss)
        
        # --- Level 2: 文字展開学習 ---
        char_loss = 0.0
        char_tokens = 0
        
        np.random.shuffle(char_train_data)
        for word_id, char_ids in char_train_data[:batch_size]:
            if len(char_ids) < 2:
                continue
            
            n_seq = len(char_ids) - 1
            
            # 単語埋め込みを文字モデルの初期状態に注入
            word_emb = word_embedding[word_id]
            initial_h = word_to_char_proj @ word_emb
            
            logits, all_h, all_m = parallel_forward(
                char_embedding_scaled, char_W_res, char_W_spike, char_W_membrane, char_bias,
                char_ids[:-1], HIDDEN_DIM, char_vocab_size, LEAK
            )
            
            errors = np.zeros((n_seq, char_vocab_size), dtype=np.float32)
            for t in range(n_seq):
                probs = softmax_numba(logits[t])
                target = char_ids[t + 1]
                if target >= char_vocab_size:
                    target = 1
                char_loss += -np.log(probs[target] + 1e-10)
                char_tokens += 1
                errors[t] = probs.copy()
                errors[t, target] -= 1.0
            
            parallel_weight_update(char_W_spike, char_W_membrane, char_bias, errors, all_h, all_m, LR_CHAR)
        
        avg_char_loss = char_loss / max(char_tokens, 1)
        char_losses.append(avg_char_loss)
        
        # 進捗表示
        if epoch % 500 == 0:
            elapsed = time.time() - start_time
            speed = epoch / elapsed
            remaining = (TOTAL_EPOCHS - epoch) / speed
            
            # 生成テスト
            test_prompt = [word_to_idx.get(w, 1) for w in ["人工", "知能"]]
            h = np.zeros(HIDDEN_DIM, dtype=np.float32)
            m = np.zeros(HIDDEN_DIM, dtype=np.float32)
            
            generated_words = list(test_prompt)
            for _ in range(10):
                tid = generated_words[-1] if generated_words[-1] < word_vocab_size else 0
                x = word_embedding_scaled[tid]
                m = LEAK * m + x + word_W_res @ h
                h = np.where(m > THRESHOLD, 1.0, 0.0).astype(np.float32)
                m = np.where(m > THRESHOLD, 0.0, m)
                
                logits_gen = word_W_spike @ h + word_W_membrane @ m + word_bias
                probs = softmax_numba(logits_gen / 0.8)
                try:
                    next_word = np.random.choice(len(probs), p=probs)
                except:
                    next_word = np.argmax(logits_gen)
                generated_words.append(next_word)
            
            gen_text = "".join([idx_to_word.get(t, '?') for t in generated_words])
            
            print(f"Epoch {epoch:6d}/{TOTAL_EPOCHS} | 単語Loss: {avg_word_loss:.3f} | 文字Loss: {avg_char_loss:.3f} | "
                  f"速度: {speed:.1f} ep/s | 残り: {remaining/60:.0f}分")
            print(f"  生成: {gen_text[:60]}...")
        
        # チェックポイント
        if epoch % 2000 == 0:
            cp_data = {
                'epoch': epoch,
                'word_vocab_size': word_vocab_size,
                'char_vocab_size': char_vocab_size,
                'word_embedding': word_embedding,
                'word_W_res': word_W_res,
                'word_W_spike': word_W_spike,
                'word_W_membrane': word_W_membrane,
                'word_bias': word_bias,
                'char_embedding': char_embedding,
                'char_W_res': char_W_res,
                'char_W_spike': char_W_spike,
                'char_W_membrane': char_W_membrane,
                'char_bias': char_bias,
                'word_to_char_proj': word_to_char_proj,
                'word_to_idx': word_to_idx,
                'char_to_idx': char_to_idx,
                'word_losses': word_losses,
                'char_losses': char_losses,
            }
            cp_path = CHECKPOINT_DIR / f'model_hierarchical_{epoch}epochs.pkl'
            with open(cp_path, 'wb') as f:
                pickle.dump(cp_data, f)
            print(f"  💾 チェックポイント: {cp_path.name}")
    
    # 最終保存
    total_time = time.time() - start_time
    
    final_data = {
        'epoch': TOTAL_EPOCHS,
        'word_vocab_size': word_vocab_size,
        'char_vocab_size': char_vocab_size,
        'word_embedding': word_embedding,
        'word_W_res': word_W_res,
        'word_W_spike': word_W_spike,
        'word_W_membrane': word_W_membrane,
        'word_bias': word_bias,
        'char_embedding': char_embedding,
        'char_W_res': char_W_res,
        'char_W_spike': char_W_spike,
        'char_W_membrane': char_W_membrane,
        'char_bias': char_bias,
        'word_to_char_proj': word_to_char_proj,
        'word_to_idx': word_to_idx,
        'char_to_idx': char_to_idx,
        'word_losses': word_losses,
        'char_losses': char_losses,
    }
    
    final_path = CHECKPOINT_DIR / 'model_hierarchical_final.pkl'
    with open(final_path, 'wb') as f:
        pickle.dump(final_data, f)
    
    # 結果保存
    results = {
        'total_epochs': TOTAL_EPOCHS,
        'final_word_loss': float(word_losses[-1]),
        'final_char_loss': float(char_losses[-1]),
        'word_vocab_size': word_vocab_size,
        'char_vocab_size': char_vocab_size,
        'training_time_hours': total_time / 3600,
    }
    with open(RESULTS_DIR / 'hierarchical_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 60)
    print("📊 階層学習完了！")
    print("=" * 60)
    print(f"最終単語Loss: {word_losses[-1]:.3f}")
    print(f"最終文字Loss: {char_losses[-1]:.3f}")
    print(f"総時間: {total_time/3600:.1f}時間")
    print(f"モデル: {final_path}")


if __name__ == "__main__":
    main()
