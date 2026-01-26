#!/usr/bin/env python3
"""
1文丸暗記テスト
Deep Thinkの提案: 「吾輩は猫である」だけを1000回学習させて再現できるか

これができない = SNNの構造に問題
これができる = データの問題（解決しやすい）
"""

import numpy as np
import pickle
import json
from pathlib import Path
from numba import njit, prange, set_num_threads

set_num_threads(24)

MODEL_PATH = Path(__file__).parent.parent / "checkpoints" / "model_v4_parallel_final.pkl"
TOKENIZER_PATH = Path(__file__).parent.parent / "checkpoints" / "tokenizer_v4_parallel.json"

# テスト文
TEST_SENTENCE = "吾輩は猫である。"


@njit(cache=True)
def softmax_numba(x):
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / (np.sum(exp_x) + 1e-10)


@njit(parallel=True, cache=True)
def parallel_forward(embedding, W_res, W_spike, W_membrane, bias, 
                     token_ids, hidden_dim, vocab_size):
    n_tokens = len(token_ids)
    h = np.zeros(hidden_dim, dtype=np.float32)
    m = np.zeros(hidden_dim, dtype=np.float32)
    
    all_logits = np.zeros((n_tokens, vocab_size), dtype=np.float32)
    
    for t in range(n_tokens):
        tid = token_ids[t]
        x = embedding[tid]
        
        for i in prange(hidden_dim):
            m[i] = 0.9 * m[i] + x[i]
            for j in range(hidden_dim):
                m[i] += W_res[i, j] * h[j]
        
        for i in prange(hidden_dim):
            if m[i] > 1.0:
                h[i] = 1.0
                m[i] = 0.0
            else:
                h[i] = 0.0
        
        for i in prange(vocab_size):
            all_logits[t, i] = 0.0
            for j in range(hidden_dim):
                all_logits[t, i] += W_spike[i, j] * h[j]
                all_logits[t, i] += W_membrane[i, j] * h[j]
            all_logits[t, i] += bias[i]
    
    return all_logits, h, m


def main():
    print("=" * 60)
    print("🧪 1文丸暗記テスト")
    print(f"   テスト文: 「{TEST_SENTENCE}」")
    print("=" * 60)
    
    # Load fresh model (not the overtrained one)
    print("\n新しいモデルをロード中...")
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
    
    with open(TOKENIZER_PATH, 'r', encoding='utf-8') as f:
        tokenizer_data = json.load(f)
    
    token_to_idx = tokenizer_data['token_to_idx']
    idx_to_token = {v: k for k, v in token_to_idx.items()}
    vocab_size = model_data['vocab_size']
    hidden_dim = model_data['hidden_dim']
    
    embedding = model_data['embedding'].astype(np.float32)
    W_res = model_data['W_res'].astype(np.float32)
    W_spike = model_data['W_spike'].astype(np.float32)
    W_membrane = model_data['W_membrane'].astype(np.float32)
    bias = model_data['bias'].astype(np.float32)
    
    print(f"語彙サイズ: {vocab_size}")
    print(f"隠れ次元: {hidden_dim}")
    
    # Tokenize
    tokens = np.array([token_to_idx.get(c, 0) for c in TEST_SENTENCE], dtype=np.int64)
    print(f"トークン数: {len(tokens)}")
    print(f"トークン: {[idx_to_token.get(t, '?') for t in tokens]}")
    
    # JIT compile
    print("\nJITコンパイル中...")
    _ = parallel_forward(embedding, W_res, W_spike, W_membrane, bias,
                         tokens[:3], hidden_dim, vocab_size)
    print("JIT完了!")
    
    # Training
    print("\n" + "=" * 60)
    print("🧠 1000エポック学習開始！")
    print("=" * 60)
    
    lr = 0.01
    losses = []
    
    for epoch in range(1, 1001):
        logits, _, _ = parallel_forward(embedding, W_res, W_spike, W_membrane, bias,
                                        tokens[:-1], hidden_dim, vocab_size)
        
        epoch_loss = 0.0
        for t in range(len(tokens) - 1):
            probs = softmax_numba(logits[t])
            target = tokens[t + 1]
            
            loss = -np.log(probs[target] + 1e-10)
            epoch_loss += loss
            
            # Gradient update
            error = probs.copy()
            error[target] -= 1.0
            
            W_spike -= lr * np.outer(error, np.ones(hidden_dim, dtype=np.float32))
            W_membrane -= lr * np.outer(error, np.ones(hidden_dim, dtype=np.float32))
            bias -= lr * error
        
        avg_loss = epoch_loss / (len(tokens) - 1)
        losses.append(avg_loss)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d} | Loss: {avg_loss:.4f}")
    
    # Test: Can it reproduce?
    print("\n" + "=" * 60)
    print("📝 再現テスト")
    print("=" * 60)
    
    # Generate from first character
    prompt = TEST_SENTENCE[0]  # "吾"
    gen_tokens = [token_to_idx.get(prompt, 0)]
    h = np.zeros(hidden_dim, dtype=np.float32)
    m = np.zeros(hidden_dim, dtype=np.float32)
    
    for _ in range(len(TEST_SENTENCE) + 5):
        tid = gen_tokens[-1]
        x = embedding[tid]
        
        m = 0.9 * m + x + W_res @ h
        h = np.where(m > 1.0, 1.0, 0.0).astype(np.float32)
        m = np.where(m > 1.0, 0.0, m)
        
        logits = W_spike @ h + W_membrane @ h + bias
        next_t = np.argmax(logits)  # Greedy
        gen_tokens.append(next_t)
        
        if idx_to_token.get(next_t, '') == '。':
            break
    
    generated = "".join([idx_to_token.get(t, '?') for t in gen_tokens])
    
    print(f"\n目標: 「{TEST_SENTENCE}」")
    print(f"生成: 「{generated}」")
    
    # Check match
    match = generated == TEST_SENTENCE
    print(f"\n結果: {'✅ 完全一致！' if match else '❌ 不一致'}")
    
    if not match:
        # Check partial match
        correct_chars = sum(1 for a, b in zip(generated, TEST_SENTENCE) if a == b)
        print(f"一致文字数: {correct_chars}/{len(TEST_SENTENCE)}")
    
    print("\n" + "=" * 60)
    if match:
        print("🎉 SNNの構造は正常！データの問題を解決すれば良い！")
    else:
        print("⚠️ SNNの構造に問題の可能性あり")
        print("   → フォワードパスや重み更新のロジックを確認")
    print("=" * 60)


if __name__ == "__main__":
    main()
