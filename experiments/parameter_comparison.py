#!/usr/bin/env python3
"""
SNN発火パラメータの全パターン比較テスト

変更候補:
1. 閾値を下げる: 1.0 → 0.1
2. 入力スケールを上げる: embedding * 10
3. 膜電位の蓄積を強化: 0.9 → 0.99

テストパターン:
- 現状（ベースライン）
- 1のみ
- 2のみ
- 3のみ
- 1+2
- 1+3
- 2+3
- 1+2+3
"""

import numpy as np
import pickle
import json
from pathlib import Path
import time

MODEL_PATH = Path(__file__).parent.parent / "checkpoints" / "model_v4_parallel_final.pkl"
TOKENIZER_PATH = Path(__file__).parent.parent / "checkpoints" / "tokenizer_v4_parallel.json"

TEST_SENTENCE = "吾輩は猫である。"
EPOCHS = 500  # 早めに結果を見るため


def softmax(x):
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / (np.sum(exp_x) + 1e-10)


def train_and_test(embedding, W_res, W_spike, W_membrane, bias, tokens, idx_to_token,
                   threshold=1.0, input_scale=1.0, leak=0.9, epochs=EPOCHS):
    """学習してテスト"""
    hidden_dim = embedding.shape[1]
    vocab_size = len(bias)
    
    # コピーして変更
    W_spike = W_spike.copy()
    W_membrane = W_membrane.copy()
    bias = bias.copy()
    embedding_scaled = embedding * input_scale
    
    lr = 0.01
    
    # 学習
    for epoch in range(epochs):
        h = np.zeros(hidden_dim, dtype=np.float32)
        m = np.zeros(hidden_dim, dtype=np.float32)
        
        for t_idx in range(len(tokens) - 1):
            tid = tokens[t_idx]
            x = embedding_scaled[tid]
            
            # 膜電位更新
            m = leak * m + x + W_res @ h
            
            # 発火判定
            h = np.where(m > threshold, 1.0, 0.0).astype(np.float32)
            m = np.where(m > threshold, 0.0, m)
            
            # 出力計算
            logits = W_spike @ h + W_membrane @ h + bias
            probs = softmax(logits)
            target = tokens[t_idx + 1]
            
            # 勾配更新
            error = probs.copy()
            error[target] -= 1.0
            
            W_spike -= lr * np.outer(error, np.ones(hidden_dim, dtype=np.float32))
            W_membrane -= lr * np.outer(error, np.ones(hidden_dim, dtype=np.float32))
            bias -= lr * error
    
    # 生成テスト
    h = np.zeros(hidden_dim, dtype=np.float32)
    m = np.zeros(hidden_dim, dtype=np.float32)
    
    generated = [tokens[0]]
    firing_counts = []
    
    for _ in range(len(TEST_SENTENCE) + 3):
        tid = generated[-1]
        x = embedding_scaled[tid]
        
        m = leak * m + x + W_res @ h
        h = np.where(m > threshold, 1.0, 0.0).astype(np.float32)
        m = np.where(m > threshold, 0.0, m)
        
        firing_counts.append(int(np.sum(h > 0)))
        
        logits = W_spike @ h + W_membrane @ h + bias
        next_token = np.argmax(logits)
        generated.append(next_token)
        
        if idx_to_token.get(next_token) == '。':
            break
    
    gen_text = "".join([idx_to_token.get(t, '?') for t in generated])
    avg_firing = np.mean(firing_counts) if firing_counts else 0
    
    return gen_text, avg_firing


def main():
    print("=" * 70)
    print("🔬 SNN発火パラメータ 全パターン比較テスト")
    print("=" * 70)
    
    # Load
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
    with open(TOKENIZER_PATH, 'r', encoding='utf-8') as f:
        tokenizer_data = json.load(f)
    
    token_to_idx = tokenizer_data['token_to_idx']
    idx_to_token = {v: k for k, v in token_to_idx.items()}
    
    embedding = model_data['embedding'].astype(np.float32)
    W_res = model_data['W_res'].astype(np.float32)
    W_spike = model_data['W_spike'].astype(np.float32)
    W_membrane = model_data['W_membrane'].astype(np.float32)
    bias = model_data['bias'].astype(np.float32)
    
    tokens = [token_to_idx.get(c, 0) for c in TEST_SENTENCE]
    
    print(f"\n目標: 「{TEST_SENTENCE}」")
    print(f"トークン: {[idx_to_token.get(t) for t in tokens]}")
    print(f"学習エポック: {EPOCHS}")
    
    # パターン定義
    patterns = [
        ("ベースライン", {"threshold": 1.0, "input_scale": 1.0, "leak": 0.9}),
        ("1. 閾値↓", {"threshold": 0.1, "input_scale": 1.0, "leak": 0.9}),
        ("2. スケール↑", {"threshold": 1.0, "input_scale": 10.0, "leak": 0.9}),
        ("3. リーク↓", {"threshold": 1.0, "input_scale": 1.0, "leak": 0.99}),
        ("1+2", {"threshold": 0.1, "input_scale": 10.0, "leak": 0.9}),
        ("1+3", {"threshold": 0.1, "input_scale": 1.0, "leak": 0.99}),
        ("2+3", {"threshold": 1.0, "input_scale": 10.0, "leak": 0.99}),
        ("1+2+3", {"threshold": 0.1, "input_scale": 10.0, "leak": 0.99}),
    ]
    
    print("\n" + "=" * 70)
    print("📊 結果")
    print("=" * 70)
    
    results = []
    
    for name, params in patterns:
        start = time.time()
        gen_text, avg_firing = train_and_test(
            embedding, W_res, W_spike, W_membrane, bias, tokens, idx_to_token,
            **params
        )
        elapsed = time.time() - start
        
        # 一致度計算
        match_count = sum(1 for a, b in zip(gen_text, TEST_SENTENCE) if a == b)
        match_ratio = match_count / len(TEST_SENTENCE) * 100
        
        results.append({
            'name': name,
            'gen_text': gen_text,
            'match_ratio': match_ratio,
            'avg_firing': avg_firing,
            'time': elapsed
        })
        
        status = "✅" if match_ratio == 100 else "⚠️" if match_ratio >= 50 else "❌"
        print(f"\n{status} {name}")
        print(f"   生成: 「{gen_text[:20]}{'...' if len(gen_text) > 20 else ''}」")
        print(f"   一致: {match_ratio:.0f}% ({match_count}/{len(TEST_SENTENCE)})")
        print(f"   発火: {avg_firing:.1f}/2048")
        print(f"   時間: {elapsed:.1f}秒")
    
    # サマリー
    print("\n" + "=" * 70)
    print("📈 サマリー（一致度順）")
    print("=" * 70)
    
    sorted_results = sorted(results, key=lambda x: x['match_ratio'], reverse=True)
    
    for i, r in enumerate(sorted_results):
        print(f"{i+1}. {r['name']:12} | 一致: {r['match_ratio']:5.1f}% | 発火: {r['avg_firing']:6.1f}")
    
    best = sorted_results[0]
    print(f"\n🏆 ベスト: {best['name']} ({best['match_ratio']:.0f}%一致)")


if __name__ == "__main__":
    main()
