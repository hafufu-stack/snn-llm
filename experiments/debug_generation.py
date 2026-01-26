#!/usr/bin/env python3
"""
SNN生成ロジックのデバッグ
問題: 学習時はLoss下がるのに、生成時は「は」がループする

デバッグポイント:
1. 学習時と生成時のフォワードパスが違う？
2. 状態(h, m)の扱いが違う？
3. トークン履歴の使い方が違う？
"""

import numpy as np
import pickle
import json
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "checkpoints" / "model_v4_parallel_final.pkl"
TOKENIZER_PATH = Path(__file__).parent.parent / "checkpoints" / "tokenizer_v4_parallel.json"

TEST_SENTENCE = "吾輩は猫である。"


def main():
    print("=" * 60)
    print("🔍 SNN生成ロジック デバッグ")
    print("=" * 60)
    
    # Load
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
    
    tokens = [token_to_idx.get(c, 0) for c in TEST_SENTENCE]
    print(f"テスト文: {TEST_SENTENCE}")
    print(f"トークン: {tokens} → {[idx_to_token.get(t) for t in tokens]}")
    
    # === 学習フォワードパスの再現 ===
    print("\n" + "=" * 60)
    print("📚 学習時のフォワードパス（全トークン入力）")
    print("=" * 60)
    
    h = np.zeros(hidden_dim, dtype=np.float32)
    m = np.zeros(hidden_dim, dtype=np.float32)
    
    for t_idx, tid in enumerate(tokens[:-1]):
        x = embedding[tid]
        
        # 学習時のロジック
        m_new = 0.9 * m + x
        for j in range(hidden_dim):
            m_new += W_res[:, j] * h[j]
        
        h_new = np.where(m_new > 1.0, 1.0, 0.0).astype(np.float32)
        m_new = np.where(m_new > 1.0, 0.0, m_new)
        
        h = h_new
        m = m_new
        
        # 予測
        logits = W_spike @ h + W_membrane @ h + bias
        pred_token = np.argmax(logits)
        actual_next = tokens[t_idx + 1]
        
        print(f"  入力: '{idx_to_token.get(tid)}' → 予測: '{idx_to_token.get(pred_token)}' (正解: '{idx_to_token.get(actual_next)}')")
        print(f"    h活性: {np.sum(h > 0)}/{hidden_dim}, m平均: {np.mean(m):.4f}")
    
    # === 生成フォワードパスの再現 ===
    print("\n" + "=" * 60)
    print("🤖 生成時のフォワードパス（1トークンずつ自己回帰）")
    print("=" * 60)
    
    # 生成用にリセット
    h = np.zeros(hidden_dim, dtype=np.float32)
    m = np.zeros(hidden_dim, dtype=np.float32)
    
    # 最初のトークンから開始
    current_token = tokens[0]  # "吾"
    generated = [current_token]
    
    for step in range(len(TEST_SENTENCE) + 3):
        x = embedding[current_token]
        
        # 生成時のロジック（同じはず）
        m = 0.9 * m + x + W_res @ h
        h = np.where(m > 1.0, 1.0, 0.0).astype(np.float32)
        m = np.where(m > 1.0, 0.0, m)
        
        # 予測
        logits = W_spike @ h + W_membrane @ h + bias
        next_token = np.argmax(logits)
        
        print(f"  Step {step}: 入力: '{idx_to_token.get(current_token)}' → 予測: '{idx_to_token.get(next_token)}'")
        print(f"    h活性: {np.sum(h > 0)}/{hidden_dim}, m平均: {np.mean(m):.4f}")
        
        generated.append(next_token)
        current_token = next_token  # 自己回帰
        
        if idx_to_token.get(next_token) == '。':
            break
    
    gen_text = "".join([idx_to_token.get(t, '?') for t in generated])
    print(f"\n生成結果: 「{gen_text}」")
    
    # === 違いを分析 ===
    print("\n" + "=" * 60)
    print("🔬 問題の分析")
    print("=" * 60)
    
    # 学習時と生成時のW_res適用方法を比較
    print("\n1. W_res @ h の形状確認:")
    test_h = np.ones(hidden_dim, dtype=np.float32)
    res_result = W_res @ test_h
    print(f"   W_res.shape: {W_res.shape}")
    print(f"   h.shape: {test_h.shape}")
    print(f"   (W_res @ h).shape: {res_result.shape}")
    
    # 学習時の正しいロジック
    print("\n2. 学習時のロジック（正しい）:")
    print("   for j in range(hidden_dim):")
    print("       m_new += W_res[:, j] * h[j]")
    print("   → 各列をスカラー倍して足し合わせ")
    
    print("\n3. 生成時のロジック（簡略化）:")
    print("   m = 0.9 * m + x + W_res @ h")
    print("   → 行列ベクトル積（同じ結果のはず）")
    
    # 実際に比較
    print("\n4. 実際の計算結果比較:")
    test_h = np.random.randn(hidden_dim).astype(np.float32)
    
    # 方法1: ループ
    result1 = np.zeros(hidden_dim, dtype=np.float32)
    for j in range(hidden_dim):
        result1 += W_res[:, j] * test_h[j]
    
    # 方法2: 行列積
    result2 = W_res @ test_h
    
    diff = np.max(np.abs(result1 - result2))
    print(f"   最大差: {diff}")
    print(f"   → {'✅ 同一' if diff < 1e-5 else '❌ 違う！'}")


if __name__ == "__main__":
    main()
