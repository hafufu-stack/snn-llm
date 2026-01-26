#!/usr/bin/env python3
"""
10000エポックモデルのテスト
様々なプロンプトで生成品質を確認
"""

import numpy as np
import pickle
import json
from pathlib import Path
from numba import njit, set_num_threads

set_num_threads(24)

MODEL_PATH = Path(__file__).parent.parent / "checkpoints" / "model_teacher_10000epochs_parallel.pkl"
TOKENIZER_PATH = Path(__file__).parent.parent / "checkpoints" / "tokenizer_v4_parallel.json"


@njit(cache=True)
def softmax_numba(x):
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / (np.sum(exp_x) + 1e-10)


def main():
    print("=" * 60)
    print("🧪 10000エポックモデル テスト")
    print("=" * 60)
    
    # Load model
    print("\nモデルをロード中...")
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
    hidden_dim = model_data['hidden_dim']
    vocab_size = model_data['vocab_size']
    
    print(f"語彙サイズ: {vocab_size}")
    print(f"隠れ次元: {hidden_dim}")
    
    def tokenize(text):
        return [token_to_idx.get(c, 0) for c in text]
    
    def generate(prompt, max_length=50, temperature=0.7):
        tokens = tokenize(prompt)
        h = np.zeros(hidden_dim, dtype=np.float32)
        m = np.zeros(hidden_dim, dtype=np.float32)
        
        for _ in range(max_length):
            # Process last tokens
            for tid in tokens[-10:]:
                x = embedding[tid] if tid < vocab_size else np.zeros(hidden_dim, dtype=np.float32)
                m = 0.9 * m + x
                for j in range(hidden_dim):
                    m += W_res[:, j] * h[j]
                h = np.where(m > 1.0, 1.0, 0.0).astype(np.float32)
                m = np.where(m > 1.0, 0.0, m)
            
            logits = W_spike @ h + W_membrane @ h + bias
            logits = logits / temperature
            probs = softmax_numba(logits)
            
            # Sample
            try:
                next_token = np.random.choice(len(probs), p=probs)
            except:
                next_token = np.argmax(logits)
            
            tokens.append(next_token)
            
            # Stop on EOS or period
            if idx_to_token.get(next_token, '') in ['。', '<EOS>', '<PAD>']:
                break
        
        # Decode
        result = ""
        for t in tokens[len(tokenize(prompt)):]:
            if t in idx_to_token:
                char = idx_to_token[t]
                if char not in ['<PAD>', '<UNK>', '<EOS>']:
                    result += char
        
        return result
    
    # Test prompts - including ones NOT in training data
    test_prompts = [
        # Training dataにある
        ("人工知能は", "（学習済み）"),
        ("スパイキングニューラルネットワークとは", "（学習済み）"),
        ("深層学習とは", "（学習済み）"),
        
        # Training dataにない（汎化テスト）
        ("私の名前は", "（未学習）"),
        ("今日の天気は", "（未学習）"),
        ("日本の首都は", "（未学習）"),
        ("猫は", "（未学習）"),
        ("プログラミングとは", "（未学習）"),
        ("吾輩は猫で", "（未学習・夏目漱石）"),
    ]
    
    print("\n" + "=" * 60)
    print("📝 生成テスト")
    print("=" * 60)
    
    for prompt, note in test_prompts:
        result = generate(prompt, max_length=60)
        print(f"\n【{note}】「{prompt}」")
        print(f"  → {result[:70]}...")
    
    print("\n" + "=" * 60)
    print("テスト完了!")
    print("=" * 60)


if __name__ == "__main__":
    main()
