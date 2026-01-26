#!/usr/bin/env python3
"""Super Teacher v2 生成テスト"""

import numpy as np
import pickle
import json
from pathlib import Path
from numba import njit, set_num_threads

set_num_threads(20)

MODEL_PATH = Path(__file__).parent.parent / "checkpoints" / "model_super_teacher_v2.pkl"
TOKENIZER_PATH = Path(__file__).parent.parent / "checkpoints" / "tokenizer_v4_parallel.json"

@njit(cache=True)
def softmax_numba(x):
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / (np.sum(exp_x) + 1e-10)

print("モデルをロード中...")
with open(MODEL_PATH, 'rb') as f:
    model_data = pickle.load(f)

with open(TOKENIZER_PATH, 'r', encoding='utf-8') as f:
    tokenizer_data = json.load(f)

token_to_idx = tokenizer_data['token_to_idx']
idx_to_token = {v: k for k, v in token_to_idx.items()}

embedding = model_data['embedding'].astype(np.float64)
W_res = model_data['W_res'].astype(np.float64)
W_spike = model_data['W_spike'].astype(np.float64)
W_membrane = model_data['W_membrane'].astype(np.float64)
bias = model_data['bias'].astype(np.float64)
hidden_dim = model_data['hidden_dim']
vocab_size = model_data['vocab_size']

print(f"モデルロード完了: {vocab_size}語彙, {hidden_dim}次元")

def generate(prompt, max_length=50, temperature=0.8):
    """テキスト生成"""
    tokens = [token_to_idx.get(c, 0) for c in prompt]
    state = np.zeros(hidden_dim)
    
    for _ in range(max_length):
        # Forward pass
        for token_id in tokens[-10:]:  # Last 10 tokens context
            x = embedding[token_id] if token_id < len(embedding) else np.zeros(hidden_dim)
            new_state = 0.9 * state + 0.1 * (x + W_res @ state)
            state = np.where(new_state > 0.5, 1.0, 0.0)
        
        # Compute logits
        logits = W_spike @ state + W_membrane @ state + bias
        
        # Temperature sampling
        logits = logits / temperature
        probs = softmax_numba(logits)
        
        # Sample
        try:
            next_token = np.random.choice(len(probs), p=probs)
        except:
            next_token = np.argmax(logits)
        
        tokens.append(next_token)
        
        if next_token == 0:
            break
    
    # Decode
    result = ""
    for t in tokens[len(prompt):]:
        if t in idx_to_token:
            result += idx_to_token[t]
    return result

print("\n" + "="*60)
print("📝 Super Teacher v2 生成テスト")
print("="*60)

test_prompts = [
    "人工知能は",
    "日本語の",
    "スパイキングニューラルネットワークは",
    "脳と",
    "吾輩は猫で",
    "ニューロモーフィック",
    "11次元",
    "食べた",  # 文法ドリルのテスト
]

for prompt in test_prompts:
    result = generate(prompt, max_length=40)
    print(f"\n「{prompt}」")
    print(f"  → {result[:60]}...")
