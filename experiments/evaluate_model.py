#!/usr/bin/env python3
"""
SNN-LLM 定量評価スクリプト
Perplexity, 精度, 多様性などを測定
"""

import numpy as np
import pickle
import json
import math
from pathlib import Path
from collections import Counter
from numba import njit

# Paths
MODEL_PATH = Path(__file__).parent.parent / "checkpoints" / "model_word_v3_final.pkl"
TOKENIZER_PATH = Path(__file__).parent.parent / "checkpoints" / "tokenizer_word_v3.json"
CORPUS_PATH = Path(__file__).parent.parent / "data" / "japanese_corpus.txt"


@njit(cache=True)
def softmax_numba(x):
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / (np.sum(exp_x) + 1e-10)


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║    📊 SNN-LLM 定量評価                                        ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Load model
    print("モデルをロード中...")
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
    
    with open(TOKENIZER_PATH, 'r', encoding='utf-8') as f:
        tokenizer_data = json.load(f)
    
    word_to_idx = tokenizer_data['word_to_idx']
    idx_to_word = {int(v): k for k, v in word_to_idx.items()}
    
    vocab_size = model_data['vocab_size']
    hidden_dim = model_data['hidden_dim']
    embedding = model_data['embedding'].astype(np.float32)
    W_res = model_data['W_res'].astype(np.float32)
    W_spike = model_data['W_spike'].astype(np.float32)
    W_membrane = model_data['W_membrane'].astype(np.float32)
    bias = model_data['bias'].astype(np.float32)
    input_scale = model_data.get('input_scale', 10.0)
    leak = model_data.get('leak', 0.99)
    
    embedding_scaled = embedding * input_scale
    
    # MeCab
    import fugashi
    tagger = fugashi.Tagger()
    
    def tokenize(text):
        words = []
        for word in tagger(text):
            if word.surface.strip():
                words.append(word.surface)
        return words
    
    def text_to_ids(text):
        words = tokenize(text)
        return [word_to_idx.get(w, 1) for w in words]
    
    # Load test corpus
    print("テストコーパスをロード中...")
    with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
        corpus_lines = [line.strip() for line in f if line.strip()]
    
    # Use last 200 lines as test set
    test_lines = corpus_lines[-200:]
    print(f"テストサンプル数: {len(test_lines)}")
    
    # === 1. Perplexity (PPL) ===
    print("\n" + "=" * 50)
    print("1️⃣ Perplexity (PPL) 計算")
    print("=" * 50)
    
    total_loss = 0.0
    total_tokens = 0
    
    for line in test_lines:
        token_ids = text_to_ids(line)
        if len(token_ids) < 2:
            continue
        
        h = np.zeros(hidden_dim, dtype=np.float32)
        m = np.zeros(hidden_dim, dtype=np.float32)
        
        for t in range(len(token_ids) - 1):
            tid = token_ids[t]
            if tid >= vocab_size:
                tid = 1
            
            x = embedding_scaled[tid]
            m = leak * m + x + W_res @ h
            h = np.where(m > 1.0, 1.0, 0.0).astype(np.float32)
            m = np.where(m > 1.0, 0.0, m)
            
            logits = W_spike @ h + W_membrane @ m + bias
            probs = softmax_numba(logits)
            
            target = token_ids[t + 1]
            if target >= vocab_size:
                target = 1
            
            loss = -np.log(probs[target] + 1e-10)
            total_loss += loss
            total_tokens += 1
    
    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(avg_loss)
    
    print(f"平均Loss: {avg_loss:.4f}")
    print(f"Perplexity (PPL): {perplexity:.2f}")
    
    # === 2. Top-k Accuracy ===
    print("\n" + "=" * 50)
    print("2️⃣ Top-k 精度")
    print("=" * 50)
    
    top1_correct = 0
    top5_correct = 0
    top10_correct = 0
    total_preds = 0
    
    for line in test_lines[:50]:  # サンプル制限
        token_ids = text_to_ids(line)
        if len(token_ids) < 2:
            continue
        
        h = np.zeros(hidden_dim, dtype=np.float32)
        m = np.zeros(hidden_dim, dtype=np.float32)
        
        for t in range(len(token_ids) - 1):
            tid = token_ids[t]
            if tid >= vocab_size:
                tid = 1
            
            x = embedding_scaled[tid]
            m = leak * m + x + W_res @ h
            h = np.where(m > 1.0, 1.0, 0.0).astype(np.float32)
            m = np.where(m > 1.0, 0.0, m)
            
            logits = W_spike @ h + W_membrane @ m + bias
            
            target = token_ids[t + 1]
            if target >= vocab_size:
                target = 1
            
            top_k = np.argsort(logits)[::-1]
            
            if target == top_k[0]:
                top1_correct += 1
            if target in top_k[:5]:
                top5_correct += 1
            if target in top_k[:10]:
                top10_correct += 1
            
            total_preds += 1
    
    print(f"Top-1 精度: {100 * top1_correct / total_preds:.1f}%")
    print(f"Top-5 精度: {100 * top5_correct / total_preds:.1f}%")
    print(f"Top-10 精度: {100 * top10_correct / total_preds:.1f}%")
    
    # === 3. Generation Diversity ===
    print("\n" + "=" * 50)
    print("3️⃣ 生成多様性")
    print("=" * 50)
    
    def generate(prompt_ids, max_words=15):
        h = np.zeros(hidden_dim, dtype=np.float32)
        m = np.zeros(hidden_dim, dtype=np.float32)
        
        for tid in prompt_ids:
            if tid >= vocab_size:
                tid = 1
            x = embedding_scaled[tid]
            m = leak * m + x + W_res @ h
            h = np.where(m > 1.0, 1.0, 0.0).astype(np.float32)
            m = np.where(m > 1.0, 0.0, m)
        
        generated = list(prompt_ids)
        for _ in range(max_words):
            tid = generated[-1] if generated[-1] < vocab_size else 0
            x = embedding_scaled[tid]
            m = leak * m + x + W_res @ h
            h = np.where(m > 1.0, 1.0, 0.0).astype(np.float32)
            m = np.where(m > 1.0, 0.0, m)
            
            logits = W_spike @ h + W_membrane @ m + bias
            probs = softmax_numba(logits / 0.7)
            
            try:
                next_token = np.random.choice(len(probs), p=probs)
            except:
                next_token = np.argmax(logits)
            
            generated.append(next_token)
        
        return generated
    
    # Generate multiple samples
    prompt = "人工知能"
    prompt_ids = text_to_ids(prompt)
    
    all_tokens = []
    unique_outputs = set()
    
    for _ in range(20):
        gen = generate(prompt_ids, max_words=15)
        output_text = "".join([idx_to_word.get(t, '?') for t in gen])
        unique_outputs.add(output_text)
        all_tokens.extend(gen[len(prompt_ids):])
    
    # Calculate distinct-n
    unigrams = all_tokens
    bigrams = [(all_tokens[i], all_tokens[i+1]) for i in range(len(all_tokens)-1)]
    
    distinct_1 = len(set(unigrams)) / max(len(unigrams), 1)
    distinct_2 = len(set(bigrams)) / max(len(bigrams), 1)
    
    print(f"ユニーク出力数 (20サンプル): {len(unique_outputs)}")
    print(f"Distinct-1 (unigram多様性): {distinct_1:.3f}")
    print(f"Distinct-2 (bigram多様性): {distinct_2:.3f}")
    
    # === 4. Repetition Rate ===
    print("\n" + "=" * 50)
    print("4️⃣ 反復率（低いほど良い）")
    print("=" * 50)
    
    repetition_count = 0
    total_pairs = 0
    
    for _ in range(20):
        gen = generate(prompt_ids, max_words=20)
        for i in range(len(gen) - 1):
            if gen[i] == gen[i + 1]:
                repetition_count += 1
            total_pairs += 1
    
    repetition_rate = repetition_count / max(total_pairs, 1)
    print(f"連続反復率: {100 * repetition_rate:.1f}%")
    
    # === Summary ===
    print("\n" + "=" * 50)
    print("📋 サマリー")
    print("=" * 50)
    print(f"""
    ┌────────────────────────────────┐
    │ Perplexity (PPL)    : {perplexity:>7.2f} │
    │ Top-1 精度          : {100 * top1_correct / total_preds:>6.1f}% │
    │ Top-5 精度          : {100 * top5_correct / total_preds:>6.1f}% │
    │ Distinct-1          : {distinct_1:>7.3f} │
    │ Distinct-2          : {distinct_2:>7.3f} │
    │ 反復率              : {100 * repetition_rate:>6.1f}% │
    └────────────────────────────────┘
    """)
    
    # Save results
    results = {
        'perplexity': perplexity,
        'avg_loss': avg_loss,
        'top1_accuracy': top1_correct / total_preds,
        'top5_accuracy': top5_correct / total_preds,
        'top10_accuracy': top10_correct / total_preds,
        'distinct_1': distinct_1,
        'distinct_2': distinct_2,
        'repetition_rate': repetition_rate,
        'unique_outputs_20': len(unique_outputs),
    }
    
    results_path = Path(__file__).parent.parent / "results" / "word_v3_evaluation.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"結果保存: {results_path}")


if __name__ == "__main__":
    main()
