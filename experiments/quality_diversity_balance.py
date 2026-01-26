#!/usr/bin/env python3
"""
品質-多様性バランス探索
低温度から徐々に上げて、品質を維持しつつ多様性が出るポイントを探す
"""

import numpy as np
import pickle
from pathlib import Path
from numba import njit

import fugashi
tagger = fugashi.Tagger()
print("✅ fugashi (MeCab) ロード成功")

MODEL_PATH = Path(__file__).parent.parent / "checkpoints" / "model_large_dropout_final.pkl"

INPUT_SCALE = 10.0
LEAK = 0.99
THRESHOLD = 1.0


@njit(cache=True)
def softmax_numba(x):
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / (np.sum(exp_x) + 1e-10)


@njit(cache=True)
def nucleus_sampling(logits, p=0.9, temperature=1.0):
    """Nucleus (top-p) sampling"""
    scaled_logits = logits / temperature
    probs = np.exp(scaled_logits - np.max(scaled_logits))
    probs = probs / (np.sum(probs) + 1e-10)
    
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    
    cumsum = np.cumsum(sorted_probs)
    cutoff_idx = 1
    for i in range(len(cumsum)):
        if cumsum[i] > p:
            cutoff_idx = i + 1
            break
    else:
        cutoff_idx = len(cumsum)
    
    cutoff_idx = max(1, cutoff_idx)
    selected_probs = sorted_probs[:cutoff_idx]
    selected_probs = selected_probs / (np.sum(selected_probs) + 1e-10)
    
    cumsum2 = np.cumsum(selected_probs)
    r = np.random.random()
    for i in range(len(cumsum2)):
        if r < cumsum2[i]:
            return sorted_indices[i]
    return sorted_indices[0]


def tokenize_words(text):
    return [w.surface for w in tagger(text) if w.surface.strip()]


def main():
    print(f"""
    ╔═══════════════════════════════════════════════════════════════╗
    ║    ⚖️ 品質-多様性バランス探索                                 ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # モデルロード
    print("モデルをロード中...")
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    vocab_size = model['vocab_size']
    embedding = model['embedding'].astype(np.float32)
    W_res = model['W_res'].astype(np.float32)
    W_spike = model['W_spike'].astype(np.float32)
    W_membrane = model['W_membrane'].astype(np.float32)
    bias = model['bias'].astype(np.float32)
    word_to_idx = model['word_to_idx']
    
    idx_to_word = {v: k for k, v in word_to_idx.items()}
    embedding_scaled = embedding * INPUT_SCALE
    hidden_dim = embedding.shape[1]
    
    print(f"語彙サイズ: {vocab_size}")
    
    def generate(prompt, max_words=12, temperature=1.0, top_p=0.9):
        prompt_words = tokenize_words(prompt)
        prompt_ids = [word_to_idx.get(w, 1) for w in prompt_words]
        if len(prompt_ids) == 0:
            prompt_ids = [0]
        
        h = np.zeros(hidden_dim, dtype=np.float32)
        m = np.zeros(hidden_dim, dtype=np.float32)
        
        for tid in prompt_ids:
            if tid >= vocab_size:
                tid = 1
            x = embedding_scaled[tid]
            m = LEAK * m + x + W_res @ h
            h = np.where(m > THRESHOLD, 1.0, 0.0).astype(np.float32)
            m = np.where(m > THRESHOLD, 0.0, m)
        
        generated = list(prompt_ids)
        for _ in range(max_words):
            tid = generated[-1] if generated[-1] < vocab_size else 0
            x = embedding_scaled[tid]
            m = LEAK * m + x + W_res @ h
            h = np.where(m > THRESHOLD, 1.0, 0.0).astype(np.float32)
            m = np.where(m > THRESHOLD, 0.0, m)
            
            logits = W_spike @ h + W_membrane @ m + bias
            next_token = nucleus_sampling(logits, p=top_p, temperature=temperature)
            generated.append(next_token)
        
        return "".join([idx_to_word.get(t, '?') for t in generated])
    
    def evaluate_quality(text):
        """
        簡易品質スコア
        - 連続「。」の少なさ
        - 「を」「が」「は」等の助詞の適切さ
        """
        score = 100
        
        # 連続句点ペナルティ
        score -= text.count("。。") * 10
        score -= text.count("。。。") * 15
        
        # 連続助詞ペナルティ
        bad_patterns = ["をを", "がが", "はは", "にに", "のの", "をが", "をは", "がを"]
        for p in bad_patterns:
            score -= text.count(p) * 8
        
        # 文字の多様性ボーナス
        unique_chars = len(set(text))
        if unique_chars > 20:
            score += 10
        
        return max(0, min(100, score))
    
    # === 温度による変化を調べる ===
    print("\n" + "=" * 70)
    print("🌡️ Temperature探索 (Nucleus p=0.9)")
    print("=" * 70)
    
    prompt = "人工知能"
    temperatures = [0.7, 0.8, 0.9, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3]
    
    results = []
    
    for temp in temperatures:
        unique_outputs = set()
        quality_scores = []
        samples = []
        
        for _ in range(10):
            text = generate(prompt, max_words=12, temperature=temp, top_p=0.9)
            unique_outputs.add(text)
            quality_scores.append(evaluate_quality(text))
            if len(samples) < 3:
                samples.append(text)
        
        diversity = len(unique_outputs)
        avg_quality = np.mean(quality_scores)
        
        results.append({
            'temp': temp,
            'diversity': diversity,
            'quality': avg_quality,
            'samples': samples
        })
        
        print(f"\nT={temp:.2f} | 多様性: {diversity:2d}/10 | 品質: {avg_quality:.1f}")
        for i, s in enumerate(samples):
            print(f"  {i+1}. {s[:55]}...")
    
    # === 最適点を探す ===
    print("\n" + "=" * 70)
    print("🎯 最適バランス探索")
    print("=" * 70)
    
    # 多様性と品質の両方を考慮したスコア
    # diversity(0-10) * quality(0-100) の積を最大化
    best = max(results, key=lambda r: r['diversity'] * r['quality'])
    
    print(f"""
    ╔═══════════════════════════════════════════════════════════════╗
    ║  🏆 最適設定                                                  ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║  Temperature: {best['temp']:.2f}                                         ║
    ║  多様性: {best['diversity']}/10                                          ║
    ║  品質スコア: {best['quality']:.1f}                                       ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    print("最適設定でのサンプル:")
    for i, s in enumerate(best['samples']):
        print(f"  {i+1}. {s}")
    
    # === 他のプロンプトでテスト ===
    print("\n" + "=" * 70)
    print("📝 最適設定で様々なプロンプト")
    print("=" * 70)
    
    opt_temp = best['temp']
    prompts = ["機械学習", "深層学習", "脳型", "日本語", "ロボット"]
    
    for p in prompts:
        print(f"\n【{p}】")
        for i in range(3):
            text = generate(p, max_words=12, temperature=opt_temp, top_p=0.9)
            quality = evaluate_quality(text)
            print(f"  {i+1}. (Q:{quality:3.0f}) {text[:55]}...")
    
    # === Top-p探索 ===
    print("\n" + "=" * 70)
    print("🎚️ Top-p探索 (Temperature={:.2f})".format(opt_temp))
    print("=" * 70)
    
    top_ps = [0.7, 0.8, 0.85, 0.9, 0.95]
    
    for top_p in top_ps:
        unique_outputs = set()
        quality_scores = []
        
        for _ in range(10):
            text = generate(prompt, max_words=12, temperature=opt_temp, top_p=top_p)
            unique_outputs.add(text)
            quality_scores.append(evaluate_quality(text))
        
        print(f"p={top_p:.2f} | 多様性: {len(unique_outputs):2d}/10 | 品質: {np.mean(quality_scores):.1f}")


if __name__ == "__main__":
    main()
