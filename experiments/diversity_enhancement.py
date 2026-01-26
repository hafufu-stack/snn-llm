#!/usr/bin/env python3
"""
多様性向上実験 - 3つの手法を同時適用

1. Top-k / Nucleus (top-p) sampling
2. 推論時膜電位ノイズ
3. 高温度 (Temperature 1.5+)
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
def top_k_sampling(logits, k=10, temperature=1.0):
    """Top-k sampling: 上位k個からのみサンプリング"""
    scaled_logits = logits / temperature
    
    # 上位k個のインデックスを取得
    indices = np.argsort(scaled_logits)[::-1][:k]
    
    # 上位k個のlogitsのみでsoftmax
    top_logits = scaled_logits[indices]
    probs = np.exp(top_logits - np.max(top_logits))
    probs = probs / (np.sum(probs) + 1e-10)
    
    # 累積確率でサンプリング
    cumsum = np.cumsum(probs)
    r = np.random.random()
    for i in range(len(cumsum)):
        if r < cumsum[i]:
            return indices[i]
    return indices[0]


@njit(cache=True)
def nucleus_sampling(logits, p=0.9, temperature=1.0):
    """Nucleus (top-p) sampling: 累積確率がpに達するまでの候補からサンプリング"""
    scaled_logits = logits / temperature
    probs = np.exp(scaled_logits - np.max(scaled_logits))
    probs = probs / (np.sum(probs) + 1e-10)
    
    # 確率順にソート
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    
    # 累積確率がpを超えるまでの候補を取得
    cumsum = np.cumsum(sorted_probs)
    cutoff_idx = 0
    for i in range(len(cumsum)):
        if cumsum[i] > p:
            cutoff_idx = i + 1
            break
    else:
        cutoff_idx = len(cumsum)
    
    cutoff_idx = max(1, cutoff_idx)  # 最低1つは選ぶ
    
    # カットオフまでの確率で再正規化
    selected_probs = sorted_probs[:cutoff_idx]
    selected_probs = selected_probs / (np.sum(selected_probs) + 1e-10)
    
    # サンプリング
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
    ║    🎯 多様性向上実験 - 3手法同時適用                          ║
    ║                                                               ║
    ║    1. Top-k / Nucleus sampling                                ║
    ║    2. 推論時膜電位ノイズ                                      ║
    ║    3. 高温度 (Temperature 1.5+)                               ║
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
    
    def generate_enhanced(prompt, max_words=15, 
                         temperature=1.5, 
                         top_k=20, 
                         top_p=0.9, 
                         membrane_noise=0.3,
                         sampling_method='nucleus'):
        """
        3つの多様性向上手法を組み合わせた生成
        
        Args:
            temperature: 高温度でよりランダムに
            top_k: Top-k samplingのk値
            top_p: Nucleus samplingのp値
            membrane_noise: 膜電位に加えるノイズの標準偏差
            sampling_method: 'topk', 'nucleus', 'standard'
        """
        prompt_words = tokenize_words(prompt)
        prompt_ids = [word_to_idx.get(w, 1) for w in prompt_words]
        if len(prompt_ids) == 0:
            prompt_ids = [0]
        
        h = np.zeros(hidden_dim, dtype=np.float32)
        m = np.zeros(hidden_dim, dtype=np.float32)
        
        # プロンプト処理
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
            
            # 🔥 推論時ノイズ: 膜電位にランダムノイズを追加
            if membrane_noise > 0:
                m = m + np.random.randn(hidden_dim).astype(np.float32) * membrane_noise
            
            h = np.where(m > THRESHOLD, 1.0, 0.0).astype(np.float32)
            m = np.where(m > THRESHOLD, 0.0, m)
            
            logits = W_spike @ h + W_membrane @ m + bias
            
            # 🔥 サンプリング手法選択
            if sampling_method == 'topk':
                next_token = top_k_sampling(logits, k=top_k, temperature=temperature)
            elif sampling_method == 'nucleus':
                next_token = nucleus_sampling(logits, p=top_p, temperature=temperature)
            else:
                probs = softmax_numba(logits / temperature)
                try:
                    next_token = np.random.choice(len(probs), p=probs)
                except:
                    next_token = np.argmax(logits)
            
            generated.append(next_token)
        
        return "".join([idx_to_word.get(t, '?') for t in generated])
    
    # === テスト1: 従来手法との比較 ===
    print("\n" + "=" * 60)
    print("📊 手法比較テスト（プロンプト: 人工知能）")
    print("=" * 60)
    
    prompt = "人工知能"
    
    # 従来手法（低温度、ノイズなし）
    print("\n【従来手法】T=0.9, ノイズなし, 標準サンプリング")
    unique_standard = set()
    for i in range(10):
        r = generate_enhanced(prompt, temperature=0.9, membrane_noise=0, sampling_method='standard')
        unique_standard.add(r)
        print(f"  {i+1}: {r[:50]}...")
    print(f"  ユニーク: {len(unique_standard)}/10")
    
    # 新手法: 全部盛り
    print("\n【新手法】T=1.5, ノイズ0.3, Nucleus p=0.9")
    unique_enhanced = set()
    for i in range(10):
        r = generate_enhanced(prompt, temperature=1.5, membrane_noise=0.3, 
                             top_p=0.9, sampling_method='nucleus')
        unique_enhanced.add(r)
        print(f"  {i+1}: {r[:50]}...")
    print(f"  ユニーク: {len(unique_enhanced)}/10")
    
    # === テスト2: パラメータ探索 ===
    print("\n" + "=" * 60)
    print("🔬 パラメータ探索")
    print("=" * 60)
    
    configs = [
        # (temperature, membrane_noise, top_p, method, description)
        (1.2, 0.0, 0.9, 'nucleus', "T=1.2, ノイズ0, nucleus"),
        (1.5, 0.0, 0.9, 'nucleus', "T=1.5, ノイズ0, nucleus"),
        (2.0, 0.0, 0.9, 'nucleus', "T=2.0, ノイズ0, nucleus"),
        (1.5, 0.2, 0.9, 'nucleus', "T=1.5, ノイズ0.2, nucleus"),
        (1.5, 0.5, 0.9, 'nucleus', "T=1.5, ノイズ0.5, nucleus"),
        (1.5, 0.3, 0.8, 'nucleus', "T=1.5, ノイズ0.3, p=0.8"),
        (1.5, 0.3, 0.95, 'nucleus', "T=1.5, ノイズ0.3, p=0.95"),
        (1.5, 0.3, 0.9, 'topk', "T=1.5, ノイズ0.3, top-k=20"),
    ]
    
    best_config = None
    best_unique = 0
    
    for temp, noise, top_p, method, desc in configs:
        unique = set()
        for _ in range(10):
            r = generate_enhanced(prompt, temperature=temp, membrane_noise=noise,
                                 top_p=top_p, sampling_method=method, top_k=20)
            unique.add(r)
        
        print(f"\n{desc}")
        print(f"  ユニーク: {len(unique)}/10")
        
        if len(unique) > best_unique:
            best_unique = len(unique)
            best_config = (temp, noise, top_p, method, desc)
    
    # === 最良設定でサンプル表示 ===
    print("\n" + "=" * 60)
    print("🏆 最良設定でのサンプル")
    print("=" * 60)
    
    if best_config:
        temp, noise, top_p, method, desc = best_config
        print(f"\n設定: {desc}")
        print(f"ユニーク数: {best_unique}/10\n")
        
        for i in range(5):
            r = generate_enhanced(prompt, temperature=temp, membrane_noise=noise,
                                 top_p=top_p, sampling_method=method)
            print(f"  {i+1}: {r}")
    
    # === 他のプロンプトでテスト ===
    print("\n" + "=" * 60)
    print("📝 様々なプロンプトでテスト（最良設定）")
    print("=" * 60)
    
    if best_config:
        temp, noise, top_p, method, _ = best_config
        
        prompts = ["機械学習", "深層学習", "ロボット", "日本語"]
        for p in prompts:
            print(f"\n【{p}】")
            for i in range(3):
                r = generate_enhanced(p, temperature=temp, membrane_noise=noise,
                                     top_p=top_p, sampling_method=method)
                print(f"  {i+1}: {r[:60]}...")
    
    # === サマリー ===
    print("\n" + "=" * 60)
    print("📊 結果サマリー")
    print("=" * 60)
    print(f"""
    【従来】ユニーク数: {len(unique_standard)}/10
    【新手法】ユニーク数: {len(unique_enhanced)}/10
    
    最良設定: {best_config[4] if best_config else 'N/A'}
    最高ユニーク数: {best_unique}/10
    """)


if __name__ == "__main__":
    main()
