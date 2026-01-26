#!/usr/bin/env python3
"""
階層型SNN-LLM テスト・評価スクリプト
"""

import numpy as np
import pickle
import json
from pathlib import Path
from numba import njit

# MeCab
import fugashi
tagger = fugashi.Tagger()

MODEL_PATH = Path(__file__).parent.parent / "checkpoints" / "model_hierarchical_final.pkl"

# パラメータ
INPUT_SCALE = 10.0
LEAK = 0.99
THRESHOLD = 1.0


@njit(cache=True)
def softmax_numba(x):
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / (np.sum(exp_x) + 1e-10)


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║    🧪 階層型 SNN-LLM 評価                                     ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # モデルロード
    print("モデルをロード中...")
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    # 各種パラメータ取得
    word_vocab_size = model['word_vocab_size']
    char_vocab_size = model['char_vocab_size']
    word_embedding = model['word_embedding'].astype(np.float32)
    word_W_res = model['word_W_res'].astype(np.float32)
    word_W_spike = model['word_W_spike'].astype(np.float32)
    word_W_membrane = model['word_W_membrane'].astype(np.float32)
    word_bias = model['word_bias'].astype(np.float32)
    word_to_idx = model['word_to_idx']
    char_to_idx = model['char_to_idx']
    
    idx_to_word = {v: k for k, v in word_to_idx.items()}
    idx_to_char = {v: k for k, v in char_to_idx.items()}
    
    word_embedding_scaled = word_embedding * INPUT_SCALE
    hidden_dim = word_embedding.shape[1]
    
    print(f"単語語彙サイズ: {word_vocab_size}")
    print(f"文字語彙サイズ: {char_vocab_size}")
    print(f"隠れ層次元: {hidden_dim}")
    
    # 単語の損失情報
    if 'word_losses' in model:
        word_losses = model['word_losses']
        print(f"\n学習履歴:")
        print(f"  初期Loss: {word_losses[0]:.3f}")
        print(f"  最終Loss: {word_losses[-1]:.3f}")
        print(f"  改善率: {(1 - word_losses[-1]/word_losses[0])*100:.1f}%")
    
    def tokenize_words(text):
        words = []
        for word in tagger(text):
            if word.surface.strip():
                words.append(word.surface)
        return words
    
    def generate_words(prompt, max_words=20, temperature=0.8):
        """単語レベルで生成"""
        prompt_words = tokenize_words(prompt)
        prompt_ids = [word_to_idx.get(w, 1) for w in prompt_words]
        
        h = np.zeros(hidden_dim, dtype=np.float32)
        m = np.zeros(hidden_dim, dtype=np.float32)
        
        # プロンプト処理
        for tid in prompt_ids:
            if tid >= word_vocab_size:
                tid = 1
            x = word_embedding_scaled[tid]
            m = LEAK * m + x + word_W_res @ h
            h = np.where(m > THRESHOLD, 1.0, 0.0).astype(np.float32)
            m = np.where(m > THRESHOLD, 0.0, m)
        
        # 生成
        generated = list(prompt_ids)
        for _ in range(max_words):
            tid = generated[-1] if generated[-1] < word_vocab_size else 0
            x = word_embedding_scaled[tid]
            m = LEAK * m + x + word_W_res @ h
            h = np.where(m > THRESHOLD, 1.0, 0.0).astype(np.float32)
            m = np.where(m > THRESHOLD, 0.0, m)
            
            logits = word_W_spike @ h + word_W_membrane @ m + word_bias
            probs = softmax_numba(logits / temperature)
            
            try:
                next_token = np.random.choice(len(probs), p=probs)
            except:
                next_token = np.argmax(logits)
            
            generated.append(next_token)
            
            word = idx_to_word.get(next_token, '')
            if word in ['。', '<EOS>']:
                break
        
        return "".join([idx_to_word.get(t, '?') for t in generated])
    
    # === テスト1: 様々なプロンプトで生成 ===
    print("\n" + "=" * 60)
    print("📝 生成テスト")
    print("=" * 60)
    
    test_prompts = [
        "人工知能",
        "深層学習",
        "脳",
        "ニューラルネットワーク",
        "日本語",
        "機械学習",
        "データ",
        "スパイキング",
    ]
    
    for prompt in test_prompts:
        result = generate_words(prompt, max_words=15, temperature=0.8)
        print(f"\n【{prompt}】")
        print(f"  → {result}")
    
    # === テスト2: 多様性テスト ===
    print("\n" + "=" * 60)
    print("🎲 多様性テスト（同じプロンプトで10回生成）")
    print("=" * 60)
    
    prompt = "人工知能"
    unique_outputs = set()
    print(f"\nプロンプト: 「{prompt}」\n")
    
    for i in range(10):
        result = generate_words(prompt, max_words=15, temperature=0.9)
        unique_outputs.add(result)
        print(f"  {i+1}: {result[:50]}...")
    
    print(f"\nユニーク出力数: {len(unique_outputs)}/10")
    
    # === テスト3: Temperature比較 ===
    print("\n" + "=" * 60)
    print("🌡️ Temperature比較")
    print("=" * 60)
    
    prompt = "機械学習"
    for temp in [0.5, 0.7, 0.9, 1.2]:
        result = generate_words(prompt, max_words=15, temperature=temp)
        print(f"\nT={temp}: {result[:60]}...")
    
    # === テスト4: 長文生成 ===
    print("\n" + "=" * 60)
    print("📄 長文生成テスト")
    print("=" * 60)
    
    long_result = generate_words("人工知能", max_words=50, temperature=0.8)
    print(f"\n{long_result}")
    
    # === サマリー ===
    print("\n" + "=" * 60)
    print("📊 評価サマリー")
    print("=" * 60)
    print(f"""
    単語語彙サイズ: {word_vocab_size}
    文字語彙サイズ: {char_vocab_size}
    多様性 (10回中ユニーク): {len(unique_outputs)}/10
    モデルパス: {MODEL_PATH.name}
    """)


if __name__ == "__main__":
    main()
