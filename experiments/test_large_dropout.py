#!/usr/bin/env python3
"""
大規模コーパス + Dropout モデル 多様性テスト
"""

import numpy as np
import pickle
from pathlib import Path
from numba import njit

import fugashi
tagger = fugashi.Tagger()

MODEL_PATH = Path(__file__).parent.parent / "checkpoints" / "model_large_dropout_final.pkl"

INPUT_SCALE = 10.0
LEAK = 0.99
THRESHOLD = 1.0


@njit(cache=True)
def softmax_numba(x):
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / (np.sum(exp_x) + 1e-10)


def tokenize_words(text):
    return [w.surface for w in tagger(text) if w.surface.strip()]


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║    🧪 大規模コーパス + Dropout モデル 多様性テスト            ║
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
    print(f"Dropout率: {model.get('dropout_rate', 'N/A')}")
    
    def generate(prompt, max_words=15, temperature=0.9):
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
            h = np.where(m > THRESHOLD, 1.0, 0.0).astype(np.float32)
            m = np.where(m > THRESHOLD, 0.0, m)
            
            logits = W_spike @ h + W_membrane @ m + bias
            probs = softmax_numba(logits / temperature)
            
            try:
                next_token = np.random.choice(len(probs), p=probs)
            except:
                next_token = np.argmax(logits)
            
            generated.append(next_token)
        
        return "".join([idx_to_word.get(t, '?') for t in generated])
    
    # === 多様性テスト ===
    print("\n" + "=" * 60)
    print("🎲 多様性テスト（同じプロンプトで20回生成）")
    print("=" * 60)
    
    prompt = "人工知能"
    unique_outputs = set()
    results = []
    
    print(f"\nプロンプト: 「{prompt}」\n")
    for i in range(20):
        result = generate(prompt, max_words=15, temperature=0.9)
        results.append(result)
        unique_outputs.add(result)
        print(f"  {i+1:2d}: {result[:60]}...")
    
    print(f"\n🎯 ユニーク出力数: {len(unique_outputs)}/20")
    
    # === 様々なプロンプトで生成 ===
    print("\n" + "=" * 60)
    print("📝 様々なプロンプトで生成")
    print("=" * 60)
    
    prompts = ["機械学習", "深層学習", "脳", "日本語", "ロボット", "エネルギー"]
    for p in prompts:
        r = generate(p, max_words=12, temperature=0.9)
        print(f"\n【{p}】→ {r[:50]}...")
    
    # === 比較 ===
    print("\n" + "=" * 60)
    print("📊 結果サマリー")
    print("=" * 60)
    print(f"""
    語彙サイズ: {vocab_size}
    Dropout率: {model.get('dropout_rate', 0.25)*100:.0f}%
    多様性 (20回中ユニーク): {len(unique_outputs)}/20
    
    【旧モデルとの比較】
    階層型モデル: 1/10 (10%)
    今回モデル: {len(unique_outputs)}/20 ({len(unique_outputs)/20*100:.0f}%)
    """)


if __name__ == "__main__":
    main()
