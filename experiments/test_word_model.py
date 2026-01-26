#!/usr/bin/env python3
"""
単語単位SNN-LLM テストスクリプト
様々なプロンプトで生成テスト
"""

import numpy as np
import pickle
import json
from pathlib import Path
from numba import njit

# Paths
MODEL_PATH = Path(__file__).parent.parent / "checkpoints" / "model_word_v3_final.pkl"
TOKENIZER_PATH = Path(__file__).parent.parent / "checkpoints" / "tokenizer_word_v3.json"


@njit(cache=True)
def softmax_numba(x):
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / (np.sum(exp_x) + 1e-10)


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║    🧪 単語単位 SNN-LLM テスト                                  ║
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
    
    print(f"語彙サイズ: {vocab_size}")
    print(f"入力スケール: {input_scale}")
    print(f"リーク係数: {leak}")
    
    # MeCab for tokenization
    try:
        import fugashi
        tagger = fugashi.Tagger()
        print("✅ fugashi (MeCab) ロード成功")
    except:
        print("❌ fugashi ロード失敗")
        return
    
    def tokenize(text):
        words = []
        for word in tagger(text):
            if word.surface.strip():
                words.append(word.surface)
        return words
    
    def generate(prompt, max_words=20, temperature=0.7):
        """テキスト生成"""
        prompt_words = tokenize(prompt)
        prompt_ids = [word_to_idx.get(w, 1) for w in prompt_words]
        
        h = np.zeros(hidden_dim, dtype=np.float32)
        m = np.zeros(hidden_dim, dtype=np.float32)
        
        # Process prompt
        for tid in prompt_ids:
            if tid >= vocab_size:
                tid = 1
            x = embedding_scaled[tid]
            m = leak * m + x + W_res @ h
            h = np.where(m > 1.0, 1.0, 0.0).astype(np.float32)
            m = np.where(m > 1.0, 0.0, m)
        
        # Generate
        generated = list(prompt_ids)
        for _ in range(max_words):
            tid = generated[-1] if generated[-1] < vocab_size else 0
            x = embedding_scaled[tid]
            m = leak * m + x + W_res @ h
            h = np.where(m > 1.0, 1.0, 0.0).astype(np.float32)
            m = np.where(m > 1.0, 0.0, m)
            
            logits = W_spike @ h + W_membrane @ m + bias
            probs = softmax_numba(logits / temperature)
            
            try:
                next_token = np.random.choice(len(probs), p=probs)
            except:
                next_token = np.argmax(logits)
            
            generated.append(next_token)
            
            # Stop at EOS or period
            word = idx_to_word.get(next_token, '')
            if word in ['。', '<EOS>']:
                break
        
        return "".join([idx_to_word.get(t, '?') for t in generated])
    
    # Test prompts
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
    
    print("\n" + "=" * 60)
    print("📝 生成テスト")
    print("=" * 60)
    
    for prompt in test_prompts:
        result = generate(prompt, max_words=25, temperature=0.7)
        print(f"\n【{prompt}】")
        print(f"  → {result}")
    
    # Interactive mode
    print("\n" + "=" * 60)
    print("💬 インタラクティブモード (終了: quit)")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("\nプロンプト: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            if not user_input:
                continue
            
            result = generate(user_input, max_words=30, temperature=0.7)
            print(f"  → {result}")
        except KeyboardInterrupt:
            break
        except EOFError:
            break
    
    print("\n👋 テスト終了！")


if __name__ == "__main__":
    main()
