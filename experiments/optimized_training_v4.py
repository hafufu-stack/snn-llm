#!/usr/bin/env python3
"""
最適化SNN学習 v4
発見した最適パラメータを適用

変更点:
- input_scale = 10.0 (埋め込みを10倍)
- leak = 0.99 (膜電位の漏れを減少)
- threshold = 1.0 (そのまま)
- 24スレッド並列処理
"""

import numpy as np
import pickle
import json
import time
from pathlib import Path
from numba import njit, prange, set_num_threads

set_num_threads(24)  # 全コア使用

# Paths
MODEL_PATH = Path(__file__).parent.parent / "checkpoints" / "model_v4_parallel_final.pkl"
TOKENIZER_PATH = Path(__file__).parent.parent / "checkpoints" / "tokenizer_v4_parallel.json"
RESULTS_DIR = Path(__file__).parent.parent / "results"
CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"

# 最適パラメータ
INPUT_SCALE = 10.0
LEAK = 0.99
THRESHOLD = 1.0

# Teacher examples (Claude風)
TEACHER_EXAMPLES = [
    ("人工知能は", "人工知能は、人間の知能を模倣するコンピュータシステムです。機械学習や深層学習によって実現されます。"),
    ("スパイキングニューラルネットワークとは", "スパイキングニューラルネットワークとは、生物の神経細胞を模倣した計算モデルです。スパイクで情報を伝達します。"),
    ("深層学習とは", "深層学習とは、多層のニューラルネットワークを用いた機械学習手法です。画像認識や自然言語処理に使われます。"),
    ("日本語の特徴は", "日本語の特徴は、ひらがな、カタカナ、漢字の三種類の文字体系を持つことです。文法はSOV型です。"),
    ("脳とコンピュータの違いは", "脳とコンピュータの違いは、処理方式にあります。脳は並列分散処理を行い、低消費電力で動作します。"),
    ("機械学習とは", "機械学習とは、データからパターンを学習するアルゴリズムです。教師あり学習、教師なし学習があります。"),
    ("ニューロンとは", "ニューロンとは、神経系を構成する細胞です。電気信号を発火して情報を伝達します。"),
    ("吾輩は猫で", "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。"),
    ("今日の天気は", "今日の天気は晴れです。気温は二十度で過ごしやすい一日になるでしょう。"),
    ("プログラミングとは", "プログラミングとは、コンピュータに指示を与えるためのコードを書くことです。様々な言語があります。"),
    ("Transformerとは", "Transformerとは、注意機構を用いた深層学習モデルです。自然言語処理で広く使われています。"),
    ("GPUとは", "GPUとは、グラフィック処理用に設計された並列処理装置です。深層学習の高速化に使われます。"),
    ("言語モデルとは", "言語モデルとは、テキストの確率分布を学習するモデルです。次の単語を予測することで文章を生成します。"),
    ("強化学習とは", "強化学習とは、報酬を最大化する行動を学習する手法です。ゲームやロボット制御に応用されます。"),
    ("11次元ハイパーキューブとは", "11次元ハイパーキューブとは、脳のニューロン構造を模倣した接続パターンです。2048ノードを効率的に接続します。"),
]


@njit(cache=True)
def softmax_numba(x):
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / (np.sum(exp_x) + 1e-10)


@njit(parallel=True, cache=True)
def optimized_forward(embedding_scaled, W_res, W_spike, W_membrane, bias,
                      token_ids, hidden_dim, vocab_size, leak):
    """最適化されたforward pass"""
    n_tokens = len(token_ids)
    h = np.zeros(hidden_dim, dtype=np.float32)
    m = np.zeros(hidden_dim, dtype=np.float32)
    
    all_logits = np.zeros((n_tokens, vocab_size), dtype=np.float32)
    all_h = np.zeros((n_tokens, hidden_dim), dtype=np.float32)
    
    for t in range(n_tokens):
        tid = token_ids[t]
        x = embedding_scaled[tid]
        
        # 膜電位更新（漏れを減少）
        for i in prange(hidden_dim):
            m[i] = leak * m[i] + x[i]
            for j in range(hidden_dim):
                m[i] += W_res[i, j] * h[j]
        
        # 発火判定
        for i in prange(hidden_dim):
            if m[i] > 1.0:
                h[i] = 1.0
                m[i] = 0.0
            else:
                h[i] = 0.0
        
        all_h[t] = h.copy()
        
        # 出力計算
        for i in prange(vocab_size):
            all_logits[t, i] = bias[i]
            for j in range(hidden_dim):
                all_logits[t, i] += W_spike[i, j] * h[j]
                all_logits[t, i] += W_membrane[i, j] * h[j]
    
    return all_logits, all_h


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║    🚀 最適化SNN学習 v4 - 発火パラメータ最適化版 🚀             ║
    ║                                                               ║
    ║    変更: input_scale=10, leak=0.99                           ║
    ║    並列処理: 24スレッド                                       ║
    ║    目標: 1文丸暗記テストをクリア                              ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Load model
    print("モデルをロード中...")
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
    
    with open(TOKENIZER_PATH, 'r', encoding='utf-8') as f:
        tokenizer_data = json.load(f)
    
    token_to_idx = tokenizer_data['token_to_idx']
    idx_to_token = {v: k for k, v in token_to_idx.items()}
    vocab_size = model_data['vocab_size']
    hidden_dim = model_data['hidden_dim']
    
    # 重み取得＆スケーリング
    embedding = model_data['embedding'].astype(np.float32)
    embedding_scaled = embedding * INPUT_SCALE  # スケールアップ！
    W_res = model_data['W_res'].astype(np.float32)
    W_spike = model_data['W_spike'].astype(np.float32)
    W_membrane = model_data['W_membrane'].astype(np.float32)
    bias = model_data['bias'].astype(np.float32)
    
    print(f"語彙サイズ: {vocab_size}")
    print(f"隠れ次元: {hidden_dim}")
    print(f"入力スケール: {INPUT_SCALE}x")
    print(f"リーク係数: {LEAK}")
    print(f"Teacher例: {len(TEACHER_EXAMPLES)}種類")
    
    # JIT compile
    print("\nJITコンパイル中...")
    dummy_ids = np.array([0, 1, 2], dtype=np.int64)
    _ = optimized_forward(embedding_scaled, W_res, W_spike, W_membrane, bias,
                          dummy_ids, hidden_dim, vocab_size, LEAK)
    print("JIT完了!")
    
    # Tokenize
    def tokenize(text):
        return np.array([token_to_idx.get(c, 0) for c in text], dtype=np.int64)
    
    teacher_data = []
    for prompt, response in TEACHER_EXAMPLES:
        full_text = prompt + response
        tokens = tokenize(full_text)
        teacher_data.append(tokens)
    
    print("\n" + "=" * 60)
    print("🧠 10000エポック学習開始！")
    print("=" * 60)
    
    lr = 0.005  # 学習率
    epochs = 10000
    losses = []
    best_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        total_firing = 0
        total_steps = 0
        
        np.random.shuffle(teacher_data)
        
        for tokens in teacher_data:
            if len(tokens) < 2:
                continue
            
            logits, h_all = optimized_forward(
                embedding_scaled, W_res, W_spike, W_membrane, bias,
                tokens[:-1], hidden_dim, vocab_size, LEAK
            )
            
            total_firing += np.sum(h_all > 0)
            total_steps += len(h_all)
            
            for t in range(len(tokens) - 1):
                probs = softmax_numba(logits[t])
                target = tokens[t + 1]
                
                loss = -np.log(probs[target] + 1e-10)
                epoch_loss += loss
                
                error = probs.copy()
                error[target] -= 1.0
                
                W_spike -= lr * np.outer(error, np.ones(hidden_dim, dtype=np.float32))
                W_membrane -= lr * np.outer(error, np.ones(hidden_dim, dtype=np.float32))
                bias -= lr * error
        
        avg_loss = epoch_loss / sum(len(t) - 1 for t in teacher_data)
        avg_firing = total_firing / total_steps if total_steps > 0 else 0
        losses.append(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
        
        if epoch % 100 == 0:
            elapsed = time.time() - start_time
            speed = epoch / elapsed
            remaining = (epochs - epoch) / speed
            
            # Generate sample
            sample_prompt = "人工知能"
            sample_tokens = list(tokenize(sample_prompt))
            h = np.zeros(hidden_dim, dtype=np.float32)
            m = np.zeros(hidden_dim, dtype=np.float32)
            
            for _ in range(30):
                tid = sample_tokens[-1]
                x = embedding_scaled[tid]
                m = LEAK * m + x + W_res @ h
                h = np.where(m > THRESHOLD, 1.0, 0.0).astype(np.float32)
                m = np.where(m > THRESHOLD, 0.0, m)
                
                logits_gen = W_spike @ h + W_membrane @ h + bias
                probs = softmax_numba(logits_gen / 0.7)
                next_token = np.random.choice(len(probs), p=probs)
                sample_tokens.append(next_token)
                
                if idx_to_token.get(next_token) == '。':
                    break
            
            sample_text = "".join([idx_to_token.get(t, '?') for t in sample_tokens[len(sample_prompt):]])
            
            print(f"Epoch {epoch:5d}/10000 | Loss: {avg_loss:.3f} | "
                  f"発火: {avg_firing:.1f} | "
                  f"速度: {speed:.1f} ep/s | "
                  f"残り: {remaining/60:.0f}分")
            print(f"  Sample: 「{sample_prompt}」→ {sample_text[:40]}...")
        
        # Checkpoint
        if epoch % 1000 == 0:
            checkpoint_data = {
                'vocab_size': vocab_size,
                'hidden_dim': hidden_dim,
                'embedding': embedding,  # オリジナル（スケールは学習時に適用）
                'W_res': W_res,
                'W_spike': W_spike,
                'W_membrane': W_membrane,
                'bias': bias,
                'input_scale': INPUT_SCALE,
                'leak': LEAK,
            }
            cp_path = CHECKPOINT_DIR / f'model_optimized_v4_{epoch}epochs.pkl'
            with open(cp_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            print(f"  💾 チェックポイント: {cp_path.name}")
    
    # Final save
    final_data = {
        'vocab_size': vocab_size,
        'hidden_dim': hidden_dim,
        'embedding': embedding,
        'W_res': W_res,
        'W_spike': W_spike,
        'W_membrane': W_membrane,
        'bias': bias,
        'input_scale': INPUT_SCALE,
        'leak': LEAK,
    }
    
    final_path = CHECKPOINT_DIR / 'model_optimized_v4_10000epochs.pkl'
    with open(final_path, 'wb') as f:
        pickle.dump(final_data, f)
    
    # Results
    results = {
        'epochs': epochs,
        'final_loss': losses[-1],
        'best_loss': best_loss,
        'input_scale': INPUT_SCALE,
        'leak': LEAK,
        'training_time_sec': time.time() - start_time,
    }
    
    with open(RESULTS_DIR / 'optimized_v4_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 60)
    print("📊 学習完了!")
    print("=" * 60)
    print(f"最終Loss: {losses[-1]:.3f}")
    print(f"ベストLoss: {best_loss:.3f}")
    print(f"総時間: {(time.time() - start_time)/3600:.1f}時間")
    print(f"モデル保存: {final_path}")


if __name__ == "__main__":
    main()
