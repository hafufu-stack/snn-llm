#!/usr/bin/env python3
"""
Super Teacher Learning v3 - 10000エポック長期学習
並列処理で高速化、Super Teacher v2モデルから継続

目標: Loss 5.14 → さらに低く！
"""

import numpy as np
import pickle
import json
import time
from pathlib import Path
from numba import njit, prange, set_num_threads

set_num_threads(24)  # Ryzen AI 9 HX 375 全コア

# Paths
MODEL_PATH = Path(__file__).parent.parent / "checkpoints" / "model_super_teacher_v2.pkl"
TOKENIZER_PATH = Path(__file__).parent.parent / "checkpoints" / "tokenizer_v4_parallel.json"
RESULTS_DIR = Path(__file__).parent.parent / "results"
CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"

# Teacher examples (Claude風)
TEACHER_EXAMPLES = [
    ("人工知能とは", "人工知能とは、人間の知能を模倣するコンピュータシステムです。"),
    ("スパイキングニューラルネットワークは", "スパイキングニューラルネットワークは、生物の神経細胞を模倣した計算モデルです。"),
    ("深層学習とは", "深層学習とは、多層のニューラルネットワークを用いた機械学習手法です。"),
    ("自然言語処理とは", "自然言語処理とは、人間の言語をコンピュータで処理する技術です。"),
    ("Transformerとは", "Transformerとは、注意機構を用いた深層学習モデルです。"),
    ("日本語の文法は", "日本語の文法は、主語・目的語・動詞の順序を持つSOV型言語です。"),
    ("機械学習とは", "機械学習とは、データから規則性を学習するアルゴリズムです。"),
    ("ニューロンとは", "ニューロンとは、神経系を構成する細胞で、電気信号を伝達します。"),
    ("GPUとは", "GPUとは、グラフィック処理用に設計された並列処理装置です。"),
    ("Pythonとは", "Pythonとは、読みやすさを重視したプログラミング言語です。"),
    ("脳科学とは", "脳科学とは、脳の構造と機能を研究する学問分野です。"),
    ("11次元ハイパーキューブとは", "11次元ハイパーキューブとは、脳のニューロン構造を模倣した接続パターンです。"),
    ("海馬とは", "海馬とは、記憶の形成に重要な役割を持つ脳の部位です。"),
    ("歯状回とは", "歯状回とは、海馬の入口に位置し、パターン分離を行う脳領域です。"),
    ("長期増強とは", "長期増強とは、シナプスの伝達効率が持続的に増加する現象です。"),
    ("バックプロパゲーションとは", "バックプロパゲーションとは、誤差を逆伝播させて重みを更新する学習法です。"),
    ("活性化関数とは", "活性化関数とは、ニューラルネットワークに非線形性を導入する関数です。"),
    ("損失関数とは", "損失関数とは、モデルの予測と正解の誤差を測る関数です。"),
    ("過学習とは", "過学習とは、訓練データに過剰適合し汎化性能が低下する現象です。"),
    ("正則化とは", "正則化とは、過学習を防ぐためにモデルの複雑さを制約する手法です。"),
    ("バッチ正規化とは", "バッチ正規化とは、層の入力を正規化して学習を安定させる手法です。"),
    ("注意機構とは", "注意機構とは、入力の重要な部分に集中する仕組みです。"),
    ("埋め込みとは", "埋め込みとは、離散的なデータを連続的なベクトルに変換する手法です。"),
    ("言語モデルとは", "言語モデルとは、テキストの確率分布を学習するモデルです。"),
    ("強化学習とは", "強化学習とは、報酬を最大化する行動を学習する手法です。"),
]


@njit(cache=True)
def softmax_numba(x):
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / (np.sum(exp_x) + 1e-10)


@njit(parallel=True, cache=True)
def parallel_forward(embedding, W_res, W_spike, W_membrane, bias, 
                     token_ids, hidden_dim, vocab_size):
    """高速並列forward (float32)"""
    n_tokens = len(token_ids)
    state = np.zeros(hidden_dim, dtype=np.float32)
    total_logits = np.zeros((n_tokens, vocab_size), dtype=np.float32)
    
    for t in range(n_tokens):
        token_id = token_ids[t]
        if token_id < embedding.shape[0]:
            x = embedding[token_id]
        else:
            x = np.zeros(hidden_dim, dtype=np.float32)
        
        new_state = np.zeros(hidden_dim, dtype=np.float32)
        for i in prange(hidden_dim):
            res_sum = np.float32(0.0)
            for j in range(hidden_dim):
                res_sum += W_res[i, j] * state[j]
            new_state[i] = np.float32(0.9) * state[i] + np.float32(0.1) * (x[i] + res_sum)
            if new_state[i] > np.float32(0.5):
                new_state[i] = np.float32(1.0)
            else:
                new_state[i] = np.float32(0.0)
        
        state = new_state
        
        logits = np.zeros(vocab_size, dtype=np.float32)
        for v in prange(vocab_size):
            for h in range(hidden_dim):
                logits[v] += W_spike[v, h] * state[h]
                logits[v] += W_membrane[v, h] * state[h]
            logits[v] += bias[v]
        
        total_logits[t] = logits
    
    return total_logits, state


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║    🎓 Super Teacher Learning v3 - 10000エポック長期学習 🎓     ║
    ║                                                               ║
    ║    ベース: Super Teacher v2 (Loss 5.14)                       ║
    ║    目標: さらなるLoss低下と生成品質向上                       ║
    ║    並列処理: 24スレッド (Ryzen AI 9 HX 375)                   ║
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
    
    # Get weights - float32 for better CPU cache efficiency
    embedding = model_data['embedding'].astype(np.float32)
    W_res = model_data['W_res'].astype(np.float32)
    W_spike = model_data['W_spike'].astype(np.float32)
    W_membrane = model_data['W_membrane'].astype(np.float32)
    bias = model_data['bias'].astype(np.float32)
    
    print(f"語彙サイズ: {vocab_size}")
    print(f"隠れ次元: {hidden_dim}")
    print(f"パラメータ数: {embedding.size + W_res.size + W_spike.size + W_membrane.size + bias.size:,}")
    print(f"Teacher例: {len(TEACHER_EXAMPLES)}種類")
    
    # JIT compile
    print("\nJITコンパイル中...")
    dummy_ids = np.array([0, 1, 2])
    _ = parallel_forward(embedding, W_res, W_spike, W_membrane, bias, 
                         dummy_ids, hidden_dim, vocab_size)
    print("JIT完了!")
    
    # Tokenize teacher examples
    def tokenize(text):
        return np.array([token_to_idx.get(c, 0) for c in text])
    
    teacher_data = []
    for prompt, response in TEACHER_EXAMPLES:
        full_text = prompt + response
        tokens = tokenize(full_text)
        teacher_data.append(tokens)
    
    print("\n" + "="*60)
    print("🧠 10000エポック学習開始！")
    print("="*60)
    
    lr = 0.0005  # 学習率（少し下げて安定化）
    epochs = 10000
    losses = []
    best_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        
        # Shuffle
        np.random.shuffle(teacher_data)
        
        for tokens in teacher_data:
            if len(tokens) < 2:
                continue
            
            # Forward
            logits, _ = parallel_forward(
                embedding, W_res, W_spike, W_membrane, bias,
                tokens[:-1], hidden_dim, vocab_size
            )
            
            # Calculate loss and update
            for t in range(len(tokens) - 1):
                probs = softmax_numba(logits[t])
                target = tokens[t + 1]
                
                loss = -np.log(probs[target] + 1e-10)
                epoch_loss += loss
                
                # Gradient update
                error = probs.copy()
                error[target] -= 1.0
                
                W_spike -= lr * np.outer(error, np.ones(hidden_dim))
                W_membrane -= lr * np.outer(error, np.ones(hidden_dim))
                bias -= lr * error
        
        avg_loss = epoch_loss / sum(len(t)-1 for t in teacher_data)
        losses.append(avg_loss)
        
        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
        
        # Progress
        if epoch % 100 == 0:
            elapsed = time.time() - start_time
            speed = epoch / elapsed
            remaining = (epochs - epoch) / speed
            
            # Generate sample
            sample_prompt = "人工知能"
            sample_tokens = list(tokenize(sample_prompt))
            state = np.zeros(hidden_dim)
            
            for _ in range(30):
                logits, state = parallel_forward(
                    embedding, W_res, W_spike, W_membrane, bias,
                    np.array(sample_tokens[-10:]), hidden_dim, vocab_size
                )
                probs = softmax_numba(logits[-1] / 0.8)
                next_token = np.random.choice(len(probs), p=probs)
                sample_tokens.append(next_token)
            
            sample_text = "".join([idx_to_token.get(t, '?') for t in sample_tokens[len(sample_prompt):]])
            
            print(f"Epoch {epoch:5d}/10000 | Loss: {avg_loss:.3f} | "
                  f"Best: {best_loss:.3f} | "
                  f"速度: {speed:.1f} ep/s | "
                  f"残り: {remaining/60:.0f}分")
            print(f"  Sample: 「{sample_prompt}」→ {sample_text[:35]}...")
        
        # Checkpoint every 1000
        if epoch % 1000 == 0:
            checkpoint_data = {
                'vocab_size': vocab_size,
                'hidden_dim': hidden_dim,
                'embedding': embedding,
                'W_res': W_res,
                'W_spike': W_spike,
                'W_membrane': W_membrane,
                'bias': bias,
            }
            cp_path = CHECKPOINT_DIR / f'model_teacher_v3_{epoch}epochs.pkl'
            with open(cp_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            print(f"  💾 チェックポイント保存: {cp_path.name}")
    
    # Final save
    final_data = {
        'vocab_size': vocab_size,
        'hidden_dim': hidden_dim,
        'embedding': embedding,
        'W_res': W_res,
        'W_spike': W_spike,
        'W_membrane': W_membrane,
        'bias': bias,
    }
    
    final_path = CHECKPOINT_DIR / 'model_teacher_v3_10000epochs.pkl'
    with open(final_path, 'wb') as f:
        pickle.dump(final_data, f)
    
    # Results
    results = {
        'epochs': epochs,
        'final_loss': losses[-1],
        'best_loss': best_loss,
        'losses': losses,
        'training_time_sec': time.time() - start_time,
    }
    
    with open(RESULTS_DIR / 'teacher_v3_10000epochs_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*60)
    print("📊 学習完了!")
    print("="*60)
    print(f"最終Loss: {losses[-1]:.3f}")
    print(f"ベストLoss: {best_loss:.3f}")
    print(f"総時間: {(time.time() - start_time)/3600:.1f}時間")
    print(f"モデル保存: {final_path}")


if __name__ == "__main__":
    main()
