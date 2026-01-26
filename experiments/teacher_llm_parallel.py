#!/usr/bin/env python3
"""
Teacher LLM Learning - Parallel Version
========================================
Uses Numba parallelization for faster training.
Resume from epoch 21 to epoch 50.
"""

import numpy as np
import pickle
import json
from pathlib import Path
import time
from numba import njit, prange, set_num_threads

# Use 24 threads (全コア使用 for 95% CPU)
set_num_threads(24)

# Paths
MODEL_PATH = Path(__file__).parent.parent / "checkpoints" / "model_super_teacher_v2.pkl"
TOKENIZER_PATH = Path(__file__).parent.parent / "checkpoints" / "tokenizer_v4_parallel.json"
RESULTS_DIR = Path(__file__).parent.parent / "results"

# Teacher examples - expanded with more diverse content
TEACHER_EXAMPLES = [
    # Original examples
    ("人工知能は", "人間の知能を模倣するコンピュータシステムです。機械学習や深層学習によって、画像認識や自然言語処理などの複雑なタスクを実行できます。"),
    ("脳とコンピュータの違いは", "処理方式にあります。脳は並列分散処理を行い、低消費電力で柔軟な学習が可能です。一方コンピュータは逐次処理が基本で、高速ですが消費電力が大きいです。"),
    ("スパイキングニューラルネットワークとは", "生物の神経細胞の動作を模倣したAIモデルです。ニューロンがスパイク（電気信号）を発火することで情報を処理します。従来のニューラルネットワークより省エネルギーです。"),
    ("言語を理解するとは", "単語の意味だけでなく、文脈や話者の意図、文化的背景までを把握することです。人間は経験を通じて言語を習得しますが、AIは大量のテキストデータから統計的パターンを学習します。"),
    ("機械学習の仕組みは", "データからパターンを見つけ出し、新しい入力に対して予測や分類を行うことです。教師あり学習、教師なし学習、強化学習の三種類があります。"),
    ("ハイパーキューブトポロジーの利点は", "ノード間の最短経路が短く、効率的な情報伝播が可能です。11次元ハイパーキューブでは、2048ノードが11ホップ以内で接続され、スケーラビリティと通信効率のバランスが取れています。"),
    ("自然言語処理において", "テキストから意味を抽出し、適切な応答を生成することが重要です。形態素解析、構文解析、意味解析などの段階を経て、人間の言語を機械が理解できる形に変換します。"),
    ("深層学習が成功した理由は", "ビッグデータの利用可能性、GPUによる計算能力の向上、そしてアルゴリズムの改良です。特にバックプロパゲーションと確率的勾配降下法が効率的な学習を可能にしました。"),
    ("今後のAI研究の方向性は", "説明可能なAI、省エネルギー型AI、そして汎用人工知能の実現です。倫理的な課題にも取り組む必要があり、人間と協調できるAIの開発が求められています。"),
    ("私たちの研究では", "スパイキングニューラルネットワークを用いた大規模言語モデルの構築を目指しています。11次元ハイパーキューブトポロジーと階層的メモリ構造により、効率的な学習が可能になりました。"),
    # NEW examples - technical AI concepts
    ("ニューロモーフィックコンピューティングとは", "脳の構造と機能を模倣したコンピューティングパラダイムです。従来のフォン・ノイマンアーキテクチャと異なり、メモリと処理が一体化されており、省電力で並列処理が可能です。"),
    ("GPUが深層学習に適している理由は", "大量の行列演算を並列に実行できるからです。数千のコアが同時に計算を行い、ニューラルネットワークの学習を大幅に高速化します。"),
    ("Transformerアーキテクチャの特徴は", "自己注意機構によって入力シーケンス全体の依存関係を捉えることができます。これにより長距離の文脈を効率的に学習でき、現代の大規模言語モデルの基盤となっています。"),
    ("強化学習において", "エージェントは環境と相互作用しながら報酬を最大化する方策を学習します。試行錯誤を通じて最適な行動を発見し、ゲームやロボット制御などに応用されています。"),
    ("埋め込み表現とは", "単語や文を高次元ベクトル空間に変換したものです。意味的に類似した単語は近い位置にマッピングされ、言語の意味的関係を数学的に表現できます。"),
    # NEW examples - Japanese language and culture
    ("日本語の特徴は", "ひらがな、カタカナ、漢字の三種類の文字体系を持つことです。また、主語を省略することが多く、敬語表現が発達しています。文法構造は英語と異なりSOV型です。"),
    ("日本のAI研究の歴史は", "1980年代の第五世代コンピュータプロジェクトから始まりました。現在は自動運転、医療診断、創薬など幅広い分野で研究が進められています。"),
    ("日本語の形態素解析では", "MeCabやJumanなどのツールが広く使われています。日本語は単語の区切りがないため、正確な分かち書きが自然言語処理の重要な前処理となります。"),
    # NEW examples - SNN specifics
    ("スパイクタイミング依存可塑性とは", "ニューロン間のシナプス結合強度がスパイクのタイミングに依存して変化する現象です。これはヘブの法則を時間的に精密化したもので、学習の生物学的基盤となっています。"),
    ("リーキーインテグレートアンドファイアモデルは", "ニューロンをシンプルに模倣した数理モデルです。入力を膜電位として積分し、閾値を超えるとスパイクを発火します。ニューロン群のダイナミクスを効率的にシミュレートできます。"),
    ("ニューロモーフィックチップの例として", "IntelのLoihiやIBMのTrueNorthがあります。これらは従来のGPUより数桁少ない消費電力でスパイキングニューラルネットワークを実行できます。"),
    # NEW examples - research methodology
    ("パープレキシティとは", "言語モデルの性能を評価する指標の一つです。モデルがテキストをどれだけ予測できたかを測り、値が低いほど性能が良いことを意味します。"),
    ("ファインチューニングとは", "事前学習済みモデルを特定のタスクに適応させる手法です。少量のデータでも高い性能を達成でき、転移学習の主要な手法となっています。"),
    ("モデルの軽量化手法として", "知識蒸留、量子化、プルーニングがあります。これにより大規模モデルをエッジデバイスで実行可能にし、応答時間と消費電力を削減できます。"),
    ("研究論文を書く際に重要なのは", "問題設定を明確にし、提案手法の新規性を示すことです。実験結果は再現可能な形式で報告し、関連研究との比較を通じて貢献を明確にします。"),
]


@njit(cache=True)
def softmax_numba(x):
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / (np.sum(exp_x) + 1e-10)


@njit(parallel=True, cache=True)
def parallel_forward(embedding, W_res, W_spike, W_membrane, bias, 
                     token_ids, hidden_dim, vocab_size):
    """Parallel forward pass through sequence."""
    n_tokens = len(token_ids)
    h = np.zeros(hidden_dim, dtype=np.float32)
    m = np.zeros(hidden_dim, dtype=np.float32)
    
    all_logits = np.zeros((n_tokens, vocab_size), dtype=np.float32)
    all_h = np.zeros((n_tokens, hidden_dim), dtype=np.float32)
    all_m = np.zeros((n_tokens, hidden_dim), dtype=np.float32)
    
    for t in range(n_tokens):
        tid = token_ids[t]
        x = embedding[tid] * 10.0  # 入力スケール10倍（最適化）
        
        # Parallel membrane update
        for i in prange(hidden_dim):
            m[i] = 0.99 * m[i] + x[i]  # リーク0.99（最適化）
            for j in range(hidden_dim):
                m[i] += W_res[i, j] * h[j]
        
        # Spiking
        for i in prange(hidden_dim):
            if m[i] > 1.0:
                h[i] = 1.0
                m[i] = 0.0
            else:
                h[i] = 0.0
        
        all_h[t] = h.copy()
        all_m[t] = m.copy()
        
        # Parallel logits
        for i in prange(vocab_size):
            logit = bias[i]
            for j in range(hidden_dim):
                logit += W_spike[i, j] * h[j] + W_membrane[i, j] * m[j]
            all_logits[t, i] = logit
    
    return all_logits, all_h, all_m


@njit(parallel=True, cache=True)
def parallel_weight_update(W_spike, W_membrane, bias, 
                           errors, all_h, all_m, lr):
    """Parallel weight updates."""
    n_tokens, hidden_dim = all_h.shape
    vocab_size = len(bias)
    
    for t in range(n_tokens):
        error = errors[t]
        h = all_h[t]
        m = all_m[t]
        
        for i in prange(vocab_size):
            if abs(error[i]) > 1e-6:
                bias[i] += lr * error[i] * 0.01
                for j in range(hidden_dim):
                    W_spike[i, j] += lr * error[i] * h[j] * 0.01
                    W_membrane[i, j] += lr * error[i] * m[j] * 0.01


class ParallelStudentSNN:
    def __init__(self, model_path, tokenizer_path, lr=0.01):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            tok_data = json.load(f)
        
        self.token_to_id = tok_data['token_to_idx']
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.vocab_size = model['vocab_size']
        self.hidden_dim = model['hidden_dim']
        self.embedding = np.ascontiguousarray(model['embedding'].astype(np.float32))
        self.W_res = np.ascontiguousarray(model['W_res'].astype(np.float32))
        self.W_spike = np.ascontiguousarray(model['W_spike'].astype(np.float32))
        self.W_membrane = np.ascontiguousarray(model['W_membrane'].astype(np.float32))
        self.bias = np.ascontiguousarray(model['bias'].astype(np.float32))
        self.lr = lr
        self.losses = []
    
    def encode(self, text):
        tokens = []
        i = 0
        while i < len(text):
            found = False
            for length in range(min(5, len(text) - i), 0, -1):
                if text[i:i+length] in self.token_to_id:
                    tokens.append(self.token_to_id[text[i:i+length]])
                    i += length
                    found = True
                    break
            if not found:
                tokens.append(self.token_to_id.get('<UNK>', 1))
                i += 1
        return np.array(tokens, dtype=np.int32)
    
    def decode(self, ids):
        return ''.join(self.id_to_token.get(int(t), '') for t in ids 
                      if not self.id_to_token.get(int(t), '').startswith('<'))
    
    def train_example(self, prompt_tokens, completion_tokens):
        """Train on a single example using parallel ops."""
        all_tokens = np.concatenate([prompt_tokens, completion_tokens])
        
        # Forward pass
        all_logits, all_h, all_m = parallel_forward(
            self.embedding, self.W_res, self.W_spike, self.W_membrane, self.bias,
            all_tokens, self.hidden_dim, self.vocab_size
        )
        
        # Compute errors for completion part
        start_idx = len(prompt_tokens)
        n_completion = len(completion_tokens)
        errors = np.zeros((n_completion, self.vocab_size), dtype=np.float32)
        total_loss = 0.0
        
        for i in range(n_completion):
            logits = all_logits[start_idx + i - 1]  # Previous position
            probs = softmax_numba(logits)
            target = completion_tokens[i]
            loss = -np.log(probs[target] + 1e-10)
            total_loss += loss
            errors[i, target] = 1.0 - probs[target]
        
        # Update weights
        parallel_weight_update(
            self.W_spike, self.W_membrane, self.bias,
            errors, all_h[start_idx-1:start_idx-1+n_completion], 
            all_m[start_idx-1:start_idx-1+n_completion], self.lr
        )
        
        return total_loss / max(n_completion, 1)
    
    def generate(self, prompt, max_length=50, temperature=0.7):
        tokens = self.encode(prompt)
        h = np.zeros(self.hidden_dim, dtype=np.float32)
        m = np.zeros(self.hidden_dim, dtype=np.float32)
        
        for tid in tokens:
            x = self.embedding[tid] * 10.0  # スケール10倍
            m = 0.99 * m + self.W_res @ h + x  # リーク0.99
            h = (m > 1.0).astype(np.float32)
            m = m * (1 - h)
        
        generated = []
        last = int(tokens[-1]) if len(tokens) > 0 else 0
        
        for _ in range(max_length):
            x = self.embedding[last] * 10.0  # スケール10倍
            m = 0.99 * m + self.W_res @ h + x  # リーク0.99
            h = (m > 1.0).astype(np.float32)
            m = m * (1 - h)
            
            logits = (self.W_spike @ h + self.W_membrane @ m + self.bias) / temperature
            logits = np.nan_to_num(logits, nan=0.0, posinf=100, neginf=-100)
            logits = logits - np.max(logits)
            exp_logits = np.exp(np.clip(logits, -50, 50))
            probs = exp_logits / (np.sum(exp_logits) + 1e-10)
            
            # Check for NaN and fallback to greedy
            if np.any(np.isnan(probs)) or np.abs(probs.sum() - 1.0) > 0.01:
                next_token = int(np.argmax(logits))
            else:
                probs = probs / probs.sum()
                next_token = np.random.choice(len(probs), p=probs)
            
            generated.append(next_token)
            last = next_token
            
            if self.id_to_token.get(next_token, '') in ['。', '<EOS>']:
                break
        
        return self.decode(generated)


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║    ⚡ 教師LLM学習 並列版 (Epoch 1-10000) ⚡                    ║
    ║    24スレッド並列処理 + 25教師例 (95% CPU目標)              ║
    ║    ベース: Super Teacher v2 (Loss 5.14)                      ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    if not MODEL_PATH.exists():
        print(f"❌ モデルが見つかりません: {MODEL_PATH}")
        return
    
    print("モデルをロード中...")
    student = ParallelStudentSNN(MODEL_PATH, TOKENIZER_PATH, lr=0.01)
    
    print("JITコンパイル中（初回のみ）...")
    # Warm up JIT
    dummy_prompt = student.encode("テスト")
    dummy_completion = student.encode("これはテストです。")
    student.train_example(dummy_prompt, dummy_completion)
    print("JIT完了!\n")
    
    # Prepare tokenized examples
    tokenized_examples = [
        (student.encode(p), student.encode(c)) 
        for p, c in TEACHER_EXAMPLES
    ]
    
    start_time = time.time()
    results = []
    
    for epoch in range(1, 10001):  # 1 to 10000
        epoch_loss = 0
        
        for prompt_tokens, completion_tokens in tokenized_examples:
            loss = student.train_example(prompt_tokens, completion_tokens)
            epoch_loss += loss
        
        avg_loss = epoch_loss / len(TEACHER_EXAMPLES)
        student.losses.append(avg_loss)
        
        if epoch % 100 == 0:
            elapsed = time.time() - start_time
            speed = epoch / elapsed
            remaining = (10000 - epoch) / speed if speed > 0 else 0
            sample = student.generate("人工知能は", max_length=40)
            print(f"Epoch {epoch}/10000 | Loss: {avg_loss:.3f} | 速度: {speed:.2f} ep/s | 残り: {remaining/60:.0f}分")
            print(f"  生成: {sample[:50]}...")
            
            results.append({
                'epoch': epoch,
                'loss': float(avg_loss),
                'sample': sample[:60]
            })
    
    total_time = time.time() - start_time
    print(f"\n⏱️ 学習時間: {total_time:.1f}秒 ({total_time/60:.1f}分)")
    
    # Final test
    print("\n" + "="*50)
    print("📊 最終評価")
    print("="*50)
    
    for prompt in ["人工知能は", "脳とコンピュータの違いは", "私たちの研究では"]:
        output = student.generate(prompt, max_length=50)
        print(f"📝 {prompt}")
        print(f"🤖 {prompt}{output[:60]}")
        print()
    
    # Save
    final_path = RESULTS_DIR.parent / "checkpoints" / "model_teacher_10000epochs_parallel.pkl"
    model_data = {
        'vocab_size': student.vocab_size,
        'hidden_dim': student.hidden_dim,
        'hypercube_dim': 11,
        'embedding': student.embedding,
        'W_res': student.W_res,
        'W_spike': student.W_spike,
        'W_membrane': student.W_membrane,
        'bias': student.bias,
        'lr': student.lr
    }
    with open(final_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    with open(RESULTS_DIR / 'teacher_10000epochs_results.json', 'w', encoding='utf-8') as f:
        json.dump({
            'start_epoch': 1,
            'end_epoch': 10000,
            'num_teacher_examples': len(TEACHER_EXAMPLES),
            'training_time_sec': total_time,
            'final_loss': float(student.losses[-1]) if student.losses else None,
            'results': results,
            'losses': [float(l) for l in student.losses]
        }, f, ensure_ascii=False, indent=2)
    
    print(f"✅ モデル保存: {final_path}")
    print("🎉 並列学習完了！")


if __name__ == "__main__":
    main()
