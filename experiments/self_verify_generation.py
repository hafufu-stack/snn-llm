#!/usr/bin/env python3
"""
Self-Verifying SNN Language Model
「考えてから話す」生成アーキテクチャ

人間の会話プロセスを模倣:
1. 候補を生成
2. 自分で読み返して理解できるかチェック
3. 理解できたら出力、できなければ再生成
"""

import numpy as np
import pickle
import json
import time
from pathlib import Path
from numba import njit, prange, set_num_threads

set_num_threads(20)

# Paths
MODEL_PATH = Path(__file__).parent.parent / "checkpoints" / "model_super_teacher_v2.pkl"
TOKENIZER_PATH = Path(__file__).parent.parent / "checkpoints" / "tokenizer_v4_parallel.json"
RESULTS_DIR = Path(__file__).parent.parent / "results"


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
    state = np.zeros(hidden_dim)
    total_logits = np.zeros((n_tokens, vocab_size))
    
    for t in range(n_tokens):
        token_id = token_ids[t]
        if token_id < embedding.shape[0]:
            x = embedding[token_id]
        else:
            x = np.zeros(hidden_dim)
        
        new_state = np.zeros(hidden_dim)
        for i in prange(hidden_dim):
            res_sum = 0.0
            for j in range(hidden_dim):
                res_sum += W_res[i, j] * state[j]
            new_state[i] = 0.9 * state[i] + 0.1 * (x[i] + res_sum)
            if new_state[i] > 0.5:
                new_state[i] = 1.0
            else:
                new_state[i] = 0.0
        
        state = new_state
        
        logits = np.zeros(vocab_size)
        for v in prange(vocab_size):
            spike_sum = 0.0
            membrane_sum = 0.0
            for h in range(hidden_dim):
                spike_sum += W_spike[v, h] * state[h]
                membrane_sum += W_membrane[v, h] * state[h]
            logits[v] = spike_sum + membrane_sum + bias[v]
        
        total_logits[t] = logits
    
    return total_logits, state


class SelfVerifyingSNN:
    """自己検証型SNN言語モデル"""
    
    def __init__(self, vocab_size, hidden_dim, token_to_idx):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.token_to_idx = token_to_idx
        self.idx_to_token = {v: k for k, v in token_to_idx.items()}
        
        # Weights
        self.embedding = None
        self.W_res = None
        self.W_spike = None
        self.W_membrane = None
        self.bias = None
        
        # Stats
        self.regeneration_count = 0
        self.total_generations = 0
        
    def load_weights(self, model_data):
        self.embedding = model_data['embedding'].astype(np.float64)
        self.W_res = model_data['W_res'].astype(np.float64)
        self.W_spike = model_data['W_spike'].astype(np.float64)
        self.W_membrane = model_data['W_membrane'].astype(np.float64)
        self.bias = model_data['bias'].astype(np.float64)
    
    def tokenize(self, text):
        tokens = []
        for char in text:
            if char in self.token_to_idx:
                tokens.append(self.token_to_idx[char])
            else:
                tokens.append(self.token_to_idx.get('<unk>', 0))
        return tokens
    
    def detokenize(self, token_ids):
        result = ""
        for t in token_ids:
            if t in self.idx_to_token:
                result += self.idx_to_token[t]
        return result
    
    def forward(self, token_ids):
        logits, state = parallel_forward(
            self.embedding, self.W_res, self.W_spike, self.W_membrane, self.bias,
            np.array(token_ids), self.hidden_dim, self.vocab_size
        )
        return logits, state
    
    def compute_coherence_score(self, text):
        """
        文章の「理解可能性」スコアを計算
        低いPerplexityほど高いスコア（理解しやすい）
        """
        tokens = self.tokenize(text)
        if len(tokens) < 2:
            return 0.0
        
        logits, _ = self.forward(tokens[:-1])
        
        total_log_prob = 0.0
        for t in range(len(tokens) - 1):
            probs = softmax_numba(logits[t])
            next_token = tokens[t + 1]
            total_log_prob += np.log(probs[next_token] + 1e-10)
        
        # 平均log確率（高いほど良い）
        avg_log_prob = total_log_prob / (len(tokens) - 1)
        
        # 0-1のスコアに変換（高いほど良い）
        coherence = np.exp(avg_log_prob) 
        return min(coherence, 1.0)
    
    def check_grammar_patterns(self, text):
        """
        基本的な文法パターンチェック
        日本語の特徴を利用
        """
        score = 0.0
        checks = 0
        
        # ひらがなで終わる（自然な文末）
        if len(text) > 0 and '\u3040' <= text[-1] <= '\u309f':
            score += 1.0
        checks += 1
        
        # 漢字→ひらがなの流れがある（自然な日本語）
        for i in range(len(text) - 1):
            if '\u4e00' <= text[i] <= '\u9fff':  # 漢字
                if '\u3040' <= text[i+1] <= '\u309f':  # 次がひらがな
                    score += 0.5
                    checks += 1
        
        # 連続する同じ文字を避ける
        for i in range(len(text) - 2):
            if text[i] == text[i+1] == text[i+2]:
                score -= 0.5
        
        return score / max(checks, 1)
    
    def generate_candidate(self, prompt, max_length=30, temperature=1.0):
        """候補テキストを生成"""
        tokens = self.tokenize(prompt)
        
        for _ in range(max_length):
            logits, _ = self.forward(tokens)
            
            # Temperature sampling
            last_logits = logits[-1] / temperature
            probs = softmax_numba(last_logits)
            
            try:
                next_token = np.random.choice(len(probs), p=probs)
            except:
                next_token = np.argmax(last_logits)
            
            tokens.append(next_token)
            
            if next_token == 0:
                break
        
        return self.detokenize(tokens[len(self.tokenize(prompt)):])
    
    def generate_with_verification(self, prompt, max_length=30, 
                                   num_candidates=5, 
                                   coherence_threshold=0.01,
                                   max_retries=3):
        """
        自己検証付き生成
        
        1. 複数の候補を生成
        2. 各候補の理解可能性をスコアリング
        3. 最もスコアの高い候補を選択
        4. スコアが閾値以下なら再生成
        """
        self.total_generations += 1
        
        best_candidate = ""
        best_score = -float('inf')
        
        for retry in range(max_retries):
            candidates = []
            scores = []
            
            # 複数候補を生成
            for i in range(num_candidates):
                temp = 0.7 + 0.1 * i  # 様々なtemperature
                candidate = self.generate_candidate(prompt, max_length, temp)
                
                # スコア計算
                full_text = prompt + candidate
                coherence = self.compute_coherence_score(full_text)
                grammar = self.check_grammar_patterns(candidate)
                
                # 総合スコア
                total_score = 0.6 * coherence + 0.4 * grammar
                
                candidates.append(candidate)
                scores.append(total_score)
            
            # ベストを選択
            best_idx = np.argmax(scores)
            if scores[best_idx] > best_score:
                best_score = scores[best_idx]
                best_candidate = candidates[best_idx]
            
            # 閾値を超えたら終了
            if best_score >= coherence_threshold:
                break
            
            self.regeneration_count += 1
        
        return best_candidate, best_score
    
    def generate_with_token_verification(self, prompt, max_length=30):
        """
        トークンごとに検証しながら生成
        各トークンで複数候補を評価し、最も自然なものを選ぶ
        多様性ペナルティ付き（同じ文字の繰り返しを防ぐ）
        """
        tokens = self.tokenize(prompt)
        recent_tokens = []  # 最近使ったトークンを記録
        
        for _ in range(max_length):
            logits, _ = self.forward(tokens)
            last_logits = logits[-1]
            
            # Top-K候補を取得
            top_k = 15  # 増やす
            top_indices = np.argsort(last_logits)[-top_k:][::-1]
            
            best_token = top_indices[0]
            best_score = -float('inf')
            
            # 各候補をシミュレーション
            for candidate_token in top_indices:
                # この候補で続けた場合のcoherenceを予測
                test_tokens = tokens + [candidate_token]
                test_logits, _ = self.forward(test_tokens)
                
                # 次のトークンの確信度（エントロピー逆数）
                next_probs = softmax_numba(test_logits[-1])
                entropy = -np.sum(next_probs * np.log(next_probs + 1e-10))
                confidence = 1.0 / (entropy + 1e-10)
                
                # このトークンの確率
                token_prob = softmax_numba(last_logits)[candidate_token]
                
                # 多様性ペナルティ（最近使ったトークンは低スコア）
                repetition_penalty = 1.0
                if candidate_token in recent_tokens[-5:]:  # 直近5トークン
                    repetition_penalty = 0.3  # 70%ペナルティ
                if len(recent_tokens) >= 2 and candidate_token == recent_tokens[-1] == recent_tokens[-2]:
                    repetition_penalty = 0.1  # 90%ペナルティ（3連続防止）
                
                # 総合スコア
                score = (0.5 * token_prob + 0.5 * (confidence / 10.0)) * repetition_penalty
                
                if score > best_score:
                    best_score = score
                    best_token = candidate_token
            
            tokens.append(best_token)
            recent_tokens.append(best_token)
            
            if best_token == 0:
                break
        
        return self.detokenize(tokens[len(self.tokenize(prompt)):])


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║    🧠 Self-Verifying SNN Language Model 🧠                    ║
    ║    「考えてから話す」生成アーキテクチャ                        ║
    ║                                                               ║
    ║    1. 複数候補生成                                            ║
    ║    2. 理解可能性スコアリング                                  ║
    ║    3. ベスト候補選択                                          ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Load model
    if not MODEL_PATH.exists():
        print(f"⚠️ モデルファイルが見つかりません: {MODEL_PATH}")
        return
    
    print("モデルをロード中...")
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
    
    with open(TOKENIZER_PATH, 'r', encoding='utf-8') as f:
        tokenizer_data = json.load(f)
    
    token_to_idx = tokenizer_data['token_to_idx']
    vocab_size = model_data['vocab_size']
    hidden_dim = model_data['hidden_dim']
    
    print(f"語彙サイズ: {vocab_size}")
    print(f"隠れ次元: {hidden_dim}")
    
    # Create model
    model = SelfVerifyingSNN(vocab_size, hidden_dim, token_to_idx)
    model.load_weights(model_data)
    
    print("\nJITコンパイル中...")
    _ = model.forward([0, 1, 2])
    print("JIT完了!")
    
    # Test prompts
    test_prompts = [
        "人工知能は",
        "日本語の",
        "スパイキング",
        "脳と",
        "吾輩は",
    ]
    
    print("\n" + "="*60)
    print("📝 生成比較テスト")
    print("="*60)
    
    for prompt in test_prompts:
        print(f"\n【プロンプト】「{prompt}」")
        
        # 通常生成
        start = time.time()
        normal = model.generate_candidate(prompt, max_length=30)
        normal_time = time.time() - start
        
        # 自己検証付き生成
        start = time.time()
        verified, score = model.generate_with_verification(
            prompt, max_length=30, num_candidates=5
        )
        verified_time = time.time() - start
        
        # トークンごと検証
        start = time.time()
        token_verified = model.generate_with_token_verification(prompt, max_length=30)
        token_time = time.time() - start
        
        print(f"  通常生成      ({normal_time:.2f}s): {normal[:40]}...")
        print(f"  候補選択型    ({verified_time:.2f}s, score={score:.3f}): {verified[:40]}...")
        print(f"  トークン検証型({token_time:.2f}s): {token_verified[:40]}...")
    
    print(f"\n\n📊 統計:")
    print(f"  総生成回数: {model.total_generations}")
    print(f"  再生成回数: {model.regeneration_count}")
    
    # Save results
    results = {
        'method': 'self_verifying_generation',
        'prompts': test_prompts,
        'regeneration_count': model.regeneration_count,
        'total_generations': model.total_generations,
    }
    
    with open(RESULTS_DIR / 'self_verify_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n結果保存: {RESULTS_DIR / 'self_verify_results.json'}")


if __name__ == "__main__":
    main()
