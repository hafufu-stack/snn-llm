#!/usr/bin/env python3
"""
MD-LD Pattern Recognition Enhancement
先輩の研究を応用したパターン識別強化

先輩の知見:
- MDに特定パターンが入ったとき、LD入力があると発火回数が顕著に増加
- 20Hz LD入力はMDのパターン識別機能を増強
- ランダムな20Hzより一定間隔のレギュラー刺激が効果的

LLMへの応用:
- MD = 文脈情報（前の単語列）
- LD = 重要トークンへの「レギュラー刺激」信号
- 結果 = 正しい次トークンの識別能力向上
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


def is_important_token(char):
    """
    重要なトークンかどうかを判定
    漢字、カタカナ = 内容語（名詞、動詞語幹など）→ 重要
    ひらがな = 機能語（助詞、接続詞など）→ 通常
    """
    if '\u4e00' <= char <= '\u9fff':  # 漢字
        return True
    if '\u30a0' <= char <= '\u30ff':  # カタカナ
        return True
    return False


def generate_regular_ld_signal(t, frequency=20.0, amplitude=1.0):
    """
    レギュラー刺激を生成（20Hz = 50ms周期）
    
    先輩の知見: ランダムよりレギュラーの方が効果的
    """
    period = 1000.0 / frequency  # 20Hz → 50ms周期
    phase = (t % period) / period
    
    # パルス幅10%のレギュラー刺激
    if phase < 0.1:
        return amplitude
    return 0.0


def generate_random_ld_signal(amplitude=1.0, probability=0.2):
    """
    ランダム刺激（比較用）
    """
    if np.random.random() < probability:
        return amplitude
    return 0.0


class MDLDEnhancedSNN:
    """
    MD-LD相互作用を組み込んだSNN
    
    先輩の研究:
    - MD入力 + LD入力 → 発火増加
    - レギュラー刺激 → 特定パターンだけ発火
    """
    
    def __init__(self, vocab_size, hidden_dim, token_to_idx):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.token_to_idx = token_to_idx
        self.idx_to_token = {v: k for k, v in token_to_idx.items()}
        
        # 重み
        self.embedding = None
        self.W_res = None
        self.W_spike = None
        self.W_membrane = None
        self.bias = None
        
        # LD刺激パラメータ
        self.ld_frequency = 20.0  # Hz（先輩の設定）
        self.ld_amplitude = 0.3   # 刺激の強さ
        
        # 統計
        self.ld_triggered_count = 0
        self.total_tokens = 0
    
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
    
    def forward_with_ld(self, token_ids, use_regular=True, chars=None):
        """
        LD刺激付きforward
        
        use_regular=True: レギュラー刺激（先輩推奨）
        use_regular=False: ランダム刺激（比較用）
        """
        state = np.zeros(self.hidden_dim)
        logits_list = []
        
        # シミュレーション時間（各トークンに10msを割り当て）
        t = 0.0
        dt = 10.0  # ms per token
        
        for i, token_id in enumerate(token_ids):
            # MD入力（通常の埋め込み）
            x_md = self.embedding[token_id] if token_id < len(self.embedding) else np.zeros(self.hidden_dim)
            
            # LD刺激（重要トークンの場合のみ）
            ld_signal = 0.0
            if chars and i < len(chars):
                if is_important_token(chars[i]):
                    self.ld_triggered_count += 1
                    if use_regular:
                        ld_signal = generate_regular_ld_signal(t, self.ld_frequency, self.ld_amplitude)
                    else:
                        ld_signal = generate_random_ld_signal(self.ld_amplitude)
            
            self.total_tokens += 1
            
            # MD + LD の統合（先輩の知見: LD入力で発火増加）
            # LD刺激はリザバー状態に加算される
            x_combined = x_md + ld_signal * np.ones(self.hidden_dim)
            
            # リザバー更新
            new_state = 0.9 * state + 0.1 * (x_combined + self.W_res @ state)
            
            # スパイキング（LD刺激がある場合は閾値を少し下げる効果）
            threshold = 0.5 - 0.1 * ld_signal  # LD刺激で閾値低下
            state = np.where(new_state > threshold, 1.0, 0.0)
            
            # 出力計算
            logits = self.W_spike @ state + self.W_membrane @ state + self.bias
            logits_list.append(logits)
            
            t += dt
        
        return np.array(logits_list), state
    
    def generate(self, prompt, max_length=30, temperature=0.8, use_regular=True):
        """テキスト生成"""
        tokens = self.tokenize(prompt)
        chars = list(prompt)
        state = np.zeros(self.hidden_dim)
        
        t = 0.0
        dt = 10.0
        
        for _ in range(max_length):
            # 最近のコンテキストで処理
            context_tokens = tokens[-10:]
            context_chars = chars[-10:]
            
            for i, token_id in enumerate(context_tokens):
                x_md = self.embedding[token_id] if token_id < len(self.embedding) else np.zeros(self.hidden_dim)
                
                ld_signal = 0.0
                if i < len(context_chars) and is_important_token(context_chars[i]):
                    if use_regular:
                        ld_signal = generate_regular_ld_signal(t, self.ld_frequency, self.ld_amplitude)
                    else:
                        ld_signal = generate_random_ld_signal(self.ld_amplitude)
                
                x_combined = x_md + ld_signal * np.ones(self.hidden_dim)
                new_state = 0.9 * state + 0.1 * (x_combined + self.W_res @ state)
                threshold = 0.5 - 0.1 * ld_signal
                state = np.where(new_state > threshold, 1.0, 0.0)
                t += dt
            
            logits = self.W_spike @ state + self.W_membrane @ state + self.bias
            logits = logits / temperature
            probs = softmax_numba(logits)
            
            try:
                next_token = np.random.choice(len(probs), p=probs)
            except:
                next_token = np.argmax(logits)
            
            tokens.append(next_token)
            next_char = self.idx_to_token.get(next_token, '?')
            chars.append(next_char)
            
            if next_token == 0:
                break
        
        result = ""
        for t in tokens[len(self.tokenize(prompt)):]:
            if t in self.idx_to_token:
                result += self.idx_to_token[t]
        
        return result
    
    def compute_loss(self, text, use_regular=True):
        """損失計算"""
        tokens = self.tokenize(text)
        chars = list(text)
        
        if len(tokens) < 2:
            return 0.0
        
        logits, _ = self.forward_with_ld(tokens[:-1], use_regular, chars[:-1])
        
        total_loss = 0.0
        for t in range(len(tokens) - 1):
            probs = softmax_numba(logits[t])
            target = tokens[t + 1]
            total_loss -= np.log(probs[target] + 1e-10)
        
        return total_loss / (len(tokens) - 1)


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║    🧠 MD-LD Pattern Recognition Enhancement 🧠                ║
    ║    先輩の研究を応用したパターン識別強化                        ║
    ║                                                               ║
    ║    MD = 文脈情報（通常の埋め込み）                            ║
    ║    LD = 重要トークンへのレギュラー刺激（20Hz）                ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # モデルロード
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
    
    # モデル作成
    model = MDLDEnhancedSNN(vocab_size, hidden_dim, token_to_idx)
    model.load_weights(model_data)
    
    # テスト文
    test_texts = [
        "人工知能は人間の知能を模倣する",
        "スパイキングニューラルネットワークは生物の神経細胞を模倣する",
        "日本語の文法は主語が省略されることが多い",
    ]
    
    print("\n" + "="*60)
    print("📊 レギュラー刺激 vs ランダム刺激 比較")
    print("="*60)
    
    for text in test_texts:
        print(f"\nテキスト: {text[:30]}...")
        
        # レギュラー刺激
        model.ld_triggered_count = 0
        model.total_tokens = 0
        loss_regular = model.compute_loss(text, use_regular=True)
        ld_ratio_regular = model.ld_triggered_count / max(model.total_tokens, 1)
        
        # ランダム刺激
        model.ld_triggered_count = 0
        model.total_tokens = 0
        loss_random = model.compute_loss(text, use_regular=False)
        ld_ratio_random = model.ld_triggered_count / max(model.total_tokens, 1)
        
        print(f"  レギュラー刺激: Loss = {loss_regular:.3f}, LD発火率 = {ld_ratio_regular:.1%}")
        print(f"  ランダム刺激:   Loss = {loss_random:.3f}, LD発火率 = {ld_ratio_random:.1%}")
        
        if loss_regular < loss_random:
            print(f"  → レギュラー刺激が {(1 - loss_regular/loss_random)*100:.1f}% 優位")
        elif loss_random < loss_regular:
            print(f"  → ランダム刺激が {(1 - loss_random/loss_regular)*100:.1f}% 優位")
        else:
            print(f"  → 同等")
    
    # 生成テスト
    print("\n" + "="*60)
    print("📝 生成テスト（レギュラー刺激あり）")
    print("="*60)
    
    test_prompts = [
        "人工知能は",
        "日本語の",
        "脳と",
    ]
    
    for prompt in test_prompts:
        result = model.generate(prompt, max_length=30, use_regular=True)
        print(f"\n「{prompt}」→ {result[:40]}...")
    
    # 結果保存
    results = {
        'method': 'md_ld_enhancement',
        'ld_frequency': 20.0,
        'ld_amplitude': 0.3,
        'insight': '先輩の知見: レギュラー刺激がパターン識別を強化',
    }
    
    with open(RESULTS_DIR / 'md_ld_enhancement_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n結果保存: {RESULTS_DIR / 'md_ld_enhancement_results.json'}")


if __name__ == "__main__":
    main()
