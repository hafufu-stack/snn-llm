#!/usr/bin/env python3
"""
Temporal Coding SNN-LLM
時間コーディングによる言語モデル

ろーるさんの研究を応用:
- 発火タイミング（位相）と発火間隔（ISI）で情報を符号化
- 少ないニューロンで大量の単語を表現可能

従来: 1トークン = 2048次元ベクトル（静的）
提案: 1トークン = 時間パターン（位相 + ISI）
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


class TemporalEncoder:
    """
    時間コーディングエンコーダー
    
    ろーるさんの研究を参考に:
    - 位相（phase）: 0〜50ms、0.5ms刻み → 100通り
    - ISI（間隔）: 5〜15ms、0.5ms刻み → 20通り
    - 1トークン = 位相 × ISI = 2000パターン
    
    4ニューロン（1基準+3自由）で 2000^3 = 8,000,000,000 パターン
    → 4000語彙に対して十分すぎる容量
    """
    
    def __init__(self, vocab_size, num_encoding_neurons=4):
        self.vocab_size = vocab_size
        self.num_neurons = num_encoding_neurons
        
        # 時間パラメータ（ろーるさんの設定を参考）
        self.phase_min = 0.0
        self.phase_max = 50.0
        self.phase_step = 0.5
        self.phases = np.arange(self.phase_min, self.phase_max, self.phase_step)
        
        self.isi_min = 5.0
        self.isi_max = 15.0
        self.isi_step = 0.5
        self.isis = np.arange(self.isi_min, self.isi_max, self.isi_step)
        
        # 1ニューロンあたりのパターン数
        self.patterns_per_neuron = len(self.phases) * len(self.isis)
        
        # 複数ニューロンで表現できる総パターン数
        # 8ニューロン（1つは基準）で: patterns^8 通り
        self.free_neurons = num_encoding_neurons - 1
        self.total_patterns = self.patterns_per_neuron ** self.free_neurons
        
        print(f"=== 時間コーディングエンコーダー ===")
        print(f"位相パターン: {len(self.phases)}通り ({self.phase_min}-{self.phase_max}ms)")
        print(f"ISIパターン: {len(self.isis)}通り ({self.isi_min}-{self.isi_max}ms)")
        print(f"1ニューロンあたり: {self.patterns_per_neuron}パターン")
        print(f"エンコード用ニューロン: {num_encoding_neurons}個（基準1個 + 自由{self.free_neurons}個）")
        print(f"理論的表現可能数: {self.total_patterns:.2e}")
        print(f"語彙サイズ: {vocab_size}")
        
        if self.total_patterns >= vocab_size:
            print(f"✅ {num_encoding_neurons}ニューロンで{vocab_size}語彙を表現可能!")
        else:
            print(f"⚠️ ニューロン数が足りません")
    
    def encode_token(self, token_id):
        """
        トークンIDを時間パターンに変換
        Returns: [(phase_0, isi_0), (phase_1, isi_1), ...]
        """
        patterns = []
        
        # Neuron 0は基準（phase=0, isi=10固定）
        patterns.append((0.0, 10.0))
        
        # Neuron 1以降でトークンIDを符号化
        remaining = token_id
        for _ in range(self.free_neurons):
            digit = remaining % self.patterns_per_neuron
            remaining //= self.patterns_per_neuron
            
            phase_idx = digit // len(self.isis)
            isi_idx = digit % len(self.isis)
            
            phase = self.phases[phase_idx]
            isi = self.isis[isi_idx]
            
            patterns.append((phase, isi))
        
        return patterns
    
    def decode_pattern(self, patterns):
        """
        時間パターンからトークンIDを復元
        """
        token_id = 0
        
        for i in range(1, len(patterns)):
            phase, isi = patterns[i]
            
            # 最も近いインデックスを探す
            phase_idx = np.argmin(np.abs(self.phases - phase))
            isi_idx = np.argmin(np.abs(self.isis - isi))
            
            digit = phase_idx * len(self.isis) + isi_idx
            weight = self.patterns_per_neuron ** (i - 1)
            token_id += digit * weight
        
        return token_id
    
    def pattern_to_embedding(self, patterns, hidden_dim):
        """
        時間パターンを埋め込みベクトルに変換
        
        位相とISIを周期関数（sin/cos）で連続空間にマッピング
        これにより似た時間パターンは似た埋め込みになる
        """
        embedding = np.zeros(hidden_dim)
        
        for i, (phase, isi) in enumerate(patterns):
            # 周期関数でエンコード（Transformer的な位置エンコーディング風）
            # 各ニューロンに hidden_dim / num_neurons 次元を割り当て
            dims_per_neuron = hidden_dim // self.num_neurons
            start_dim = i * dims_per_neuron
            
            for d in range(dims_per_neuron):
                # 周波数を変えながらsin/cosでエンコード
                freq_phase = (d + 1) * 2 * np.pi / self.phase_max
                freq_isi = (d + 1) * 2 * np.pi / self.isi_max
                
                if d % 2 == 0:
                    embedding[start_dim + d] = np.sin(phase * freq_phase) + np.sin(isi * freq_isi)
                else:
                    embedding[start_dim + d] = np.cos(phase * freq_phase) + np.cos(isi * freq_isi)
        
        # 正規化
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm
        
        return embedding


class TemporalCodingSNN:
    """
    時間コーディングSNN言語モデル
    """
    
    def __init__(self, vocab_size, hidden_dim, token_to_idx, num_encoding_neurons=10):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.token_to_idx = token_to_idx
        self.idx_to_token = {v: k for k, v in token_to_idx.items()}
        
        # 時間エンコーダー
        self.encoder = TemporalEncoder(vocab_size, num_encoding_neurons)
        
        # 時間エンコード埋め込みをキャッシュ
        print("\n時間埋め込みを事前計算中...")
        self.temporal_embeddings = np.zeros((vocab_size, hidden_dim))
        for token_id in range(vocab_size):
            patterns = self.encoder.encode_token(token_id)
            self.temporal_embeddings[token_id] = self.encoder.pattern_to_embedding(patterns, hidden_dim)
        print("時間埋め込み完了!")
        
        # SNNの重み（既存モデルから）
        self.W_res = None
        self.W_spike = None
        self.W_membrane = None
        self.bias = None
        
    def load_weights(self, model_data):
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
    
    def forward(self, token_ids):
        """
        時間コーディング埋め込みを使ったforward
        """
        state = np.zeros(self.hidden_dim)
        logits_list = []
        
        for token_id in token_ids:
            # 時間コーディング埋め込みを取得
            x = self.temporal_embeddings[token_id]
            
            # リザバー更新
            new_state = 0.9 * state + 0.1 * (x + self.W_res @ state)
            state = np.where(new_state > 0.5, 1.0, 0.0)
            
            # 出力計算
            logits = self.W_spike @ state + self.W_membrane @ state + self.bias
            logits_list.append(logits)
        
        return np.array(logits_list), state
    
    def generate(self, prompt, max_length=30, temperature=0.8):
        tokens = self.tokenize(prompt)
        state = np.zeros(self.hidden_dim)
        
        for _ in range(max_length):
            # 時間コーディング埋め込みで処理
            for token_id in tokens[-10:]:
                x = self.temporal_embeddings[token_id]
                new_state = 0.9 * state + 0.1 * (x + self.W_res @ state)
                state = np.where(new_state > 0.5, 1.0, 0.0)
            
            logits = self.W_spike @ state + self.W_membrane @ state + self.bias
            logits = logits / temperature
            probs = softmax_numba(logits)
            
            try:
                next_token = np.random.choice(len(probs), p=probs)
            except:
                next_token = np.argmax(logits)
            
            tokens.append(next_token)
            
            if next_token == 0:
                break
        
        result = ""
        for t in tokens[len(self.tokenize(prompt)):]:
            if t in self.idx_to_token:
                result += self.idx_to_token[t]
        
        return result


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║    ⏱️ Temporal Coding SNN-LLM ⏱️                              ║
    ║    時間コーディングによる高効率言語モデル                      ║
    ║                                                               ║
    ║    ろーるさんの研究を応用:                                    ║
    ║    - 位相 + ISI で情報を符号化                                ║
    ║    - 少ないニューロンで大量の語彙を表現                       ║
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
    
    # 時間コーディングモデル作成
    model = TemporalCodingSNN(vocab_size, hidden_dim, token_to_idx, num_encoding_neurons=4)
    model.load_weights(model_data)
    
    # テスト
    print("\n" + "="*60)
    print("📝 時間コーディング生成テスト")
    print("="*60)
    
    test_prompts = [
        "人工知能は",
        "日本語の",
        "スパイキング",
        "脳と",
        "吾輩は猫で",
    ]
    
    for prompt in test_prompts:
        start = time.time()
        result = model.generate(prompt, max_length=30)
        elapsed = time.time() - start
        print(f"\n「{prompt}」")
        print(f"  → {result[:40]}... ({elapsed:.2f}s)")
    
    # エンコード/デコードテスト
    print("\n\n" + "="*60)
    print("🔬 時間エンコード/デコードテスト")
    print("="*60)
    
    test_tokens = [0, 100, 500, 1000, vocab_size - 1]
    for token_id in test_tokens:
        patterns = model.encoder.encode_token(token_id)
        decoded = model.encoder.decode_pattern(patterns)
        status = "✅" if token_id == decoded else "❌"
        print(f"{status} Token {token_id} → Patterns → Decoded {decoded}")
    
    # 結果保存
    results = {
        'method': 'temporal_coding',
        'num_encoding_neurons': 10,
        'patterns_per_neuron': model.encoder.patterns_per_neuron,
        'total_patterns': float(model.encoder.total_patterns),
        'vocab_size': vocab_size,
    }
    
    with open(RESULTS_DIR / 'temporal_coding_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n結果保存: {RESULTS_DIR / 'temporal_coding_results.json'}")


if __name__ == "__main__":
    main()
