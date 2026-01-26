#!/usr/bin/env python3
"""
Growing SNN - 成長型スパイキングニューラルネットワーク

ろーるさんのアイデア:
- 学習中にパラメータだけでなくニューロン数も増やす
- 細胞分裂のように成長するネットワーク
- 各SNNが協力し合う（人体のように）

生物学的背景:
- 海馬での神経新生（成人でも新しいニューロンが生まれる）
- シナプス刈り込み（不要な接続を削除）
- 発達過程での脳の成長
"""

import numpy as np
import pickle
import json
import time
from pathlib import Path
from numba import njit, set_num_threads

set_num_threads(20)

# Paths
TOKENIZER_PATH = Path(__file__).parent.parent / "checkpoints" / "tokenizer_v4_parallel.json"
CORPUS_PATH = Path(__file__).parent.parent / "data" / "japanese_corpus.txt"
RESULTS_DIR = Path(__file__).parent.parent / "results"
CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"


@njit(cache=True)
def softmax_numba(x):
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / (np.sum(exp_x) + 1e-10)


class GrowingSNN:
    """
    成長型SNN
    
    特徴:
    1. 学習中にニューロン数を動的に増加
    2. 使われていないニューロンを刈り込み
    3. 容量が足りなくなったら成長
    """
    
    def __init__(self, vocab_size, initial_hidden_dim=256, 
                 max_hidden_dim=2048, growth_rate=64):
        self.vocab_size = vocab_size
        self.hidden_dim = initial_hidden_dim
        self.max_hidden_dim = max_hidden_dim
        self.growth_rate = growth_rate
        
        # 初期重みを小さめに
        scale = 0.01
        self.embedding = np.random.randn(vocab_size, initial_hidden_dim) * scale
        self.W_res = np.random.randn(initial_hidden_dim, initial_hidden_dim) * scale / np.sqrt(initial_hidden_dim)
        self.W_spike = np.random.randn(vocab_size, initial_hidden_dim) * scale
        self.W_membrane = np.random.randn(vocab_size, initial_hidden_dim) * scale
        self.bias = np.zeros(vocab_size)
        
        # 成長履歴
        self.growth_history = []
        self.prune_history = []
        
        # ニューロン使用率追跡
        self.neuron_activation_count = np.zeros(initial_hidden_dim)
        self.total_forward_count = 0
        
        print(f"=== Growing SNN 初期化 ===")
        print(f"語彙サイズ: {vocab_size}")
        print(f"初期ニューロン数: {initial_hidden_dim}")
        print(f"最大ニューロン数: {max_hidden_dim}")
        print(f"成長単位: {growth_rate}ニューロン")
    
    def grow(self, num_new_neurons=None):
        """
        ニューロンを追加（細胞分裂）
        """
        if num_new_neurons is None:
            num_new_neurons = self.growth_rate
        
        if self.hidden_dim + num_new_neurons > self.max_hidden_dim:
            num_new_neurons = self.max_hidden_dim - self.hidden_dim
            if num_new_neurons <= 0:
                print("⚠️ 最大サイズに到達、成長できません")
                return False
        
        old_dim = self.hidden_dim
        new_dim = old_dim + num_new_neurons
        
        print(f"🌱 成長中: {old_dim} → {new_dim} ニューロン (+{num_new_neurons})")
        
        # embedding拡張
        new_embedding = np.zeros((self.vocab_size, new_dim))
        new_embedding[:, :old_dim] = self.embedding
        new_embedding[:, old_dim:] = np.random.randn(self.vocab_size, num_new_neurons) * 0.01
        self.embedding = new_embedding
        
        # W_res拡張
        new_W_res = np.zeros((new_dim, new_dim))
        new_W_res[:old_dim, :old_dim] = self.W_res
        # 新しいニューロン間の接続を初期化
        new_W_res[old_dim:, :] = np.random.randn(num_new_neurons, new_dim) * 0.01 / np.sqrt(new_dim)
        new_W_res[:, old_dim:] = np.random.randn(new_dim, num_new_neurons) * 0.01 / np.sqrt(new_dim)
        self.W_res = new_W_res
        
        # W_spike, W_membrane拡張
        new_W_spike = np.zeros((self.vocab_size, new_dim))
        new_W_spike[:, :old_dim] = self.W_spike
        new_W_spike[:, old_dim:] = np.random.randn(self.vocab_size, num_new_neurons) * 0.01
        self.W_spike = new_W_spike
        
        new_W_membrane = np.zeros((self.vocab_size, new_dim))
        new_W_membrane[:, :old_dim] = self.W_membrane
        new_W_membrane[:, old_dim:] = np.random.randn(self.vocab_size, num_new_neurons) * 0.01
        self.W_membrane = new_W_membrane
        
        # 活性化カウント拡張
        new_activation_count = np.zeros(new_dim)
        new_activation_count[:old_dim] = self.neuron_activation_count
        self.neuron_activation_count = new_activation_count
        
        self.hidden_dim = new_dim
        self.growth_history.append({
            'from': old_dim,
            'to': new_dim,
            'added': num_new_neurons
        })
        
        return True
    
    def prune(self, threshold=0.01):
        """
        使われていないニューロンを刈り込み
        """
        if self.total_forward_count == 0:
            return False
        
        # 活性化率を計算
        activation_rate = self.neuron_activation_count / self.total_forward_count
        
        # 閾値以下のニューロンを特定
        inactive_neurons = activation_rate < threshold
        num_inactive = np.sum(inactive_neurons)
        
        if num_inactive == 0 or num_inactive >= self.hidden_dim - 10:
            # 全部消すわけにはいかない
            return False
        
        # アクティブなニューロンだけ残す
        active_mask = ~inactive_neurons
        active_indices = np.where(active_mask)[0]
        
        old_dim = self.hidden_dim
        new_dim = len(active_indices)
        
        print(f"✂️ 刈り込み中: {old_dim} → {new_dim} ニューロン (-{old_dim - new_dim})")
        
        self.embedding = self.embedding[:, active_indices]
        self.W_res = self.W_res[np.ix_(active_indices, active_indices)]
        self.W_spike = self.W_spike[:, active_indices]
        self.W_membrane = self.W_membrane[:, active_indices]
        self.neuron_activation_count = self.neuron_activation_count[active_indices]
        
        self.hidden_dim = new_dim
        self.prune_history.append({
            'from': old_dim,
            'to': new_dim,
            'removed': old_dim - new_dim
        })
        
        return True
    
    def forward(self, token_ids, track_activation=True):
        """
        Forward pass with activation tracking
        """
        state = np.zeros(self.hidden_dim)
        logits_list = []
        
        for token_id in token_ids:
            if token_id < self.vocab_size:
                x = self.embedding[token_id]
            else:
                x = np.zeros(self.hidden_dim)
            
            # リザバー更新
            new_state = 0.9 * state + 0.1 * (x + self.W_res @ state)
            state = np.where(new_state > 0.5, 1.0, 0.0)
            
            # 活性化追跡
            if track_activation:
                self.neuron_activation_count += state
                self.total_forward_count += 1
            
            # 出力計算
            logits = self.W_spike @ state + self.W_membrane @ state + self.bias
            logits_list.append(logits)
        
        return np.array(logits_list), state
    
    def train_step(self, token_ids, lr=0.001):
        """
        1ステップの学習
        """
        if len(token_ids) < 2:
            return 0.0
        
        logits, _ = self.forward(token_ids[:-1])
        
        total_loss = 0.0
        for t in range(len(token_ids) - 1):
            probs = softmax_numba(logits[t])
            target = token_ids[t + 1]
            
            # Cross-entropy loss
            loss = -np.log(probs[target] + 1e-10)
            total_loss += loss
            
            # Gradient
            error = probs.copy()
            error[target] -= 1.0
            
            # Weight update
            state = np.zeros(self.hidden_dim)
            if t > 0:
                for i in range(t):
                    if token_ids[i] < self.vocab_size:
                        x = self.embedding[token_ids[i]]
                    else:
                        x = np.zeros(self.hidden_dim)
                    new_state = 0.9 * state + 0.1 * (x + self.W_res @ state)
                    state = np.where(new_state > 0.5, 1.0, 0.0)
            
            self.W_spike -= lr * np.outer(error, state)
            self.W_membrane -= lr * np.outer(error, state)
            self.bias -= lr * error
        
        return total_loss / (len(token_ids) - 1)
    
    def should_grow(self, recent_losses, stagnation_threshold=0.01):
        """
        成長すべきかどうかを判断
        - 損失が停滞している
        - まだ最大サイズに達していない
        """
        if len(recent_losses) < 10:
            return False
        
        if self.hidden_dim >= self.max_hidden_dim:
            return False
        
        # 損失の変化率
        loss_change = abs(recent_losses[-1] - recent_losses[-10]) / (recent_losses[-10] + 1e-10)
        
        # 停滞していたら成長
        return loss_change < stagnation_threshold


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║    🌱 Growing SNN - 成長型スパイキングニューラルネットワーク 🌱 ║
    ║                                                               ║
    ║    ろーるさんのアイデア:                                      ║
    ║    - 学習中にニューロン数を動的に増加                         ║
    ║    - 細胞分裂のように成長                                     ║
    ║    - 人体のように各部が協力                                   ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # トークナイザーロード
    print("トークナイザーをロード中...")
    with open(TOKENIZER_PATH, 'r', encoding='utf-8') as f:
        tokenizer_data = json.load(f)
    
    token_to_idx = tokenizer_data['token_to_idx']
    vocab_size = len(token_to_idx)
    
    # コーパスロード
    print("コーパスをロード中...")
    with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
        corpus = f.read()
    
    print(f"語彙サイズ: {vocab_size}")
    print(f"コーパスサイズ: {len(corpus):,}文字")
    
    # Growing SNN作成
    model = GrowingSNN(
        vocab_size=vocab_size,
        initial_hidden_dim=256,  # 小さく始める
        max_hidden_dim=1024,     # 最大1024まで成長
        growth_rate=128          # 128ニューロンずつ成長
    )
    
    # 学習設定
    epochs = 100
    seq_length = 50
    
    # サンプル作成
    samples = []
    for i in range(0, min(len(corpus) - seq_length, 10000), seq_length):
        text = corpus[i:i+seq_length]
        tokens = [token_to_idx.get(c, 0) for c in text]
        samples.append(tokens)
    
    print(f"サンプル数: {len(samples)}")
    
    print("\n" + "="*60)
    print("🧠 成長型SNN学習開始！")
    print("="*60)
    
    recent_losses = []
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        
        # シャッフル
        np.random.shuffle(samples)
        
        for sample in samples[:100]:  # 100サンプル/エポック
            loss = model.train_step(sample, lr=0.001)
            epoch_loss += loss
        
        avg_loss = epoch_loss / 100
        recent_losses.append(avg_loss)
        
        # 成長チェック
        if model.should_grow(recent_losses):
            model.grow()
        
        # 進捗表示
        if epoch % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:3d} | Loss: {avg_loss:.3f} | "
                  f"ニューロン: {model.hidden_dim} | "
                  f"成長回数: {len(model.growth_history)} | "
                  f"時間: {elapsed:.1f}s")
    
    # 最終結果
    print("\n" + "="*60)
    print("📊 学習完了!")
    print("="*60)
    
    print(f"\n成長履歴:")
    for i, g in enumerate(model.growth_history):
        print(f"  {i+1}. {g['from']} → {g['to']} (+{g['added']})")
    
    print(f"\n最終ニューロン数: {model.hidden_dim}")
    print(f"総成長回数: {len(model.growth_history)}")
    print(f"最終Loss: {recent_losses[-1]:.3f}")
    
    # モデル保存
    model_data = {
        'vocab_size': vocab_size,
        'hidden_dim': model.hidden_dim,
        'embedding': model.embedding,
        'W_res': model.W_res,
        'W_spike': model.W_spike,
        'W_membrane': model.W_membrane,
        'bias': model.bias,
        'growth_history': model.growth_history,
    }
    
    save_path = CHECKPOINT_DIR / 'model_growing_snn.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"\nモデル保存: {save_path}")
    
    # 結果保存
    results = {
        'method': 'growing_snn',
        'initial_neurons': 256,
        'final_neurons': model.hidden_dim,
        'growth_history': model.growth_history,
        'final_loss': recent_losses[-1],
        'losses': recent_losses,
    }
    
    with open(RESULTS_DIR / 'growing_snn_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"結果保存: {RESULTS_DIR / 'growing_snn_results.json'}")


if __name__ == "__main__":
    main()
