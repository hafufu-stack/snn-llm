#!/usr/bin/env python3
"""
Super Teacher Learning v2 - 3つのAI提案を統合

1. 漢字部首分解入力 (Deep Think提案)
2. 文法ドリルデータ (全員推奨)  
3. Teacher審判付きFriendly Network (Deep Think改良案)
"""

import numpy as np
import pickle
import json
import time
from pathlib import Path
from numba import njit, prange, set_num_threads

set_num_threads(20)

# Paths
MODEL_PATH = Path(__file__).parent.parent / "checkpoints" / "model_teacher_1000epochs_parallel.pkl"
TOKENIZER_PATH = Path(__file__).parent.parent / "checkpoints" / "tokenizer_v4_parallel.json"
RESULTS_DIR = Path(__file__).parent.parent / "results"

# =============================================================================
# 1. 漢字部首分解データベース (Deep Think提案)
# =============================================================================
RADICAL_DB = {
    # サンズイ (水関連)
    '海': ['氵', '每'], '洗': ['氵', '先'], '清': ['氵', '青'], '河': ['氵', '可'],
    '流': ['氵', '㐬'], '波': ['氵', '皮'], '湖': ['氵', '胡'], '深': ['氵', '罙'],
    '浅': ['氵', '戔'], '温': ['氵', '昷'], '泳': ['氵', '永'], '漢': ['氵', '𦰩'],
    # ニンベン (人関連)
    '体': ['亻', '本'], '休': ['亻', '木'], '作': ['亻', '乍'], '使': ['亻', '吏'],
    '住': ['亻', '主'], '何': ['亻', '可'], '他': ['亻', '也'], '代': ['亻', '弋'],
    # ゴンベン (言葉関連)
    '語': ['言', '吾'], '話': ['言', '舌'], '説': ['言', '兌'], '読': ['言', '売'],
    '記': ['言', '己'], '計': ['言', '十'], '訳': ['言', '尺'], '訓': ['言', '川'],
    # キヘン (木関連)
    '林': ['木', '木'], '森': ['木', '林'], '村': ['木', '寸'], '机': ['木', '几'],
    '板': ['木', '反'], '根': ['木', '艮'], '植': ['木', '直'], '横': ['木', '黄'],
    # AI/脳関連の漢字
    '脳': ['月', '⺍', '凶'], '神': ['示', '申'], '経': ['糸', '圣'], '知': ['矢', '口'],
    '能': ['厶', '月', '匕'], '学': ['⺍', '子'], '習': ['羽', '白'],
}

# 部首の意味カテゴリ
RADICAL_MEANINGS = {
    '氵': 'water',    # 水関連
    '亻': 'person',   # 人関連  
    '言': 'speech',   # 言葉関連
    '木': 'tree',     # 木関連
    '月': 'body',     # 体の部分
    '心': 'heart',    # 心関連
    '糸': 'thread',   # 糸・関係
    '金': 'metal',    # 金属
    '火': 'fire',     # 火関連
    '土': 'earth',    # 土地関連
}


# =============================================================================
# 2. 文法ドリルデータ (全員推奨)
# =============================================================================
GRAMMAR_DRILLS = [
    # 動詞活用 - 五段活用
    ("書く", "過去", "書いた"),
    ("書く", "否定", "書かない"),
    ("書く", "可能", "書ける"),
    ("書く", "受身", "書かれる"),
    ("読む", "過去", "読んだ"),
    ("読む", "否定", "読まない"),
    ("走る", "過去", "走った"),
    ("走る", "否定", "走らない"),
    ("話す", "過去", "話した"),
    ("話す", "否定", "話さない"),
    ("待つ", "過去", "待った"),
    ("待つ", "否定", "待たない"),
    # 動詞活用 - 上一段・下一段
    ("食べる", "過去", "食べた"),
    ("食べる", "否定", "食べない"),
    ("食べる", "可能", "食べられる"),
    ("見る", "過去", "見た"),
    ("見る", "否定", "見ない"),
    ("起きる", "過去", "起きた"),
    ("起きる", "否定", "起きない"),
    # 形容詞活用
    ("大きい", "過去", "大きかった"),
    ("大きい", "否定", "大きくない"),
    ("美しい", "過去", "美しかった"),
    ("美しい", "否定", "美しくない"),
    ("楽しい", "過去", "楽しかった"),
    ("楽しい", "否定", "楽しくない"),
    # 形容動詞活用
    ("静かだ", "過去", "静かだった"),
    ("静かだ", "否定", "静かではない"),
    ("便利だ", "過去", "便利だった"),
    ("便利だ", "否定", "便利ではない"),
    # 敬語変換
    ("食べる", "敬語", "召し上がる"),
    ("見る", "敬語", "ご覧になる"),
    ("行く", "敬語", "いらっしゃる"),
    ("言う", "敬語", "おっしゃる"),
    ("する", "敬語", "なさる"),
    # 助詞の使い分け
    ("私_学生です", "は", "私は学生です"),
    ("彼_来た", "が", "彼が来た"),
    ("本_読む", "を", "本を読む"),
    ("東京_行く", "に", "東京に行く"),
    ("電車_行く", "で", "電車で行く"),
]


# =============================================================================
# 3. Teacher審判付き対話データ (Deep Think改良案)
# =============================================================================
TEACHER_DIALOGUES = [
    # (プロンプト, 理想的な返答, 評価基準)
    ("人工知能とは何ですか？", 
     "人工知能は、人間の知能を模倣するコンピュータシステムです。機械学習や深層学習によって、画像認識や自然言語処理などの複雑なタスクを実行できます。",
     {"clarity": 0.9, "accuracy": 0.9, "grammar": 0.95}),
    
    ("脳とコンピュータの違いは？",
     "脳は並列分散処理を行い、低消費電力で柔軟な学習が可能です。一方コンピュータは逐次処理が基本で、高速ですが消費電力が大きいです。",
     {"clarity": 0.85, "accuracy": 0.9, "grammar": 0.95}),
    
    ("スパイキングニューラルネットワークの利点は？",
     "生物の神経細胞を模倣し、スパイク信号で情報処理するため、従来のニューラルネットワークより省エネルギーで効率的です。",
     {"clarity": 0.85, "accuracy": 0.95, "grammar": 0.9}),
    
    ("11次元ハイパーキューブとは？",
     "2048個のノードが11ホップ以内で接続される効率的なネットワーク構造です。スケーラビリティと通信効率のバランスが優れています。",
     {"clarity": 0.8, "accuracy": 0.95, "grammar": 0.95}),
     
    ("日本語の特徴は？",
     "ひらがな、カタカナ、漢字の三種類の文字体系を持ち、主語省略が多く、敬語が発達しています。文法はSOV型です。",
     {"clarity": 0.9, "accuracy": 0.9, "grammar": 0.95}),
     
    ("ニューロモーフィックコンピューティングとは？",
     "脳の構造と機能を模倣したコンピューティングパラダイムで、メモリと処理が一体化し、省電力で並列処理が可能です。",
     {"clarity": 0.85, "accuracy": 0.9, "grammar": 0.95}),
]


# =============================================================================
# Numba並列計算関数
# =============================================================================
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
        
        # Reservoir update with parallel computation
        new_state = np.zeros(hidden_dim)
        for i in prange(hidden_dim):
            res_sum = 0.0
            for j in range(hidden_dim):
                res_sum += W_res[i, j] * state[j]
            new_state[i] = 0.9 * state[i] + 0.1 * (x[i] + res_sum)
            # Spiking activation
            if new_state[i] > 0.5:
                new_state[i] = 1.0
            else:
                new_state[i] = 0.0
        
        state = new_state
        
        # Compute logits
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


@njit(parallel=True, cache=True)
def parallel_weight_update(W, grad, lr, hidden_dim, vocab_size):
    """Parallel weight update."""
    for i in prange(vocab_size):
        for j in range(hidden_dim):
            W[i, j] -= lr * grad[i, j]
    return W


# =============================================================================
# SuperTeacherStudent クラス
# =============================================================================
class SuperTeacherStudent:
    """3つの学習法を統合したSNNモデル"""
    
    def __init__(self, vocab_size, hidden_dim, token_to_idx):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.token_to_idx = token_to_idx
        self.idx_to_token = {v: k for k, v in token_to_idx.items()}
        
        # Weights (will be loaded)
        self.embedding = None
        self.W_res = None
        self.W_spike = None
        self.W_membrane = None
        self.bias = None
        
        self.losses = []
        self.grammar_scores = []
        self.dialogue_scores = []
        
    def load_weights(self, model_data):
        """Load pretrained weights."""
        self.embedding = model_data['embedding'].astype(np.float64)
        self.W_res = model_data['W_res'].astype(np.float64)
        self.W_spike = model_data['W_spike'].astype(np.float64)
        self.W_membrane = model_data['W_membrane'].astype(np.float64)
        self.bias = model_data['bias'].astype(np.float64)
        
    def get_char_type(self, char):
        """文字種別を判定（品詞ヒント）"""
        # 漢字 (CJK統合漢字)
        if '\u4e00' <= char <= '\u9fff':
            return 'NOUN'  # 漢字 → ほぼ名詞/動詞語幹
        # カタカナ
        elif '\u30a0' <= char <= '\u30ff':
            return 'NOUN'  # カタカナ → 外来語名詞
        # ひらがな
        elif '\u3040' <= char <= '\u309f':
            return 'FUNC'  # ひらがな → 助詞/接続詞/活用語尾
        return 'OTHER'
    
    def tokenize_with_radicals(self, text):
        """漢字を部首分解してトークン化 + 文字種別ヒント"""
        tokens = []
        char_types = []  # 文字種別を記録
        
        for char in text:
            char_type = self.get_char_type(char)
            
            if char in RADICAL_DB:
                # 部首に分解
                radicals = RADICAL_DB[char]
                for r in radicals:
                    if r in self.token_to_idx:
                        tokens.append(self.token_to_idx[r])
                        char_types.append(char_type)
                    elif char in self.token_to_idx:
                        tokens.append(self.token_to_idx[char])
                        char_types.append(char_type)
                        break
            elif char in self.token_to_idx:
                tokens.append(self.token_to_idx[char])
                char_types.append(char_type)
            else:
                tokens.append(self.token_to_idx.get('<unk>', 0))
                char_types.append('OTHER')
        
        # 文字種別をインスタンス変数に保存（forward時に使用）
        self.last_char_types = char_types
        return tokens
    
    def forward(self, token_ids):
        """Forward pass."""
        logits, state = parallel_forward(
            self.embedding, self.W_res, self.W_spike, self.W_membrane, self.bias,
            np.array(token_ids), self.hidden_dim, self.vocab_size
        )
        return logits, state
    
    def train_grammar_drill(self, drill, lr=0.001):
        """文法ドリルで学習"""
        base_word, transform, expected = drill
        
        # 入力: "食べる [過去]" 形式
        input_text = f"{base_word} [{transform}]"
        input_tokens = self.tokenize_with_radicals(input_text)
        target_tokens = self.tokenize_with_radicals(expected)
        
        if len(input_tokens) == 0 or len(target_tokens) == 0:
            return 0.0
            
        # Forward
        logits, state = self.forward(input_tokens)
        
        # Loss計算と重み更新
        loss = 0.0
        for t, target_id in enumerate(target_tokens[:len(logits)]):
            probs = softmax_numba(logits[min(t, len(logits)-1)])
            loss -= np.log(probs[target_id] + 1e-10)
            
            # Gradient-based weight update
            error = probs.copy()
            error[target_id] -= 1.0  # Cross-entropy gradient
            
            # Update W_spike and W_membrane
            for h in range(self.hidden_dim):
                if state[h] > 0:  # Only update for active neurons
                    for v in range(self.vocab_size):
                        self.W_spike[v, h] -= lr * error[v] * state[h]
                        self.W_membrane[v, h] -= lr * error[v] * state[h]
            
            # Update bias
            self.bias -= lr * error
            
        return loss / max(len(target_tokens), 1)
    
    def train_dialogue(self, dialogue, lr=0.001):
        """Teacher審判付き対話学習"""
        prompt, ideal_response, scores = dialogue
        
        # プロンプトから応答を生成
        prompt_tokens = self.tokenize_with_radicals(prompt)
        response_tokens = self.tokenize_with_radicals(ideal_response)
        
        if len(prompt_tokens) == 0 or len(response_tokens) == 0:
            return 0.0, 0.0
            
        # Forward on prompt + response
        all_tokens = prompt_tokens + response_tokens
        logits, state = self.forward(all_tokens)
        
        # Teacher score (clarity * accuracy * grammar の平均)
        teacher_score = (scores['clarity'] + scores['accuracy'] + scores['grammar']) / 3
        
        # Loss on response part only with weight update
        loss = 0.0
        start_idx = len(prompt_tokens)
        for t, target_id in enumerate(response_tokens):
            idx = start_idx + t
            if idx < len(logits):
                probs = softmax_numba(logits[idx])
                loss -= np.log(probs[target_id] + 1e-10)
                
                # Gradient-based weight update (scaled by teacher score)
                error = probs.copy()
                error[target_id] -= 1.0
                
                # Scale learning rate by teacher score (higher score = more learning)
                scaled_lr = lr * teacher_score
                
                # Update weights
                for h in range(self.hidden_dim):
                    if state[h] > 0:
                        for v in range(self.vocab_size):
                            self.W_spike[v, h] -= scaled_lr * error[v] * state[h]
                            self.W_membrane[v, h] -= scaled_lr * error[v] * state[h]
                
                self.bias -= scaled_lr * error
        
        return loss / max(len(response_tokens), 1), teacher_score
    
    def generate(self, prompt, max_length=50):
        """テキスト生成"""
        tokens = self.tokenize_with_radicals(prompt)
        if len(tokens) == 0:
            tokens = [0]
            
        for _ in range(max_length):
            logits, _ = self.forward(tokens)
            
            # Get probabilities for next token
            probs = logits[-1]
            probs = probs - np.max(probs)
            probs = np.exp(probs)
            probs = np.clip(probs, 1e-10, None)
            probs = probs / np.sum(probs)
            probs = np.nan_to_num(probs, nan=1e-10)
            
            if np.sum(probs) <= 0 or not np.isfinite(np.sum(probs)):
                next_token = np.argmax(logits[-1])
            else:
                probs = probs / np.sum(probs)
                try:
                    next_token = np.random.choice(len(probs), p=probs)
                except:
                    next_token = np.argmax(logits[-1])
            
            tokens.append(next_token)
            
            # Stop on EOS or special token
            if next_token == 0:
                break
                
        # Decode
        result = ""
        for t in tokens[len(self.tokenize_with_radicals(prompt)):]:
            if t in self.idx_to_token:
                result += self.idx_to_token[t]
        return result


# =============================================================================
# メイン学習ループ
# =============================================================================
def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║    🌟 Super Teacher Learning v2 🌟                            ║
    ║    3つのAI提案を統合した最強学習！                            ║
    ║                                                               ║
    ║    1. 漢字部首分解入力 (Deep Think)                           ║
    ║    2. 文法ドリルデータ (全員推奨)                             ║
    ║    3. Teacher審判付き対話 (改良Friendly)                      ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Load model
    if not MODEL_PATH.exists():
        print(f"⚠️ モデルファイルが見つかりません: {MODEL_PATH}")
        print("先に5000エポック学習を完了させてください")
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
    print(f"部首DB: {len(RADICAL_DB)}漢字")
    print(f"文法ドリル: {len(GRAMMAR_DRILLS)}パターン")
    print(f"対話例: {len(TEACHER_DIALOGUES)}件")
    
    # Create student
    student = SuperTeacherStudent(vocab_size, hidden_dim, token_to_idx)
    student.load_weights(model_data)
    
    print("\nJITコンパイル中（初回のみ）...")
    # Warmup
    test_tokens = [0, 1, 2]
    _ = student.forward(test_tokens)
    print("JIT完了!")
    
    # Training
    print("\n" + "="*60)
    print("🎓 Super Teacher Learning 開始！")
    print("="*60)
    
    n_epochs = 1000
    start_time = time.time()
    results = []
    
    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0.0
        grammar_correct = 0
        dialogue_total_score = 0.0
        
        # Phase 1: 文法ドリル学習
        for drill in GRAMMAR_DRILLS:
            loss = student.train_grammar_drill(drill)
            epoch_loss += loss
            
        # Phase 2: Teacher審判付き対話学習
        for dialogue in TEACHER_DIALOGUES:
            loss, score = student.train_dialogue(dialogue)
            epoch_loss += loss
            dialogue_total_score += score
            
        avg_loss = epoch_loss / (len(GRAMMAR_DRILLS) + len(TEACHER_DIALOGUES))
        avg_score = dialogue_total_score / len(TEACHER_DIALOGUES)
        student.losses.append(avg_loss)
        
        if epoch % 10 == 0:
            elapsed = time.time() - start_time
            speed = epoch / elapsed
            remaining = (n_epochs - epoch) / speed if speed > 0 else 0
            
            sample = student.generate("人工知能は", max_length=30)
            
            print(f"Epoch {epoch}/{n_epochs} | Loss: {avg_loss:.3f} | Teacher得点: {avg_score:.2f} | 残り: {remaining:.0f}秒")
            print(f"  生成: {sample[:40]}...")
            
            results.append({
                'epoch': epoch,
                'loss': float(avg_loss),
                'teacher_score': float(avg_score),
                'sample': sample[:60]
            })
    
    total_time = time.time() - start_time
    print(f"\n🎉 学習完了！ 総時間: {total_time/60:.1f}分")
    
    # Save model
    final_path = RESULTS_DIR.parent / "checkpoints" / "model_super_teacher_v2.pkl"
    model_data = {
        'vocab_size': student.vocab_size,
        'hidden_dim': student.hidden_dim,
        'embedding': student.embedding,
        'W_res': student.W_res,
        'W_spike': student.W_spike,
        'W_membrane': student.W_membrane,
        'bias': student.bias,
    }
    with open(final_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"モデル保存: {final_path}")
    
    # Save results
    with open(RESULTS_DIR / 'super_teacher_v2_results.json', 'w', encoding='utf-8') as f:
        json.dump({
            'training_time_sec': total_time,
            'final_loss': float(student.losses[-1]) if student.losses else None,
            'features': ['radical_decomposition', 'grammar_drills', 'teacher_judged_dialogue'],
            'results': results,
            'losses': [float(l) for l in student.losses],
        }, f, ensure_ascii=False, indent=2)
    print(f"結果保存: {RESULTS_DIR / 'super_teacher_v2_results.json'}")
    
    # Final generation test
    print("\n" + "="*60)
    print("📝 最終生成テスト")
    print("="*60)
    
    test_prompts = [
        "人工知能は",
        "日本語の",
        "スパイキング",
        "脳と",
    ]
    
    for prompt in test_prompts:
        result = student.generate(prompt, max_length=40)
        print(f"「{prompt}」→ {result}")


if __name__ == "__main__":
    main()
