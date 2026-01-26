#!/usr/bin/env python3
"""
単語単位SNN-LLM学習 v4 (多様性強化版)

改善点:
1. より大きなコーパス生成（AI生成テキストで拡充）
2. Dropout追加（過学習抑制）
3. 高Temperature生成（多様性向上）
"""

import numpy as np
import pickle
import json
import time
from pathlib import Path
from numba import njit, prange, set_num_threads

# MeCab
import fugashi
tagger = fugashi.Tagger()
print("✅ fugashi (MeCab) ロード成功")

set_num_threads(24)

# Paths
CORPUS_PATH = Path(__file__).parent.parent / "data" / "japanese_corpus.txt"
CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"
RESULTS_DIR = Path(__file__).parent.parent / "results"

# 最適パラメータ
INPUT_SCALE = 10.0
LEAK = 0.99
THRESHOLD = 1.0

# モデルパラメータ
HIDDEN_DIM = 2048
MAX_VOCAB_SIZE = 4000
EPOCHS = 5000
LR = 0.003  # 少し下げる
DROPOUT_RATE = 0.1  # 新規追加


@njit(cache=True)
def softmax_numba(x):
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / (np.sum(exp_x) + 1e-10)


@njit(parallel=True, cache=True)
def parallel_forward_dropout(embedding_scaled, W_res, W_spike, W_membrane, bias,
                              token_ids, hidden_dim, vocab_size, leak, dropout_mask):
    """並列フォワードパス（Dropout付き）"""
    n_tokens = len(token_ids)
    h = np.zeros(hidden_dim, dtype=np.float32)
    m = np.zeros(hidden_dim, dtype=np.float32)
    
    all_logits = np.zeros((n_tokens, vocab_size), dtype=np.float32)
    all_h = np.zeros((n_tokens, hidden_dim), dtype=np.float32)
    all_m = np.zeros((n_tokens, hidden_dim), dtype=np.float32)
    
    for t in range(n_tokens):
        tid = token_ids[t]
        if tid >= vocab_size:
            tid = 0
        x = embedding_scaled[tid]
        
        for i in prange(hidden_dim):
            m[i] = leak * m[i] + x[i]
            for j in range(hidden_dim):
                m[i] += W_res[i, j] * h[j]
        
        for i in prange(hidden_dim):
            if m[i] > 1.0:
                h[i] = 1.0 * dropout_mask[i]  # Dropout適用
                m[i] = 0.0
            else:
                h[i] = 0.0
        
        all_h[t] = h.copy()
        all_m[t] = m.copy()
        
        for i in prange(vocab_size):
            all_logits[t, i] = bias[i]
            for j in range(hidden_dim):
                all_logits[t, i] += W_spike[i, j] * h[j]
                all_logits[t, i] += W_membrane[i, j] * m[j]
    
    return all_logits, all_h, all_m


@njit(parallel=True, cache=True)
def parallel_weight_update(W_spike, W_membrane, bias, errors, all_h, all_m, lr):
    """並列重み更新"""
    n_tokens, hidden_dim = all_h.shape
    vocab_size = len(bias)
    
    for t in range(n_tokens):
        error = errors[t]
        h = all_h[t]
        m = all_m[t]
        
        for i in prange(vocab_size):
            bias[i] -= lr * error[i]
            for j in range(hidden_dim):
                W_spike[i, j] -= lr * error[i] * h[j]
                W_membrane[i, j] -= lr * error[i] * m[j]


def tokenize_words(text):
    words = []
    for word in tagger(text):
        if word.surface.strip():
            words.append(word.surface)
    return words


def generate_expanded_corpus():
    """多様なコーパスを生成"""
    corpus = []
    
    # 1. 既存コーパス読み込み
    with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                corpus.append(line.strip())
    
    # 2. 追加テキスト（多様なトピック）
    additional_texts = [
        # 科学技術
        "量子コンピュータは従来のコンピュータとは異なる原理で動作します。",
        "ブラックホールは強力な重力を持つ天体です。",
        "DNAは生物の遺伝情報を担う分子です。",
        "気候変動は地球規模の課題となっています。",
        "再生可能エネルギーの普及が進んでいます。",
        
        # 文化・社会
        "伝統的な日本文化は世界中で注目されています。",
        "グローバル化により国際交流が活発化しています。",
        "教育のデジタル化が急速に進展しています。",
        "多様性を尊重する社会への変革が求められています。",
        "持続可能な社会の実現に向けた取り組みが広がっています。",
        
        # 日常・生活
        "健康的な食生活は心身の健康に重要です。",
        "運動習慣は生活の質を向上させます。",
        "睡眠は脳と体の回復に不可欠です。",
        "ストレス管理は現代人にとって重要な課題です。",
        "趣味は人生を豊かにする活動です。",
        
        # 技術・AI
        "自然言語処理は人間の言語を理解する技術です。",
        "画像認識はコンピュータビジョンの基盤技術です。",
        "ロボット工学は自動化の実現に貢献しています。",
        "IoTは様々なデバイスをインターネットに接続します。",
        "エッジコンピューティングはデータ処理を分散化します。",
        
        # 歴史・人物
        "アインシュタインは相対性理論を提唱しました。",
        "ニュートンは万有引力の法則を発見しました。",
        "ダーウィンは進化論を体系化しました。",
        "チューリングは計算理論の基礎を築きました。",
        "フォン・ノイマンはコンピュータ・アーキテクチャを設計しました。",
        
        # 感情・表現
        "喜びは人生を彩る感情のひとつです。",
        "悲しみは成長の糧となることがあります。",
        "怒りをコントロールすることは重要なスキルです。",
        "驚きは新しい発見への扉を開きます。",
        "感謝の気持ちは人間関係を豊かにします。",
        
        # 抽象概念
        "時間は過去から未来へと流れていきます。",
        "空間は三次元の広がりを持っています。",
        "エネルギーは形態を変えて保存されます。",
        "情報は知識を構成する基本要素です。",
        "システムは複数の要素が連携して機能します。",
        
        # 文学的表現
        "桜の花びらが春風に舞い散ります。",
        "夕暮れの空が赤く染まっています。",
        "静かな夜に星が輝いています。",
        "雪が静かに降り積もります。",
        "波が穏やかに浜辺を洗います。",
        
        # 質問形式
        "なぜ空は青いのでしょうか。",
        "人工知能は意識を持つことができるでしょうか。",
        "宇宙の果てはどこにあるのでしょうか。",
        "言語はどのように進化してきたのでしょうか。",
        "創造性とは何でしょうか。",
        
        # 説明文
        "この技術は様々な分野で応用されています。",
        "研究者たちは新しい発見を続けています。",
        "実験結果は仮説を裏付けました。",
        "データ分析により傾向が明らかになりました。",
        "改善の余地はまだ残されています。",
    ]
    
    corpus.extend(additional_texts)
    
    # 3. バリエーション生成（既存文を少し変形）
    variations = []
    templates = [
        "{}について説明します。",
        "{}は重要な概念です。",
        "{}の研究が進んでいます。",
        "{}を理解することが大切です。",
        "{}に関する知識が求められています。",
    ]
    
    topics = [
        "人工知能", "機械学習", "深層学習", "ニューラルネットワーク",
        "自然言語処理", "画像認識", "音声認識", "強化学習",
        "データサイエンス", "統計学", "確率論", "線形代数",
        "物理学", "化学", "生物学", "数学", "情報科学",
        "哲学", "心理学", "言語学", "認知科学", "神経科学",
    ]
    
    for topic in topics:
        for template in templates:
            variations.append(template.format(topic))
    
    corpus.extend(variations)
    
    return corpus


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║    🔤 単語単位 SNN-LLM 学習 v4 (多様性強化版)                 ║
    ║                                                               ║
    ║    改善1: コーパス拡充（テンプレート生成）                    ║
    ║    改善2: Dropout 10%（過学習抑制）                          ║
    ║    改善3: 高Temperature生成                                  ║
    ║    並列: 24スレッド                                           ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # 拡充コーパス生成
    print("コーパスを拡充中...")
    corpus_lines = generate_expanded_corpus()
    print(f"コーパス行数: {len(corpus_lines)}")
    
    # 事前トークン化
    print("\n🔄 事前トークン化中...")
    word_freq = {}
    all_tokenized = []
    
    for line in corpus_lines:
        words = tokenize_words(line)
        all_tokenized.append(words)
        for w in words:
            word_freq[w] = word_freq.get(w, 0) + 1
    
    print(f"✅ 総単語種類: {len(word_freq)}")
    
    # 語彙構築
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    actual_vocab_size = min(MAX_VOCAB_SIZE, len(sorted_words) + 3)
    
    word_to_idx = {'<PAD>': 0, '<UNK>': 1, '<EOS>': 2}
    for word, freq in sorted_words[:actual_vocab_size - 3]:
        word_to_idx[word] = len(word_to_idx)
    
    idx_to_word = {v: k for k, v in word_to_idx.items()}
    vocab_size = len(word_to_idx)
    print(f"語彙サイズ: {vocab_size}")
    
    # データ変換
    train_data = []
    for words in all_tokenized:
        ids = np.array([word_to_idx.get(w, 1) for w in words], dtype=np.int64)
        if len(ids) >= 3:
            train_data.append(ids)
    print(f"学習サンプル数: {len(train_data)}")
    
    # モデル初期化
    print(f"\nモデル初期化（語彙: {vocab_size}）...")
    embedding = np.random.randn(vocab_size, HIDDEN_DIM).astype(np.float32) * 0.01
    W_res = np.random.randn(HIDDEN_DIM, HIDDEN_DIM).astype(np.float32) * 0.01
    W_spike = np.random.randn(vocab_size, HIDDEN_DIM).astype(np.float32) * 0.01
    W_membrane = np.random.randn(vocab_size, HIDDEN_DIM).astype(np.float32) * 0.01
    bias = np.zeros(vocab_size, dtype=np.float32)
    
    embedding_scaled = embedding * INPUT_SCALE
    
    params = vocab_size * HIDDEN_DIM * 3 + HIDDEN_DIM * HIDDEN_DIM + vocab_size
    print(f"パラメータ数: {params:,}")
    print(f"Dropout率: {DROPOUT_RATE * 100}%")
    
    # JIT compile
    print("\nJITコンパイル中...")
    dummy_ids = np.array([0, 1, 2], dtype=np.int64)
    dummy_mask = np.ones(HIDDEN_DIM, dtype=np.float32)
    _ = parallel_forward_dropout(embedding_scaled, W_res, W_spike, W_membrane, bias,
                                  dummy_ids, HIDDEN_DIM, vocab_size, LEAK, dummy_mask)
    print("JIT完了!")
    
    # 学習開始
    print("\n" + "=" * 60)
    print(f"🧠 {EPOCHS}エポック学習開始！")
    print("=" * 60)
    
    start_time = time.time()
    losses = []
    batch_size = 100
    
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        total_tokens = 0
        
        np.random.shuffle(train_data)
        
        for token_ids in train_data[:batch_size]:
            if len(token_ids) < 2:
                continue
            
            n_seq = len(token_ids) - 1
            
            # Dropoutマスク生成
            dropout_mask = (np.random.rand(HIDDEN_DIM) > DROPOUT_RATE).astype(np.float32)
            
            logits, all_h, all_m = parallel_forward_dropout(
                embedding_scaled, W_res, W_spike, W_membrane, bias,
                token_ids[:-1], HIDDEN_DIM, vocab_size, LEAK, dropout_mask
            )
            
            errors = np.zeros((n_seq, vocab_size), dtype=np.float32)
            
            for t in range(n_seq):
                probs = softmax_numba(logits[t])
                target = token_ids[t + 1]
                if target >= vocab_size:
                    target = 1
                
                loss = -np.log(probs[target] + 1e-10)
                epoch_loss += loss
                total_tokens += 1
                
                errors[t] = probs.copy()
                errors[t, target] -= 1.0
            
            parallel_weight_update(W_spike, W_membrane, bias, errors, all_h, all_m, LR)
        
        avg_loss = epoch_loss / max(total_tokens, 1)
        losses.append(avg_loss)
        
        if epoch % 100 == 0:
            elapsed = time.time() - start_time
            speed = epoch / elapsed
            remaining = (EPOCHS - epoch) / speed
            
            # 生成テスト（高Temperature）
            prompt_ids = [word_to_idx.get(w, 1) for w in ["人工", "知能"]]
            
            h = np.zeros(HIDDEN_DIM, dtype=np.float32)
            m = np.zeros(HIDDEN_DIM, dtype=np.float32)
            
            generated = list(prompt_ids)
            for _ in range(15):
                tid = generated[-1] if generated[-1] < vocab_size else 0
                x = embedding_scaled[tid]
                m = LEAK * m + x + W_res @ h
                h = np.where(m > THRESHOLD, 1.0, 0.0).astype(np.float32)
                m = np.where(m > THRESHOLD, 0.0, m)
                
                logits_gen = W_spike @ h + W_membrane @ m + bias
                # 高Temperature (1.0) で多様性向上
                probs = softmax_numba(logits_gen / 1.0)
                try:
                    next_token = np.random.choice(len(probs), p=probs)
                except:
                    next_token = np.argmax(logits_gen)
                generated.append(next_token)
            
            gen_text = "".join([idx_to_word.get(t, '?') for t in generated])
            
            print(f"Epoch {epoch:5d}/{EPOCHS} | Loss: {avg_loss:.3f} | "
                  f"速度: {speed:.1f} ep/s | 残り: {remaining/60:.0f}分")
            print(f"  生成: {gen_text[:60]}...")
        
        # チェックポイント
        if epoch % 1000 == 0:
            cp_data = {
                'vocab_size': vocab_size,
                'hidden_dim': HIDDEN_DIM,
                'embedding': embedding,
                'W_res': W_res,
                'W_spike': W_spike,
                'W_membrane': W_membrane,
                'bias': bias,
                'word_to_idx': word_to_idx,
                'input_scale': INPUT_SCALE,
                'leak': LEAK,
                'dropout_rate': DROPOUT_RATE,
            }
            cp_path = CHECKPOINT_DIR / f'model_word_v4_{epoch}epochs.pkl'
            with open(cp_path, 'wb') as f:
                pickle.dump(cp_data, f)
            print(f"  💾 チェックポイント: {cp_path.name}")
    
    # 最終保存
    final_data = {
        'vocab_size': vocab_size,
        'hidden_dim': HIDDEN_DIM,
        'embedding': embedding,
        'W_res': W_res,
        'W_spike': W_spike,
        'W_membrane': W_membrane,
        'bias': bias,
        'word_to_idx': word_to_idx,
        'input_scale': INPUT_SCALE,
        'leak': LEAK,
        'dropout_rate': DROPOUT_RATE,
    }
    
    final_path = CHECKPOINT_DIR / 'model_word_v4_final.pkl'
    with open(final_path, 'wb') as f:
        pickle.dump(final_data, f)
    
    tokenizer_path = CHECKPOINT_DIR / 'tokenizer_word_v4.json'
    with open(tokenizer_path, 'w', encoding='utf-8') as f:
        json.dump({'word_to_idx': word_to_idx}, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 60)
    print("📊 学習完了!")
    print("=" * 60)
    print(f"最終Loss: {losses[-1]:.3f}")
    print(f"総時間: {(time.time() - start_time)/3600:.1f}時間")
    print(f"モデル: {final_path}")


if __name__ == "__main__":
    main()
