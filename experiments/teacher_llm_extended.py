#!/usr/bin/env python3
"""
Teacher LLM Learning - Extended Training
=========================================
Resume training from epoch 6 to epoch 50.
"""

import numpy as np
import pickle
import json
from pathlib import Path
import time

# Paths
MODEL_PATH = Path(__file__).parent.parent / "checkpoints" / "model_teacher_trained.pkl"
TOKENIZER_PATH = Path(__file__).parent.parent / "checkpoints" / "tokenizer_v4_parallel.json"
RESULTS_DIR = Path(__file__).parent.parent / "results"

# Teacher examples (same as before)
TEACHER_EXAMPLES = [
    {"prompt": "人工知能は", "completion": "人間の知能を模倣するコンピュータシステムです。機械学習や深層学習によって、画像認識や自然言語処理などの複雑なタスクを実行できます。"},
    {"prompt": "脳とコンピュータの違いは", "completion": "処理方式にあります。脳は並列分散処理を行い、低消費電力で柔軟な学習が可能です。一方コンピュータは逐次処理が基本で、高速ですが消費電力が大きいです。"},
    {"prompt": "スパイキングニューラルネットワークとは", "completion": "生物の神経細胞の動作を模倣したAIモデルです。ニューロンがスパイク（電気信号）を発火することで情報を処理します。従来のニューラルネットワークより省エネルギーです。"},
    {"prompt": "言語を理解するとは", "completion": "単語の意味だけでなく、文脈や話者の意図、文化的背景までを把握することです。人間は経験を通じて言語を習得しますが、AIは大量のテキストデータから統計的パターンを学習します。"},
    {"prompt": "機械学習の仕組みは", "completion": "データからパターンを見つけ出し、新しい入力に対して予測や分類を行うことです。教師あり学習、教師なし学習、強化学習の三種類があります。"},
    {"prompt": "ハイパーキューブトポロジーの利点は", "completion": "ノード間の最短経路が短く、効率的な情報伝播が可能です。11次元ハイパーキューブでは、2048ノードが11ホップ以内で接続され、スケーラビリティと通信効率のバランスが取れています。"},
    {"prompt": "自然言語処理において", "completion": "テキストから意味を抽出し、適切な応答を生成することが重要です。形態素解析、構文解析、意味解析などの段階を経て、人間の言語を機械が理解できる形に変換します。"},
    {"prompt": "深層学習が成功した理由は", "completion": "ビッグデータの利用可能性、GPUによる計算能力の向上、そしてアルゴリズムの改良です。特にバックプロパゲーションと確率的勾配降下法が効率的な学習を可能にしました。"},
    {"prompt": "今後のAI研究の方向性は", "completion": "説明可能なAI、省エネルギー型AI、そして汎用人工知能の実現です。倫理的な課題にも取り組む必要があり、人間と協調できるAIの開発が求められています。"},
    {"prompt": "私たちの研究では", "completion": "スパイキングニューラルネットワークを用いた大規模言語モデルの構築を目指しています。11次元ハイパーキューブトポロジーと階層的メモリ構造により、効率的な学習が可能になりました。"},
]

GRAMMAR_LESSONS = [
    {"topic": "助詞", "content": "は - 私は学生です\nが - 鳥が飛ぶ\nを - 本を読む\nに - 学校に行く"},
    {"topic": "動詞", "content": "書く - 書か/書き/書く\n見る - 見/見/見る\n食べる - 食べ/食べ/食べる"},
    {"topic": "敬語", "content": "いらっしゃる - 尊敬語\nまいる - 謙譲語\nです - 丁寧語"},
]


def softmax(x):
    x = x - np.max(x)
    return np.exp(x) / (np.sum(np.exp(x)) + 1e-10)


class StudentSNN:
    def __init__(self, model_path, tokenizer_path, lr=0.01):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            tok_data = json.load(f)
        
        self.token_to_id = tok_data['token_to_idx']
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.vocab_size = model['vocab_size']
        self.hidden_dim = model['hidden_dim']
        self.embedding = model['embedding']
        self.W_res = model['W_res']
        self.W_spike = model['W_spike']
        self.W_membrane = model['W_membrane']
        self.bias = model['bias']
        self.lr = lr
        self.h = np.zeros(self.hidden_dim, dtype=np.float32)
        self.m = np.zeros(self.hidden_dim, dtype=np.float32)
        self.losses = []
    
    def reset_state(self):
        self.h = np.zeros(self.hidden_dim, dtype=np.float32)
        self.m = np.zeros(self.hidden_dim, dtype=np.float32)
    
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
        return tokens
    
    def decode(self, ids):
        return ''.join(self.id_to_token.get(t, '') for t in ids 
                      if not self.id_to_token.get(t, '').startswith('<'))
    
    def forward_step(self, token_id):
        x = self.embedding[token_id]
        self.m = 0.9 * self.m + self.W_res @ self.h + x
        self.h = (self.m > 1.0).astype(np.float32)
        self.m = self.m * (1 - self.h)
        return self.W_spike @ self.h + self.W_membrane @ self.m + self.bias
    
    def train_step(self, prompt, completion):
        self.reset_state()
        prompt_tokens = self.encode(prompt)
        completion_tokens = self.encode(completion)
        
        for tid in prompt_tokens:
            self.forward_step(tid)
        
        total_loss = 0
        last_token = prompt_tokens[-1] if prompt_tokens else 0
        
        for target in completion_tokens:
            logits = self.forward_step(last_token)
            probs = softmax(logits)
            loss = -np.log(probs[target] + 1e-10)
            total_loss += loss
            
            error = np.zeros(self.vocab_size)
            error[target] = 1.0 - probs[target]
            self.W_spike += self.lr * np.outer(error, self.h) * 0.01
            self.W_membrane += self.lr * np.outer(error, self.m) * 0.01
            self.bias += self.lr * error * 0.01
            
            last_token = target
        
        return total_loss / max(len(completion_tokens), 1)
    
    def generate(self, prompt, max_length=50, temperature=0.7):
        self.reset_state()
        tokens = self.encode(prompt)
        generated = []
        
        for tid in tokens:
            self.forward_step(tid)
        
        last = tokens[-1] if tokens else 0
        for _ in range(max_length):
            logits = self.forward_step(last) / temperature
            probs = softmax(logits)
            next_token = np.random.choice(len(probs), p=probs)
            generated.append(next_token)
            last = next_token
            if self.id_to_token.get(next_token, '') in ['。', '<EOS>']:
                break
        
        return self.decode(generated)


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║      👨‍🏫 教師LLM学習 拡張版 (Epoch 6-50) 👨‍🏫                 ║
    ║      Resume from checkpoint                                  ║
    ║      推定時間: 約33分                                         ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    if not MODEL_PATH.exists():
        print(f"❌ 学習済みモデルが見つかりません: {MODEL_PATH}")
        print("   先に teacher_llm_learning.py を実行してください")
        return
    
    print("前回のモデルをロード中 (Epoch 5)...")
    student = StudentSNN(MODEL_PATH, TOKENIZER_PATH, lr=0.01)
    
    start_time = time.time()
    results = []
    
    for epoch in range(6, 51):  # 6 to 50
        epoch_loss = 0
        
        for example in TEACHER_EXAMPLES:
            loss = student.train_step(example['prompt'], example['completion'])
            epoch_loss += loss
        
        # Grammar training
        for lesson in GRAMMAR_LESSONS:
            for line in lesson['content'].split('\n'):
                if '-' in line:
                    parts = line.split('-')
                    if len(parts) >= 2:
                        example_text = parts[1].strip()
                        tokens = student.encode(example_text)
                        if len(tokens) >= 2:
                            student.reset_state()
                            for i in range(len(tokens)-1):
                                student.forward_step(tokens[i])
        
        avg_loss = epoch_loss / len(TEACHER_EXAMPLES)
        student.losses.append(avg_loss)
        
        if epoch % 5 == 0:
            elapsed = time.time() - start_time
            remaining = (50 - epoch) * (elapsed / (epoch - 5))
            sample = student.generate("人工知能は", max_length=40)
            print(f"Epoch {epoch}/50 | Loss: {avg_loss:.3f} | 残り: {remaining/60:.1f}分")
            print(f"  生成: {sample[:50]}...")
            
            results.append({
                'epoch': epoch,
                'loss': float(avg_loss),
                'sample': sample[:60]
            })
    
    total_time = time.time() - start_time
    print(f"\n⏱️ 学習時間: {total_time/60:.1f}分")
    
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
    final_path = RESULTS_DIR.parent / "checkpoints" / "model_teacher_50epochs.pkl"
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
    
    # Save results
    with open(RESULTS_DIR / 'teacher_50epochs_results.json', 'w', encoding='utf-8') as f:
        json.dump({
            'total_epochs': 50,
            'training_time_sec': total_time,
            'final_loss': float(student.losses[-1]) if student.losses else None,
            'results': results,
            'losses': [float(l) for l in student.losses]
        }, f, ensure_ascii=False, indent=2)
    
    print(f"✅ モデル保存: {final_path}")
    print("🎉 50エポック学習完了！")


if __name__ == "__main__":
    main()
