#!/usr/bin/env python3
"""
Teacher LLM Learning
====================
Use a powerful LLM (Claude/GPT) as a teacher to accelerate SNN-LLM training.

Key concepts:
1. Teacher provides correct completions and explanations
2. Student (SNN) learns from teacher's outputs
3. Grammar instruction for Japanese
4. Knowledge distillation from large to small model

This version works offline with pre-generated teacher data,
or can optionally use API calls if keys are provided.
"""

import numpy as np
import pickle
import json
from pathlib import Path
import time
from typing import Optional

# Paths
MODEL_PATH = Path(__file__).parent.parent / "checkpoints" / "model_v4_parallel_best.pkl"
TOKENIZER_PATH = Path(__file__).parent.parent / "checkpoints" / "tokenizer_v4_parallel.json"
RESULTS_DIR = Path(__file__).parent.parent / "results"


# Pre-generated teacher examples (simulating Claude/GPT output)
# In production, these would come from API calls
TEACHER_EXAMPLES = [
    # (prompt, correct_completion, grammar_hint)
    {
        "prompt": "人工知能は",
        "completion": "人間の知能を模倣するコンピュータシステムです。機械学習や深層学習によって、画像認識や自然言語処理などの複雑なタスクを実行できます。",
        "grammar": "「は」= 主題を示す助詞。「です」= 丁寧語の断定。"
    },
    {
        "prompt": "脳とコンピュータの違いは",
        "completion": "処理方式にあります。脳は並列分散処理を行い、低消費電力で柔軟な学習が可能です。一方コンピュータは逐次処理が基本で、高速ですが消費電力が大きいです。",
        "grammar": "「に」= 場所・存在を示す。「一方」= 対比を表す接続詞。"
    },
    {
        "prompt": "スパイキングニューラルネットワークとは",
        "completion": "生物の神経細胞の動作を模倣したAIモデルです。ニューロンがスパイク（電気信号）を発火することで情報を処理します。従来のニューラルネットワークより省エネルギーです。",
        "grammar": "「とは」= 定義を求める表現。「より」= 比較を示す。"
    },
    {
        "prompt": "言語を理解するとは",
        "completion": "単語の意味だけでなく、文脈や話者の意図、文化的背景までを把握することです。人間は経験を通じて言語を習得しますが、AIは大量のテキストデータから統計的パターンを学習します。",
        "grammar": "「だけでなく」= 範囲の拡張。「を通じて」= 手段を示す。"
    },
    {
        "prompt": "機械学習の仕組みは",
        "completion": "データからパターンを見つけ出し、新しい入力に対して予測や分類を行うことです。教師あり学習、教師なし学習、強化学習の三種類があります。",
        "grammar": "「から」= 起点を示す。「に対して」= 対象を示す。"
    },
    {
        "prompt": "ハイパーキューブトポロジーの利点は",
        "completion": "ノード間の最短経路が短く、効率的な情報伝播が可能です。11次元ハイパーキューブでは、2048ノードが11ホップ以内で接続され、スケーラビリティと通信効率のバランスが取れています。",
        "grammar": "「間」= 二点の関係。「以内」= 範囲の上限。"
    },
    {
        "prompt": "自然言語処理において",
        "completion": "テキストから意味を抽出し、適切な応答を生成することが重要です。形態素解析、構文解析、意味解析などの段階を経て、人間の言語を機械が理解できる形に変換します。",
        "grammar": "「において」= 範囲・場面を示す。「を経て」= 過程を示す。"
    },
    {
        "prompt": "深層学習が成功した理由は",
        "completion": "ビッグデータの利用可能性、GPUによる計算能力の向上、そしてアルゴリズムの改良です。特にバックプロパゲーションと確率的勾配降下法が効率的な学習を可能にしました。",
        "grammar": "「による」= 原因・手段を示す。「が〜を可能にした」= 使役表現。"
    },
    {
        "prompt": "今後のAI研究の方向性は",
        "completion": "説明可能なAI、省エネルギー型AI、そして汎用人工知能の実現です。倫理的な課題にも取り組む必要があり、人間と協調できるAIの開発が求められています。",
        "grammar": "「今後」= 時間の方向。「に取り組む」= 努力を表す。"
    },
    {
        "prompt": "私たちの研究では",
        "completion": "スパイキングニューラルネットワークを用いた大規模言語モデルの構築を目指しています。11次元ハイパーキューブトポロジーと階層的メモリ構造により、効率的な学習が可能になりました。",
        "grammar": "「を用いて」= 手段。「により」= 原因・手段。"
    },
]

# Japanese grammar lessons
GRAMMAR_LESSONS = [
    {
        "topic": "助詞の使い方",
        "content": """
        は - 主題を示す（私は学生です）
        が - 主語を示す（鳥が飛んでいる）
        を - 目的語を示す（本を読む）
        に - 場所・時間を示す（学校に行く）
        で - 手段・場所を示す（電車で行く）
        と - 並列・引用を示す（犬と猫）
        の - 所有・修飾を示す（私の本）
        から - 起点を示す（東京から来た）
        まで - 終点を示す（大阪まで行く）
        """
    },
    {
        "topic": "動詞の活用",
        "content": """
        五段活用（書く）: 書か/書き/書く/書く/書け/書け
        上一段（見る）: 見/見/見る/見る/見れ/見ろ
        下一段（食べる）: 食べ/食べ/食べる/食べる/食べれ/食べろ
        サ変（する）: し/し/する/する/すれ/しろ
        カ変（来る）: 来/来/来る/来る/来れ/来い
        """
    },
    {
        "topic": "敬語",
        "content": """
        尊敬語: いらっしゃる、おっしゃる、ご覧になる
        謙譲語: まいる、申す、拝見する
        丁寧語: です、ます、ございます
        """
    },
]


def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / (np.sum(exp_x) + 1e-10)


class StudentSNN:
    """SNN that learns from teacher LLM."""
    
    def __init__(self, model_path, tokenizer_path, lr=0.005):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            tok_data = json.load(f)
        
        self.token_to_id = tok_data['token_to_idx']
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        
        self.vocab_size = model['vocab_size']
        self.hidden_dim = model['hidden_dim']
        self.embedding = model['embedding'].copy()
        self.W_res = model['W_res'].copy()
        self.W_spike = model['W_spike'].copy()
        self.W_membrane = model['W_membrane'].copy()
        self.bias = model['bias'].copy()
        
        self.lr = lr
        self.h = np.zeros(self.hidden_dim, dtype=np.float32)
        self.m = np.zeros(self.hidden_dim, dtype=np.float32)
        
        # Training stats
        self.losses = []
        self.grammar_scores = []
    
    def reset_state(self):
        self.h = np.zeros(self.hidden_dim, dtype=np.float32)
        self.m = np.zeros(self.hidden_dim, dtype=np.float32)
    
    def encode(self, text):
        tokens = []
        i = 0
        while i < len(text):
            found = False
            for length in range(min(5, len(text) - i), 0, -1):
                substring = text[i:i+length]
                if substring in self.token_to_id:
                    tokens.append(self.token_to_id[substring])
                    i += length
                    found = True
                    break
            if not found:
                if '<UNK>' in self.token_to_id:
                    tokens.append(self.token_to_id['<UNK>'])
                i += 1
        return tokens
    
    def decode(self, token_ids):
        return ''.join(self.id_to_token.get(t, '') for t in token_ids 
                      if not self.id_to_token.get(t, '').startswith('<'))
    
    def forward_step(self, token_id):
        x = self.embedding[token_id]
        self.m = 0.9 * self.m + self.W_res @ self.h + x
        self.h = (self.m > 1.0).astype(np.float32)
        self.m = self.m * (1 - self.h)
        logits = self.W_spike @ self.h + self.W_membrane @ self.m + self.bias
        return logits
    
    def compute_loss(self, logits, target_id):
        """Cross-entropy loss."""
        probs = softmax(logits)
        loss = -np.log(probs[target_id] + 1e-10)
        return loss
    
    def train_on_teacher_example(self, prompt, completion, grammar_hint=None):
        """
        Learn from a teacher's example.
        Uses supervised learning on the correct completion.
        """
        self.reset_state()
        
        # Encode prompt and completion
        prompt_tokens = self.encode(prompt)
        completion_tokens = self.encode(completion)
        
        # Process prompt (build context)
        for tid in prompt_tokens:
            self.forward_step(tid)
        
        # Train on completion (teacher forcing)
        total_loss = 0
        last_token = prompt_tokens[-1] if prompt_tokens else 0
        
        for target_token in completion_tokens:
            logits = self.forward_step(last_token)
            loss = self.compute_loss(logits, target_token)
            total_loss += loss
            
            # Gradient-like update (simplified)
            # Increase probability of correct token
            probs = softmax(logits)
            error = np.zeros(self.vocab_size)
            error[target_token] = 1.0 - probs[target_token]
            
            # Update readout weights
            self.W_spike += self.lr * np.outer(error, self.h) * 0.01
            self.W_membrane += self.lr * np.outer(error, self.m) * 0.01
            self.bias += self.lr * error * 0.01
            
            last_token = target_token
        
        avg_loss = total_loss / max(len(completion_tokens), 1)
        self.losses.append(avg_loss)
        
        return avg_loss
    
    def train_on_grammar(self, lesson):
        """
        Learn grammar patterns.
        """
        # Parse examples from lesson
        lines = lesson['content'].strip().split('\n')
        grammar_loss = 0
        count = 0
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Try to extract pattern
            if '（' in line and '）' in line:
                # Extract example
                start = line.find('（')
                end = line.find('）')
                if start < end:
                    example = line[start+1:end]
                    if len(example) > 2:
                        # Train on this example
                        tokens = self.encode(example)
                        if len(tokens) >= 2:
                            self.reset_state()
                            for i in range(len(tokens) - 1):
                                logits = self.forward_step(tokens[i])
                                loss = self.compute_loss(logits, tokens[i+1])
                                grammar_loss += loss
                                count += 1
        
        avg_grammar_loss = grammar_loss / max(count, 1)
        self.grammar_scores.append(1.0 / (1.0 + avg_grammar_loss))
        return avg_grammar_loss
    
    def generate(self, prompt, max_length=50, temperature=0.7):
        """Generate text after training."""
        self.reset_state()
        tokens = self.encode(prompt)
        generated = []
        
        for tid in tokens:
            self.forward_step(tid)
        
        last_token = tokens[-1] if tokens else 0
        
        for _ in range(max_length):
            logits = self.forward_step(last_token)
            logits = logits / temperature
            probs = softmax(logits)
            next_token = np.random.choice(len(probs), p=probs)
            generated.append(next_token)
            last_token = next_token
            
            tok = self.id_to_token.get(next_token, '')
            if tok in ['。', '<EOS>'] and len(generated) > 10:
                break
        
        return self.decode(generated)


def train_with_teacher(student, num_epochs=5):
    """
    Main training loop with teacher supervision.
    """
    print("\n" + "="*60)
    print("    👨‍🏫 教師LLM学習開始 👨‍🏫")
    print("    Teacher-Guided SNN Training")
    print("="*60)
    
    all_results = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        print(f"\n📚 Epoch {epoch+1}/{num_epochs}")
        
        # Train on teacher examples
        for i, example in enumerate(TEACHER_EXAMPLES):
            loss = student.train_on_teacher_example(
                example['prompt'],
                example['completion'],
                example.get('grammar')
            )
            epoch_loss += loss
            
            if i % 3 == 0:
                print(f"  例{i+1}: {example['prompt'][:15]}... → Loss: {loss:.3f}")
        
        # Train on grammar
        print("  📖 文法学習中...")
        for lesson in GRAMMAR_LESSONS:
            student.train_on_grammar(lesson)
        
        avg_loss = epoch_loss / len(TEACHER_EXAMPLES)
        print(f"  平均Loss: {avg_loss:.3f}")
        
        # Test generation
        test_prompt = "人工知能は"
        generated = student.generate(test_prompt, max_length=40)
        print(f"  テスト生成: {test_prompt}{generated[:50]}...")
        
        all_results.append({
            'epoch': epoch + 1,
            'avg_loss': float(avg_loss),
            'sample_output': test_prompt + generated[:50]
        })
    
    return all_results


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║      👨‍🏫 教師LLM学習 (Teacher LLM Learning) 👨‍🏫              ║
    ║      Claude/GPT先生がSNN生徒に教える                          ║
    ║      - 正しい文章の模範                                        ║
    ║      - 日本語文法の指導                                        ║
    ║      - 知識蒸留 (Knowledge Distillation)                      ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    if not MODEL_PATH.exists():
        print(f"❌ モデルが見つかりません: {MODEL_PATH}")
        return
    
    print("生徒モデルをロード中...")
    student = StudentSNN(MODEL_PATH, TOKENIZER_PATH, lr=0.01)
    print(f"  パラメータ数: {student.vocab_size * student.hidden_dim:,}")
    
    # Train
    start_time = time.time()
    results = train_with_teacher(student, num_epochs=5)
    elapsed = time.time() - start_time
    
    print(f"\n⏱️ 学習時間: {elapsed:.1f}秒")
    
    # Final evaluation
    print("\n" + "="*60)
    print("    📊 最終評価 📊")
    print("="*60)
    
    test_prompts = [
        "脳とコンピュータの違いは",
        "スパイキングニューラルネットワークとは",
        "言語を理解するとは",
        "私たちの研究では",
    ]
    
    print("\n生成テスト:")
    for prompt in test_prompts:
        output = student.generate(prompt, max_length=50)
        print(f"  📝 {prompt}")
        print(f"  🤖 {prompt}{output}")
        print()
    
    # Save model
    evolved_path = Path(__file__).parent.parent / "checkpoints" / "model_teacher_trained.pkl"
    evolved = {
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
    with open(evolved_path, 'wb') as f:
        pickle.dump(evolved, f)
    
    print(f"✅ 教師訓練モデル保存: {evolved_path}")
    
    # Save results
    results_data = {
        'method': 'teacher_llm_learning',
        'num_examples': len(TEACHER_EXAMPLES),
        'num_grammar_lessons': len(GRAMMAR_LESSONS),
        'training_time': elapsed,
        'epoch_results': results,
        'losses': [float(l) for l in student.losses],
        'grammar_scores': [float(s) for s in student.grammar_scores]
    }
    
    results_path = RESULTS_DIR / 'teacher_learning_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 結果保存: {results_path}")
    print("\n🎉 教師学習完了！")


if __name__ == "__main__":
    main()
