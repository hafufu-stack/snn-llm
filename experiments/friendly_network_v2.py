#!/usr/bin/env python3
"""
Friendly Network v2 - Enhanced Cooperative Learning
====================================================
Improvements over v1:
1. Empathy (共感) instead of just happiness
2. Diversity bonus to prevent mode collapse
3. Topic reset when conversation degrades
4. Better reward shaping

Key insight: 共感 = understanding + emotional resonance
"""

import numpy as np
import pickle
import json
from pathlib import Path
import time
from collections import deque

# Paths
MODEL_PATH = Path(__file__).parent.parent / "checkpoints" / "model_v4_parallel_best.pkl"
TOKENIZER_PATH = Path(__file__).parent.parent / "checkpoints" / "tokenizer_v4_parallel.json"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / (np.sum(exp_x) + 1e-10)


def sample_token(logits, temperature=0.8, top_k=50):
    logits = logits / max(temperature, 0.1)
    if top_k > 0 and top_k < len(logits):
        indices = np.argsort(logits)[-top_k:]
        mask = np.ones_like(logits) * (-1e10)
        mask[indices] = logits[indices]
        logits = mask
    probs = softmax(logits)
    return np.random.choice(len(probs), p=probs)


class EmpathicSNN:
    """SNN-LLM with empathic learning capabilities."""
    
    def __init__(self, name, model_path, tokenizer_path, lr=0.001):
        self.name = name
        self.lr = lr
        
        # Load model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            tok_data = json.load(f)
        
        self.token_to_id = tok_data['token_to_idx']
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        
        # Model weights
        self.vocab_size = model['vocab_size']
        self.hidden_dim = model['hidden_dim']
        self.embedding = model['embedding'].copy()
        self.W_res = model['W_res'].copy()
        self.W_spike = model['W_spike'].copy()
        self.W_membrane = model['W_membrane'].copy()
        self.bias = model['bias'].copy()
        
        # State
        self.h = np.zeros(self.hidden_dim, dtype=np.float32)
        self.m = np.zeros(self.hidden_dim, dtype=np.float32)
        
        # History for diversity tracking
        self.recent_outputs = deque(maxlen=5)
        
        # Learning stats
        self.empathy_history = []
        self.clarity_history = []
        self.diversity_history = []
        
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
        chars = []
        for tid in token_ids:
            if tid in self.id_to_token:
                tok = self.id_to_token[tid]
                if not tok.startswith('<'):
                    chars.append(tok)
        return ''.join(chars)
    
    def forward_step(self, token_id):
        x = self.embedding[token_id]
        self.m = 0.9 * self.m + self.W_res @ self.h + x
        self.h = (self.m > 1.0).astype(np.float32)
        self.m = self.m * (1 - self.h)
        logits = self.W_spike @ self.h + self.W_membrane @ self.m + self.bias
        return logits
    
    def generate(self, prompt, max_length=40, temperature=0.9):
        """Generate with higher temperature for diversity."""
        self.reset_state()
        tokens = self.encode(prompt)
        generated = []
        
        for tid in tokens:
            self.forward_step(tid)
        
        last_token = tokens[-1] if tokens else 0
        seen_tokens = set()
        
        for i in range(max_length):
            logits = self.forward_step(last_token)
            
            # Penalize recently used tokens slightly
            for t in seen_tokens:
                if t < len(logits):
                    logits[t] *= 0.8
            
            next_token = sample_token(logits, temperature)
            generated.append(next_token)
            seen_tokens.add(next_token)
            last_token = next_token
            
            tok = self.id_to_token.get(next_token, '')
            if tok in ['<EOS>', '。', '\n'] and i > 10:
                break
            
            # Early stop on repetition
            if len(generated) > 5:
                last_5 = generated[-5:]
                if len(set(last_5)) == 1:
                    break
        
        output = self.decode(generated)
        self.recent_outputs.append(output)
        return output
    
    def evaluate_empathy(self, text, my_context):
        """
        共感度を評価 (Empathy evaluation)
        - 相手の文脈を理解しているか
        - 感情的な共鳴があるか
        - 建設的な返答か
        """
        if len(text) < 5:
            return 0.1
        
        # Context relevance: shared vocabulary
        my_tokens = set(self.encode(my_context))
        their_tokens = set(self.encode(text))
        if len(my_tokens) > 0:
            overlap = len(my_tokens & their_tokens) / len(my_tokens | their_tokens)
        else:
            overlap = 0
        
        # Emotional markers (positive/engaged)
        positive_markers = ['です', 'ます', 'ね', 'よ', '思', '感', '理解', '確か', '面白', '素晴']
        engagement_score = sum(0.1 for m in positive_markers if m in text)
        engagement_score = min(1.0, engagement_score)
        
        # Content richness (kanji ratio)
        kanji_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        richness = kanji_count / max(len(text), 1)
        
        # Combine
        empathy = 0.35 * overlap + 0.35 * engagement_score + 0.3 * richness
        return np.clip(empathy, 0, 1)
    
    def evaluate_clarity(self, text):
        """分かりやすさ評価"""
        if len(text) < 3:
            return 0.1
        
        particles = ['は', 'が', 'を', 'に', 'で', 'と', 'の', 'も']
        has_particle = any(p in text for p in particles)
        
        # Sentence structure
        endings = ['。', 'です', 'ます', 'た', 'だ', 'る']
        has_ending = any(e in text for e in endings)
        
        # Repetition penalty
        chars = list(text)
        if len(chars) > 2:
            rep_count = sum(1 for i in range(2, len(chars)) 
                          if chars[i] == chars[i-1] == chars[i-2])
            rep_penalty = max(0, 1 - rep_count / 5)
        else:
            rep_penalty = 1.0
        
        # Symbol spam penalty
        symbol_count = sum(1 for c in text if c in '=─━│┃╔╗╚╝║')
        symbol_penalty = max(0, 1 - symbol_count / 10)
        
        clarity = (0.2 * has_particle + 0.2 * has_ending + 
                   0.3 * rep_penalty + 0.3 * symbol_penalty)
        return np.clip(clarity, 0, 1)
    
    def evaluate_diversity(self, text):
        """多様性評価 - 過去の出力と比較"""
        if len(self.recent_outputs) == 0:
            return 1.0
        
        # Compare with recent outputs
        current_set = set(text)
        total_overlap = 0
        
        for past in self.recent_outputs:
            past_set = set(past)
            if len(current_set | past_set) > 0:
                overlap = len(current_set & past_set) / len(current_set | past_set)
                total_overlap += overlap
        
        avg_overlap = total_overlap / len(self.recent_outputs)
        diversity = 1.0 - avg_overlap
        return np.clip(diversity, 0, 1)
    
    def update_weights(self, tokens, empathy, clarity, diversity):
        """
        Enhanced weight update with all three metrics.
        """
        # Combined reward
        reward = 0.4 * empathy + 0.3 * clarity + 0.3 * diversity
        
        # Only update if reward is reasonable
        if reward > 0.35 and len(tokens) > 0:
            update_strength = self.lr * (reward - 0.35)
            
            for tid in tokens[-15:]:
                if tid < self.vocab_size:
                    # Small perturbation in positive direction
                    self.embedding[tid] += update_strength * np.random.randn(self.hidden_dim) * 0.001
        
        # Record
        self.empathy_history.append(empathy)
        self.clarity_history.append(clarity)
        self.diversity_history.append(diversity)
        
        return reward


def run_dialogue_v2(model_a, model_b, num_turns=60):
    """
    Enhanced dialogue with recovery mechanism.
    """
    topics = [
        "人工知能は人間を超えることができるでしょうか",
        "脳とコンピュータは根本的に何が違うのでしょう",
        "スパイキングニューラルネットワークの可能性とは",
        "言語を理解するとはどういうことでしょう",
        "機械は本当に学習できるのでしょうか",
        "意識とは何かという問題について",
        "未来のAIはどのような形になるでしょう",
    ]
    
    dialogue_log = []
    low_quality_streak = 0
    current_topic_idx = 0
    
    print("\n" + "="*60)
    print("    🧠💬🧠 友好的ネットワーク v2 🧠💬🧠")
    print("    共感 + 多様性 + 回復メカニズム")
    print("="*60)
    
    current_text = topics[current_topic_idx]
    print(f"\n📌 開始トピック: {current_text}")
    
    for turn in range(num_turns):
        # Model A generates
        response_a = model_a.generate(current_text)
        tokens_a = model_a.encode(response_a)
        
        # Model B evaluates A with empathy
        empathy_a = model_b.evaluate_empathy(response_a, current_text)
        clarity_a = model_b.evaluate_clarity(response_a)
        diversity_a = model_a.evaluate_diversity(response_a)
        
        reward_a = model_a.update_weights(tokens_a, empathy_a, clarity_a, diversity_a)
        
        entry_a = {
            'turn': turn, 'speaker': model_a.name,
            'text': response_a, 'empathy': empathy_a,
            'clarity': clarity_a, 'diversity': diversity_a, 'reward': reward_a
        }
        dialogue_log.append(entry_a)
        
        # Model B responds
        if len(response_a) > 5 and clarity_a > 0.3:
            current_text = response_a
        
        response_b = model_b.generate(current_text)
        tokens_b = model_b.encode(response_b)
        
        empathy_b = model_a.evaluate_empathy(response_b, current_text)
        clarity_b = model_a.evaluate_clarity(response_b)
        diversity_b = model_b.evaluate_diversity(response_b)
        
        reward_b = model_b.update_weights(tokens_b, empathy_b, clarity_b, diversity_b)
        
        entry_b = {
            'turn': turn, 'speaker': model_b.name,
            'text': response_b, 'empathy': empathy_b,
            'clarity': clarity_b, 'diversity': diversity_b, 'reward': reward_b
        }
        dialogue_log.append(entry_b)
        
        # Check quality and reset if needed
        avg_quality = (reward_a + reward_b) / 2
        if avg_quality < 0.25:
            low_quality_streak += 1
        else:
            low_quality_streak = 0
        
        # Recovery: switch topic if quality drops
        if low_quality_streak >= 3:
            current_topic_idx = (current_topic_idx + 1) % len(topics)
            current_text = topics[current_topic_idx]
            model_a.reset_state()
            model_b.reset_state()
            low_quality_streak = 0
            print(f"\n🔄 Turn {turn+1}: トピック切替 → {current_text[:20]}...")
        elif turn % 10 == 0:
            print(f"\n--- Turn {turn+1} ---")
            print(f"{model_a.name}: {response_a[:40]}...")
            print(f"  共感: {empathy_a:.2f}, 明瞭: {clarity_a:.2f}, 多様: {diversity_a:.2f}")
            print(f"{model_b.name}: {response_b[:40]}...")
            print(f"  共感: {empathy_b:.2f}, 明瞭: {clarity_b:.2f}, 多様: {diversity_b:.2f}")
        
        # Use B's response as next prompt
        if len(response_b) > 5 and clarity_b > 0.3:
            current_text = response_b
        else:
            # Pick new topic
            current_topic_idx = (current_topic_idx + 1) % len(topics)
            current_text = topics[current_topic_idx]
    
    return dialogue_log


def analyze_v2(model_a, model_b, dialogue_log):
    """Analyze v2 results."""
    print("\n" + "="*60)
    print("    📊 v2 学習結果分析 📊")
    print("="*60)
    
    # Stats
    for model, name in [(model_a, "α"), (model_b, "β")]:
        emp = model.empathy_history
        cla = model.clarity_history
        div = model.diversity_history
        
        if len(emp) >= 20:
            print(f"\nSNN-{name}:")
            print(f"  共感:   最初10: {np.mean(emp[:10]):.3f} → 最後10: {np.mean(emp[-10:]):.3f}")
            print(f"  明瞭:   最初10: {np.mean(cla[:10]):.3f} → 最後10: {np.mean(cla[-10:]):.3f}")
            print(f"  多様性: 最初10: {np.mean(div[:10]):.3f} → 最後10: {np.mean(div[-10:]):.3f}")
    
    # Count good turns
    good_turns = sum(1 for e in dialogue_log if e.get('reward', 0) > 0.4)
    print(f"\n高品質ターン: {good_turns}/{len(dialogue_log)} ({good_turns/len(dialogue_log)*100:.1f}%)")
    
    # Save
    results = {
        'version': 'v2_empathy',
        'model_a': {
            'name': model_a.name,
            'empathy_history': model_a.empathy_history,
            'clarity_history': model_a.clarity_history,
            'diversity_history': model_a.diversity_history
        },
        'model_b': {
            'name': model_b.name,
            'empathy_history': model_b.empathy_history,
            'clarity_history': model_b.clarity_history,
            'diversity_history': model_b.diversity_history
        },
        'dialogue': [
            {'turn': e['turn'], 'speaker': e['speaker'], 
             'text': e['text'][:80], 'empathy': e['empathy'],
             'clarity': e['clarity'], 'diversity': e['diversity']}
            for e in dialogue_log
        ]
    }
    
    results_path = RESULTS_DIR / 'friendly_network_v2_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 結果保存: {results_path}")
    return results


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║     🧠💬🧠 友好的ネットワーク v2 🧠💬🧠                       ║
    ║     Enhanced Cooperative Learning with Empathy               ║
    ║     共感 + 多様性 + モード崩壊対策                            ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    if not MODEL_PATH.exists():
        print(f"❌ モデルが見つかりません: {MODEL_PATH}")
        return
    
    print("モデルをロード中...")
    model_a = EmpathicSNN("SNN-α", MODEL_PATH, TOKENIZER_PATH, lr=0.002)
    model_b = EmpathicSNN("SNN-β", MODEL_PATH, TOKENIZER_PATH, lr=0.002)
    
    print(f"  {model_a.name}: 共感学習モード")
    print(f"  {model_b.name}: 共感学習モード")
    
    start_time = time.time()
    dialogue_log = run_dialogue_v2(model_a, model_b, num_turns=60)
    elapsed = time.time() - start_time
    
    print(f"\n⏱️ 対話時間: {elapsed:.1f}秒")
    
    results = analyze_v2(model_a, model_b, dialogue_log)
    
    # Save evolved model
    evolved_path = Path(__file__).parent.parent / "checkpoints" / "model_friendly_v2_evolved.pkl"
    evolved = {
        'vocab_size': model_a.vocab_size,
        'hidden_dim': model_a.hidden_dim,
        'hypercube_dim': 11,
        'embedding': model_a.embedding,
        'W_res': model_a.W_res,
        'W_spike': model_a.W_spike,
        'W_membrane': model_a.W_membrane,
        'bias': model_a.bias,
        'lr': model_a.lr
    }
    with open(evolved_path, 'wb') as f:
        pickle.dump(evolved, f)
    
    print(f"✅ 進化モデル保存: {evolved_path}")
    print("\n🎉 v2実験完了！")


if __name__ == "__main__":
    main()
