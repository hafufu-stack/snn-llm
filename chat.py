"""
Interactive Chat with SNN-LLM
==============================

Load trained model and chat interactively!

Usage: python chat.py
"""

import pickle
import json
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_model_and_tokenizer():
    """Load trained model and tokenizer"""
    
    # Load tokenizer
    with open("checkpoints/tokenizer_wikitext.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    char_to_idx = data['char_to_idx']
    idx_to_char = {int(i): c for c, i in char_to_idx.items()}
    
    # Load model
    with open("checkpoints/model_wikitext_best.pkl", 'rb') as f:
        model_data = pickle.load(f)
    
    return model_data, char_to_idx, idx_to_char


class LoadedSNNLM:
    """Loaded SNN Language Model"""
    
    def __init__(self, model_data):
        self.hidden_dim = model_data['hidden_dim']
        self.vocab_size = model_data['vocab_size']
        self.embedding = model_data['embedding']
        self.W_res = model_data['W_res']
        self.W_out = model_data['W_out']
        self.b_out = model_data['b_out']
        self.state = np.zeros(self.hidden_dim, dtype=np.float32)
    
    def forward(self, token_idx):
        emb = self.embedding[token_idx]
        h_rec = np.tanh(self.W_res @ self.state)
        self.state = 0.7 * self.state + 0.3 * (emb + h_rec)
        mask = self.state > 0.5
        self.state = np.where(mask, self.state * 0.7, self.state)
        return self.W_out @ self.state + self.b_out
    
    def reset(self):
        self.state = np.zeros(self.hidden_dim, dtype=np.float32)
    
    def generate(self, start_indices, max_len=200, temperature=0.7):
        self.reset()
        generated = list(start_indices)
        
        for idx in start_indices:
            self.forward(idx)
        
        for _ in range(max_len):
            logits = self.forward(generated[-1])
            log_p = logits / temperature
            exp_l = np.exp(log_p - np.max(log_p))
            probs = exp_l / np.sum(exp_l)
            next_idx = np.random.choice(len(probs), p=probs)
            generated.append(next_idx)
        
        return generated


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║         🧠 SNN-LLM Interactive Chat 🧠                        ║
    ║         Talk to your locally-trained neural network!          ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    print("Loading model...")
    model_data, char_to_idx, idx_to_char = load_model_and_tokenizer()
    model = LoadedSNNLM(model_data)
    print(f"Model loaded! Vocab: {len(char_to_idx)}, Hidden: {model.hidden_dim}")
    print("\nType a prompt and press Enter (or 'quit' to exit)")
    print("-" * 60)
    
    while True:
        try:
            prompt = input("\n🧠 You: ")
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Goodbye! 👋")
                break
            
            if not prompt:
                continue
            
            # Encode prompt
            prompt_tokens = [char_to_idx.get(c, 0) for c in prompt]
            
            # Generate
            generated_tokens = model.generate(prompt_tokens, max_len=200, temperature=0.7)
            generated_text = ''.join(idx_to_char.get(i, '?') for i in generated_tokens)
            
            print(f"🤖 SNN: {generated_text}")
            
        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
