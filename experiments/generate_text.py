#!/usr/bin/env python3
"""
SNN-LLM v4 Text Generation Script
=================================
Uses the trained model to generate Japanese text.
"""

import numpy as np
import pickle
import json
from pathlib import Path

# Paths
MODEL_PATH = Path(__file__).parent.parent / "checkpoints" / "model_v4_parallel_best.pkl"
TOKENIZER_PATH = Path(__file__).parent.parent / "checkpoints" / "tokenizer_v4_parallel.json"


def softmax(x):
    """Numerically stable softmax."""
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / (np.sum(exp_x) + 1e-10)


def sample_token(logits, temperature=1.0, top_k=50, top_p=0.9):
    """Sample a token with temperature, top-k, and nucleus sampling."""
    logits = np.array(logits, dtype=np.float64)
    
    # Temperature scaling
    if temperature != 1.0:
        logits = logits / temperature
    
    # Top-k filtering
    if top_k > 0 and top_k < len(logits):
        indices = np.argsort(logits)[-top_k:]
        mask = np.ones_like(logits) * (-1e10)
        mask[indices] = logits[indices]
        logits = mask
    
    # Top-p (nucleus) sampling
    probs = softmax(logits)
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    cumsum = np.cumsum(sorted_probs)
    
    cutoff_idx = np.searchsorted(cumsum, top_p) + 1
    cutoff_idx = min(cutoff_idx, len(sorted_indices))
    
    allowed_indices = sorted_indices[:cutoff_idx]
    allowed_probs = probs[allowed_indices]
    allowed_probs = allowed_probs / (allowed_probs.sum() + 1e-10)
    
    return np.random.choice(allowed_indices, p=allowed_probs)


class TextGenerator:
    """Text generator using trained SNN-LLM model."""
    
    def __init__(self, model_path, tokenizer_path):
        print("Loading model and tokenizer...")
        
        # Load model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Load tokenizer
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            tok_data = json.load(f)
        
        self.token_to_id = tok_data['token_to_idx']
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        
        # Model parameters
        self.vocab_size = self.model['vocab_size']
        self.hidden_dim = self.model['hidden_dim']
        self.hypercube_dim = self.model['hypercube_dim']
        
        # Extract weights
        self.embedding = self.model['embedding']       # (vocab_size, hidden_dim)
        self.W_res = self.model['W_res']               # (hidden_dim, hidden_dim)
        self.W_spike = self.model['W_spike']           # (hidden_dim, vocab_size) - readout
        self.W_membrane = self.model['W_membrane']     # (hidden_dim, vocab_size) - membrane readout
        self.bias = self.model['bias']                 # (vocab_size,)
        
        print(f"  Model loaded:")
        print(f"    Hidden dim: {self.hidden_dim}")
        print(f"    Vocab size: {self.vocab_size}")
        print(f"    Hypercube: {self.hypercube_dim}D")
        print(f"    Parameters: {self.count_params():,}")
    
    def count_params(self):
        """Count total parameters."""
        total = self.embedding.size + self.W_res.size + self.W_spike.size
        total += self.W_membrane.size + self.bias.size
        return total
    
    def encode(self, text):
        """Simple character-based encoding."""
        tokens = []
        i = 0
        while i < len(text):
            found = False
            # Try longest match first (up to 5 chars)
            for length in range(min(5, len(text) - i), 0, -1):
                substring = text[i:i+length]
                if substring in self.token_to_id:
                    tokens.append(self.token_to_id[substring])
                    i += length
                    found = True
                    break
            if not found:
                # Unknown character - use UNK or skip
                if '<UNK>' in self.token_to_id:
                    tokens.append(self.token_to_id['<UNK>'])
                i += 1
        return tokens
    
    def decode(self, token_ids):
        """Decode token IDs to text."""
        chars = []
        for tid in token_ids:
            if tid in self.id_to_token:
                tok = self.id_to_token[tid]
                if not tok.startswith('<'):  # Skip special tokens
                    chars.append(tok)
        return ''.join(chars)
    
    def forward_step(self, token_id, h, m):
        """
        Single forward step through the SNN.
        h: spike state (hidden_dim,)
        m: membrane potential (hidden_dim,)
        Returns: new h, new m, logits
        """
        # Get embedding for current token
        x = self.embedding[token_id]  # (hidden_dim,)
        
        # Update membrane potential
        # m = 0.9 * m + W_res @ h + x
        m_new = 0.9 * m + self.W_res @ h + x
        
        # Spiking: apply threshold
        threshold = 1.0
        h_new = (m_new > threshold).astype(np.float32)
        
        # Reset membrane where spiked
        m_new = m_new * (1 - h_new)
        
        # Compute logits using hybrid readout
        # W_spike and W_membrane are (vocab_size, hidden_dim)
        logits = self.W_spike @ h_new + self.W_membrane @ m_new + self.bias
        
        return h_new, m_new, logits
    
    def generate(self, prompt, max_length=100, temperature=0.8, top_k=50, top_p=0.9, 
                 show_progress=True):
        """Generate text from a prompt."""
        # Initialize hidden state
        h = np.zeros(self.hidden_dim, dtype=np.float32)
        m = np.zeros(self.hidden_dim, dtype=np.float32)
        
        # Encode prompt
        tokens = self.encode(prompt)
        if len(tokens) == 0:
            tokens = [self.token_to_id.get('<BOS>', 0)]
        
        generated_tokens = []
        
        # Process prompt to build context
        for token_id in tokens:
            h, m, _ = self.forward_step(token_id, h, m)
        
        if show_progress:
            print(f"\nPrompt: {prompt}")
            print("Generating", end="", flush=True)
        
        # Use last token as starting point
        last_token = tokens[-1]
        
        # Generate new tokens
        for i in range(max_length):
            # Forward step
            h, m, logits = self.forward_step(last_token, h, m)
            
            # Sample next token
            next_token = sample_token(logits, temperature, top_k, top_p)
            generated_tokens.append(next_token)
            last_token = next_token
            
            # Check for EOS
            if next_token in self.id_to_token:
                tok = self.id_to_token[next_token]
                if tok == '<EOS>':
                    break
            
            if show_progress and i % 10 == 0:
                print(".", end="", flush=True)
        
        if show_progress:
            print(" Done!")
        
        # Decode generated tokens
        generated_text = self.decode(generated_tokens)
        return generated_text


def interactive_mode(generator):
    """Interactive text generation mode."""
    print("\n" + "="*60)
    print("    🤖 SNN-LLM Interactive Text Generator 🤖")
    print("="*60)
    print("\nCommands:")
    print("  Type any text as a prompt")
    print("  /temp <value>  - Set temperature (default: 0.8)")
    print("  /len <value>   - Set max length (default: 100)")
    print("  /quit          - Exit")
    print("-"*60)
    
    temperature = 0.8
    max_length = 100
    
    while True:
        try:
            user_input = input("\n📝 Prompt: ").strip()
            
            if not user_input:
                continue
            
            if user_input.startswith('/'):
                parts = user_input.split()
                cmd = parts[0].lower()
                
                if cmd == '/quit':
                    print("Goodbye! 👋")
                    break
                elif cmd == '/temp' and len(parts) > 1:
                    temperature = float(parts[1])
                    print(f"Temperature set to {temperature}")
                    continue
                elif cmd == '/len' and len(parts) > 1:
                    max_length = int(parts[1])
                    print(f"Max length set to {max_length}")
                    continue
                else:
                    print("Unknown command")
                    continue
            
            # Generate text
            generated = generator.generate(
                user_input, 
                max_length=max_length,
                temperature=temperature
            )
            
            print(f"\n🤖 Generated:\n{user_input}{generated}")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye! 👋")
            break
        except Exception as e:
            print(f"Error: {e}")


def demo_generation(generator):
    """Run demo with various prompts."""
    print("\n" + "="*60)
    print("    🎭 SNN-LLM Generation Demo 🎭")
    print("="*60)
    
    prompts = [
        "人工知能は",
        "脳とコンピュータの違いは",
        "スパイキングニューラルネットワークとは",
        "私たちの研究では",
        "今日の天気は",
    ]
    
    print("\n[Temperature = 0.8, Top-k = 50, Top-p = 0.9]\n")
    
    for prompt in prompts:
        print("-" * 40)
        generated = generator.generate(
            prompt, 
            max_length=80,
            temperature=0.8,
            show_progress=False
        )
        print(f"📝 {prompt}")
        print(f"🤖 {prompt}{generated}")
    
    print("-" * 40)
    print("\n✅ Demo complete!")


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║        🧠 SNN-LLM v4 Text Generation 🧠                       ║
    ║        PPL: 81.92 | 28.7M Parameters | 11D Hypercube          ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Check files exist
    if not MODEL_PATH.exists():
        print(f"❌ Model not found: {MODEL_PATH}")
        return
    if not TOKENIZER_PATH.exists():
        print(f"❌ Tokenizer not found: {TOKENIZER_PATH}")
        return
    
    # Load generator
    generator = TextGenerator(MODEL_PATH, TOKENIZER_PATH)
    
    # Run demo first
    demo_generation(generator)
    
    # Then interactive mode
    print("\n\n" + "="*60)
    response = input("Start interactive mode? [Y/n]: ").strip().lower()
    if response != 'n':
        interactive_mode(generator)


if __name__ == "__main__":
    main()
