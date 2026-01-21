"""
SNN Language Model (SNN-LM)
===========================

Ultra-low-power language model using Spiking Neural Networks
with 11D Hypercube topology.

Features:
- Temporal coding for high information capacity
- 11D Hypercube for efficient information propagation
- Chaotic reservoir dynamics for context memory
- 100x energy efficiency target

Author: Hiroto Funasaki (roll)
Date: 2026-01-21
"""

import numpy as np
from typing import List, Tuple, Optional
import hashlib

from .hypercube import create_hypercube_mask, create_hybrid_hypercube_mask


class LIFNeuron:
    """Leaky Integrate-and-Fire Neuron"""
    
    def __init__(self, dt: float = 0.5, tau: float = 20.0):
        self.dt = dt
        self.tau = tau
        self.v = -65.0
        self.v_rest = -65.0
        self.v_thresh = -50.0
        self.v_reset = -70.0
    
    def step(self, I_syn: float) -> float:
        """Update neuron state and return spike (0 or 1)"""
        dv = (-(self.v - self.v_rest) + I_syn) / self.tau * self.dt
        self.v += dv
        
        if self.v >= self.v_thresh:
            self.v = self.v_reset
            return 1.0
        return 0.0
    
    def reset(self):
        """Reset to resting state"""
        self.v = self.v_rest


class SNNReservoir:
    """
    Spiking Neural Network Reservoir with Hypercube Topology
    
    The reservoir maintains context through recurrent dynamics
    and produces spike patterns that encode the input history.
    """
    
    def __init__(
        self,
        num_neurons: int = 2048,
        hypercube_dim: int = 11,
        input_scale: float = 30.0,
        spectral_radius: float = 1.2,
        use_hybrid: bool = False,
        seed: int = 42
    ):
        np.random.seed(seed)
        
        self.num_neurons = num_neurons
        self.hypercube_dim = hypercube_dim
        
        # Create topology mask
        if use_hybrid:
            self.mask = create_hybrid_hypercube_mask(hypercube_dim, shortcut_prob=0.05)
        else:
            self.mask = create_hypercube_mask(hypercube_dim)
        
        # Resize mask if num_neurons != 2^hypercube_dim
        expected_size = 2 ** hypercube_dim
        if num_neurons != expected_size:
            self.mask = self._resize_mask(self.mask, num_neurons)
        
        # Initialize weights
        W_res = np.random.randn(num_neurons, num_neurons) * 0.3
        W_res *= self.mask
        
        # Scale to target spectral radius
        eigenvalues = np.linalg.eigvals(W_res + 1e-6 * np.eye(num_neurons))
        current_radius = np.max(np.abs(eigenvalues))
        if current_radius > 0:
            W_res *= spectral_radius / current_radius
        
        self.W_res = W_res
        self.W_in = np.random.randn(num_neurons) * input_scale
        
        # Create neurons
        self.neurons = [LIFNeuron() for _ in range(num_neurons)]
        
        # State
        self.fire_rate = np.zeros(num_neurons)
        self.spike_history = []
        
    def _resize_mask(self, mask: np.ndarray, target_size: int) -> np.ndarray:
        """Resize mask to target size"""
        orig_size = mask.shape[0]
        new_mask = np.zeros((target_size, target_size))
        
        for i in range(target_size):
            for j in range(target_size):
                orig_i = (i * orig_size) // target_size
                orig_j = (j * orig_size) // target_size
                new_mask[i, j] = mask[orig_i, orig_j]
        
        return new_mask
    
    def step(self, input_val: float) -> np.ndarray:
        """
        Process one input and return spike pattern
        
        Args:
            input_val: Normalized input value (-1 to 1)
        
        Returns:
            Binary spike array
        """
        # Compute currents
        I_rec = self.W_res @ self.fire_rate
        I_ext = self.W_in * input_val
        I_total = I_rec + I_ext
        
        # Add small noise for chaotic dynamics
        I_total += np.random.randn(self.num_neurons) * 0.5
        
        # Update neurons
        spikes = np.zeros(self.num_neurons)
        for i, neuron in enumerate(self.neurons):
            spikes[i] = neuron.step(I_total[i] + 25.0)  # Bias for activity
        
        # Update state
        self.fire_rate = 0.8 * self.fire_rate + 0.2 * spikes
        self.spike_history.append(spikes.copy())
        
        return spikes
    
    def reset(self):
        """Reset reservoir state"""
        for neuron in self.neurons:
            neuron.reset()
        self.fire_rate = np.zeros(self.num_neurons)
        self.spike_history = []
    
    def get_state(self) -> np.ndarray:
        """Get current reservoir state (for readout)"""
        return self.fire_rate.copy()


class SNNLM:
    """
    SNN Language Model
    
    A language model using Spiking Neural Networks for
    ultra-low-power text generation.
    
    Architecture:
    1. Token embedding → Input encoding
    2. SNN Reservoir → Context representation
    3. Linear readout → Next token prediction
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        hidden_dim: int = 2048,
        hypercube_dim: int = 11,
        embedding_dim: int = 128,
        seed: int = 42
    ):
        np.random.seed(seed)
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        
        # Token embeddings
        self.embeddings = np.random.randn(vocab_size, embedding_dim) * 0.1
        
        # Input projection (embedding → scalar for reservoir)
        self.input_proj = np.random.randn(embedding_dim) * 0.1
        
        # SNN Reservoir
        self.reservoir = SNNReservoir(
            num_neurons=hidden_dim,
            hypercube_dim=hypercube_dim,
            seed=seed
        )
        
        # Readout layer
        self.W_out = np.random.randn(vocab_size, hidden_dim) * 0.01
        self.b_out = np.zeros(vocab_size)
        
        # Learning rate
        self.lr = 0.01
        
        # Vocabulary (will be set during training)
        self.token_to_idx = {}
        self.idx_to_token = {}
    
    def build_vocab(self, text: str, min_freq: int = 1):
        """Build vocabulary from text"""
        # Character-level for now
        chars = set(text)
        self.token_to_idx = {c: i for i, c in enumerate(sorted(chars))}
        self.idx_to_token = {i: c for c, i in self.token_to_idx.items()}
        
        # Add special tokens
        self.token_to_idx['<UNK>'] = len(self.token_to_idx)
        self.token_to_idx['<PAD>'] = len(self.token_to_idx)
        
        actual_vocab = len(self.token_to_idx)
        
        # Resize embeddings and output if needed
        if actual_vocab != self.vocab_size:
            self.vocab_size = actual_vocab
            self.embeddings = np.random.randn(actual_vocab, self.embedding_dim) * 0.1
            self.W_out = np.random.randn(actual_vocab, self.hidden_dim) * 0.01
            self.b_out = np.zeros(actual_vocab)
    
    def encode_token(self, token: str) -> int:
        """Convert token to index"""
        return self.token_to_idx.get(token, self.token_to_idx.get('<UNK>', 0))
    
    def decode_token(self, idx: int) -> str:
        """Convert index to token"""
        return self.idx_to_token.get(idx, '<UNK>')
    
    def forward(self, token_idx: int) -> np.ndarray:
        """
        Forward pass for one token
        
        Args:
            token_idx: Token index
        
        Returns:
            Log probabilities over vocabulary
        """
        # Get embedding
        emb = self.embeddings[token_idx]
        
        # Project to scalar input
        input_val = np.tanh(np.dot(emb, self.input_proj))
        
        # Run reservoir
        _ = self.reservoir.step(input_val)
        state = self.reservoir.get_state()
        
        # Readout
        logits = self.W_out @ state + self.b_out
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / (np.sum(exp_logits) + 1e-10)
        
        return probs
    
    def train_step(self, input_idx: int, target_idx: int) -> float:
        """
        Train on one input-target pair
        
        Args:
            input_idx: Input token index
            target_idx: Target token index
        
        Returns:
            Cross-entropy loss
        """
        probs = self.forward(input_idx)
        
        # Cross-entropy loss
        loss = -np.log(probs[target_idx] + 1e-10)
        
        # Gradient (simplified, output layer only)
        grad = probs.copy()
        grad[target_idx] -= 1
        
        # Update output weights
        state = self.reservoir.get_state()
        self.W_out -= self.lr * np.outer(grad, state)
        self.b_out -= self.lr * grad
        
        return loss
    
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 1.0
    ) -> str:
        """
        Generate text from prompt
        
        Args:
            prompt: Starting text
            max_length: Maximum generation length
            temperature: Sampling temperature (higher = more random)
        
        Returns:
            Generated text
        """
        self.reservoir.reset()
        
        # Process prompt
        generated = list(prompt)
        
        for char in prompt:
            idx = self.encode_token(char)
            _ = self.forward(idx)
        
        # Generate
        for _ in range(max_length):
            last_idx = self.encode_token(generated[-1])
            probs = self.forward(last_idx)
            
            # Temperature scaling
            if temperature != 1.0:
                log_probs = np.log(probs + 1e-10) / temperature
                probs = np.exp(log_probs - np.max(log_probs))
                probs /= np.sum(probs)
            
            # Sample
            next_idx = np.random.choice(len(probs), p=probs)
            next_char = self.decode_token(next_idx)
            
            generated.append(next_char)
        
        return ''.join(generated)
    
    def perplexity(self, text: str) -> float:
        """
        Calculate perplexity on text
        
        Args:
            text: Input text
        
        Returns:
            Perplexity score
        """
        self.reservoir.reset()
        
        total_loss = 0.0
        n_tokens = 0
        
        for i in range(len(text) - 1):
            input_idx = self.encode_token(text[i])
            target_idx = self.encode_token(text[i + 1])
            
            probs = self.forward(input_idx)
            loss = -np.log(probs[target_idx] + 1e-10)
            
            total_loss += loss
            n_tokens += 1
        
        avg_loss = total_loss / max(n_tokens, 1)
        return np.exp(avg_loss)
    
    def get_stats(self) -> dict:
        """Get model statistics"""
        return {
            'vocab_size': self.vocab_size,
            'hidden_dim': self.hidden_dim,
            'hypercube_dim': self.reservoir.hypercube_dim,
            'total_params': (
                self.embeddings.size +
                self.input_proj.size +
                self.W_out.size +
                self.b_out.size +
                self.reservoir.W_res.size +
                self.reservoir.W_in.size
            ),
            'reservoir_connections': int(self.reservoir.mask.sum())
        }


if __name__ == "__main__":
    # Demo
    print("SNN Language Model Demo")
    print("=" * 50)
    
    # Create model
    model = SNNLM(vocab_size=100, hidden_dim=512, hypercube_dim=9)
    
    # Build vocabulary
    sample_text = "The quick brown fox jumps over the lazy dog. " * 10
    model.build_vocab(sample_text)
    
    print(f"\nModel Statistics:")
    for key, val in model.get_stats().items():
        print(f"  {key}: {val:,}")
    
    # Train a few steps
    print("\nTraining...")
    for epoch in range(3):
        total_loss = 0
        for i in range(len(sample_text) - 1):
            loss = model.train_step(
                model.encode_token(sample_text[i]),
                model.encode_token(sample_text[i + 1])
            )
            total_loss += loss
        print(f"  Epoch {epoch + 1}: Loss = {total_loss / len(sample_text):.4f}")
    
    # Generate
    print("\nGeneration:")
    output = model.generate("The ", max_length=50, temperature=0.8)
    print(f"  '{output}'")
    
    # Perplexity
    ppl = model.perplexity("The quick brown fox")
    print(f"\nPerplexity: {ppl:.2f}")
