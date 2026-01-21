"""
Quick Start Demo for SNN-LLM
============================

This script demonstrates:
1. Creating an SNN Language Model
2. Training on sample text
3. Generating text
4. Measuring perplexity

Run: python demo.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.snn_lm import SNNLM
import time


def get_sample_text():
    """Get sample training text"""
    return """
    The brain is the most complex organ in the human body. It controls thought, 
    memory, emotion, touch, motor skills, vision, breathing, temperature, hunger 
    and every process that regulates our body. The brain consists of about 86 
    billion neurons. These neurons communicate through synapses, forming complex 
    networks. The structure of the brain includes 11-dimensional cliques, as 
    discovered by the Blue Brain Project. This high-dimensional topology enables 
    efficient information processing with minimal energy consumption. Spiking 
    Neural Networks attempt to mimic this biological efficiency. Unlike traditional 
    artificial neural networks, SNNs use temporal coding to represent information.
    This allows for massive information capacity increases while reducing energy
    consumption by up to 100 times compared to von Neumann architectures.
    """ * 5


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                    🧠 SNN-LLM Demo 🧠                         ║
    ║         Ultra-Low-Power Language Model with SNNs              ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Configuration
    hidden_dim = 512  # 9D hypercube
    hypercube_dim = 9
    epochs = 5
    
    print(f"Configuration:")
    print(f"  Hidden Dimension: {hidden_dim}")
    print(f"  Hypercube Dimension: {hypercube_dim}D")
    print(f"  Epochs: {epochs}")
    print("-" * 60)
    
    # Get training data
    text = get_sample_text()
    print(f"\nTraining Data: {len(text)} characters")
    
    # Create model
    print("\nCreating SNN-LM model...")
    t0 = time.time()
    model = SNNLM(
        vocab_size=100,
        hidden_dim=hidden_dim,
        hypercube_dim=hypercube_dim
    )
    model.build_vocab(text)
    print(f"  Created in {time.time() - t0:.2f}s")
    
    # Print model stats
    stats = model.get_stats()
    print(f"\nModel Statistics:")
    print(f"  Vocabulary Size: {stats['vocab_size']}")
    print(f"  Hidden Dimension: {stats['hidden_dim']}")
    print(f"  Total Parameters: {stats['total_params']:,}")
    print(f"  Reservoir Connections: {stats['reservoir_connections']:,}")
    
    # Training
    print(f"\nTraining for {epochs} epochs...")
    print("-" * 60)
    
    for epoch in range(epochs):
        model.reservoir.reset()
        total_loss = 0
        n_chars = len(text) - 1
        
        t0 = time.time()
        for i in range(n_chars):
            loss = model.train_step(
                model.encode_token(text[i]),
                model.encode_token(text[i + 1])
            )
            total_loss += loss
        
        avg_loss = total_loss / n_chars
        ppl = model.perplexity(text[:200])
        elapsed = time.time() - t0
        
        print(f"  Epoch {epoch + 1}/{epochs}: Loss={avg_loss:.4f}, PPL={ppl:.1f}, Time={elapsed:.1f}s")
    
    # Generation
    print("\n" + "=" * 60)
    print("TEXT GENERATION")
    print("=" * 60)
    
    prompts = [
        "The brain ",
        "Neural networks ",
        "Spiking ",
    ]
    
    for prompt in prompts:
        model.reservoir.reset()
        output = model.generate(prompt, max_length=100, temperature=0.7)
        print(f"\nPrompt: '{prompt}'")
        print(f"Output: '{output[:150]}...'")
    
    # Final perplexity
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    
    test_text = "The brain consists of neurons that communicate through synapses."
    final_ppl = model.perplexity(test_text)
    print(f"\nTest Perplexity: {final_ppl:.2f}")
    
    # Energy estimate (theoretical)
    ops_per_char = stats['reservoir_connections'] * 3  # Rough estimate
    pj_per_op = 0.1  # Neuromorphic estimate
    energy_per_char = ops_per_char * pj_per_op
    
    print(f"\nEnergy Estimate (theoretical):")
    print(f"  Operations per character: {ops_per_char:,}")
    print(f"  Energy per character: {energy_per_char:.1f} pJ")
    print(f"  Energy per 1000 chars: {energy_per_char * 1000 / 1e6:.3f} µJ")
    
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                      Demo Complete! 🎉                        ║
    ║                                                               ║
    ║  This is just the beginning! Future improvements:             ║
    ║  - Better tokenization (BPE, WordPiece)                       ║
    ║  - Larger training data                                       ║
    ║  - Backpropagation through time                               ║
    ║  - Hardware deployment (Loihi, SpiNNaker)                     ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
