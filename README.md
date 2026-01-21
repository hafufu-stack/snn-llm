# 🧠 SNN-LLM: Spiking Neural Network Language Model

**Ultra-Low-Power Local Language Model Using Brain-Inspired Computing**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

---

## 🎯 Vision

> **"A language model that runs on your device without cloud, using 100x less energy than traditional LLMs"**

SNN-LLM is an experimental project to create a practical, deployable language model using Spiking Neural Networks (SNNs). Unlike massive transformer-based LLMs that require cloud infrastructure, SNN-LLM aims to:

- 🔋 **100x Energy Efficiency**: Brain-inspired spike-based computation
- 🏠 **100% Local**: No cloud, no API, no privacy concerns  
- 📱 **Edge Deployable**: Run on Raspberry Pi, smartphones, IoT devices
- 🧠 **11D Hypercube Topology**: Based on Blue Brain Project discoveries

---

## 🧬 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      SNN-LLM Architecture                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Input Text  ──►  Spike Encoder  ──►  SNN Reservoir        │
│                         │                    │               │
│                         ▼                    ▼               │
│                   [Temporal Code]     [11D Hypercube]        │
│                         │                    │               │
│                         └────────┬───────────┘               │
│                                  ▼                           │
│                          Readout Layer  ──►  Output Token    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

1. **Spike Encoder**: Converts text tokens to spike trains using temporal coding
2. **SNN Reservoir**: 11D Hypercube topology (2048 neurons) with chaotic dynamics
3. **Readout Layer**: Linear decoder for next-token prediction

---

## 📊 Target Specifications

| Metric | Traditional LLM | SNN-LLM Target |
|--------|-----------------|----------------|
| Parameters | 7B+ | 10M-100M |
| Memory | 14GB+ | 100MB-1GB |
| Power | 200W+ (GPU) | 1-10W (CPU) |
| Latency | Cloud-dependent | Real-time local |
| Privacy | Data sent to cloud | 100% on-device |

---

## 🚀 Getting Started

### Installation

```bash
git clone https://github.com/hafufu-stack/snn-llm.git
cd snn-llm
pip install -r requirements.txt
```

### Quick Demo

```python
from models.snn_lm import SNNLM

# Create model
model = SNNLM(vocab_size=10000, hidden_dim=2048, hypercube_dim=11)

# Generate text
prompt = "The future of AI is"
output = model.generate(prompt, max_length=50)
print(output)
```

---

## 📁 Project Structure

```
snn-llm/
├── README.md           # This file
├── requirements.txt    # Python dependencies
├── models/             # SNN-based language models
│   ├── snn_lm.py       # Core SNN-LM implementation
│   ├── hypercube.py    # 11D Hypercube topology
│   └── spike_encoder.py # Text to spike conversion
├── experiments/        # Training and evaluation scripts
├── benchmarks/         # Comparison with other LLMs
├── data/               # Training data
├── results/            # Experiment results
└── docs/               # Documentation
```

---

## 🔬 Research Background

This project builds on our previous research:

1. **Brain vs Neumann** (2026): Demonstrated 93 trillion times more information capacity with temporal coding
2. **11D Brain Structure** (2026): Verified that 11D hypercube achieves 8x faster information propagation
3. **SNN-Comprypto** (2026): Proved cryptographic-grade randomness with 65% fewer connections

### Key Hypothesis

> The brain's 11-dimensional structure is optimized for language-like sequential processing, making SNNs ideal for ultra-efficient language models.

---

## 📈 Roadmap

### Phase 1: Proof of Concept (Current)
- [x] Project setup
- [ ] Basic SNN-LM implementation
- [ ] Character-level text generation
- [ ] Perplexity evaluation

### Phase 2: Enhancement
- [ ] Word-level tokenization
- [ ] 11D Hypercube integration
- [ ] Energy consumption measurement
- [ ] Comparison with TinyLLM, MobileLLM

### Phase 3: Deployment
- [ ] Raspberry Pi demo
- [ ] ONNX export
- [ ] Mobile app prototype
- [ ] Neuromorphic hardware (Intel Loihi)

---

## 📚 Related Publications

- [Brain vs Neumann (Zenodo)](https://zenodo.org/records/18319152)
- [SNN Language Model](https://github.com/hafufu-stack/snn-language-model)
- [SNN Comprypto](https://github.com/hafufu-stack/temporal-coding-simulation)

---

## 🤝 Contributing

This is an experimental research project. Contributions, ideas, and discussions are welcome!

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

## 👤 Author

**Hiroto Funasaki (roll)**
- GitHub: [@hafufu-stack](https://github.com/hafufu-stack)
- ORCID: [0009-0004-2517-0177](https://orcid.org/0009-0004-2517-0177)
- Zenn: [@cell_activation](https://zenn.dev/cell_activation)

---

> 🧠 *"The brain is the most energy-efficient computer in the universe. Let's learn from it."*
