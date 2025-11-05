# Generative Pre-Trained Transformer (GPT)

This repository contains an educational and experimental implementation of a **Generative Pre-Trained Transformer (GPT)** model from scratch. It demonstrates the core ideas behind modern Transformer-based architectures: character-level tokenization, attention, scaled dot-product mechanics, and training for autoregressive text generation.

---

## 📘 Overview

The notebook `generative-pre-trained-transformer.ipynb` walks through a minimal yet complete GPT pipeline:

1. **Character-Level Tokenization**

   - Build a vocabulary from unique characters.
   - Implement `encode(s: str) -> list[int]` and `decode(ids: list[int]) -> str`.

2. **Transformer Components**

   - Multi-Head Self-Attention
   - Scaled Dot-Product Attention
   - Position embeddings
   - Feed-Forward blocks
   - Residual connections & layer norm

3. **Model Shapes & Terminology**

   - **Batch size** — number of sequences processed together.
   - **Sequence length / Block size** — fixed number of tokens per sequence.
   - **Feature dimension** — embedding size per token.

4. **Attention Head Walkthrough**

   - Linear projections to Queries (Q), Keys (K), Values (V): shape `(B, T, D_h)`.
   - Transpose K to align for `Q @ Kᵀ` → raw attention scores `(B, T, T)`.
   - Scale by `√d_k`, apply masks (causal/padding), softmax, dropout.
   - Multiply by V to aggregate context → `(B, T, D_h)`.

5. **Why Scale by √dₖ**
   - Prevent softmax saturation from large dot products in high dimensions.
   - Keep gradients stable and training efficient.

---

## 🧠 Concepts Highlighted

- Difference (and equivalence here) between **sequence length** and **block size**
- Role of **causal masks** for autoregressive generation
- Effect of **scaled dot-product** on gradient stability
- Interpreting tensor shapes like `(8, 32, 64)` in attention
- End-to-end path from tokens → embeddings → attention → logits

---

## ⚙️ Requirements

Install dependencies:

```bash
pip install torch numpy matplotlib jupyter
```

(Optional visualization/experiments)

```bash
pip install seaborn
```

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/generative-pre-trained-transformer.git
cd generative-pre-trained-transformer
```

### 2. Launch the Notebook

```bash
jupyter notebook generative-pre-trained-transformer.ipynb
```

### 3. Run the Notebook

Execute each cell in order to:

Initialize the tokenizer

Construct the attention mechanism

Train the minimal GPT model

Generate sample text outputs

Once completed, you’ll have a working understanding of how a Transformer-based language model is built and trained from scratch.

## 🧑‍💻 Author

Nitin Sai Varma Indukuri
