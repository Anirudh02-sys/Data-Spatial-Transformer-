# Data Spatial Transformer

This repository contains two implementations of the Transformer encoder logic:
1. **JacLang** (`transformer.jac`)
2. **Python (using PyTorch)** (`transformer.py`)

The goal is to explore JacLang's Data Spatial capabilities by applying the Transformer architecture within its framework.

## Introduction

### What is JacLang?
Data Spatial Transformer
This repository showcases an exploration of implementing the Transformer encoder logic using JacLang, alongside a reference implementation in Python (PyTorch). The project leverages JacLang’s Data Spatial programming model, a novel approach designed to enhance data structure manipulation, traversal, and interaction.

Introduction
Why JacLang for Transformers?
JacLang is built to be an extension of Python, introducing a higher abstraction level known as Data Spatial Programming. This paradigm is focused on modeling data as nodes, edges, and walkers, facilitating graph-based computation. It allows us to create a spatial representation of neural network architectures like Transformers, making JacLang an ideal candidate for implementing models that rely on interconnected data flows, such as attention mechanisms.

### What is a Transformer?
The **Transformer** architecture is a state-of-the-art deep learning model, particularly known for its effectiveness in handling sequence data. It uses mechanisms like **multi-head attention**, **positional encoding**, and **feed-forward networks** to process input sequences efficiently. This model has set benchmarks in NLP, powering models like GPT, BERT, etc.

## Project Overview

This project aims to **recreate the Transformer encoder logic** in JacLang by modeling tokens as **nodes**, using walkers to apply transformations like **attention**, and exploring the Data Spatial paradigm through graph-like traversal. The Python implementation serves as a baseline, highlighting how similar logic can be expressed differently using JacLang's syntax.

## File Structure
- `transformer.jac`: JacLang implementation of the Transformer encoder.
- `transformer.py`: Python (PyTorch) implementation for comparison and validation.

## How to Run

### 1. Python Implementation
Ensure you have the necessary dependencies installed:
```bash
pip install torch
```

To run the Python version:
```bash
python transformer.py
```

### 2. JacLang Implementation
Ensure you have **JacLang** installed and properly set up. If not, you can find the installation guide [here](https://www.jac-lang.org/docs/installation).

To run the JacLang version:
```bash
jac run transformer.jac
```

## Implementation Details

### Key Components

1. **Token Nodes**:
   - Tokens are represented as `TokenNode`s in JacLang, with attributes like `word`, `embedding`, and `pos_encoding`.

2. **Embeddings**:
   - Each word is converted to a 16-dimensional embedding vector using basic character-based transformations.
   - Positional encodings are calculated using the sine-cosine formula to maintain sequence information.

3. **Multi-Head Attention**:
   - Implemented using PyTorch tensors in JacLang. It processes token embeddings by computing attention scores, applying them to the input vectors.

4. **Graph-Based Traversal**:
   - Walkers traverse connected nodes (tokens) in JacLang, applying transformations to simulate attention and positional updates.

### Sample Output
Here’s a sample output when running the code:
```
Embedding: tensor([0.4062, 0.3945, 0.4219, ...])
Word: hello
Embedding with attention: tensor([...])
Positional Encoding: tensor([...])
```

### Next Steps
- Implement **feed-forward networks**, **residual connections**, and **layer normalization** in JacLang.
- Explore multi-layer stacking for the encoder to fully replicate a Transformer.
- Use JacLang's **Data Spatial** capabilities to model inter-token relationships through edges and complex traversals.

## Learnings and Observations

This project demonstrates the flexibility and power of JacLang’s spatial architecture in handling NLP tasks, specifically sequence modeling. While JacLang has a different programming paradigm compared to Python, it aligns well with graph-based AI models like Transformers, making it a unique tool for exploring neural architectures spatially.

## Contributing
Feel free to fork this repository, open issues, or submit pull requests to improve the JacLang implementation or add new features!

## Acknowledgements
- **JacLang** documentation and tutorials for guiding the implementation.
- **PyTorch** for the reference implementation of the Transformer.
