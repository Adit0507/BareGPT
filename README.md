## BareGPT

<b>BareGPT</b> is my attempt to build a  GPT-style large language model entirely from scratch using <b>Pytorch</b>. The goal is to understand the inner workings of LLMs by manually implementing every core component, from tokenization and attention mechanisms to full model architecture without relying on high level abstractions.


### Overview

Large language models (LLMs) like GPT have reshaped natural language processing by leveraging transformers and self-attention. BareGPT aims to reconstruct this architecture from the ground up, focusing on clarity, correctness, and learning.

### Motivation
BareGPT is a hands-on initiative to break down the complexity of modern LLMs and gain practical insight into how generative models work at a fundamental level. It’s an exploration in learning by building.

### Concepts Explored
- Tokenization and Embeddings
- Self-attention and multi-head attention
- Implemented scaled dot product attention for self-attention mechanism.
- Built Multi-head attention modules
- Added Causal attention mask to prevent tokens from attending to future positions.
