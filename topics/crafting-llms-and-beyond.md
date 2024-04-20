
# Crafting LLMs and Beyond

- LLMs is a miracle to me, when trained using the next token prediction (ntp) loss, all kinds of human-like language understanding ability emerges via their generative behavior. What a human-made phenomenon instead of all the miracle phenomana of our mother nature. This document is particularlly dedicated to how to craft a LLM in a low-resource setting. Here the word low-resource indicates the relative scarcity of both compute and data resources, which is commom in many companies without thorough AI technique background. So the topics I am mainly interested in are:
  - scaling laws
  - continual pre-training using ntp loss or beyond
  - pre-training and instruction-tuning data _curation_, _elicitation_, _synthesis_, _mixing_, _selection_ under the goal of building an expert LLM in certain domain or industry $\mathcal{I}$
  - model inheritance, merging, pruning, distillation
  - systematic training process control, monitoring, model selection and evaluation

## Data curation

- [A Pretrainer’s Guide to Training Data: Measuring the Effects of Data Age, Domain Coverage, Quality, & Toxicity](https://arxiv.org/pdf/2305.13169.pdf), Nov. 13 2023.
- [Scaling Laws for Data Filtering — Data Curation cannot be Compute Agnostic](https://arxiv.org/pdf/2404.07177.pdf), Apr. 10 2024.

## Scaling laws and performance prediction

- [Predicting the performance of foundation models via agreement-on-the-line](https://arxiv.org/pdf/2404.01542.pdf), Apr. 2 2024.

## Unlearning

- [Digital Forgetting in Large Language Models: A Survey of Unlearning Methods](https://arxiv.org/pdf/2404.02062.pdf), Apr. 2 2024.

## Architectural design

- [Repetition Improves Language Model Embeddings](https://arxiv.org/pdf/2402.15449.pdf), Feb. 23 2024.

## Alignment

- [Generative AI Security: Challenges and Countermeasures](https://arxiv.org/pdf/2402.12617.pdf), Feb. 20 2024.
- [What’s in Your “Safe” Data?: Identifying Benign Data that Breaks Safety](https://arxiv.org/pdf/2404.01099.pdf), Apr. 1 2024.
- [Improving Dialogue Agents by Decomposing One Global Explicit Annotation with Local Implicit Multimodal Feedback](https://arxiv.org/pdf/2403.11330.pdf), Mar. 17 2024. `dialoguing`.
- [Towards Practical Tool Usage for Continually Learning LLMs](https://arxiv.org/pdf/2404.09339.pdf), Apr. 14 2024. `agent`.
- [GENAUDIT: Fixing Factual Errors in Language Model Outputs with Evidence](https://arxiv.org/pdf/2402.12566.pdf), Mar. 16 2024. `factuality`.
- [Data-driven Discovery with Large Generative Models](https://arxiv.org/pdf/2402.13610.pdf), Feb. 21 2024. `scientific discovery`.

## Understanding llms, transformers and beyond

- [Localizing Paragraph Memorization in Language Models](https://arxiv.org/pdf/2403.19851.pdf), Mar. 28 2024.

## Multimodality

- [Tables as Images? Exploring the Strengths and Limitations of LLMs on Multimodal Representations of Tabular Data](https://arxiv.org/pdf/2402.12424.pdf), Feb. 23 2024.
