
## Crafting LLMs and Beyond

- LLMs is a miracle to me. When trained using the next token prediction (ntp) loss, all kinds of human-like language understanding ability emerges via their generative behavior. What a human-made phenomenon instead of all the miracle phenomana of our mother nature!
- This document is particularlly dedicated to the art of crafting a LLM in a low-resource setting. Here the word low-resource indicates the relative scarcity of both compute and data resources, which is commom in many companies without thorough AI background.
- Under the ambition of building an expert LLM in certain domain or business **B**, I'd like to survey some topics that I most care about:
  - **scaling laws**
  - **continual pre-training** using ntp loss or beyond
  - pre-training and instruction-tuning **data sources, _curation_, _elicitation_, _synthesis_, _mixing_, _selection_**
  - **model inheritance, merging, pruning, distillation**
  - systematic **training process control, monitoring, model selection and evaluation**

> _Note that: the **B** above might be industry, e-commerce, finance, health and medicine, logistic etc._

### Scaling laws

> Scaling law is an outstanding artifact as reserchers continue to scale the compute (model and data size).

- _[TODO: add paper links here.]_
- [Predicting the performance of foundation models via agreement-on-the-line](https://arxiv.org/pdf/2404.01542.pdf), Apr. 2 2024.

### Data sources

- _[TODO: refer to the Dolma paper]_

### Data curation and selection

- [A Pretrainer’s Guide to Training Data: Measuring the Effects of Data Age, Domain Coverage, Quality, & Toxicity](https://arxiv.org/pdf/2305.13169.pdf), Nov. 13 2023.
- [D4: Improving LLM Pretraining via Document De-Duplication and Diversification](https://proceedings.neurips.cc/paper_files/paper/2023/file/a8f8cbd7f7a5fb2c837e578c75e5b615-Paper-Datasets_and_Benchmarks.pdf), NeurIPS 2023.
- [Domain Specialization as the Key to Make Large Language Models Disruptive: A Comprehensive Survey](https://arxiv.org/pdf/2305.18703), Mar. 29 2024.
- [Scaling Laws for Data Filtering — Data Curation cannot be Compute Agnostic](https://arxiv.org/pdf/2404.07177.pdf), Apr. 10 2024.
- [Text Quaility-Based Pruning for Efficient Training of Language Models](https://arxiv.org/pdf/2405.01582), May 10 2024.
- [LMD3: Language Model Data Density Dependence](https://arxiv.org/pdf/2405.06331), May 10 2024.
- [DataComp-LM: In search of the next generation of training sets for language models](https://arxiv.org/pdf/2406.11794), Jun. 17 2024. `data selection`.
- [CoLoR-Filter: Conditional Loss Reduction Filtering for Targeted Language Model Pre-training](https://arxiv.org/pdf/2406.10670), Jun. 15 2024. `data selection`.
- [Data Shapley in One Training Run](https://arxiv.org/pdf/2406.11011), Jun. 19 2024. `data selection`.
- [Rephrasing the Web: A Recipe for Compute and Data-Efficient Language Modeling](https://arxiv.org/pdf/2401.16380), Jan. 29 2024. `data curation`.
- [Automated Data Curation for Robust Language Model Fine-Tuning](https://arxiv.org/pdf/2403.12776), Mar. 19 2024.
- [AutoPureData: Automated Filtering of Web Data for LLM Fine-tuning](https://arxiv.org/abs/2406.19271), Jun. 27 2024. `data filtering`. [code](https://github.com/Pro-GenAI/AutoPureData).
- [RegMix: Data Mixture as Regression for Language Model Pre-training](https://arxiv.org/abs/2407.01492), Jul. 1 2024.

### Alignment

- [Digital Forgetting in Large Language Models: A Survey of Unlearning Methods](https://arxiv.org/pdf/2404.02062.pdf), Apr. 2 2024.
- [Generative AI Security: Challenges and Countermeasures](https://arxiv.org/pdf/2402.12617.pdf), Feb. 20 2024.
- [What’s in Your “Safe” Data?: Identifying Benign Data that Breaks Safety](https://arxiv.org/pdf/2404.01099.pdf), Apr. 1 2024.
- [Improving Dialogue Agents by Decomposing One Global Explicit Annotation with Local Implicit Multimodal Feedback](https://arxiv.org/pdf/2403.11330.pdf), Mar. 17 2024. `dialoguing`.
- [Towards Practical Tool Usage for Continually Learning LLMs](https://arxiv.org/pdf/2404.09339.pdf), Apr. 14 2024. `agent`.
- [GENAUDIT: Fixing Factual Errors in Language Model Outputs with Evidence](https://arxiv.org/pdf/2402.12566.pdf), Mar. 16 2024. `factuality`.
- [Data-driven Discovery with Large Generative Models](https://arxiv.org/pdf/2402.13610.pdf), Feb. 21 2024. `scientific discovery`.

### Understanding llms, transformers and beyond

- [Localizing Paragraph Memorization in Language Models](https://arxiv.org/pdf/2403.19851.pdf), Mar. 28 2024.

### Multimodality

- [Tables as Images? Exploring the Strengths and Limitations of LLMs on Multimodal Representations of Tabular Data](https://arxiv.org/pdf/2402.12424.pdf), Feb. 23 2024.

### Architectural design

- [Repetition Improves Language Model Embeddings](https://arxiv.org/pdf/2402.15449.pdf), Feb. 23 2024.


