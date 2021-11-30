
### Open-Domain Question Answering

Some directions and topics:
- **Low-storage cost retriever**: passage embedding compression
- **Various data (question) settings**: entity-centric questions, questions with contradicting contexts, questions requires multiple evidence passages
- **Improved formulation for retriever**: phrase retrieval, document-passage interactive model, retriever-reranker-reader formulation
- **Improved training of DPR**: negative example selection, semi-supervised, weakly-supervised training
- **Novel information retrieval methods**
- **opdq over structured data**: KG-based data, Table-based data etc.


---

- [Latent Retrieval for Weakly Supervised Open Domain Question Answering](https://arxiv.org/pdf/1906.00300.pdf), `emnlp2019` `inverse close task` `pretraining`
- [Simple Entity-Centric Questions Challenge Dense Retrievers](https://arxiv.org/pdf/2109.08535.pdf), Danqi Chen et al. `emnlp2021`
- [Phrase-retrieval Learns Passage Retrieval, Too](https://arxiv.org/abs/2109.08133), Danqi Chen et al. `emnlp2021`
- [Unsupervised Open Domain Question Answering](https://arxiv.org/pdf/2108.13817.pdf), Hai Zhao's group.
- [Adaptive Information Seeking for Open-Domain Question Answering](https://arxiv.org/pdf/2109.06747.pdf), Xueqi Cheng et al. `rl`
- [What’s in a Name? Answer Equivalence For Open-Domain Question Answering](https://arxiv.org/pdf/2109.05289.pdf), Jordan Boyd-Graber et al. `answer expansion`
- [Entity-Based Knowledge Conflicts in Question Answering](https://arxiv.org/pdf/2109.05052.pdf), Sameer Singh et al. `analysis`
- [SPARTA: Efficient Open-Domain Question Answering via Sparse Transformer Matching Retrieval](https://aclanthology.org/2021.naacl-main.47.pdf), `naacl2021`
- [Mr. TYDI: A Multi-lingual Benchmark for Dense Retrieval](https://arxiv.org/pdf/2108.08787.pdf), Jimmy Lin's group, `dense retrieval` `evaluation`
- [Question and Answer Test-Train Overlap in Open-Domain Question Answering Datasets](https://aclanthology.org/2021.eacl-main.86.pdf), `eacl`
- [Relation-Guided Pre-Training for Open-Domain Question Answering](https://arxiv.org/pdf/2109.10346.pdf), Kai-Wei Chang's group.
- [Single-dataset Experts for Multi-dataset Question Answering](https://arxiv.org/pdf/2109.13880.pdf), `emnlp2021`
- [Adversarial Retriever-Reranker for Dense Text Retrieval](https://arxiv.org/pdf/2110.03611.pdf), iclr2022  submitted.
- [Distantly-Supervised Evidence Retrieval Enables Question Answering without Evidence Annotation](https://arxiv.org/pdf/2110.04889.pdf), `Hal Daume III`, Oct. 10 2021. [code](https://github.com/henryzhao5852/DistDR). 
- [CONTRAQA: QUESTION ANSWERING UNDER CONTRADICTING CONTEXTS](https://arxiv.org/pdf/2110.07803.pdf), `dataset creation`, Oct. 2021.
- [Open Domain Question Answering over Virtual Documents: A Unified Approach for Data and Text](https://arxiv.org/pdf/2110.08417.pdf), Oct. 16 2021.
- [Simple and Effective Unsupervised Redundancy Elimination to Compress Denses for Passage Retrieval](https://cs.uwaterloo.ca/~jimmylin/publications/Ma_etal_EMNLP2021.pdf), `emnlp2021`, Jimmy Lin's group.
- [Domain-matched Pre-training Tasks for Dense Retrieval](https://arxiv.org/pdf/2107.13602.pdf), Jul. 28 2021. [github, dpr-scale](https://github.com/facebookresearch/dpr-scale).
- [Answering Open-Domain Questions of Varying Reasoning Steps from Text](https://arxiv.org/pdf/2010.12527.pdf), Oct. 29 2021, `emnlp2021`, [code & benchmark](https://beerqa.github.io/)
- [End-to-End Training of Multi-Document Reader and Retriever for Open-Domain Question Answering](https://arxiv.org/pdf/2106.05346.pdf), Jun. 9 2021.
- [One Question Answering Model for Many Languages with Cross-lingual Dense Passage Retrieval](https://arxiv.org/pdf/2107.11976.pdf), Oct. 28 2021. [code](https://github.com/AkariAsai/CORA).
- [Dense Hierarchical Retrieval for Open-Domain Question Answering](https://arxiv.org/pdf/2110.15439.pdf), Oct. 28 2021.
  - information intactness of passage formulation of DPR
- [Joint Passage Ranking for Diverse Multi-Answer Retrieval](https://arxiv.org/pdf/2104.08445.pdf), Sep. 21 2021.

#### Improved DPR training

**Negative sample selection**

- [Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval](https://arxiv.org/pdf/2007.00808.pdf), Oct. 20 2021. `iclr2020`
  - problems of in-batch negatives: uninformative negatives --> diminishing gradient norms, large stochastic gradient variances and slow learning convergence.
- [RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/pdf/2010.08191.pdf), May 12 2021. `emnlp2021`
  - interactive q-p encoder to supervise (denoise) negative examples for DPR training
  - semi-supervised learning supervised by interactive q-p encoder

**Semi-supervised learning**

- RocketQA paper
- [Relevance-guided Supervision for OpenQA with ColBERT](https://arxiv.org/pdf/2007.00814.pdf), Aug. 2 2021. `tacl2021`



#### Retriever

- [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://dl.acm.org/doi/pdf/10.1145/3397271.3401075?casa_token=Er5SHkW6pggAAAAA:i0UwW9LxTMZmoF5k-HM6leeqIezjih8X9KBXb0ZXrt5PGZ05d-oX2Lur_TC5nkVEHzG_Pb1TV4Wfuvo), `sigir2020`.
  - "crucially, ColBERT's pruning-friendly interaction mechanism enables leveraging vector-similarity indexes for end-to-end retrieval directly from millions of documents."
- [Condenser: a Pre-training Architecture for Dense Retrieval](https://aclanthology.org/2021.emnlp-main.75.pdf), `emnlp2021`.

#### Information retrieval

- [BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://arxiv.org/abs/2104.08663), Oct. 21 2021. `nips2021`

### Knowledge-Base Question Answering, KB Completion

- [RNG-KBQA: Generation Augmented Iterative Ranking for Knowledge Base Question Answering](https://arxiv.org/pdf/2109.08678.pdf), Caiming Xiong et al. `arXiv`
- [Benchmarking the Combinatorial Generalizability of Complex Query Answering on Knowledge Graphs](https://arxiv.org/pdf/2109.08925.pdf), Yangqiu Song et al. `nips2021`
- [SMORE: Knowledge Graph Completion and Multi-hop Reasoning in Massive Knowledge Graphs](https://arxiv.org/pdf/2110.14890.pdf), Oct. 28 2021, `nips2021`. [code](https://github.com/google-research/smore)
- [Vector-valued Distance and Gyrocalculus on the Space of Symmetric Positive Definite Matrices](https://arxiv.org/pdf/2110.13475.pdf), Oct. 26 2021. `nips2021`
- [Metadata Shaping: Natural Language Annotations for the Tail](https://arxiv.org/pdf/2110.08430.pdf), Oct. 16 2021. `handling long-tail` `knowledge injection`
- [SQALER: Scaling Question Answering by Decoupling Multi-Hop and Logical Reasoning](https://proceedings.neurips.cc/paper/2021/file/68bd22864919297c8c8a8c32378e89b4-Paper.pdf), `nips2021`.

### Machine Reading Comprehension

- [Numerical reasoning in machine reading comprehension tasks: are we there yet?](https://arxiv.org/abs/2109.08207), `emnlp2021`
- [Improving Unsupervised Question Answering via Summarization-Informed Question Generation](https://arxiv.org/pdf/2109.07954.pdf), `emnlp2021` `qg`
- [Using Knowledge Distillation and Active Learning](https://arxiv.org/pdf/2109.12662.pdf), `arXiv 2021 Sep`
- [GNN is a Counter: Revisiting GNN for Question Answering](https://arxiv.org/pdf/2110.03192.pdf), iclr2022  submitted.
- [A Dataset for Answering Time-Sensitive Questions](https://arxiv.org/pdf/2108.06314.pdf), `nips2021`.
- [Improving Question Answering Model Robustness with Synthetic Adversarial Data Generation](https://arxiv.org/pdf/2104.08678.pdf), Sep. 16 2021, `emnlp2021` `adversarial` `robustness`
- [Do We Know What We Don’t Know? Studying Unanswerable Questions beyond SQuAD 2.0](https://aclanthology.org/2021.findings-emnlp.385/), `emnlp2021`

### Reasoning, Commonsense and Knowledge

- [BeliefBank: Adding Memory to a Pre-Trained Language Model for a Systematic Notion of Belief](https://arxiv.org/pdf/2109.14723.pdf), Sep. 2021. `compositionality`
- [Conversational Multi-Hop Reasoning with Neural Commonsense Knowledge and Symbolic Logic Rules](https://arxiv.org/pdf/2109.08544.pdf), `emnlp2021`
- [How Much Coffee Was Consumed During EMNLP 2019? Fermi Problems: A New Reasoning Challenge for AI](https://arxiv.org/pdf/2110.14207.pdf), Oct. 27, `emnlp2021`. [data](https://allenai.org/data/fermi)
- [Symbolic Knowledge Distillation: from General Language Models to Commonsense Models](https://arxiv.org/pdf/2110.07178.pdf), Oct. 14 2021.

### Visual Question Answering

- [Abductive Visual Question Answering for Label Efficient Learning](https://karans.github.io/assets/pdf/Papers/AB-VQA.pdf), Le Song's group. `abduction`


### Long-anwer Question Answering

- [TruthfulQA: Measuring How Models Mimic Human Falsehoods](https://owainevans.github.io/pdfs/truthfulQA_lin_evans.pdf), OpenAI, Oxford Univ. `benchmark`


### Analysis

- [The Effect of Natural Distribution Shift on Question Answering Models](http://proceedings.mlr.press/v119/miller20a/miller20a.pdf), `icml2020`
- [Evaluation Paradigms in Question Answering](https://research.fb.com/wp-content/uploads/2021/09/Evaluation-Paradigms-in-Question-Answering.pdf), Sep. 2021. Facebook.

### QA and social bias

- [BBQ: A Hand-Built Bias Benchmark for Question Answering](https://arxiv.org/pdf/2110.08193.pdf), Oct. 15 2021. Sam Bowman's group.
