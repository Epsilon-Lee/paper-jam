
### Open-Domain Question Answering

Some directions and topics:
- **Low-storage cost retriever**: passage embedding compression
- **Various data (question) settings**: entity-centric questions, questions with contradicting contexts, questions requires multiple evidence passages
- **Improved formulation for retriever**: phrase retrieval, document-passage interactive model, retriever-reranker-reader formulation
- **Improved (unsupervised/pre or supervised-) training of DPR**: negative example selection, semi-supervised, weakly-supervised training
- **Novel information retrieval methods**
- **odqa over structured data**: KG-based data, Table-based data etc.

---

- :white_heart: [Latent Retrieval for Weakly Supervised Open Domain Question Answering](https://arxiv.org/pdf/1906.00300.pdf), `emnlp2019` `inverse close task` `pretraining`
- [Don’t Read Too Much Into It: Adaptive Computation for Open-Domain Question Answering](https://arxiv.org/pdf/2011.05435.pdf), Nov. 10 2021.
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
- [ContraQA: Question Answering under Contradicting Contexts](https://arxiv.org/pdf/2110.07803.pdf), `dataset creation`, Oct. 2021.
- [Open Domain Question Answering over Virtual Documents: A Unified Approach for Data and Text](https://arxiv.org/pdf/2110.08417.pdf), Oct. 16 2021.
- [Simple and Effective Unsupervised Redundancy Elimination to Compress Denses for Passage Retrieval](https://cs.uwaterloo.ca/~jimmylin/publications/Ma_etal_EMNLP2021.pdf), `emnlp2021`, Jimmy Lin's group.
- :white_heart: [Domain-matched Pre-training Tasks for Dense Retrieval](https://arxiv.org/pdf/2107.13602.pdf), Jul. 28 2021. [github, dpr-scale](https://github.com/facebookresearch/dpr-scale).
- [Answering Open-Domain Questions of Varying Reasoning Steps from Text](https://arxiv.org/pdf/2010.12527.pdf), Oct. 29 2021, `emnlp2021`, [code & benchmark](https://beerqa.github.io/)
- [End-to-End Training of Multi-Document Reader and Retriever for Open-Domain Question Answering](https://arxiv.org/pdf/2106.05346.pdf), Jun. 9 2021. `acl2021`
- [One Question Answering Model for Many Languages with Cross-lingual Dense Passage Retrieval](https://arxiv.org/pdf/2107.11976.pdf), Oct. 28 2021. [code](https://github.com/AkariAsai/CORA).
- :white_heart: [Dense Hierarchical Retrieval for Open-Domain Question Answering](https://arxiv.org/pdf/2110.15439.pdf), Oct. 28 2021.
  - information intactness of passage formulation of DPR
- :white_heart: [Joint Passage Ranking for Diverse Multi-Answer Retrieval](https://arxiv.org/pdf/2104.08445.pdf), Sep. 21 2021.
- [Evidentiality-guided Generation for Knowledge-Intensive NLP Tasks](https://akariasai.github.io/files/evidentiality_arxiv_2021.pdf), Dec. 16 2021. UW.
- :green_heart: [You Only Need One Model for Open-domain Question Answering](https://arxiv.org/pdf/2112.07381.pdf), Dec. 14 2021. Stanford.
  - ![image](https://user-images.githubusercontent.com/7335618/147050611-a64f79a9-e50e-4718-85cc-198a3e58b1a0.png)
    - it is interesting to see that eMSS pretraining can boot retrieval or reranking task about 7-12%
    - this pre-training method is able create input-passage-output triples, where the input is the query, i.e. a sentence with a named entity being masked, the passage is the positive evidence passage that contains the masked named entity and other common named entities, the output is the answer, i.e. the masked named entity.
- [Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering](https://aclanthology.org/2021.eacl-main.74.pdf), `eacl21`
- [Distilling Knowledge from Reader to Retreiver for Question Answering](https://openreview.net/pdf?id=NTEz-6wysdb), `iclr2021`
- [Sparsifying Sparse Representations for Passage Retrieval by Top-k Masking](https://arxiv.org/pdf/2112.09628.pdf), Dec. 17 2021.


#### Phrase-based retrieval model

- [Phrase-Indexed Question Answering: A New Challenge for Scalable Document Comprehension](https://aclanthology.org/D18-1052.pdf), `emnlp2018`
- [Real-Time Open-Domain Question Answering with Dense-Sparse Phrase Index](https://aclanthology.org/P19-1436.pdf), `acl2019`
- [Learning Dense Representations of Phrases at Scale](https://aclanthology.org/2021.acl-long.518.pdf), `acl2021`.

#### Improved DPR training

**Negative sample selection**

- [Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval](https://arxiv.org/pdf/2007.00808.pdf), Oct. 20 2021. `iclr2020`
  - problems of in-batch negatives: uninformative negatives --> diminishing gradient norms, large stochastic gradient variances and slow learning convergence.
- :white_heart: [RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/pdf/2010.08191.pdf), May 12 2021. `emnlp2021`
  - interactive q-p encoder to supervise (denoise) negative examples for DPR training
  - semi-supervised learning supervised by interactive q-p encoder
- :white_heart: [Learning Robust Dense Retrieval Models from Incomplete Relevance Labels](https://ciir-publications.cs.umass.edu/getpdf.php?id=1429), SIGIR 2021.
  - Improve upon ANCE, analyze the issue of unjudged relevance of negative example with ANCE.

**Semi-supervised learning**

- RocketQA paper
- :white_heart: [Relevance-guided Supervision for OpenQA with ColBERT](https://arxiv.org/pdf/2007.00814.pdf), Aug. 2 2021. `tacl2021`

#### Understanding datasets

- [Question and Answer Test-Train Overlap in Open-Domain Question Answering Datasets](https://arxiv.org/pdf/2008.02637.pdf), Aug. 6 2020.
- [Undersensitivity in Neural Reading Comprehension](https://arxiv.org/pdf/2003.04808.pdf), Feb. 15 2020.
- :white_heart: [Challenges in Generalization in Open Domain Question Answering](https://arxiv.org/pdf/2109.01156.pdf), Sep. 2 2021. 

#### Retriever, unsupervised training

- [Passage Ranking with BERT](https://arxiv.org/pdf/1901.04085.pdf), Apr. 14 2020.
- :white_heart: [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://dl.acm.org/doi/pdf/10.1145/3397271.3401075?casa_token=Er5SHkW6pggAAAAA:i0UwW9LxTMZmoF5k-HM6leeqIezjih8X9KBXb0ZXrt5PGZ05d-oX2Lur_TC5nkVEHzG_Pb1TV4Wfuvo), `sigir2020`.
  - "crucially, ColBERT's pruning-friendly interaction mechanism enables leveraging vector-similarity indexes for end-to-end retrieval directly from millions of documents."
- :white_heart: [Condenser: a Pre-training Architecture for Dense Retrieval](https://aclanthology.org/2021.emnlp-main.75.pdf), `emnlp2021`.
  - the issue with using representation of [CLS] token to represent dense vector, due to the illed attention pattern of BERT, so this paper takes some architectural change to the Transformer to make [CLS] representation more powerful 
- :green_heart: [Learning to Retrieve Passages without Supervision](http://www.cs.tau.ac.il/~oriram/spider.pdf), `spider` Dec. 16 2021 [tweet](https://twitter.com/ori__ram/status/1471111673398366208), [code](https://github.com/oriram/spider)
  - how to learn self-supervised dense retriever? => ***recurrent span retrieval*** - using recurring spans across passages in a document to create pseudo examples for contrastive learning
  - comparative with DPR (supervised dense retrieval model), see the following figure
  - ![image](https://user-images.githubusercontent.com/7335618/146286112-9b54b854-8d4e-4dc4-951a-a50330f6903f.png)
  - The idea of the method for creating self-supervsion signals are shown in the following figure
  - ![image](https://user-images.githubusercontent.com/7335618/146286438-46374c52-612c-4cd0-87af-3207974717b8.png)
  - specific procedure of creating positive examples: query transformation, splan filtering (a group of heuristics), training (contrastive learning loss), hybrid dense-sparse retrieval
  - The **analysis** compare different choice of query transformation, they find that query transformations that resemble more the natural query and adopt kee/remove (span) strategy works the best
  - **My Two Cents**
    - how many pseudo examples do they need to achieve the comparable performance of DPR? `implementation`! batch size of 512 with 512 hard negatives
    - the above question can be transformed to: how many (q, p+, p-) pairs are created during preprocessing?
- [Towards Unsupervised Dense Information Retrieval with Contrastive Learning](https://arxiv.org/pdf/2112.09118.pdf), Dec. 16 2021.

#### Information retrieval

- [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://arxiv.org/abs/2004.12832), Apr. 27 2020. `sigir2020`
- [ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/pdf/2112.01488.pdf), Dec. 2 2021.
- [BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://arxiv.org/abs/2104.08663), Oct. 21 2021. `nips2021`
- :green_heart: [Boosted Dense Retriever](https://arxiv.org/pdf/2112.07771.pdf), Dec. 14 2021.
  - ![image](https://user-images.githubusercontent.com/7335618/147015967-d78c96c0-ef09-43f1-bddf-dade83d06242.png)
- [Design Challenges for a Multi-Perspective Search Engine](https://arxiv.org/pdf/2112.08357.pdf), Dec. 15 2021.
- [Sublinear Time Approximation of Text Similarity Matrices](https://arxiv.org/pdf/2112.09631.pdf), Dec. 17 2021.
  - How to efficiently compute `n * n` similarity matrices over large `n` documents?

#### ACL ARR 2021

**pretraining, training of retriever**
- [Hyperlink-induced Pre-training for Passage Retrieval of Open-domain Question Answering](https://openreview.net/forum?id=5BtSP5Wi7gn)
- [Augmenting Document Representations for Dense Retrieval with Interpolation and Perturbation](https://openreview.net/forum?id=AgGygsClU1a)
- [Retrieval Data Augmentation Informed by Downstream Question Answering Performance](https://openreview.net/forum?id=e3ujixkK9yS)
- [Sentence-aware Contrastive Learning for Open-Domain Passage Retrieval](https://openreview.net/forum?id=t4RMD0mI_k)
- [CCQA: A New Web-Scale Question Answering Dataset for Model Pre-Training](https://openreview.net/forum?id=4CwYXIpRYe0)
- [C-MORE: Pretraining to Answer Open-Domain Questions by Consulting Millions of References](https://openreview.net/forum?id=kL8xTwwFMCT)
- [Question Answering Infused Pre-training of General-Purpose Contextualized Representations](https://openreview.net/forum?id=WmU4nT2Avy-)
- [Domain-matched Pre-training Tasks for Dense Retrieval](https://arxiv.org/pdf/2107.13602.pdf), Jul. 28 2021.

**QG**
- [A Simple and Effective Model for Multi-Hop Question Generation](https://openreview.net/forum?id=IV5YUaQ4pzG)
- [A Unified Abstractive Model for Generating Question-Answer Pairs](https://openreview.net/forum?id=WO4a5buL8OV)
- [A Copy-Augmented Generative Model for Open-Domain Question Answering](https://openreview.net/forum?id=9RHCjj-vhq3)
- [Entity-Conditioned Question Generation for Robust Attention Distribution in Neural Information Retrieval](https://openreview.net/forum?id=dEDH-_vQ2Wb)

**New benchmark**
- [CCQA: A New Web-Scale Question Answering Dataset for Model Pre-Training](https://openreview.net/forum?id=4CwYXIpRYe0)
- [ArchivalQA: A Large-scale Benchmark Dataset for Open Domain Question Answering over Archival News Collections](https://openreview.net/forum?id=4zUWJyt0Ja4)

**Analysis**
- [Salient Phrase Aware Dense Retrieval: Can a Dense Retriever Imitate a Sparse One?](https://openreview.net/forum?id=bkagwUGBYU1)
- [What Makes Machine Reading Comprehension Questions Difficult? Investigating Variation in Passage Sources and Question Types](https://openreview.net/forum?id=npVogfg9Vuw)
- [Probing Difficulty and Discrimination of Natural Language Questions With Item Response Theory](https://openreview.net/forum?id=fUahDQidjl_)
- [Last to Learn Bias: Analyzing and Mitigating a Shortcut in Question Matching](https://openreview.net/forum?id=peJrpYeuzEA)
- [Delving Deep into Extractive Question Answering Data](https://openreview.net/forum?id=mgu6JpUzgD)

**Multi-hop**
- [Modeling Multi-hop Question Answering as Single Sequence Prediction](https://openreview.net/forum?id=C1XEENowywW)

**Others: domain adaptation, etc.**
- [Synthetic Question Value Estimation for Domain Adaptation of Question Answering](https://openreview.net/forum?id=U-e3OTlgXwW)
- [Answering Open-Domain Multi-Answer Questions via a Recall-then-Verify Framework](https://openreview.net/forum?id=MVcv5KxTqu_)




### Knowledge-Base Question Answering, KB Completion

- [RNG-KBQA: Generation Augmented Iterative Ranking for Knowledge Base Question Answering](https://arxiv.org/pdf/2109.08678.pdf), Caiming Xiong et al. `arXiv`
- [Benchmarking the Combinatorial Generalizability of Complex Query Answering on Knowledge Graphs](https://arxiv.org/pdf/2109.08925.pdf), Yangqiu Song et al. `nips2021`
- [SMORE: Knowledge Graph Completion and Multi-hop Reasoning in Massive Knowledge Graphs](https://arxiv.org/pdf/2110.14890.pdf), Oct. 28 2021, `nips2021`. [code](https://github.com/google-research/smore)
- [Vector-valued Distance and Gyrocalculus on the Space of Symmetric Positive Definite Matrices](https://arxiv.org/pdf/2110.13475.pdf), Oct. 26 2021. `nips2021`
- [Metadata Shaping: Natural Language Annotations for the Tail](https://arxiv.org/pdf/2110.08430.pdf), Oct. 16 2021. `handling long-tail` `knowledge injection`
- [SQALER: Scaling Question Answering by Decoupling Multi-Hop and Logical Reasoning](https://proceedings.neurips.cc/paper/2021/file/68bd22864919297c8c8a8c32378e89b4-Paper.pdf), `nips2021`.

### Machine Reading Comprehension

- [A Survey on Explainability in Machine Reading Comprehension](https://arxiv.org/pdf/2010.00389.pdf), Oct. 1 2020.
- [Numerical reasoning in machine reading comprehension tasks: are we there yet?](https://arxiv.org/abs/2109.08207), `emnlp2021`
- [Improving Unsupervised Question Answering via Summarization-Informed Question Generation](https://arxiv.org/pdf/2109.07954.pdf), `emnlp2021` `qg`
- [Using Knowledge Distillation and Active Learning](https://arxiv.org/pdf/2109.12662.pdf), `arXiv 2021 Sep`
- [GNN is a Counter: Revisiting GNN for Question Answering](https://arxiv.org/pdf/2110.03192.pdf), iclr2022  submitted.
- [A Dataset for Answering Time-Sensitive Questions](https://arxiv.org/pdf/2108.06314.pdf), `nips2021`.
- [Improving Question Answering Model Robustness with Synthetic Adversarial Data Generation](https://arxiv.org/pdf/2104.08678.pdf), Sep. 16 2021, `emnlp2021` `adversarial` `robustness`
- [Do We Know What We Don’t Know? Studying Unanswerable Questions beyond SQuAD 2.0](https://aclanthology.org/2021.findings-emnlp.385/), `emnlp2021`

#### Old goodies

- [Machine Comprehension using Match-LSTM and Answer Pointer](https://arxiv.org/pdf/1608.07905.pdf), ICLR 2017.
- [Bi-Directional Attention Flow for Machine Comprehension](https://arxiv.org/pdf/1611.01603.pdf), ICLR 2017.

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
- [Ditch the Gold Standard: Re-evaluating Conversational Question Answering](https://arxiv.org/pdf/2112.08812.pdf), Dec. 16 2021.


### QA and social bias

- [BBQ: A Hand-Built Bias Benchmark for Question Answering](https://arxiv.org/pdf/2110.08193.pdf), Oct. 15 2021. Sam Bowman's group.

### Dataset

- [QA Dataset Explosion: A Taxonomy of NLP Resources for Question Answering and Reading Comprehension](https://arxiv.org/pdf/2107.12708.pdf), Jul. 27 2021.
- [Models in the Loop: Aiding Crowdworkers with Generative Annotation Assistants](https://arxiv.org/pdf/2112.09062.pdf), Dec. 16 2021. `crowdsourcing`
