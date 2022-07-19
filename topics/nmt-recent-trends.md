### Pretraining-based NMT

- [Language Models are Good Translators](https://arxiv.org/pdf/2106.13627.pdf), Jun. 25.
- [Cross-Attention is All You Need: Adapting Pretrained Transformers for Machine Translation](https://aclanthology.org/2021.emnlp-main.132.pdf), `emnlp2021`
- [BERT, MBERT, or BIBERT? A Study on Contextualized Embeddings for Neural Machine Translation](https://aclanthology.org/2021.emnlp-main.534.pdf), `emnlp2021` [code](https://github.com/fe1ixxu/BiBERT).
- [DEEP: DEnoising Entity Pre-training for Neural Machine Translation](https://arxiv.org/pdf/2111.07393.pdf), Neubig et al. `emnlp2021`
- [Towards Making the Most of BERT in Neural Machine Translation](https://arxiv.org/abs/1908.05672), `aaai2020`.


### Data Augmentation for NMT

- [mixSeq: A Simple Data Augmentation Method for Neural Machine Translation](https://aclanthology.org/2021.iwslt-1.23.pdf), IWSLT 2021.
- [BITEXTEDIT: Automatic Bitext Editing for Improved Low-Resource Machine Translation](https://arxiv.org/pdf/2111.06787.pdf), Nov. 12 2021.
- [Phrase-level Adversarial Example Generation for Neural Machine Translation](https://arxiv.org/pdf/2201.02009.pdf), Jan. 6 2021.

### Char NMT

- [Why don’t people use character-level machine translation?](https://arxiv.org/pdf/2110.08191.pdf), `analysis`
- [Noisy UGC Translation at the Character Level: Revisiting Open-Vocabulary Capabilities and Robustness of Char-Based Models](https://arxiv.org/pdf/2110.12552.pdf), `evaluation under noise`.

### Multilingual NMT

- [Alternative Input Signals Ease Transfer in Multilingual Machine Translation](https://arxiv.org/pdf/2110.07804.pdf), Oct. 15 2021 `analysis` of transfer ability of multilingual nmt models
- [Breaking Down Multilingual Machine Translation](https://arxiv.org/pdf/2110.08130.pdf), Oct. 15 `analysis` plus improvement methods
- [Tricks for Training Sparse Translation Models](https://arxiv.org/pdf/2110.08246.pdf), Oct. 15 `multitask learning` `unbalanced learning`
- [Multilingual Neural Machine Translation: Can Linguistic Hierarchies Help?](https://arxiv.org/pdf/2110.07816.pdf), Oct. 15, proposed a new training paradigm taking into account the language hierarchy, handling `negative transfer`
- [Multilingual Domain Adaptation for NMT: Decoupling Language and Domain Information with Adapters](https://arxiv.org/abs/2110.09574), Oct. 18 2021. `wmt2021`
- [Continual Learning in Multilingual NMT via Language-Specific Embeddings](https://arxiv.org/abs/2110.10478), Oct. 20 2021. `wmt2021` 
- [Parameter Differentiation based Multilingual Neural Machine Translation](https://arxiv.org/pdf/2112.13619.pdf), Dec. 27 2021.

### Multilinguality

- [Beyond Distillation: Task-level Mixture-of-Experts for Efficient Inference](https://arxiv.org/pdf/2110.03742.pdf), `emnlp2021`
- [Using Optimal Transport as Alignment Objective for fine-tuning Multilingual Contextualized Embeddings](https://arxiv.org/pdf/2110.02887.pdf), `emnlp2021`
- [When is BERT Multilingual? Isolating Crucial Ingredients for Cross-lingual Transfer](https://arxiv.org/pdf/2110.14782.pdf), Oct. 27, `emnlp2021`
- [Few-shot Learning with Multilingual Language Models](https://arxiv.org/pdf/2112.10668.pdf), Dec. 23 2021.
- [On Cross-Lingual Retrieval with Multilingual Text Encoders](https://arxiv.org/pdf/2112.11031.pdf), Dec. 21 2021.
- [No Language Left Behind: Scaling Human-Centered Machine Translation](https://arxiv.org/pdf/2207.04672.pdf), Jul. 2022.

### Efficiency

- [ABC: Attention with Bounded-Memory Control](https://arxiv.org/pdf/2110.02488.pdf), `deepmind`

### Low-resource

- [The Low-Resource Double Bind: An Empirical Study of Pruning for Low-Resource Machine Translation](https://arxiv.org/pdf/2110.03036.pdf), `pruning` `emnlp2021`
- [Phrase-level Active Learning for Neural Machine Translation](https://arxiv.org/abs/2106.11375#), Jun. 21 2021.
- [A Preordered RNN Layer Boosts Neural Machine Translation in Low Resource Settings](https://arxiv.org/pdf/2112.13960.pdf), Dec. 28 2021.
- [Exploring Diversity in Back Translation for Low-Resource Machine Translation](https://arxiv.org/pdf/2206.00564.pdf), Jun. 1 2022. `lexical/syntactic diversity`

### Scaling Up

- [Scaling Laws for Neural Machine Translation](https://arxiv.org/pdf/2109.07740.pdf), Google.
- [Data and Parameter Scaling Laws for Neural Machine Translation](https://aclanthology.org/2021.emnlp-main.478.pdf), JHU, `emnlp2021`

### Quality Estimation

- [Levenshtein Training for Word-level Quality Estimation](https://arxiv.org/pdf/2109.05611.pdf), Shuoyang Ding et al.
- [Beyond Glass-Box Features: Uncertainty Quantification Enhanced Quality Estimation for Neural Machine Translation](https://arxiv.org/pdf/2109.07141.pdf), Alibaba Group.
- [Practical Perspectives on Quality Estimation for Machine Translation](https://arxiv.org/pdf/2005.03519.pdf), CMU & Google Inc. May 2 2020.

### Translationese

- [Comparing Feature-Engineering and Feature-Learning Approaches for Multilingual Translationese Classification](https://arxiv.org/pdf/2109.07604.pdf).

### Training Strategy

- [Improving Neural Machine Translation by Bidirectional Training](https://arxiv.org/pdf/2109.07780.pdf), Liang Ding et al.

### Analysis and Evaluation

- [Relations between comprehensibility and adequacy errors in machine translation output](https://aclanthology.org/2020.conll-1.19.pdf), `error analysis`.
- [Beyond Noise: Mitigating the Impact of Fine-grained Semantic Divergences on Neural Machine Translation](https://aclanthology.org/2021.acl-long.562.pdf), `acl2021`.
- [Translation Transformers Rediscover Inherent Data Domains](https://arxiv.org/pdf/2109.07864.pdf), 2021.
- [On the Limits of Minimal Pairs in Contrastive Evaluation](https://arxiv.org/pdf/2109.07465.pdf), 2021
- [On Neurons Invariant To Sentence Structural Changes in Neural Machine Translation](https://arxiv.org/pdf/2110.03067.pdf), `iclr`, 2021.
- [Understanding the Impact of UGC Specificities on Translation Quality](https://arxiv.org/pdf/2110.12551.pdf), Oct. 24 2021.
- [On the Limits of Minimal Pairs in Contrastive Evaluation](https://aclanthology.org/2021.blackboxnlp-1.5.pdf), `blackboxnlp 2021`
- [Variance-Aware Machine Translation Test Sets](https://arxiv.org/pdf/2111.04079.pdf), Nov. 7 2021. `nips2021`
- [HOPE: A Task-Oriented and Human-Centric Evaluation Framework Using Professional Post-Editing Towards More Effective MT Evaluation](https://arxiv.org/pdf/2112.13833.pdf), Dec. 27 2021.
- [Data Troubles in Sentence Level Confidence Estimation for Machine Translation](https://arxiv.org/pdf/2010.13856.pdf), Oct. 26 2020. Google.
- [How do lexical semantics affect translation? An empirical study](https://arxiv.org/pdf/2201.00075.pdf), Dec. 31 2021. Amazon Alexa AI.

### Discourse

- [When Does Translation Require Context? A Data-driven, Multilingual Exploration](https://arxiv.org/pdf/2109.07446.pdf).
- [DOCmT5: Document-Level Pretraining of Multilingual Language Models](https://arxiv.org/abs/2112.08709), Dec. 16 2021.
- [SMDT: Selective Memory-Augmented Neural Document Translation](https://arxiv.org/pdf/2201.01631.pdf), Jan. 5 2022.

### New Paradigm

- [Nearest Neighbor Machine Translation](https://arxiv.org/pdf/2010.00710.pdf), `iclr 2021`

### Understanding and Interpretation

- [Scaling Up Influence Functions](https://arxiv.org/pdf/2112.03052.pdf), Dec. 6 2021. `aaai2022`


### Inference, decoding

- [Amortized Noisy Channel Neural Machine Translation](https://arxiv.org/pdf/2112.08670.pdf), Dec. 16 2021.
- [Massive-scale Decoding for Text Generation using Lattices](https://arxiv.org/pdf/2112.07660.pdf), Dec. 14 2021.
- [Learning and Analyzing Generation Order for Undirected Sequence Models](https://arxiv.org/pdf/2112.09097.pdf), Dec. 16 2021.
- [Characterizing and addressing the issue of oversmoothing in neural autoregressive sequence modeling](https://arxiv.org/pdf/2112.08914.pdf), Dec. 16 2021.
