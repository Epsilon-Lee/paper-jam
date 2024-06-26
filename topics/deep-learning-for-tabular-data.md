
A table of contents for different topics around the modeling and processing of tabular data.

- [Awesome deep learning for tabular data](#awesome-deep-learning-for-tabular-data)
  - [Surveys](#surveys)
  - [Methods](#methods)
    - [Generative modeling and imputation](#generative-modeling-and-imputation)
    - [Featue processing and selection](#feature-processing-and-selection)
    - [Architectures](#architectures)
    - [Critics of deep learning](#critics-of-deep-learning)
    - [Pretraining and representation learning](#pretraining-and-representation-learning)
    - [LLMs for tabular data](#llms-for-tabular-data)
  - [Deep learning based tabular anomaly detection](#deep-learning-based-tabular-anomaly-detection)
  - [Interpretability](#interpretability)
  - [Benchmarks](#benchmarks)
  - [Codebases](#codebases)
- [Classic methods for tabular data](#classic-methods-for-tabular-data)

---

## Awesome deep learning for tabular data

### Surveys

- [Deep Neural Networks and Tabular Data: A Survey](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9998482), Feb. 2022.
- [A Short Chronology Of Deep Learning For Tabular Data](https://sebastianraschka.com/blog/2022/deep-learning-for-tabular-data.html), Jul. 24 2022.
- [Tabular Data: Deep learning is not all you need](https://arxiv.org/pdf/2106.03253.pdf), Nov. 23 2021.
- [Revisiting Deep Learning Models for Tabular Data](https://arxiv.org/abs/2106.11959), `nips2021`.
  - _revised Jul. 26 2023_

### Methods

#### Generative modeling and imputation

- [Modeling Tabular data using Conditional GAN](https://arxiv.org/abs/1907.00503), Jul. 2019.
- [A supervised generative optimization approach for tabular data](https://arxiv.org/pdf/2309.05079.pdf), Sep. 10 2023.
- [TabMT: Generating tabular data with masked Transformers](https://arxiv.org/pdf/2312.06089.pdf), Dec. 11 2023. `nips2023`.
- [Invertible Tabular GANs: Killing Two Birds with One Stone for Tabular Data Synthesis](https://proceedings.neurips.cc/paper/2021/file/22456f4b545572855c766df5eefc9832-Paper.pdf), `nips2021`.
- [Tabular data imputation: choose kNN over deep learning](https://openreview.net/pdf?id=_MRiKN8-sw), `iclr2022 rejected`.
- [Diffusion models for missing value imputation in tabular data](https://arxiv.org/pdf/2210.17128.pdf), Mar. 11 2023.
- [TabRet: Pre-training transformer-based tabular models for unseen columns](https://arxiv.org/pdf/2303.15747.pdf), Mar. 28 2023.
- [CasTGAN: cascaded generative adversarial networks for realistic tabular data synthesis](https://arxiv.org/pdf/2307.00384.pdf), Jul. 1 2023.
- [Generating and Imputing Tabular Data via Diffusion and Flow-based Gradient-Boosted Trees](https://arxiv.org/pdf/2309.09968.pdf), Sep. 18 2023.
- [Continuous Diffusion for Mixed-Type Tabular Data](https://arxiv.org/pdf/2312.10431.pdf), Dec. 16 2023.
- [Data Augmentation Techniques for Tabular Data](https://www.mphasis.com/content/dam/mphasis-com/global/en/home/innovation/next-lab/Mphasis_Data-Augmentation-for-Tabular-Data_Whitepaper.pdf). `whitepaper`.
- [DeltaPy⁠⁠ — Tabular Data Augmentation & Feature Engineering](https://github.com/firmai/deltapy/tree/master), 2020.
- [Data Augmentation for Compositional Data: Advancing Predictive Models of the Microbiome](https://proceedings.neurips.cc/paper_files/paper/2022/file/81a28be483155f802ddef448d6fc4b57-Paper-Conference.pdf), `nips2022`. [code](https://github.com/cunningham-lab/AugCoDa).
- [OpenFE: Automated Feature Generation with Expert-level Performance](https://openreview.net/attachment?id=1H1irbEaGV&name=pdf), `icml2023`. `automatic feature engineering`.
- [Rethinking Data Augmentation for Tabular Data in Deep Learning](https://arxiv.org/pdf/2305.10308.pdf), May 22 2023.
- [Semi-Supervised Learning with Data Augmentation for Tabular Data](https://web.archive.org/web/20221021061539id_/https://dl.acm.org/doi/pdf/10.1145/3511808.3557699), `cikm2022`.
- [Programmable Synthetic Tabular Data Generation](https://arxiv.org/pdf/2307.03577.pdf), Jun. 7 2023.
- [Structured Evaluation of Synthetic Tabular Data](https://arxiv.org/pdf/2403.10424.pdf), Mar. 29 2024.

#### Feature processing and selection

- [Exploiting Field Dependencies for Learning on Categorical Data](https://arxiv.org/pdf/2307.09321.pdf), Jul. 18 2023.
- [A benchmark of categorical encoders for binary classification](https://arxiv.org/pdf/2307.09191.pdf), Jul. 19 2023.
- [FeatGeNN: Improving Model Performance for Tabular Data with Correlation-based Feature Extraction](https://arxiv.org/pdf/2308.07527.pdf), Aug. 15 2023.
- [A Performance-Driven Benchmark for Feature Selection in Tabular Deep Learning](https://arxiv.org/pdf/2311.05877.pdf), Nov. 10 2023.

#### Architectures

- [DeepGBM: A Deep Learning Framework Distilled by GBDT for Online Prediction Tasks](https://www.microsoft.com/en-us/research/uploads/prod/2019/08/deepgbm_kdd2019__CR_.pdf). `kdd2019`. [code](https://github.com/motefly/DeepGBM/tree/master).
- [Modeling Tabular Data using Conditional GAN](https://proceedings.neurips.cc/paper/2019/file/254ed7d2de3b23ab10936522dd547b78-Paper.pdf), `nips2019`. [code](https://github.com/sdv-dev/CTGAN).
- [TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442), Aug. 20, 2019 v1. [code](https://github.com/dreamquark-ai/tabnet).
- [TabTransformer: Tabular Data Modeling Using Contextual Embeddings](https://arxiv.org/pdf/2012.06678.pdf), Dec. 11 2020. [code](https://github.com/lucidrains/tab-transformer-PyTorch).
- [Gradient Boosting Neural Networks: GrowNet](https://arxiv.org/pdf/2002.07971.pdf), Jun. 14 2020. [code](https://github.com/sbadirli/GrowNet/tree/master).
- [Well-tuned Simple Nets Excel on Tabular Datasets](https://proceedings.neurips.cc/paper/2021/file/c902b497eb972281fb5b4e206db38ee6-Paper.pdf), `nips2021`. [code](https://github.com/releaunifreiburg/WellTunedSimpleNets).
- [SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training](https://arxiv.org/pdf/2106.01342.pdf), Jun. 2 2021. [code](https://github.com/somepago/saintv). [openreview](https://openreview.net/forum?id=nL2lDlsrZU).
- [Self-Attention Between Datapoints: Going Beyond Individual Input-Output Pairs in Deep Learning](https://proceedings.neurips.cc/paper/2021/file/f1507aba9fc82ffa7cc7373c58f8a613-Paper.pdf), `nips2021`. [code]()
- [DeepTLF](https://openreview.net/pdf?id=PaQhL90tLmX), `iclr2022 rejected` [reviews](https://openreview.net/forum?id=PaQhL90tLmX). [revised and accepted here](https://link.springer.com/article/10.1007/s41060-022-00350-z). [code](https://github.com/unnir/DeepTLF).
- [TabPFN: A Transformer that solves small tabular classification problems in a second](https://table-representation-learning.github.io/assets/papers/tabpfn_a_transformer_that_solv.pdf), `nips2022`. [code](https://huggingface.co/spaces/TabPFN/TabPFN).
- [HyperTab: Hypernetwork Approach for Deep Learning on Small Tabular Datasets](https://arxiv.org/pdf/2304.03543.pdf), Apr. 7 2023.
- [Enabling tabular deep learning when d ≫ n with an auxiliary knowledge graph](https://arxiv.org/pdf/2306.04766.pdf), Jun. 7 2023.
- [NCART: Neural Classification and Regression Tree for Tabular Data](https://arxiv.org/pdf/2307.12198.pdf), Jul. 23 2023.
- [TabR: Unlocking the power of retrieval-augmented tabular deep learning](https://arxiv.org/pdf/2307.14338.pdf), Jul. 26 2023. [code](https://github.com/yandex-research/tabular-dl-tabr).
- [SHAPNN: Shapley Value Regularized Tabular Neural Network](https://arxiv.org/pdf/2309.08799.pdf), Sep. 15 2023.
- [Unlocking the transferability of tokens in deep models for tabular data](https://arxiv.org/pdf/2310.15149.pdf), Oct. 23 2023.
- [Scaling TabPFN: Sketching and Feature Selection for Tabular Prior-Data Fitted Networks](https://arxiv.org/pdf/2311.10609.pdf), Nov. 17 2023.
- [Meditab: Scaling medical tabular data predictors via data consolidation, enrichment and refinement](https://arxiv.org/pdf/2305.12081.pdf), Oct. 5 2023.
- [MotherNet: A Foundational Hypernetwork for Tabular Classification](https://arxiv.org/pdf/2312.08598.pdf), Dec. 14 2023.
- [Anytime neural architecture search on tabular data](https://arxiv.org/pdf/2403.10318.pdf), Mar. 15 2024.

#### Critics of deep learning

- [Why do tree-based models still outperform deep learning on tabular data?](https://arxiv.org/pdf/2207.08815.pdf), Jul. 18 2022. [benchmark](https://github.com/LeoGrin/tabular-benchmark).
- [When Do Neural Nets Outperform Boosted Trees on Tabular Data?](https://arxiv.org/pdf/2305.02997.pdf), May 4 2023. [v3](https://arxiv.org/pdf/2305.02997.pdf).

#### Pretraining and representation learning

- [Table Pre-training: A Survey on Model Architectures, Pre-training Objectives, and Downstream Tasks](https://arxiv.org/pdf/2201.09745.pdf), Apr. 29 2022.
- [SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning](https://proceedings.neurips.cc/paper/2021/file/9c8661befae6dbcd08304dbf4dcaf0db-Paper.pdf), `nips2021`.
- [PTab: Using the Pre-trained Language Model for Modeling Tabular Data](https://arxiv.org/pdf/2209.08060.pdf), Sep. 15 2022.
  - Using key-value format for including semantics of table headers and utilizing pretrained language models like BERT.
- [Transfer Learning with Deep Tabular Models](https://table-representation-learning.github.io/assets/papers/transfer_learning_with_deep_ta.pdf), `nips2022`.
  - It seems that dl can outperform XGBoost via transfer learning.
  - [new version link](https://arxiv.org/pdf/2206.15306.pdf), `iclr2023` version.
- [Explaining Anomalies using Denoising Autoencoders for Financial Tabular Data](https://arxiv.org/pdf/2209.10658.pdf), Oct. 3 2023. `anomaly detection`.
- [Embeddings for Tabular Data: A Survey](https://arxiv.org/pdf/2302.11777.pdf), Feb. 23 2023.
- [XTab: Cross-table Pretraining for Tabular Transformers](https://arxiv.org/pdf/2305.06090.pdf), May 10 2023. [code](https://github.com/BingzhaoZhu/XTab).
- [Generative Table Pre-training Empowers Models for Tabular Prediction](https://arxiv.org/pdf/2305.09696.pdf), May 16 2023. [code](https://github.com/ZhangTP1996/TapTap).
- [RoTaR: Efficient Row-Based Table Representation Learning via Teacher-Student Training](https://arxiv.org/pdf/2306.11696.pdf), Jun. 20 2023.
- [Enhancing Representation Learning on High-Dimensional, Small-Size Tabular Data: A Divide and Conquer Method with Ensembled VAEs](https://arxiv.org/pdf/2306.15661.pdf),Jun. 27 2023.
- [HYTREL: Hypergraph-enhanced Tabular Data Representation Learning](https://arxiv.org/pdf/2307.08623.pdf), Jul. 14 2023.
- [CT-BERT: Learning Better Tabular Representations Through Cross-Table Pre-training](https://arxiv.org/pdf/2307.04308.pdf), Jul. 10 2023.
- [UniTabE: Pretraining a Unified Tabular Encoder for Heterogeneous Tabular Data](https://arxiv.org/pdf/2307.09249.pdf), Jul. 18 2023.
- [CAST: Cluster-aware self-training for tabular data](https://arxiv.org/pdf/2310.06380.pdf), Oct. 10 2023.
- [Training-free generalization on heterogeneous tabular data via meta-representation](https://arxiv.org/pdf/2311.00055.pdf), Oct. 31 2023.
- [High dimensional, tabular deep learning with an auxiliary knowledge graph](https://openreview.net/pdf?id=GGylthmehy), `nips2023`.
- [Tabular few-shot generalization across heterogeneous feature spaces](https://arxiv.org/pdf/2311.10051.pdf), Nov. 16 2023.
- [Classification of Tabular Data by Text Processing](https://arxiv.org/pdf/2311.12521.pdf), Nov. 21 2023.
- [Relational Deep Learning: Graph Representation Learning on Relational Databases](https://arxiv.org/abs/2312.04615), Dec. 7 2023.
- [PyTorch Frame: A Modular Framework for Multi-Modal Tabular Learning](https://arxiv.org/pdf/2404.00776.pdf), Mar. 31 2024.

#### LLMs for tabular data

- [Towards Parameter-Efficient Automation of Data Wrangling Tasks with Prefix-Tuning](https://openreview.net/pdf?id=8kyYJs2YkFH), `nips2022`.
- [Can Foundation Models Wrangle Your Data?](https://arxiv.org/pdf/2205.09911.pdf), Dec. 24 2022.
- [TabLLM: Few-shot Classification of Tabular Data with Large Language Models](https://arxiv.org/pdf/2210.10723.pdf), Oct. 19 2022.
- [Ask me anything: A simple strategy for prompting language models](https://arxiv.org/pdf/2210.02441.pdf), Nov. 20 2022.
- [Symphony: Towards Natural Language Query Answering over Multi-modal Data Lakes](https://www2.cs.arizona.edu/~caolei/papers/SYMPHONY.pdf), `cidr2023`.
- [KAER: A Knowledge Augmented Pre-Trained Language Model for Entity Resolution](https://arxiv.org/pdf/2301.04770.pdf), Jan. 12 2023.
- [TABLET: Learning From Instructions For Tabular Data](https://arxiv.org/pdf/2304.13188.pdf), Apr. 25 2023.
- [Trompt: Towards a Better Deep Neural Network for Tabular Data](https://arxiv.org/pdf/2305.18446.pdf), May. 29 2023.
- [Language models are weak learners](https://arxiv.org/pdf/2306.14101.pdf), Jun. 25 2023.
- [Incorporating LLM Priors into Tabular Learners](https://arxiv.org/pdf/2311.11628.pdf), Nov. 20 2023.
- [Rethinking tabular data understanding with large language models](https://arxiv.org/pdf/2312.16702.pdf), Dec. 27 2023.
- [Unleashing the Potential of Large Language Models for Predictive Tabular Tasks in Data Science](https://arxiv.org/pdf/2403.20208.pdf), Mar. 29 2024.
- [Large Language Models(LLMs) on Tabular Data: Prediction, Generation, and Understanding - A Survey](https://arxiv.org/pdf/2402.17944.pdf), Mar. 1 2024.
- [Elephants Never Forget: Memorization and Learning of Tabular Data in Large Language Models](https://arxiv.org/pdf/2404.06209.pdf), Apr. 9 2024. [code](https://github.com/interpretml/LLM-Tabular-Memorization-Checker).

#### Automl for tabular

- [Automated model selection for tabular data](https://arxiv.org/pdf/2401.00961.pdf), Jan. 1 2024.

### Deep learning based tabular anomaly detection

- [TabADM: Unsupervised Tabular Anomaly Detection with Diffusion Models](https://arxiv.org/pdf/2307.12336.pdf), Jul. 23 2023.
- [Understanding the limitations of self-supervised learning for tabular anomaly detection](https://arxiv.org/pdf/2309.08374.pdf), Sep. 15 2023.

### Interpretability

- [Stable and Interpretable Deep Learning for Tabular Data: Introducing InterpreTabNet with the Novel InterpreStability Metric](https://browse.arxiv.org/pdf/2310.02870.pdf), Oct. 4 2023.
- [Refutation of Shapley Values for XAI – Additional Evidence](https://arxiv.org/pdf/2310.00416.pdf), Sep. 30 2023.
- [TabCBM: Concept-based interpretable neural networks for tabular data](https://openreview.net/pdf?id=TIsrnWpjQ0), `tmlr2023`.

### Benchmarks

- [TabRepo: A large scale repository of tabular model evaluations and its automl applications](https://arxiv.org/pdf/2311.02971.pdf), Nov. 6 2023.

### Codebases

- [pytorch_tabular](https://github.com/manujosephv/pytorch_tabular).

---

## Classic methods for tabular data

- [CatBoost Versus XGBoost and LightGBM: Developing Enhanced Predictive Models for Zero-Inflated Insurance Claim Data](https://arxiv.org/pdf/2307.07771.pdf), Jul. 15 2023. `feature interaction strength`.
- [LCE - An Augmented Combination of Bagging and Boosting in Python](https://arxiv.org/pdf/2308.07250.pdf), Aug. 14 2023.
- [Benchmarking state-of-the-art gradient boosting algorithms for classification](https://arxiv.org/pdf/2305.17094.pdf), May 26 2023.
  - _with a focus on hyperparameter search with two methods: randomized search and Bayesian optimization using the Tree-stuctured Parzen Estimator_
- [Rethinking Logic Minimization for Tabular Machine Learning](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9964348), 2022.
- [The Conditioning Bias in Binary Decision Trees and Random Forests and Its Elimination](https://arxiv.org/pdf/2312.10708.pdf), Dec. 17 2023.
  - The formulation of an issue named 'conditioning bias' in decision tree learning
- [Invariant Random Forest: Tree-Based Model Solution for OOD Generalization](https://arxiv.org/pdf/2312.04273.pdf), Dec. 20 2023. `ood generalization`.
- [Robust Loss Functions for Training Decision Trees with Noisy Labels](https://arxiv.org/pdf/2312.12937.pdf), Dec. 20 2023.


