

# Awesome deep learning for tabular data

## Surveys

- [Deep Neural Networks and Tabular Data: A Survey](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9998482), Feb. 2022.
- [A Short Chronology Of Deep Learning For Tabular Data](https://sebastianraschka.com/blog/2022/deep-learning-for-tabular-data.html), Jul. 24 2022.
- [Tabular Data: Deep learning is not all you need](https://arxiv.org/pdf/2106.03253.pdf), Nov. 23 2021.

## Classic methods

- [Benchmarking state-of-the-art gradient boosting algorithms for classification](https://arxiv.org/pdf/2305.17094.pdf), May 26 2023.
  - _with a focus on hyperparameter search with two methods: randomized search and Bayesian optimization using the Tree-stuctured Parzen Estimator_

## Methods

### Feature processing

- [Exploiting Field Dependencies for Learning on Categorical Data](https://arxiv.org/pdf/2307.09321.pdf), Jul. 18 2023.
- [A benchmark of categorical encoders for binary classification](https://arxiv.org/pdf/2307.09191.pdf), Jul. 19 2023.

### Architectures

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

### Missing values

- [Invertible Tabular GANs: Killing Two Birds with One Stone for Tabular Data Synthesis](https://proceedings.neurips.cc/paper/2021/file/22456f4b545572855c766df5eefc9832-Paper.pdf), `nips2021`.
- [Tabular data imputation: choose kNN over deep learning](https://openreview.net/pdf?id=_MRiKN8-sw), `iclr2022 rejected`.
- [Diffusion models for missing value imputation in tabular data](https://arxiv.org/pdf/2210.17128.pdf), Mar. 11 2023.
- [TabRet: Pre-training transformer-based tabular models for unseen columns](https://arxiv.org/pdf/2303.15747.pdf), Mar. 28 2023.
- [CasTGAN: cascaded generative adversarial networks for realistic tabular data synthesis](https://arxiv.org/pdf/2307.00384.pdf), Jul. 1 2023.

### Critics

- [When Do Neural Nets Outperform Boosted Trees on Tabular Data?](https://arxiv.org/pdf/2305.02997.pdf), May 4 2023.
- [Why do tree-based models still outperform deep learning on tabular data?](https://arxiv.org/pdf/2207.08815.pdf), Jul. 18 2022. [benchmark](https://github.com/LeoGrin/tabular-benchmark).

### Pretraining and representation learning

- [SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning](https://proceedings.neurips.cc/paper/2021/file/9c8661befae6dbcd08304dbf4dcaf0db-Paper.pdf), `nips2021`.
- [PTab: Using the Pre-trained Language Model for Modeling Tabular Data](https://arxiv.org/pdf/2209.08060.pdf), Sep. 15 2022.
  - Using key-value format for including semantics of table headers and utilizing pretrained language models like BERT.
- [Transfer Learning with Deep Tabular Models](https://table-representation-learning.github.io/assets/papers/transfer_learning_with_deep_ta.pdf), `nips2022`.
  - It seems that dl can outperform XGBoost via transfer learning.
- [Explaining Anomalies using Denoising Autoencoders for Financial Tabular Data](https://arxiv.org/pdf/2209.10658.pdf), Oct. 3 2023. `anomaly detection`.
- [Embeddings for Tabular Data: A Survey](https://arxiv.org/pdf/2302.11777.pdf), Feb. 23 2023.
- [XTab: Cross-table Pretraining for Tabular Transformers](https://arxiv.org/pdf/2305.06090.pdf), May 10 2023. [code](https://github.com/BingzhaoZhu/XTab).
- [Generative Table Pre-training Empowers Models for Tabular Prediction](https://arxiv.org/pdf/2305.09696.pdf), May 16 2023. [code](https://github.com/ZhangTP1996/TapTap).
- [RoTaR: Efficient Row-Based Table Representation Learning via Teacher-Student Training](https://arxiv.org/pdf/2306.11696.pdf), Jun. 20 2023.
- [Enhancing Representation Learning on High-Dimensional, Small-Size Tabular Data: A Divide and Conquer Method with Ensembled VAEs](https://arxiv.org/pdf/2306.15661.pdf),Jun. 27 2023.
- [HYTREL: Hypergraph-enhanced Tabular Data Representation Learning](https://arxiv.org/pdf/2307.08623.pdf), Jul. 14 2023.
- [CT-BERT: Learning Better Tabular Representations Through Cross-Table Pre-training](https://arxiv.org/pdf/2307.04308.pdf), Jul. 10 2023.
- [UniTabE: Pretraining a Unified Tabular Encoder for Heterogeneous Tabular Data](https://arxiv.org/pdf/2307.09249.pdf), Jul. 18 2023.

### Data augmentation

- [Data Augmentation Techniques for Tabular Data](https://www.mphasis.com/content/dam/mphasis-com/global/en/home/innovation/next-lab/Mphasis_Data-Augmentation-for-Tabular-Data_Whitepaper.pdf). `whitepaper`.
- [DeltaPy⁠⁠ — Tabular Data Augmentation & Feature Engineering](https://github.com/firmai/deltapy/tree/master), 2020.
- [Data Augmentation for Compositional Data: Advancing Predictive Models of the Microbiome](https://proceedings.neurips.cc/paper_files/paper/2022/file/81a28be483155f802ddef448d6fc4b57-Paper-Conference.pdf), `nips2022`. [code](https://github.com/cunningham-lab/AugCoDa).
- [OpenFE: Automated Feature Generation with Expert-level Performance](https://openreview.net/attachment?id=1H1irbEaGV&name=pdf), `icml2023`. `automatic feature engineering`.
- [Rethinking Data Augmentation for Tabular Data in Deep Learning](https://arxiv.org/pdf/2305.10308.pdf), May 22 2023.
- [Semi-Supervised Learning with Data Augmentation for Tabular Data](https://web.archive.org/web/20221021061539id_/https://dl.acm.org/doi/pdf/10.1145/3511808.3557699), `cikm2022`.
- [Programmable Synthetic Tabular Data Generation](https://arxiv.org/pdf/2307.03577.pdf), Jun. 7 2023.

### LLMs/FMs for tabular data

- [Towards Parameter-Efficient Automation of Data Wrangling Tasks with Prefix-Tuning](https://openreview.net/pdf?id=8kyYJs2YkFH), `nips2022`.
- [Can Foundation Models Wrangle Your Data?](https://arxiv.org/pdf/2205.09911.pdf), Dec. 24 2022.
- [TabLLM: Few-shot Classification of Tabular Data with Large Language Models](https://arxiv.org/pdf/2210.10723.pdf), Oct. 19 2022.
- [Ask me anything: A simple strategy for prompting language models](https://arxiv.org/pdf/2210.02441.pdf), Nov. 20 2022.
- [Symphony: Towards Natural Language Query Answering over Multi-modal Data Lakes](https://www2.cs.arizona.edu/~caolei/papers/SYMPHONY.pdf), `cidr2023`.
- [KAER: A Knowledge Augmented Pre-Trained Language Model for Entity Resolution](https://arxiv.org/pdf/2301.04770.pdf), Jan. 12 2023.
- [TABLET: Learning From Instructions For Tabular Data](https://arxiv.org/pdf/2304.13188.pdf), Apr. 25 2023.
- [Trompt: Towards a Better Deep Neural Network for Tabular Data](https://arxiv.org/pdf/2305.18446.pdf), May. 29 2023.
- [Language models are weak learners](https://arxiv.org/pdf/2306.14101.pdf), Jun. 25 2023.

## Others

- [Rethinking Logic Minimization for Tabular Machine Learning](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9964348), 2022.

## Codebase

- [pytorch_tabular](https://github.com/manujosephv/pytorch_tabular).



# Classics

- [CatBoost Versus XGBoost and LightGBM: Developing Enhanced Predictive Models for Zero-Inflated Insurance Claim Data](https://arxiv.org/pdf/2307.07771.pdf), Jul. 15 2023. `feature interaction strength`.

