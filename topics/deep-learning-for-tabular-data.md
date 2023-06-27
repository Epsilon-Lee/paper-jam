
## Surveys

- [Deep Neural Networks and Tabular Data: A Survey](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9998482), Feb. 2022.
- [A Short Chronology Of Deep Learning For Tabular Data](https://sebastianraschka.com/blog/2022/deep-learning-for-tabular-data.html), Jul. 24 2022.

## Others

- [Rethinking Logic Minimization for Tabular Machine Learning](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9964348), 2022.

## Classic methods

- [Benchmarking state-of-the-art gradient boosting algorithms for classification](https://arxiv.org/pdf/2305.17094.pdf), May 26 2023.
  - _with a focus on hyperparameter search with two methods: randomized search and Bayesian optimization using the Tree-stuctured Parzen Estimator_

## Methods and critics

### Pretraining

- [DeepGBM: A Deep Learning Framework Distilled by GBDT for Online Prediction Tasks](https://www.microsoft.com/en-us/research/uploads/prod/2019/08/deepgbm_kdd2019__CR_.pdf). `kdd2019`. [code](https://github.com/motefly/DeepGBM/tree/master).
- [TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442), Aug. 20, 2019 v1.
- [TabTransformer: Tabular Data Modeling Using Contextual Embeddings](https://arxiv.org/pdf/2012.06678.pdf), Dec. 11 2020.
- [Gradient Boosting Neural Networks: GrowNet](https://arxiv.org/pdf/2002.07971.pdf), Jun. 14 2020.
- [Self-Attention Between Datapoints: Going Beyond Individual Input-Output Pairs in Deep Learning](https://proceedings.neurips.cc/paper/2021/file/f1507aba9fc82ffa7cc7373c58f8a613-Paper.pdf), `nips2021`.
- [Well-tuned Simple Nets Excel on Tabular Datasets](https://proceedings.neurips.cc/paper/2021/file/c902b497eb972281fb5b4e206db38ee6-Paper.pdf), `nips2021`.
- [SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning](https://proceedings.neurips.cc/paper/2021/file/9c8661befae6dbcd08304dbf4dcaf0db-Paper.pdf), `nips2021`.
- [SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training](https://arxiv.org/pdf/2106.01342.pdf), Jun. 2 2021.
- [Tabular Data: Deep learning is not all you need](https://arxiv.org/pdf/2106.03253.pdf), Nov. 23 2021.
- [Why do tree-based models still outperform deep learning on tabular data?](https://arxiv.org/pdf/2207.08815.pdf), Jul. 18 2022. [benchmark](https://github.com/LeoGrin/tabular-benchmark).
- [PTab: Using the Pre-trained Language Model for Modeling Tabular Data](https://arxiv.org/pdf/2209.08060.pdf), Sep. 15 2022.
  - Using key-value format for including semantics of table headers and utilizing pretrained language models like BERT.
- [TabPFN: A Transformer that solves small tabular classification problems in a second](https://table-representation-learning.github.io/assets/papers/tabpfn_a_transformer_that_solv.pdf), `nips2022`. [code](https://huggingface.co/spaces/TabPFN/TabPFN).
- [Transfer Learning with Deep Tabular Models](https://table-representation-learning.github.io/assets/papers/transfer_learning_with_deep_ta.pdf), `nips2022`.
  - It seems that dl can outperform XGBoost via transfer learning.
- [Explaining Anomalies using Denoising Autoencoders for Financial Tabular Data](https://arxiv.org/pdf/2209.10658.pdf), Oct. 3 2023. `anomaly detection`.
- [Embeddings for Tabular Data: A Survey](https://arxiv.org/pdf/2302.11777.pdf), Feb. 23 2023.
- [TabRet: Pre-training transformer-based tabular models for unseen columns](https://arxiv.org/pdf/2303.15747.pdf), Mar. 28 2023.
- [HyperTab: Hypernetwork Approach for Deep Learning on Small Tabular Datasets](https://arxiv.org/pdf/2304.03543.pdf), Apr. 7 2023.
- [When Do Neural Nets Outperform Boosted Trees on Tabular Data?](https://arxiv.org/pdf/2305.02997.pdf), May 4 2023.
- [XTab: Cross-table Pretraining for Tabular Transformers](https://arxiv.org/pdf/2305.06090.pdf), May 10 2023. [code](https://github.com/BingzhaoZhu/XTab).
- [Rethinking Data Augmentation for Tabular Data in Deep Learning](https://arxiv.org/pdf/2305.10308.pdf), May 22 2023.
- [Generative Table Pre-training Empowers Models for Tabular Prediction](https://arxiv.org/pdf/2305.09696.pdf), May 16 2023. [code](https://github.com/ZhangTP1996/TapTap).
- [Enabling tabular deep learning when d ≫ n with an auxiliary knowledge graph](https://arxiv.org/pdf/2306.04766.pdf), Jun. 7 2023.
- [RoTaR: Efficient Row-Based Table Representation Learning via Teacher-Student Training](https://arxiv.org/pdf/2306.11696.pdf), Jun. 20 2023.
- [OpenFE: Automated Feature Generation with Expert-level Performance](https://openreview.net/attachment?id=1H1irbEaGV&name=pdf), `icml2023`. `automatic feature engineering`.
- [DeepTLF](https://openreview.net/pdf?id=PaQhL90tLmX), `iclr2022 rejected` [reviews](https://openreview.net/forum?id=PaQhL90tLmX). [revised and accepted here](https://link.springer.com/article/10.1007/s41060-022-00350-z). [code]().

## LLMs/FMs for tabular data

- [Towards Parameter-Efficient Automation of Data Wrangling Tasks with Prefix-Tuning](https://openreview.net/pdf?id=8kyYJs2YkFH), `nips2022`.
- [Can Foundation Models Wrangle Your Data?](https://arxiv.org/pdf/2205.09911.pdf), Dec. 24 2022.
- [TabLLM: Few-shot Classification of Tabular Data with Large Language Models](https://arxiv.org/pdf/2210.10723.pdf), Oct. 19 2022.
- [Ask me anything: A simple strategy for prompting language models](https://arxiv.org/pdf/2210.02441.pdf), Nov. 20 2022.
- [Symphony: Towards Natural Language Query Answering over Multi-modal Data Lakes](https://www2.cs.arizona.edu/~caolei/papers/SYMPHONY.pdf), `cidr2023`.
- [KAER: A Knowledge Augmented Pre-Trained Language Model for Entity Resolution](https://arxiv.org/pdf/2301.04770.pdf), Jan. 12 2023.
- [TABLET: Learning From Instructions For Tabular Data](https://arxiv.org/pdf/2304.13188.pdf), Apr. 25 2023.
- [Trompt: Towards a Better Deep Neural Network for Tabular Data](https://arxiv.org/pdf/2305.18446.pdf), May. 29 2023.

## Codebase

- [pytorch_tabular](https://github.com/manujosephv/pytorch_tabular).






