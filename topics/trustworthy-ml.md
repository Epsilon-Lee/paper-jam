
## Trustworthy Machine Learning

This is a huge umbrella word, while I have created interpretability and out-of-distribution generalization as two topics that previously most interested me.
And it is time to cover more important contents here.

---

### Algorithmic recourse

- [Beyond Individualized Recourse: Interpretable and Interactive Summaries of Actionable Recourses](https://papers.nips.cc/paper/2020/file/8ee7730e97c67473a424ccfeff49ab20-Paper.pdf), `nips2020`.

### Privacy

- [Reconstructing training data from model gradient, provably](https://arxiv.org/abs/2212.03714), Dec. 7 2022.

---

## Trustworthy data

> _understand sub-populations, prototypes, noise-level, adaptivity (shift that close to the recent data distribution) of data, portrait of customers_

- [Data-centric Artificial Intelligence: A Survey](https://arxiv.org/pdf/2303.10158.pdf), Apr. 2 2023.

### Dataset exploratory analysis

- [SAP-sLDA: An Interpretable Interface for Exploring Unstructured Text](https://arxiv.org/pdf/2308.01420.pdf), Jul. 28 2023. `visualization`.
- [Solving data quality problems with desbordante: a demo](https://arxiv.org/pdf/2307.14935.pdf), Jul. 27 2023.
- [QI2 - an Interactive Tool for Data Quality Assurance](https://arxiv.org/pdf/2307.03419.pdf), Jul. 10 2023. `data-centric` `toolkit`.
  - _"It quantifies neighborhood input-output relationship behaviors over a set of data points. High dimensional anomalous structure"_
- [Dataset Interfaces: Diagnosing Model Failures Using Controllable Counterfactual Generation](https://arxiv.org/pdf/2302.07865.pdf), Feb. 15 2023. `interpretability`.
  - Can I utilize this idea to in the domain of risk control for the fraud detection and credit score card datasets (tabular data).

### Dataset noise

- [Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks](https://arxiv.org/abs/2103.14749), Mar. 26 2021.
  - [Automated Data Quality at Scale](https://cleanlab.ai/blog/automated-data-quality-at-scale/), Jul. 2023.
- [Bridging Trustworthiness and Open-World Learning: An Exploratory Neural Approach for Enhancing Interpretability, Generalization, and Robustness](https://arxiv.org/pdf/2308.03666.pdf), Aug. 7 2023.

### Data selection

- [Optimizing Data Collection for Machine Learning](https://proceedings.neurips.cc/paper_files/paper/2022/file/c1449acc2e64050d79c2830964f8515f-Paper-Conference.pdf), `nips2022`.

## Trustworthy model

> _understand training, model behavior on sub-population of interest, model's strengthes and weaknesses, model iteration under distribution shift_

### Model debugging and testing

- [Explaining machine learning models with interactive natural langauge conversations using TalkToModel](https://arxiv.org/pdf/2207.04154.pdf), Mar. 2023.
- [MDB: Interactively Querying Datasets and Models](https://arxiv.org/pdf/2308.06686.pdf), Aug. 13 2023. `model debugging framework`.
- [Evaluating AI systems under uncertain ground truth: a case study in dermatology](https://arxiv.org/pdf/2307.02191.pdf), Jul. 5 2023.
- [Where Does My Model Underperform? A Human Evaluation of Slice Discovery Algorithms](https://arxiv.org/pdf/2306.08167.pdf), Jun. 13 2023. `systematic error analysis`.
- [A Data-Driven Measure of Relative Uncertainty for Misclassification Detection](https://arxiv.org/pdf/2306.01710.pdf), Jun. 2 2023. `uncertainty` `trustworthy`.
- [Auditing and Generating Synthetic Data with Controllable Trust Trade-offs](https://arxiv.org/pdf/2304.10819.pdf), May 2 2023. `trustworthy ml`.

### Robustness and spurious correlation

- [Spurious Correlations and Where to Find Them](https://arxiv.org/pdf/2308.11043.pdf), Aug. 21 2023.
  - _"We believe that the first step in developing robust solutions against spurious correlations is recognizing when the models trained using ERM succumb to spurious correlations"_
- [Provable domain adaptation using privileged information](https://openreview.net/pdf/eba28736e52d1f5686a07c9462dc52e4017ea2ad.pdf), `icml2023` `spurious correlation workshop`.
  - _"we show that access to side information about examples from the source and target domains can help relax sufficient assumptions on input variables and increase sample efficiency at the cost of collecting richer variable set"_
  - see Figure 1 in the paper for an intuitive demonstration
- [Contextual Reliability: When Different Features Matter in Different Contexts](https://arxiv.org/pdf/2307.10026.pdf), Jul. 19 2023. `spurious correlation`.

### Understand Distribution shift

- [On the Need for a Language Describing Distribution Shifts: Illustrations on Tabular Datasets](https://arxiv.org/pdf/2307.05284.pdf), Jul. 11 2023.
  - code is already available.
- [On Minimizing the Impact of Dataset Shifts on Actionable Explanations](https://arxiv.org/pdf/2306.06716.pdf), Jun. 11 2023. `distribution shift`.
- [Explaining Predictive Uncertainty with Information Theoretic Shapley Values](https://arxiv.org/pdf/2306.05724.pdf), Jun. 9 2023. `shapley values` `distribution shift`.
- [Explanation Shift: Detecting distribution shifts on tabular data via the explanation space](https://arxiv.org/pdf/2210.12369.pdf), Oct. 2022.

### Transparent model

- [Co-creating a globally interpretable model with human input](https://arxiv.org/pdf/2306.13381.pdf), Jun. 23 2023. `human-centered`.
- [DeforestVis: Behavior Analysis of Machine Learning Models with Surrogate Decision Stumps](https://arxiv.org/pdf/2304.00133.pdf), Mar. 31 2023. `transparent model`.

### Decision making architecture

- [When Are Two Lists Better than One?: Benefits and Harms in Joint Decision-making](https://arxiv.org/pdf/2308.11721.pdf), Aug. 22 2023. `decision making`.
- [Ideal Abstractions for Decision-Focused Learning](https://arxiv.org/pdf/2303.17062.pdf), Mar. 29 2023.

## Trustworthy operation

> _monitoring, and (fast) attribution of abnormal phenomenon_

- [FeedbackLogs: Recording and Incorporating Stakeholder Feedback into Machine Learning Pipelines](https://arxiv.org/pdf/2307.15475.pdf), Jul. 28 2023. `trustworthy`.
- [Perspectives on Incorporating Expert Feedback into Model Updates](https://arxiv.org/pdf/2205.06905.pdf), Jul. 16 2022.




