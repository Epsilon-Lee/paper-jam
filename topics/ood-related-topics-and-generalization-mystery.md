
# OOD-related Research Topics and Generalization Mystery

Contents of this archive:

- [Out-of-Distribution Learning and Generalization](#out-of-distribution-ood-learning-and-generalization)
  - [Survey](#survey)
  - [OOD generalization theories](#ood-generalization-theories)
  - [Empirical observations](#empirical-observations)
  - [OOD and domain generalization methods](#ood-and-domain-generalization-methods)
    - [Invariant risk minimization](#invariant-risk-minimization)
    - [Benchmarks](#benchmarks)
    - [Causal representation learning](#causal-representation-learning)
  - [Distribution shift](#distribution-shift)
    - [Detection of distribution shift](#detection-of-distribution-shift)
    - [Unsupervised domain adaptation](#unsupervised-domain-adaptation)
  - [Connection between OOD and adversarials](#connection-between-ood-and-adversarials)
- [Out-of-Distribution Detection](#out-of-distribution-detection)
  - [Surveys](#surveys)
  - [Theories](#theories)
  - [Methods](#methods)
  - [Explanations](#explanations)
  - [OOD methods for sequential data](#ood-methods-for-sequential-data)
  - [OOD methods for time-series data](#ood-methods-for-time-series-data)
- [Robustness](#robustness)
  - [Surveys](#surveys)
  - [Theories](#theories)
  - [Adversarial training](#adversarial-training)
  - [Attacks](#attacks)

---

## Out-of-Distribution (OOD) Learning and Generalization

### Survey

- [A Survey of Unsupervised Deep Domain Adaptation](https://dl.acm.org/doi/pdf/10.1145/3400066), 2020.
- [Unsupervised Multi-Class Domain Adaptation: Theory, Algorithms, and Practice](https://www4.comp.polyu.edu.hk/~cslzhang/paper/TPAMI-MultiClassUDA.pdf), `tpami2020`.
- [A Survey on Domain Adaptation Theory: Learning Bounds and Theoretical Guarantees](https://arxiv.org/pdf/2004.11829.pdf), Jul. 13 2020.
- [Towards Out-Of-Distribution Generalization: A Survey](https://arxiv.org/pdf/2108.13624.pdf), Aug. 2021.
- [Generalizing to Unseen Domains: A Survey on Domain Generalization](https://arxiv.org/pdf/2103.03097.pdf), May 24 2022.
- [Domain Generalization: A Survey](https://ieeexplore.ieee.org/abstract/document/9847099), 2022.
- [State-of-the-art generalisation research in NLP: a taxonomy and review](https://arxiv.org/pdf/2210.03050.pdf), Oct. 10 2022.

### OOD generalization theories

- [Towards a theory of out-of-distribution learning](https://arxiv.org/pdf/2109.14501.pdf), JHU 2021.
- [Towards a theoretical framework of out-of-distribution generalization](https://arxiv.org/abs/2106.04496), Nov. 2021.
  - _"theoretical understanding of what kind of invariance can guarantee OOD generalization is still limited, and generalization to arbitrary out-of-distribution is clearly impossible"_
- [Invariance Principle Meets Information Bottleneck for Out-of-Distribution Generalization](https://arxiv.org/pdf/2106.06607.pdf), Jun. 11 2021. `nips2021` `invariant risk minimization`
- [Failure Modes of Domain Generalization Algorithms](https://arxiv.org/pdf/2111.13733.pdf), Nov. 26 2021.
- [A Theory of Label Propagation for Subpopulation Shift](https://arxiv.org/pdf/2102.11203.pdf), Jul. 20 2021 `distribution shift`
- [Monotonic Risk Relationships under Distribution Shifts for Regularized Risk Minimization](https://arxiv.org/pdf/2210.11589.pdf), Oct. 20 2022.
- [On-Demand Sampling: Learning Optimally from Multiple Distributions](fhttps://arxiv.org/pdf/2210.12529.pdf), arXiv Oct. 22 2022.

#### Domain adaptation theory

- [Learning Bounds for Domain Adaptation](https://papers.nips.cc/paper/2007/file/42e77b63637ab381e8be5f8318cc28a2-Paper.pdf), `nips2007`.
- [Sample Selection Bias Correction Theory](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/34675.pdf), 2008.
- [Domain Adaptation: Learning Bounds and Algorithms](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/35391.pdf), 2009.
- [Impossibility Theorems for Domain Adaptation](http://proceedings.mlr.press/v9/david10a/david10a.pdf), `aistats2010`.
- [A theory of learning from different domains](https://link.springer.com/content/pdf/10.1007/s10994-009-5152-4.pdf), `ml2010`.

### Empirical observations

- [Understanding Out-of-distribution: A Perspective of Data Dynamics](https://arxiv.org/pdf/2111.14730.pdf), Nov. 29 2021.
- [Characterizing Generalization under Out-Of-Distribution Shifts in Deep Metric Learning](https://arxiv.org/pdf/2107.09562.pdf), Jul. 20 2021.
- [Understanding the failure modes of out-of-distribution generalization](https://openreview.net/forum?id=fSTD6NFIW_b), `iclr2021`
- [Out-of-Distribution Generalization with Deep Equilibrium Models](http://www.gatsby.ucl.ac.uk/~balaji/udl2021/accepted-papers/UDL2021-paper-072.pdf), 2021 icml workshop.
- [Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution](https://arxiv.org/pdf/2202.10054.pdf), Feb. 21 2022.

### OOD and domain generalization methods

- [An Empirical Investigation of Domain Generalization with Empirical Risk Minimizers](https://openreview.net/forum?id=Z8mLxlpSyrJ), `neurips2021`.
  - _"Recent work demonstrates that deep neural networks trained using ERM can generalize under distribution shift, outperforming specialized training algorithms for domain generalization. The goal of this paper is to further understand this phenomenon"_
  - _"Our investigation reveals that measures relating to the Fisher information, predictive entropy, and maximum mean discrepancy are good predictors of the ood generalization of ERM models"_
- [Deep Stable Learning for Out-Of-Distribution Generalization](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Deep_Stable_Learning_for_Out-of-Distribution_Generalization_CVPR_2021_paper.pdf), `cvpr2021`
- [LINDA: Unsupervised Learning to Interpolate in Natural Language Processing](https://arxiv.org/pdf/2112.13969.pdf), Dec. 28 2021.
- [Environment Inference for Invariant Learning](https://arxiv.org/pdf/2010.07249.pdf), `icml2021`. `environment inference`.
- [Heterogeneous Risk Minimization](https://arxiv.org/pdf/2105.03818.pdf), `icml2021`. `enviroment inference`.
- [Counterfactual Maximum Likelihood Estimation for Training Deep Networks](https://openreview.net/forum?id=o6s1b_-nDOE), `nips2021`.
- [Improving Out-of-Distribution Robustness via Selective Augmentation](https://arxiv.org/pdf/2201.00299.pdf), Jan. 2 2022.
- [ZIN: When and How to Learn Invariance Without Environment Partition? ](https://openreview.net/forum?id=pUPFRSxfACD), `nips2022`. `environment inference`.
- [On Distributionally Robust Optimization and Data Rebalancing](https://www.researchgate.net/profile/Agnieszka-Slowik-5/publication/358338470_On_Distributionally_Robust_Optimization_and_Data_Rebalancing/links/61fc4ca94393577abe0d75cc/On-Distributionally-Robust-Optimization-and-Data-Rebalancing.pdf), Feb. 2022.
- [Learning Causal Semantic Representation for Out-of-Distribution Prediction](https://proceedings.neurips.cc/paper/2021/file/310614fca8fb8e5491295336298c340f-Paper.pdf), `nips2021`.
- [Generalization and Robustness Implications in Object-Centric Learning](https://arxiv.org/pdf/2107.00637.pdf), `icml2021`.
- [Counterfactual Invariance to Spurious Correlations: Why and How to Pass Stress Tests](https://proceedings.neurips.cc/paper/2021/file/8710ef761bbb29a6f9d12e4ef8e4379c-Paper.pdf), `nips2021`.
- [Visual Representation Learning Does Not Generalize Strongly within the Same Domain](https://arxiv.org/pdf/2107.08221.pdf&lt;/p&gt;), `iclr2022`.
- [Probable Domain Generalization via Quantile Risk Minimization](https://arxiv.org/abs/2207.09944), Jul. 20 2022. [github](https://github.com/cianeastwood/qrm).
- [Distributionally Robust Losses for Latent Covariate Mixtures](https://arxiv.org/pdf/2007.13982.pdf), arXiv.v2 Aug. 10 2022.
- [Ensemble of Averages: Improving Model Selection and Boosting Performance in Domain Generalization](https://proceedings.neurips.cc/paper_files/paper/2022/file/372cb7805eaccb2b7eed641271a30eec-Paper-Conference.pdf), `neurips2022`.
- [Assaying out-of-distribution generalization in transfer learning](https://proceedings.neurips.cc/paper_files/paper/2022/file/2f5acc925919209370a3af4eac5cad4a-Paper-Conference.pdf), `neurips2022`.
  - _"Since ood generalization is a generally ill-posed problem, various proxy targets (e.g. calibration, adversarial robustness, algorithmic corruptions, invariance across shifts) were studied across different research programs resulting in different recommendations"_
  - _"Our findings  confirm that in- and out-of-distribution accuracies tend to increase jointly, but show that their relation is largely dataset-dependent, and in general more nuanced and more complex than posited by previous, smaller scale studies"_ 
- [Model Agnostic Sample Reweighting for Out-of-Distribution Learning](https://arxiv.org/pdf/2301.09819.pdf), Jan. 24 2023.
- [Discovering environments with XRM](https://arxiv.org/pdf/2309.16748.pdf), Sep. 28 2023. `environment inference`.
  - _"Successful ood generalization requires environment annotations. Unfortunately, these are resource-intensive to obtain, and their relevance to model performance is limited by the expectations and percetual biases of human annotators"_
  - _"Current proposals, which divide examples based on their training error, suffer from one fundamental problem. These methods add hyper-parameters and early-stopping criteria that are impossible to tune without a validatioin set with human-annotated environments"_
  - _"XRM provides a recipe for hyper-parameter tuning, does not require early stopping, and can discover environmentse for all training and validationd data"_
  - How to automatically discover environments from data?
- [ERM++: An Improved Baseline for Domain Generalization](https://arxiv.org/pdf/2304.01973.pdf), Aug. 15 2023.

#### Invariant risk minimization

The application from principle of **Invariance**.

- [The risks of invariant risk minimization](https://arxiv.org/pdf/2010.05761.pdf), Mar. 27 2021.
- [An Empirical Study of Invariant Risk Minimization on Deep Models](https://www.gatsby.ucl.ac.uk/~balaji/udl2021/accepted-papers/UDL2021-paper-044.pdf), `icml2021` `workshop`.
- [Does invariant risk minimization capture invariance](https://proceedings.mlr.press/v130/kamath21a/kamath21a.pdf), `aistats2021`.
- [Empirical or invariant risk minimization? A sample complexity perspective](https://arxiv.org/pdf/2010.16412.pdf), Aug. 19 2022.
- [DAIR: Data Augmented Invariant Regularization](https://arxiv.org/pdf/2110.11205.pdf), Oct. 21 2021.
- [IRM - when it works and when it doesn't: A test case of natural language inference](https://openreview.net/forum?id=KtvHbjCF4v), `nips2021`.
- [Adaptive risk minimization: Learning to adapt to domain shift](https://proceedings.neurips.cc/paper/2021/file/c705112d1ec18b97acac7e2d63973424-Paper.pdf), `nips2021`.
- [Heterogeneous Risk Minimization](https://proceedings.mlr.press/v139/liu21h.html), `icml2021`.
- [Kernelized Heterogeneous Risk Minimizaton](https://pengcui.thumedialab.com/papers/KernelHRM.pdf), `nips2021`.
- [Enviroment Inference for Invariant Learning](http://proceedings.mlr.press/v139/creager21a/creager21a.pdf), `icml2021`.
- [Fishr: Invariant Gradient Variances for Out-of-Distribution Generalization](https://proceedings.mlr.press/v162/rame22a/rame22a.pdf), `icml2022`.

#### Benchmarks

- [WILDS: A Benchmark of in-the-Wild Distribution Shifts](https://arxiv.org/abs/2012.07421), Dec. 14 2020. [icml version](http://proceedings.mlr.press/v139/koh21a/koh21a.pdf).
- [Extending the WILDS Benchmark for Unsupervised Adaptation](https://arxiv.org/pdf/2112.05090.pdf), Dec. 9 2021.
- [Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks](https://arxiv.org/pdf/2107.07455.pdf), Jul. 23 2021.
- [Extending the WILDS Benchmark for Unsupervised Adaptation](https://arxiv.org/abs/2112.05090), arXiv.v1 Dec. 9 2021.
- [GLUE-X: Evaluating Natural Language Understanding Models from an Out-of-Distribution Generalization Perspective](https://arxiv.org/pdf/2211.08073.pdf), May 22 2023.
- [Revisiting Out-of-distribution Robustness in NLP: Benchmark, Analysis, and LLMs Evaluations](https://arxiv.org/pdf/2306.04618.pdf), Jun. 7 2023.

#### Causal representation learning

- [Interventional Causal Representation Learning](https://arxiv.org/pdf/2209.11924.pdf), Sep. 24 2022.
- [Temporally Disentangled Representation Learning](https://arxiv.org/pdf/2210.13647.pdf), Oct. 24 2022.
- [Linear Causal Disentanglement via Interventions](https://arxiv.org/abs/2211.16467), Nov. 29 2022.

### Distribution or dataset shift adaptation

- [Domain Adaptation under Target and Conditional Shift](http://proceedings.mlr.press/v28/zhang13d.pdf), `icml2013`.
- [Detecting and Correcting for Label Shift with Black Box Predictions](https://proceedings.mlr.press/v80/lipton18a.html), `icml2018`. `label shift`
- [Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift](https://proceedings.neurips.cc/paper/2019/file/846c260d715e5b854ffad5f70a516c88-Paper.pdf), `nips2019`.
- [Domain Adaptation with Conditional Distribution Matching and Generalized Label Shift](https://arxiv.org/pdf/2003.04475.pdf), Dec. 11 2020.
- [On Label Shift in Domain Adaptation via Wasserstein Distance](https://arxiv.org/pdf/2110.15520.pdf), Oct. 29 2021.
- [An Information-theoretic Approach to Distribution Shifts](https://arxiv.org/pdf/2106.03783.pdf), Nov. 1 2021. `nips2021`
- [An opinioin from Cho about distributional robustness](https://twitter.com/kchonyc/status/1455619054786519045), Nov. 3 2021.
- [Estimating and Explaining Model Performance When Both Covariates and Labels Shift](https://arxiv.org/pdf/2209.08436.pdf), Sep. 18 2022. `nips2022` `explanable ml` `covariate shift` `label shift`
- [A Theoretical Analysis on Independence-driven Importance Weighting for Covariate-shift Generalization](https://arxiv.org/pdf/2111.02355.pdf), Jul. 11 2022. `version 3` `covariate shift`
- [Unsupervised Learning under Latent Label Shift](https://arxiv.org/pdf/2207.13179.pdf), Jul. 26 2022.
- [Data Drift Correction via Time-Varying Importance Weight Estimator](https://arxiv.org/pdf/2210.01422.pdf), Oct. 4 2022.
- [Optimal Representations for Covariate Shift](https://arxiv.org/pdf/2201.00057.pdf), Dec. 31 2021.
- [JAWS: Auditing Predictive Uncertainty Under Covariate Shift](https://arxiv.org/pdf/2207.10716v2.pdf), Nov. 23 2022. `nips2022`.
- [Agreement-on-the-Line: Predicting the Performance of Neural Networks under Distribution Shift](https://arxiv.org/abs/2206.13089), Jun. 27 2022. `nips2022`.
- [High Dimensional Binary Classification under Label Shift: Phase Transition and Regularization](https://arxiv.org/pdf/2212.00700.pdf), Dec. 5 2022.
- [Covariate-Shift Generalization via Random Sample Weighting](https://pengcui.thumedialab.com/papers/RandomSampleWeighting.pdf), `aaai2022`.
- [Learning Rate Schedules in the Presence of Distribution Shift](https://arxiv.org/pdf/2303.15634.pdf), Mar. 27 2023.
- [Beyond Invariance: Test-Time Label-Shift Adaptation for Addressing “Spurious” Correlations](https://arxiv.org/pdf/2211.15646.pdf), Feb. 2 2023.
- [Prominent Roles of Conditionally Invariant Components in Domain Adaptation: Theory and Algorithms](https://arxiv.org/pdf/2309.10301.pdf), Sep. 19 2023. `domain adaptation` `invariance`.
  - _"many DA algorithms have demonstrated considerable empirical success, blindly applying these algorithms can often lead to worse performance on new datasets [...] it is crucial to clarify the assumptions under which a DA algorithm has good target performance"_
  - conditionally invariant components can be be estimated through conditional invariante penalty
- [Invariant Probabilistic Prediction](https://arxiv.org/pdf/2309.10083.pdf), Sep. 18 2023. `uncertainty` `statistics`.
- [RLSbench: Domain Adaptation Under Relaxed Label Shift](https://proceedings.mlr.press/v202/garg23a/garg23a.pdf), `icml2023`.
  - _"a large-scale benchmark for relaxed label shift, consisting of >500 distribution shift pairs spanning vision, tabular, and language modalities, with varying label proportions. Unlike existing benchmarks, which primarily focus on shifts in class-conditional $$p(x \vert y)$$, our benchmark also focuses on label marginal shifts."_

#### Detection of distribution shift

- [Feature shift detection: Localizing which features have shifted via conditional distribution tests](https://proceedings.neurips.cc/paper/2020/file/e2d52448d36918c575fa79d88647ba66-Paper.pdf), `nips2020`.
- [Evaluating Robustness to Dataset Shift via Parametric Robustness Sets](https://arxiv.org/abs/2205.15947), May 31 2022. `nips2022`.
  - _"We give a method for proactively identifying small, plausible shifts in distribution which lead to large differences in model performance."_
  - _"These shifts are defined via parametric changes in the causal mechanisms of observed variables, where constraints on parameters yield a 'robustness set' of plausible distributions and a corresponding worst-case loss over the set"_
  - _"We apply our approach to a computer vision task (classifying gender from images), revealing sensitivity to shifts in non-causal attributes"_
- ["Why did the model fail?": Attributing Model Performance Changes to Distributional Shifts](https://arxiv.org/pdf/2210.10769.pdf), Oct. 19 2022.
- [Explanation Shift: Detecting distribution shifts on tabular data via the explanation space](https://arxiv.org/pdf/2210.12369.pdf), Oct. 22 2022.
- [Sequential Covariate Shift Detection Using Classifier Two-Sample Tests](https://trustml.github.io/docs/icml22.pdf), `icml2022`.
- [Finding Competence Regions in Domain Generalization](https://arxiv.org/pdf/2303.09989.pdf), Mar. 17 2023.
- [Towards Explaining Image-Based Distribution Shifts](https://www.seankulinski.com/publication/towards-explaining-image-based-shifts/towards-explaining-image-based-shifts.pdf), `cvpr2022`.
- [Towards Explaining Distribution Shifts](https://www.seankulinski.com/publication/towards-explaining-distribution-shifts/towards-explaining-distribution-shifts.pdf), 2022.
- [How did model change? Efficiently assessing machine learning API shifts](https://lchen001.github.io/papers/2022_APIShift_ICLR.pdf), `iclr2022`.
- [A learning based hypothesis test for harmful covariate shift](https://arxiv.org/abs/2212.02742), Dec. 6 2022. `iclr2023`.

#### Unsupervised domain adaptation

**Surveys**

- [A review of domain adaptation without target labels](https://arxiv.org/pdf/1901.05335.pdf), Jul. 24 2019.
- [A Survey of Unsupervised Deep Domain Adaptation](https://dl.acm.org/doi/pdf/10.1145/3400066), Jul. 2020.

**Methods**

- [A Dirt-T Approach for Unsupervised Domain Adaptation](https://arxiv.org/pdf/1802.08735.pdf), Mar. 19 2018. `citation: 483`.
- [A Prototype-Oriented Framework for Unsupervised Domain Adaptation](https://proceedings.neurips.cc/paper/2021/file/8edd72158ccd2a879f79cb2538568fdc-Paper.pdf), `nips2021`.

**Multisource domaiin adaptation**

- [Moment Matching for Multi-Source Domain Adaptation](https://arxiv.org/pdf/1812.01754.pdf), Aug. 27 2019.
- [Subspace Identification for Multi-Source Domain Adaptation](https://arxiv.org/pdf/2310.04723.pdf), Oct. 7 2023.
- [Benchmarking Multi-Domain Active Learning on Image Classification](https://arxiv.org/pdf/2312.00364.pdf), Dec. 1 2023.

#### Test-time adaptation and inference

- [Test-time recalibration of conformal predictors under distribution shift based on unlabeled examples](https://arxiv.org/pdf/2210.04166.pdf), Oct. 9 2022. `conformal prediction`
- [Memory-Based Model Editing at Scale](https://proceedings.mlr.press/v162/mitchell22a/mitchell22a.pdf), `icml2022`.
- [Test-time Adaptation via Self-Training with Nearest Neighbor Information](https://openreview.net/pdf?id=EzLtB4M1SbM), `iclr2023 submitted`.
- [Bag of Tricks for Fully Test-Time Adaptation](https://browse.arxiv.org/pdf/2310.02416.pdf), Oct. 3 2023.

### Connection Between OOD and Adversarials

- [Rethinking Machine Learning Robustness via its Link with the Out-of-Distribution Problem](https://arxiv.org/pdf/2202.08944.pdf), Feb. 18 2022.

---

## Out-of-Distribution Detection

### Surveys

- [A Unifying Review of Deep and Shallow Anomaly Detection](https://arxiv.org/abs/2009.11732), Sep. 24 2020.
- [Deep learning for anomaly detection: a review](https://arxiv.org/pdf/2007.02500), arXiv.v3 Dec. 5 2020.
- [A review of uncertainty quantification in deep learning: Techniques, applications and challenges](https://www.sciencedirect.com/science/article/pii/S1566253521001081), 2021.
- [Generalized Out-of-Distribution Detection: A Survey](https://arxiv.org/pdf/2110.11334.pdf), Oct. 21 2021.
- [A Unified Survey on Anomaly, Novelty, Open-Set, and Out-of-Distribution Detection: Solutions and Future Challenges](https://arxiv.org/pdf/2110.14051.pdf), Oct. 26 2021.

### Theories

> Mathematical theories and insightful rigorous understanding of ood detection problem.

- [Breaking Down Out-of-Distribution Detection: Many Methods Based on OOD Training Data Estimate a Combination of the Same Core Quantities](https://arxiv.org/pdf/2206.09880.pdf), Jun. 20 2022. `icml2022`
- [Is Out-of-Distribution Detection Learnable?](https://arxiv.org/pdf/2210.14707.pdf), Oct. 26 2022. `nips2022 outstanding paper` 
- [Understanding Out-of-distribution: A Perspective of Data Dynamics](https://proceedings.mlr.press/v163/adila22a/adila22a.pdf), I (Still) Can’t Believe It’s Not Better Workshop at NeurIPS 2021.
- [Provable Guarantees for Understanding Out-of-distribution Detection](https://arxiv.org/pdf/2112.00787.pdf), Dec. 1 2022. `aaai2022`
- [When and How Does Known Class Help Discover Unknown Ones? Provable Understanding Through Spectral Analysis](https://arxiv.org/pdf/2308.05017.pdf), Aug. 9 2023. `icml2023`.

### Methods

> OOD detection has a very braod extension, it includes the task from detecting _near ood_ instances to _outliers/novelties_ that have large distinction to the so-called in-distribution data.
> Based on my current knowledge, OOD detection methods could be divided into two categories: **supervised** and **unsupervised**, which means _whether_ the method uses _ood observations_.
> Total unsupervised methods are usually based on ***energy/density estimation*** (_generative modelling_) over the in-distribution data, and conduct statistical test based on certain **statistical** assumption to work it out. While supervised methods can be both ***generative*** and ***discriminative*** with ood data for _smartly_ tuning ood threshold.

- [A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks](https://proceedings.neurips.cc/paper/2018/file/abdeb6f575ac5c6676b747bca8d09cc2-Paper.pdf), `nips2018`. [github](https://github.com/pokaxpoka/deep_Mahalanobis_detector). `discriminative`
- [Using Self-Supervised Learning Can Improve Model Robustness and Uncertainty](https://arxiv.org/pdf/1906.12340.pdf), `nips2019`. `generative+discriminative` [github](https://github.com/hendrycks/ss-ood).
- [Can You Trust Your Model's Uncertainty Evaluating Predictive Uncertainty Under Dataset Shift](https://proceedings.neurips.cc/paper/2019/file/8558cb408c1d76621371888657d2eb1d-Paper.pdf), `nips2019`. `discriminative uncertainty`
- [Deep Anomaly Detection with Outlier Exposure](https://arxiv.org/abs/1812.04606), `iclr2019`. `citation: 800+`
  - _Intuition_: diverse image and text data are available in enormous quantities, though they are not the expected anomalous inputs of the task at hand, they can be leveraged as auxiliary tasks to extract generalizable feature for task-specific anomaly detection.
  - This is called outlier exposure technique, a multitask training technique for more generalizable outlier detection. [github](https://github.com/hendrycks/outlier-exposure).
  - ***my two cents***: how about generative detector with outlier exposures?
  - Related papers:
    - [CSI: Novelty Detection via Contrastive Learning on Distributionally Shifted Instances](https://proceedings.neurips.cc/paper/2020/file/8965f76632d7672e7d3cf29c87ecaa0c-Paper.pdf), `nips2020`.
    - [Deep semi-supervised anomaly detection](https://arxiv.org/pdf/1906.02694.pdf), `iclr2020`. Feb. 14 2020.
- [Detecting Out-of-Distribution Examples with Gram Matrices](http://proceedings.mlr.press/v119/sastry20a/sastry20a.pdf), `icml2020`. [github](https://github.com/VectorInstitute/gram-ood-detection).
- [Contrastive Training for Improved Out-of-Distribution Detection](https://arxiv.org/pdf/2007.05566.pdf), arXiv Jul. 10 2021. `unsupervised`
  - `Confusion Log Probability`
  - This paper proves that CLP score especially improves near ood classes.
- [On the Importance of Gradients for Detecting Distributional Shifts in the Wild](https://arxiv.org/pdf/2110.00218.pdf), Oct. 9 2021. `ood detection`
- [Identifying and Benchmarking Natural Out-of-Context Prediction Problems](https://arxiv.org/pdf/2110.13223.pdf), Oct. 25 2021.
- [A Fine-grained Analysis on Distribution Shift](https://arxiv.org/pdf/2110.11328.pdf), Oct. 21 2021.
- [Understanding the Role of Self-Supervised Learning in Out-of-Distribution Detection Task](https://arxiv.org/pdf/2110.13435.pdf), Oct. 26 2021.
- [Exploring the Limits of Out-of-Distribution Detection](https://arxiv.org/pdf/2106.03004.pdf), `landscape` `nips2021`
- [Can multi-label classification networks know what they don’t know?](https://arxiv.org/pdf/2109.14162.pdf), Sep. 2021. `ood detection`
- [Provable Guarantees for Understanding Out-of-distribution Detection](https://arxiv.org/pdf/2112.00787.pdf), Dec. 1 2021.
- [ReAct: Out-of-distribution Detection With Rectified Activations](https://arxiv.org/abs/2111.12797), Nov. 24 2021.
- [On the Impact of Spurious Correlation for Out-of-distribution Detection](https://arxiv.org/abs/2109.05642), Sep. 12 2021.
- [Open-Set Recognition: a Good Closed-Set Classifier is All You Need?](https://arxiv.org/abs/2110.06207), Oct. 12 2021. `iclr2022`.
- [VOS: Learning What You Don't Know by Virtual Outlier Synthesis](https://arxiv.org/abs/2202.01197), `iclr2022` [code](https://github.com/deeplearning-wisc/vos).
- [p-DkNN: Out-of-Distribution Detection Through Statistical Testing of Deep Representations](https://arxiv.org/pdf/2207.12545.pdf), Jul. 25 2022. `ood`
- [Oracle Analysis of Representations for Deep Open Set Detection](https://arxiv.org/pdf/2209.11350.pdf), Sep. 22 2022.
- [Deep Hybrid Models for Out-of-Distribution Detection](https://openaccess.thecvf.com/content/CVPR2022/papers/Cao_Deep_Hybrid_Models_for_Out-of-Distribution_Detection_CVPR_2022_paper.pdf), `cvpr2022`.
- [Watermarking for Out-of-distribution Detection](https://arxiv.org/pdf/2210.15198.pdf), Oct. 27 2022.
  - _reprogramming_ property of deep models
- 🤍 [Out-of-Distribution Detection and Selective Generation for Conditional Language Models](https://arxiv.org/pdf/2209.15558.pdf), Sep. 30 2022.
  - Current Issues
    - hightly accurate ML classifiers can degrade significantly and provide overly-confident, wrong classification predictions for OOD inputs.
    - The paper said that LM trained autoregressively may suffer more from OOD inputs due to the prediction is constructed step-by-step which might shift more.
  - _"present a highly accurate and lightweight OOD detection method for CLM, and demonstrate its effectiveness on **abstractive summarization** and **translation**"_.
- [Falsehoods that ML researchers believe about OOD detection](https://arxiv.org/pdf/2210.12767.pdf), Oct. 23 2022.
- [Revisiting Softmax for Uncertainty Approximation in Text Classification](https://arxiv.org/pdf/2210.14037.pdf), Oct. 25 2022. `uncertainty`
- [Beyond Mahalanobis-Based Scores for Textual OOD Detection](https://openreview.net/pdf?id=ReB7CCByD6U), `nips2022`.
- [Out-Of-Distribution Detection Is Not All You Need](https://arxiv.org/abs/2211.16158), Nov. 29 2022. `nips2022`. [tweet](https://twitter.com/ducha_aiki/status/1597889329321213952).
- [Useful Confidence Measures: Beyond the Max Score](https://arxiv.org/pdf/2210.14070.pdf), Oct. 25 2022.
- [A Functional Data Perspective and Baseline On Multi-Layer Out-of-Distribution Detection](https://arxiv.org/pdf/2306.03522.pdf), Jun. 6 2023.
- [Unleashing Mask: Explore the Intrinsic Out-of-Distribution Detection Capability](https://arxiv.org/pdf/2306.03715.pdf), Jun. 6 2023.
- [Neuron Activation Coverage: Rethinking Out-of-distribution Detection and Generalization](https://arxiv.org/pdf/2306.02879.pdf), Jun. 5 2023.
- [No Free Lunch: The Hazards of Over-Expressive Representations in Anomaly Detection](https://arxiv.org/pdf/2306.07284.pdf), Jun. 12 2023.
- [Non-parameteric outlier synthesis](https://arxiv.org/pdf/2303.02966.pdf), Mar. 6 2023.
- [Dream the Impossible: Outlier Imagination with Diffusion Models](https://arxiv.org/pdf/2309.13415.pdf), Sep. 23 2023.

#### Unsupervised

- [WAIC, but Why? Generative Ensembles for Robust Anomaly Detection](https://www.alexalemi.com/publications/waic.pdf), May 23 2019.
- [Likelihood Ratios for Out-of-Distribution Detection](https://proceedings.neurips.cc/paper/2019/file/1e79596878b2320cac26dd792a6c51c9-Paper.pdf), `nips2019`. `generative`
- [Input complexity and out-of-distribution detection with likelihood-based generative models](https://arxiv.org/pdf/1909.11480.pdf), `iclr2020`.
  - _"likelihood derived from such models have been shown to be problematic for detecting certain types of inputs that significantly differ from training data"_
  - _"we pose that this problem is due to the excessive influence that input complexity has in generative models' likelihoods"_
  - _"score to perform comparably to or even better than existing OOD detection approaches under a wide range of datasets"_
- [Type of Out-of-Distribution Texts and How to Detect Them](https://arxiv.org/pdf/2109.06827.pdf), Sep. 2021. Udit Arora et al. `emnlp2021` `ood issue` `analysis`
  - Motivation is "there is little consensus on formal def. of OOD examples";
  - Propose a categorization of OOD instances according to ***background shift*** or ***semantic shift***
  - Methods like *calibration* and density estimation for *OOD detection* are evaluated over 14 datasets
- [Density of States Estimation for Out-of-Distribution Detection](http://proceedings.mlr.press/v130/morningstar21a/morningstar21a.pdf), `aistats2021`.
  - proposed DoSE, using concept - "frequency of any reasonable statistic", _"the frequency is calculated using nonparametric density estimators, e.g. KDE and one-class SVM"_
  - DoSE requires neigther labeled data nor OOD examples
- [Entropic Issues in Likelihood-Based OOD Detection](https://proceedings.mlr.press/v163/caterini22a/caterini22a.pdf), I (Still) Can’t Believe It’s Not Better Workshop at NeurIPS 2021, `nips2021`. 
  - Deep generative models can assign high probability to OOD data than ID data, why?
  - _"manifold-supported models"_ achieve success recently.
  - likelihood to be decomposed into KL divergence term + entropy term
    - Likelihood - $\mathcal{L}_\theta:= - KL - H$.
    - and likelihood ratio can cancel out the above entropy term.
- [On the Out-of-distribution Generalization of Probabilistic Image Modelling](https://proceedings.neurips.cc/paper/2021/file/1f88c7c5d7d94ae08bd752aa3d82108b-Paper.pdf), `nips2021`.
  - This paper also finds that likelihood can be misleading for OOD detection, since:
    - local features are shared between image distributions
    - local features dominate the likelihood
- [Unsupervised Anomaly Detection via Nonlinear Manifold Learning](https://arxiv.org/pdf/2306.09441.pdf), Jun. 15 2023.

### Explanations

- [Concept-based Explanations for Out-Of-Distribution Detectors](https://arxiv.org/pdf/2203.02586.pdf), Mar. 4 2022. `ood` `interpretability`

### OOD methods for sequential data

>  text, audio sequence.

- [A Survey on Out-of-Distribution Detection in NLP](https://arxiv.org/pdf/2305.03236.pdf), May 5 2023. `survey`.
- 🤍 [Likelihood Ratios and Generative Classifiers for Unsupervised Out-of-Domain Detection in Task Oriented Dialog](https://arxiv.org/abs/1912.12800), `aaai2020`.
  - dataset constribution: ROSTD (Real Out-of-domain Sentence from Task-oriented Dialogue), the greatness of ROSTD is that _"examples were authored by annotators with apriori instructions to be out-of-domain w.r.t. sentences in an existing dataset"_
  - Likelihood ratio based OOD detection methods is better than plain likelihood/density estimation methods (?)
  - Combination of generative and discriminative learning to outperform simple likelihood based methods
  - This work motivates a series methods for ood detection on task-oritend dialogue
    - [A Deep Generative Distance-Based Classifier for Out-of-Domain Detection with Mahalanobis Space](https://aclanthology.org/2020.coling-main.125.pdf), `coling2020`. `generaetive` `classifier-based`
    - [Evaluating the Practical Utility of Confidence-score based Techniques for Unsupervised Open-world Intent Classification](https://aclanthology.org/2022.insights-1.3.pdf), ACL workshop 2022.
- 🤍 [Pretrained Transformers Improve Out-of-Distribution Robustness](https://aclanthology.org/2020.acl-main.244/), `acl2020`. `unsupervised`
  - Although this paper is not devoted to OOD **detection**, it systematically measures out-of-distribution (OOD) generalization for **seven NLP datasets** by constructing a new robustness **benchmark** with realistic distribution shifts.
  - They found that _"Pretrained transformers are also more effective at detecting anomalous or OOD examples"_.
  - Distillation can harm robustness, and more diverse pretraining data can enhance robustness.
- 🤍 [Contrastive Out-of-Distribution Detection for Pretrained Transformers](https://aclanthology.org/2021.emnlp-main.84.pdf), `emnlp2021`. [github](https://github.com/wzhouad/Contra-OOD). `unsupervised` `classfier-based`
  - _Research question_: How to identify semantic drift in real world scenario for text processing models?
  - The method finetunes Transformers with a contrastive loss and this can improve the compactness of representations.
  - _Mahalanobis distance_ is used (on the representation at the penultimate layer)
  - The drastic gain might be resulted form **margin-based** CL loss for compactness of representation of text.
  - Experiments are conducted on SST2, IMDB, TREC-10, 20NG datasets.
- 🤍 [Types of Out-of-Distribution Texts and How to Detect Them](https://aclanthology.org/2021.emnlp-main.835.pdf), `emnlp2021`.
  - _"Across 14 pairs of in-distribution and OOD English natural language understanding datasets, we find that density estimation methods consistently beat calibration methods in **background shift** settings, while perform worse in **semantic shift** settings"_
- [PnPOOD : Out-Of-Distribution Detection for Text Classification via Plug and Play Data Augmentation](https://arxiv.org/pdf/2111.00506.pdf), Oct. 31 2021. `workshop` of `icml2021`.
- [Novelty Detection: A Perspective from Natural Language Processing](https://aclanthology.org/2022.cl-1.3.pdf), `cl2021`.
- [On Out-of-Distribution Detection for Audio with Deep Nearest Neighbors](https://arxiv.org/pdf/2210.15283.pdf), Oct. 27 2022.
- [Towards Textual Out-of-Domain Detection without any In-Domain Labels](https://neurips2021-nlp.github.io/papers/4/CameraReady/OOD_ENLSP_NeurIPS_workshop_unsupervised.pdf), `taslp2022`.

### OOD methods for time-series data

> behavior sequence on E-commerce platform, time series etc.

**Surveys**

- [Anomaly detection for discrete sequences: a survey](https://conservancy.umn.edu/bitstream/handle/11299/215802/09-015.pdf?sequence=1&isAllowed=y), 2010. `citation: 684`.
- [Outlier detection for temporal data: a survey](https://romisatriawahono.net/lecture/rm/survey/machine%20learning/Gupta%20-%20Outlier%20Detection%20for%20Temporal%20Data%20-%202014.pdf), 2013. `citation: 976`.
- [Anomaly Detection for IoT Time-Series Data: A Survey](https://eprints.keele.ac.uk/id/eprint/7576/1/08926446.pdf), 2019. `citation: 241`.
- [A review on outlier/anomaly detection in time series data](https://arxiv.org/pdf/2002.04236.pdf), Feb. 11 2020. `citation: 274`.
  - This review provides a structured and comprehensive state-of-the-art on outlier detection techniques in the context of time series data.
  - The detection of outliers or anomalies that may represent errors or events of interest is of critical importance.
- [Anomaly detection in univariate time-series: a survey on the state-of-the-art](https://arxiv.org/pdf/2004.00433.pdf), Apr. 1 2020. `citation: 122`.
- [Deep Learning for Anomaly Detection in Time-Series Data: Review, Analysis, and Guidelines](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9523565), `IEEE Acess 2021`.

**Benchmarks and toolkits**

- [Evaluating Real-time Anomaly Detection Algorithms - the Numenta Anomaly Benchmark](https://arxiv.org/abs/1510.03336), Oct. 12 2015. [The Numenta Anomaly Benchmark (NAB)](https://github.com/numenta/NAB).
- [Revisiting Time Series Outlier Detection: Definitions and Benchmarks](https://openreview.net/pdf?id=r8IvOsnHchr), `nips2021`.
- [Exathlon: a benchmark for explainable anomaly detection over time series](http://vldb.org/pvldb/vol14/p2613-tatbul.pdf), `vldb2021`.
- [Towards a Rigorous Evaluation of Time-series Anomaly Detection](https://arxiv.org/pdf/2109.05257.pdf), Jan. 4 2022. `aaai2022`. `citation: 15`. [code](https://github.com/tuslkkk/tadpak).
- [TimeEval: a benchmarking toolkit for time series anomaly detection algotithms](https://hpi.de/fileadmin/user_upload/fachgebiete/naumann/publications/PDFs/2022_wenig_timeeval.pdf), `vldb2022`.
- [AnomalyKiTS: Anomaly Detection Toolkit for Time Series](https://ojs.aaai.org/index.php/AAAI/article/view/21730), `aaai2022`.
- [Local Evaluation of Time Series Anomaly Detection Algorithms](https://arxiv.org/pdf/2206.13167.pdf), `kdd2022`.

**Methods**

- [Generic and Scalable Framework for Automated Time-series Anomaly Detection](https://netman.aiops.org/~peidan/ANM2021/5.KPIAnomalyDetection/ReadingLists/2015KDD_Generic%20and%20Scalable%20Framework%20for%20Automated%20Time-series%20Anomaly%20Detection.pdf), `kdd2015`. `citation: 420`. [code based on java: EGADS](https://github.com/yahoo/egads).
- [Long Short Term Memory Networks for Anomaly Detection in Time Series](https://www.researchgate.net/profile/Mohamed-Mourad-Lafifi/post/Optimizing-Long-Short-Term-Memory-Model-CNN-for-anomaly-detection/attachment/5f46fa6bce377e00016f45e8/AS%3A928935898542080%401598486985261/download/Long+Short+Term+Memory+Networks+for+Anomaly+Detection+in+Time+Series.pdf), 2017. `citation: 1309`.
- [Unsupervised real-time anomaly detection for streaming data](https://www.sciencedirect.com/science/article/pii/S0925231217309864), Nov. 2017. `Neurocomputing`. `citation: 744`. [code](https://github.com/numenta/NAB). [blog](https://numenta.github.io/numenta-web/papers/unsupervised-real-time-anomaly-detection-for-streaming-data/).
- [Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding](https://arxiv.org/pdf/1802.04431.pdf), Jun. 6 2018. [code](https://github.com/khundman/telemanom). `citation: 566`.
- [A Deep Neural Network for Unsupervised Anomaly Detection and Diagnosis in Multivariate Time Series Data](https://arxiv.org/abs/1811.08055), Nov. 2018. `citation: 422`. [code-tf](https://github.com/7fantasysz/MSCRED), [code-pt](https://github.com/SKvtun/MSCRED-Pytorch) not reproducible.
- [DeepAnT: A Deep Learning Approach for Unsupervised Anomaly Detection in Time Series](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8581424), IEEE Access 2018. `citation: 336`. might-be-useful [code](https://github.com/swlee052/deep-learning-time-series-anomaly-detection), [code](https://github.com/dev-aadarsh/DeepAnT), [code](https://github.com/bmonikraj/medium-ds-unsupervised-anomaly-detection-deepant-lstmae) and [code](https://github.com/datacubeR/DeepAnt).
- [Anomaly Detection with Generative Adversarial Networks for Multivariate Time Series](https://arxiv.org/abs/1809.04758), Sep. 13 2018. `citation: 220`. [code](https://github.com/LiDan456/GAN-AD).
- [Lifelong Anomaly Detection Through Unlearning](https://dl.acm.org/doi/pdf/10.1145/3319535.3363226), `ccs2019`.
- [Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network](https://netman.aiops.org/wp-content/uploads/2019/07/OmniAnomaly_REPRESENTATION.pdf), `kdd2019`. `citation: 389`. [code](https://github.com/NetManAIOps/OmniAnomaly).
- [Time-Series Anomaly Detection Service at Microsof](https://arxiv.org/pdf/1906.03821.pdf), `kdd2019`. `citation: 248`.
- [USAD: UnSupervised Anomaly Detection on Multivariate Time Series](), `kdd2020`. `citation: 173`. [code](https://github.com/manigalati/usad).
- [Multivariate Time-series Anomaly Detection via Graph Attention Network](https://arxiv.org/abs/2009.02040), Sep. 4 2020. `icdm2020`. `citation: 129`. [code](https://github.com/ML4ITS/mtad-gat-pytorch).
- [TadGAN: Time Series Anomaly Detection Using Generative Adversarial Networks](https://arxiv.org/pdf/2009.07769.pdf), Nov. 14 2020. [github](https://github.com/sintel-dev/Orion). `citation: 100+`.
- [Neural Contextual Anomaly Detection for Time Series](https://arxiv.org/pdf/2107.07702.pdf), `nips2021`. [github](https://github.com/Francois-Aubet/gluon-ts/tree/adding_ncad_to_nursery/src/gluonts/nursery/ncad). `citation: 10+`
- [Anomaly Transformer: Time series anomaly detection with association discrepancy](https://arxiv.org/pdf/2110.02642.pdf), Jan. 29 2022. `citation: 60`. [code](https://github.com/thuml/Anomaly-Transformer).

**Time-series representation learning**

- [Time-series Generative Adversarial Networks](https://proceedings.neurips.cc/paper/2019/file/c9efe5f26cd17ba6216bbe2a7d26d490-Paper.pdf), `nips2019`. `citation: 396`.
- [A Transformer-based Framework for Multivariate Time Series Representation Learning](https://dl.acm.org/doi/pdf/10.1145/3447548.3467401), `kdd2021`. [code](https://github.com/gzerveas/mvts_transformer). `citation: 170`.
- [TS2Vec: Towards Universal Representation of Time Series](https://arxiv.org/abs/2106.10466), `aaai2022`. `citation: 23`.

**Forecasting**

- [Time Series Forecasting With Deep Learning: A Survey](https://arxiv.org/pdf/2004.13408.pdf), Sep. 27 2020.

**Methods for low-resource**

> This might be related to anomaly detection in time-series data, but related on time series processing in low-resource scenario.

- [A survey on heterogeneous transfer learning](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-017-0089-0#Sec17), Journal of Big Data, 2017.
- [Transfer learning for time series classification](https://arxiv.org/abs/1811.01533), Nov. 5 2018.
- [Reconstruction and Regression Loss for Time-Series Transfer Learning](https://kdd-milets.github.io/milets2018/papers/milets18_paper_2.pdf)， `kdd2018`.
- [Time Series Anomaly Detection Using Convolutional Neural Networks and Transfer Learning](https://arxiv.org/pdf/1905.13628.pdf), May 31 2019. `aaai2019`.
- [Time Series Data Augmentation for Deep Learning: A Survey](https://arxiv.org/pdf/2002.12478.pdf), Feb. 27 2020. `data augmentation`.
- [Meta-learning framework with applications to zero-shot time-series forecasting](https://arxiv.org/pdf/2002.02887.pdf), Dec. 14 2020. `transfer learning`. [github](https://github.com/Nixtla/transfer-learning-time-series).
- [Unsupervised transfer learning for anomaly detection: Application to complementary operating condition transfer](https://www.sciencedirect.com/science/article/pii/S0950705121000794), 2021.
- [Implementing transfer learning across different datasets for time series forecasting](https://www.sciencedirect.com/science/article/pii/S0031320320304209), 2021.
- [Spacecraft Time-Series Anomaly Detection Using Transfer Learning](https://openaccess.thecvf.com/content/CVPR2021W/AI4Space/papers/Baireddy_Spacecraft_Time-Series_Anomaly_Detection_Using_Transfer_Learning_CVPRW_2021_paper.pdf), 2021. `transfer learning`.
- [Forecasting adverse surgical events using self-supervised transfer learning for physiological signals](https://www.nature.com/articles/s41746-021-00536-y), 2021.
- [What makes instance discrimination good for transfer learning](http://nxzhao.com/projects/good_transfer/Good_transfer_ICLR21.pdf), `iclr2021`.
- [Intra-domain and cross-domain transfer learning for time series data – How transferable are the features?](https://arxiv.org/pdf/2201.04449.pdf), Jan. 13 2022.

### Toolkits

- [PyTorch-OOD: A Library for Out-of-Distribution Detection based on PyTorch](https://openaccess.thecvf.com/content/CVPR2022W/HCIS/papers/Kirchheim_PyTorch-OOD_A_Library_for_Out-of-Distribution_Detection_Based_on_PyTorch_CVPRW_2022_paper.pdf), `cvpr2022`.

---

## Robustness

### Surveys

- [Measure and Improve Robustness in NLP Models: A Survey](https://arxiv.org/abs/2112.08313), Dec. 15 2021.
- [Robust Natural Language Processing: Recent Advances, Challenges, and Future Directions](https://arxiv.org/pdf/2201.00768.pdf), Jan. 3 2021.

### Theories

- [A Universal Law of Robustness via Isoperimetry](https://nips.cc/virtual/2021/poster/27813), `nips2021` outstanding paper.
- [Robustness Implies Privacy in Statistical Estimation](https://arxiv.org/abs/2212.05015), Dec. 9 2022.
  - relationship between differential privacy and adversarial robustness.

### Adversarial training

- [Transductive Robust Learning Guarantees](https://arxiv.org/pdf/2110.10602.pdf), Oct. 20 2021, Nathan Srebro's group. `VC Dimension`
  - adversarially robust learning in the transductive learning setting
  - theory-paper
- [Indicators of Attack Failure: Debugging and Improving Optimization of Adversarial Examples](https://arxiv.org/pdf/2106.09947.pdf), Jun. 18 2021. `nips2021`
- [Distributionally Robust Recurrent Decoders with Random Network Distillation](https://arxiv.org/pdf/2110.13229.pdf), Oct. 25 2021.
- [Disrupting Deep Uncertainty Estimation Without Harming Accuracy](https://arxiv.org/abs/2110.13741), Oct. 26 2021. `uncertainty` `adversarial`
- [How and When Adversarial Robustness Transfers in Knowledge Distillation?](https://arxiv.org/pdf/2110.12072.pdf), Oct. 22 2021. `knowledge distillation` `robustness transfer`
- [Robustness of Graph Neural Networks at Scale](https://arxiv.org/pdf/2110.14038.pdf), Oct. 26 2021. `gnn`
- [Clustering Effect of (Linearized) Adversarial Robust Models](https://arxiv.org/pdf/2111.12922.pdf), Nov. 25 2021 `nips2021`
- [PixMix: Dreamlike Pictures Comprehensively Improve Safety Measures](https://openreview.net/pdf?id=WeUg_KpkFtt), `nips2021` workshop paper
- [Quantifying and Understanding Adversarial Examples in Discrete Input Spaces](https://arxiv.org/pdf/2112.06276.pdf), Dec. 12 2021.
- [On the Convergence and Robustness of Adversarial Training](https://arxiv.org/pdf/2112.08304.pdf), Dec. 15 2021.
- [Nash Equilibria and Pitfalls of Adversarial Training in Adversarial Robustness Games](https://arxiv.org/pdf/2210.12606.pdf), `aistats2022`.

### Attacks

- [Adversarial Preprocessing: Understanding and Preventing Image-Scaling Attacks in Machine Learning](https://www.usenix.org/system/files/sec20fall_quiring_prepub.pdf).
- [Manipulating SGD with Data Ordering Attacks](https://arxiv.org/pdf/2104.09667.pdf), Jun. 5 2021.
- [On Distinctive Properties of Universal Perturbations](https://arxiv.org/pdf/2112.15329.pdf), Dec. 31 2021.

### Certified Robustness

- [SAFER: A Structure-free Approach for Certified Robustness to Adversarial Word Substitutions](https://aclanthology.org/2020.acl-main.317.pdf), `acl2021`
- [Certifiable Robustness and Robust Training for Graph Convolutional Networks](https://arxiv.org/pdf/1906.12269.pdf), `kdd2019`
- [Collective Robustness Certificates: Exploiting Interdependence in Graph Neural Networks](https://openreview.net/forum?id=ULQdiUTHe3y), `iclr2021`
- [Barack: Partially Supervised Group Robustness with Guarantees](https://arxiv.org/pdf/2201.00072.pdf), Dec. 31 2021.

### Randomness transformation defenses

- [Demystifying the Adversarial Robustness of Random Transformation Defenses](https://openreview.net/pdf?id=p4SrFydwO5), Dec. 16 2021.

### Spurious correlation

- [Last Layer Re-Training is Sufficient for Robustness to Spurious Correlations](https://arxiv.org/pdf/2204.02937.pdf), Apr. 6 2022.
- [Understanding Rare Spurious Correlations in Neural Networks](https://arxiv.org/abs/2202.05189), Feb. 10 2022.
- [Are All Spurious Features in Natural Language Alike? An Analysis through a Causal Lens](https://arxiv.org/abs/2210.14011), Oct. 25 2022. `emnlp2022`.
