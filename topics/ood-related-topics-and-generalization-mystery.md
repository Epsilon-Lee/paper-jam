
# OOD-related Research Topics and Generalization Mystery

Contents of this archive:

- [Out-of-Distribution Learning and Generalization](#out-of-distribution-ood-learning-and-generalization)
  - [Survey](#survey)
  - [OOD generalization theories](#ood-generalization-theories)
  - [Empirical observations](#empirical-observations)
  - [OOD generalization methods](#ood-generalization-methods)
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
- [Invariance Principle Meets Information Bottleneck for Out-of-Distribution Generalization](https://arxiv.org/pdf/2106.06607.pdf), Jun. 11 2021. `nips2021` `invariant risk minimization`
- [Failure Modes of Domain Generalization Algorithms](https://arxiv.org/pdf/2111.13733.pdf), Nov. 26 2021.
- [A Theory of Label Propagation for Subpopulation Shift](https://arxiv.org/pdf/2102.11203.pdf), Jul. 20 2021 `distribution shift`
- [Monotonic Risk Relationships under Distribution Shifts for Regularized Risk Minimization](https://arxiv.org/pdf/2210.11589.pdf), Oct. 20 2022.
- [On-Demand Sampling: Learning Optimally from Multiple Distributions](fhttps://arxiv.org/pdf/2210.12529.pdf), arXiv Oct. 22 2022.  

### Empirical observations

- [Understanding Out-of-distribution: A Perspective of Data Dynamics](https://arxiv.org/pdf/2111.14730.pdf), Nov. 29 2021.
- [Characterizing Generalization under Out-Of-Distribution Shifts in Deep Metric Learning](https://arxiv.org/pdf/2107.09562.pdf), Jul. 20 2021.
- [Understanding the failure modes of out-of-distribution generalization](https://openreview.net/forum?id=fSTD6NFIW_b), `iclr2021`
- [Out-of-Distribution Generalization with Deep Equilibrium Models](http://www.gatsby.ucl.ac.uk/~balaji/udl2021/accepted-papers/UDL2021-paper-072.pdf), 2021 icml workshop.
- [Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution](https://arxiv.org/pdf/2202.10054.pdf), Feb. 21 2022.

### OOD generalization methods

- [Deep Stable Learning for Out-Of-Distribution Generalization](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Deep_Stable_Learning_for_Out-of-Distribution_Generalization_CVPR_2021_paper.pdf), `cvpr2021`
- [LINDA: Unsupervised Learning to Interpolate in Natural Language Processing](https://arxiv.org/pdf/2112.13969.pdf), Dec. 28 2021. 
- [Counterfactual Maximum Likelihood Estimation for Training Deep Networks](https://openreview.net/forum?id=o6s1b_-nDOE), `nips2021`
- [Improving Out-of-Distribution Robustness via Selective Augmentation](https://arxiv.org/pdf/2201.00299.pdf), Jan. 2 2022.
- [On Distributionally Robust Optimization and Data Rebalancing](https://www.researchgate.net/profile/Agnieszka-Slowik-5/publication/358338470_On_Distributionally_Robust_Optimization_and_Data_Rebalancing/links/61fc4ca94393577abe0d75cc/On-Distributionally-Robust-Optimization-and-Data-Rebalancing.pdf), Feb. 2022.
- [Learning Causal Semantic Representation for Out-of-Distribution Prediction](https://proceedings.neurips.cc/paper/2021/file/310614fca8fb8e5491295336298c340f-Paper.pdf), `nips2021`.
- [Generalization and Robustness Implications in Object-Centric Learning](https://arxiv.org/pdf/2107.00637.pdf), `icml2021`.
- [Counterfactual Invariance to Spurious Correlations: Why and How to Pass Stress Tests](https://proceedings.neurips.cc/paper/2021/file/8710ef761bbb29a6f9d12e4ef8e4379c-Paper.pdf), `nips2021`.
- [Visual Representation Learning Does Not Generalize Strongly within the Same Domain](https://arxiv.org/pdf/2107.08221.pdf&lt;/p&gt;), `iclr2022`.
- [Probable Domain Generalization via Quantile Risk Minimization](https://arxiv.org/abs/2207.09944), Jul. 20 2022. [github](https://github.com/cianeastwood/qrm).
- [Distributionally Robust Losses for Latent Covariate Mixtures](https://arxiv.org/pdf/2007.13982.pdf), arXiv.v2 Aug. 10 2022.

#### Invariant risk minimization

The application from principle of **Invariance**.

- [DAIR: Data Augmented Invariant Regularization](https://arxiv.org/pdf/2110.11205.pdf), Oct. 21 2021.
- [IRM - when it works and when it doesn't: A test case of natural language inference](https://openreview.net/forum?id=KtvHbjCF4v), `nips2021`.
- [Heterogeneous Risk Minimization](https://proceedings.mlr.press/v139/liu21h.html), `icml2021`.
- [Kernelized Heterogeneous Risk Minimizaton](https://pengcui.thumedialab.com/papers/KernelHRM.pdf), `nips2021`.
- [Enviroment Inference for Invariant Learning](http://proceedings.mlr.press/v139/creager21a/creager21a.pdf), `icml2021`.
- [Fishr: Invariant Gradient Variances for Out-of-Distribution Generalization](https://proceedings.mlr.press/v162/rame22a/rame22a.pdf), `icml2022`.

#### Benchmarks

- [WILDS: A Benchmark of in-the-Wild Distribution Shifts](https://arxiv.org/abs/2012.07421), Dec. 14 2020. [icml version](http://proceedings.mlr.press/v139/koh21a/koh21a.pdf).
- [Extending the WILDS Benchmark for Unsupervised Adaptation](https://arxiv.org/pdf/2112.05090.pdf), Dec. 9 2021.
- [Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks](https://arxiv.org/pdf/2107.07455.pdf), Jul. 23 2021.
- [Extending the WILDS Benchmark for Unsupervised Adaptation](https://arxiv.org/abs/2112.05090), arXiv.v1 Dec. 9 2021.

#### Causal representation learning

- [Interventional Causal Representation Learning](https://arxiv.org/pdf/2209.11924.pdf), Sep. 24 2022.
- [Temporally Disentangled Representation Learning](https://arxiv.org/pdf/2210.13647.pdf), Oct. 24 2022.

### Distribution Shift

- [Detecting and Correcting for Label Shift with Black Box Predictions](https://proceedings.mlr.press/v80/lipton18a.html), `icml2018`. `label shift`
- [Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift](https://proceedings.neurips.cc/paper/2019/file/846c260d715e5b854ffad5f70a516c88-Paper.pdf), `nips2019`.
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

#### Detection of distribution shift

- [Feature shift detection: Localizing which features have shifted via conditional distribution tests](https://proceedings.neurips.cc/paper/2020/file/e2d52448d36918c575fa79d88647ba66-Paper.pdf), `nips2020`.
- ["Why did the model fail?": Attributing Model Performance Changes to Distributional Shifts](https://arxiv.org/pdf/2210.10769.pdf), Oct. 19 2022.
- [Explanation Shift: Detecting distribution shifts on tabular data via the explanation space](https://arxiv.org/pdf/2210.12369.pdf), Oct. 22 2022.

#### Unsupervised Domain Adaptation

- [A Prototype-Oriented Framework for Unsupervised Domain Adaptation](https://proceedings.neurips.cc/paper/2021/file/8edd72158ccd2a879f79cb2538568fdc-Paper.pdf), `nips2021`.

#### Test-time adaptation and inference

- [Test-time recalibration of conformal predictors under distribution shift based on unlabeled examples](https://arxiv.org/pdf/2210.04166.pdf), Oct. 9 2022. `conformal prediction`
- [Memory-Based Model Editing at Scale](https://proceedings.mlr.press/v162/mitchell22a/mitchell22a.pdf), `icml2022`.
- [Test-time Adaptation via Self-Training with Nearest Neighbor Information](https://openreview.net/pdf?id=EzLtB4M1SbM), `iclr2023 submitted`.

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
- [Understanding Out-of-distribution:A Perspective of Data Dynamics](https://proceedings.mlr.press/v163/adila22a/adila22a.pdf), I (Still) Can’t Believe It’s Not Better Workshop at NeurIPS 2021.
- [Provable Guarantees for Understanding Out-of-distribution Detection](https://arxiv.org/pdf/2112.00787.pdf), Dec. 1 2022. `aaai2022`

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
- [Likelihood Ratios for Out-of-Distribution Detection](https://proceedings.neurips.cc/paper/2019/file/1e79596878b2320cac26dd792a6c51c9-Paper.pdf), `nips2019`. `generative`
- [Detecting Out-of-Distribution Examples with Gram Matrices](http://proceedings.mlr.press/v119/sastry20a/sastry20a.pdf), `icml2020`. [github](https://github.com/VectorInstitute/gram-ood-detection).
  - xxx
- [Contrastive Training for Improved Out-of-Distribution Detection](https://arxiv.org/pdf/2007.05566.pdf), arXiv Jul. 10 2021. `unsupervised`
  - `Confusion Log Probability`
  - This paper proves that CLP score especially improves near ood classes.
- [On the Importance of Gradients for Detecting Distributional Shifts in the Wild](https://arxiv.org/pdf/2110.00218.pdf), Oct. 9 2021. `ood detection`
- [Identifying and Benchmarking Natural Out-of-Context Prediction Problems](https://arxiv.org/pdf/2110.13223.pdf), Oct. 25 2021.
- [A Fine-grained Analysis on Distribution Shift](https://arxiv.org/pdf/2110.11328.pdf), Oct. 21 2021.
- [Understanding the Role of Self-Supervised Learning in Out-of-Distribution Detection Task](https://arxiv.org/pdf/2110.13435.pdf), Oct. 26 2021.
- [Exploring the Limits of Out-of-Distribution Detection](https://arxiv.org/pdf/2106.03004.pdf), `landscape` `nips2021`
- [Type of Out-of-Distribution Texts and How to Detect Them](https://arxiv.org/pdf/2109.06827.pdf), Sep. 2021. Udit Arora et al. `emnlp2021` `ood issue` `analysis`
  - Motivation is "there is little consensus on formal def. of OOD examples";
  - Propose a categorization of OOD instances according to ***background shift*** or ***semantic shift***
  - Methods like *calibration* and density estimation for *OOD detection* are evaluated over 14 datasets
- [Can multi-label classification networks know what they don’t know?](https://arxiv.org/pdf/2109.14162.pdf), Sep. 2021. `ood detection`
- [Provable Guarantees for Understanding Out-of-distribution Detection](https://arxiv.org/pdf/2112.00787.pdf), Dec. 1 2021.
- [ReAct: Out-of-distribution Detection With Rectified Activations](https://arxiv.org/abs/2111.12797), Nov. 24 2021.
- [On the Impact of Spurious Correlation for Out-of-distribution Detection](https://arxiv.org/abs/2109.05642), Sep. 12 2021.
- [Entropic Issues in Likelihood-Based OOD Detection](https://proceedings.mlr.press/v163/caterini22a/caterini22a.pdf), I (Still) Can’t Believe It’s Not Better Workshop at NeurIPS 2021, `nips2021`. 
  - Deep generative models can assign high probability to OOD data than ID data, why?
  - _"manifold-supported models"_ achieve success recently.
  - likelihood to be decomposed into KL divergence term + entropy term
    - Likelihood - $\mathcal{L}_\theta:= - KL - H$.
    - and likelihood ratio can cancel out the above entropy term.
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

### Explanations

- [Concept-based Explanations for Out-Of-Distribution Detectors](https://arxiv.org/pdf/2203.02586.pdf), Mar. 4 2022. `ood` `interpretability`

### OOD methods for sequential data

>  text, behavior sequence on E-commerce platform, time series etc.

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
- [On Out-of-Distribution Detection for Audio with Deep Nearest Neighbors](https://arxiv.org/pdf/2210.15283.pdf), Oct. 27 2022.
- [Towards Textual Out-of-Domain Detection without any In-Domain Labels](https://neurips2021-nlp.github.io/papers/4/CameraReady/OOD_ENLSP_NeurIPS_workshop_unsupervised.pdf), `taslp2022`

### Toolkits

- [PyTorch-OOD: A Library for Out-of-Distribution Detection based on PyTorch](https://openaccess.thecvf.com/content/CVPR2022W/HCIS/papers/Kirchheim_PyTorch-OOD_A_Library_for_Out-of-Distribution_Detection_Based_on_PyTorch_CVPRW_2022_paper.pdf), `cvpr2022`.

---

## Robustness

### Surveys

- [Measure and Improve Robustness in NLP Models: A Survey](https://arxiv.org/abs/2112.08313), Dec. 15 2021.
- [Robust Natural Language Processing: Recent Advances, Challenges, and Future Directions](https://arxiv.org/pdf/2201.00768.pdf), Jan. 3 2021.

### Theories

- [A Universal Law of Robustness via Isoperimetry](https://nips.cc/virtual/2021/poster/27813), `nips2021` outstanding paper

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
