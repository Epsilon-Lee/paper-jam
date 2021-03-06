
# OOD-related Research Topics and Generalization Mystery

Contents of this archive:

- [Out-of-Distribution Learning and Generalization](#out-of-distribution-ood-learning-and-generalization)
  - [OOD generalization theories](#ood-generalization-theories)
  - [Empirical observations](#empirical-observations)
  - [OOD generalization methods](#ood-generalization-methods)
    - [Invariant risk minimization](#invariant-risk-minimization)
    - [Benchmarks](#benchmarks)
  - [Connection between OOD and adversarials](#connection-between-ood-and-adversarials)
- [Out-of-Distribution Detection](#out-of-distribution-detection)
  - [Surveys](#surveys)
  - [Methods](#methods)
- [Robustness](#robustness)
  - [Surveys](#surveys)
  - [Theories](#theories)
  - [Adversarial training](#adversarial-training)
  - [Attacks](#attacks)

## Out-of-Distribution (OOD) Learning and Generalization

### OOD generalization theories

- [Towards a theory of out-of-distribution learning](https://arxiv.org/pdf/2109.14501.pdf), JHU 2021.
- [Invariance Principle Meets Information Bottleneck for Out-of-Distribution Generalization](https://arxiv.org/pdf/2106.06607.pdf), Jun. 11 2021. `nips2021` `invariant risk minimization`
- [Failure Modes of Domain Generalization Algorithms](https://arxiv.org/pdf/2111.13733.pdf), Nov. 26 2021.
- [A Theory of Label Propagation for Subpopulation Shift](https://arxiv.org/pdf/2102.11203.pdf), Jul. 20 2021 `distribution shift`

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
- [Optimal Representations for Covariate Shift](https://arxiv.org/pdf/2201.00057.pdf), Dec. 31 2021.
- [On Distributionally Robust Optimization and Data Rebalancing](https://www.researchgate.net/profile/Agnieszka-Slowik-5/publication/358338470_On_Distributionally_Robust_Optimization_and_Data_Rebalancing/links/61fc4ca94393577abe0d75cc/On-Distributionally-Robust-Optimization-and-Data-Rebalancing.pdf), Feb. 2022.

#### Invariant Risk Minimization

- [DAIR: Data Augmented Invariant Regularization](https://arxiv.org/pdf/2110.11205.pdf), Oct. 21 2021.
- [IRM - when it works and when it doesn't: A test case of natural language inference](https://openreview.net/forum?id=KtvHbjCF4v), `nips2021`

#### Benchmarks

- [WILDS: A Benchmark of in-the-Wild Distribution Shifts](https://arxiv.org/abs/2012.07421), Dec. 14 2020.
- [Extending the WILDS Benchmark for Unsupervised Adaptation](https://arxiv.org/pdf/2112.05090.pdf), Dec. 9 2021.
- [Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks](https://arxiv.org/pdf/2107.07455.pdf), Jul. 23 2021.

### Distribution Shfit

- [On Label Shift in Domain Adaptation via Wasserstein Distance](https://arxiv.org/pdf/2110.15520.pdf), Oct. 29 2021.
- [An Information-theoretic Approach to Distribution Shifts](https://arxiv.org/pdf/2106.03783.pdf), Nov. 1 2021. `nips2021`
- [An opinioin from Cho about distributional robustness](https://twitter.com/kchonyc/status/1455619054786519045), Nov. 3 2021.

### Connection Between OOD and Adversarials

- [Rethinking Machine Learning Robustness via its Link with the Out-of-Distribution Problem](https://arxiv.org/pdf/2202.08944.pdf), Feb. 18 2022.


## Out-of-Distribution Detection

### Surveys

- [Generalized Out-of-Distribution Detection: A Survey](https://arxiv.org/pdf/2110.11334.pdf), Oct. 21 2021.
- [A Unifying Review of Deep and Shallow Anomaly Detection](https://arxiv.org/abs/2009.11732), Sep. 24 2020.

### Methods

- [On the Importance of Gradients for Detecting Distributional Shifts in the Wild](https://arxiv.org/pdf/2110.00218.pdf), Oct. 9 2021. `ood detection`
- [A Unified Survey on Anomaly, Novelty, Open-Set, and Out-of-Distribution Detection: Solutions and Future Challenges](https://arxiv.org/pdf/2110.14051.pdf), Oct. 26 2021.
- [Identifying and Benchmarking Natural Out-of-Context Prediction Problems](https://arxiv.org/pdf/2110.13223.pdf), Oct. 25 2021.
- [A Fine-grained Analysis on Distribution Shift](https://arxiv.org/pdf/2110.11328.pdf), Oct. 21 2021.
- [Understanding the Role of Self-Supervised Learning in Out-of-Distribution Detection Task](https://arxiv.org/pdf/2110.13435.pdf), Oct. 26 2021.
- [Exploring the Limits of Out-of-Distribution Detection](https://arxiv.org/pdf/2106.03004.pdf), `landscape` `nips2021`
- [Type of Out-of-Distribution Texts and How to Detect Them](https://arxiv.org/pdf/2109.06827.pdf), Sep. 2021. Udit Arora et al. `emnlp2021` `ood issue` `analysis`
  - Motivation is "there is little consensus on formal def. of OOD examples";
  - Propose a categorization of OOD instances according to ***background shift*** or ***semantic shift***
  - Methods like *calibration* and density estimation for *OOD detection* are evaluated over 14 datasets
- [Can multi-label classification networks know what they don???t know?](https://arxiv.org/pdf/2109.14162.pdf), Sep. 2021. `ood detection`
- [Provable Guarantees for Understanding Out-of-distribution Detection](https://arxiv.org/pdf/2112.00787.pdf), Dec. 1 2021.
- [ReAct: Out-of-distribution Detection With Rectified Activations](https://arxiv.org/abs/2111.12797), Nov. 24 2021.
- [On the Impact of Spurious Correlation for Out-of-distribution Detection](https://arxiv.org/abs/2109.05642), Sep. 12 2021.
- [VOS: Learning What You Don't Know by Virtual Outlier Synthesis](https://arxiv.org/abs/2202.01197), `iclr2022` [code](https://github.com/deeplearning-wisc/vos).

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

