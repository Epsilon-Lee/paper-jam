
## OOD-related Research

### OOD generalization theory and methods

#### Theory
- [Towards a theory of out-of-distribution learning](https://arxiv.org/pdf/2109.14501.pdf), JHU 2021.
- [Invariance Principle Meets Information Bottleneck for Out-of-Distribution Generalization](https://arxiv.org/pdf/2106.06607.pdf), Jun. 11 2021. `nips2021` `invariant risk minimization`
- [Failure Modes of Domain Generalization Algorithms](https://arxiv.org/pdf/2111.13733.pdf), Nov. 26 2021.

#### Empiricals

- [Understanding Out-of-distribution: A Perspective of Data Dynamics](https://arxiv.org/pdf/2111.14730.pdf), Nov. 29 2021.
- [Characterizing Generalization under Out-Of-Distribution Shifts in Deep Metric Learning](https://arxiv.org/pdf/2107.09562.pdf), Jul. 20 2021.
- [Understanding the failure modes of out-of-distribution generalization](https://openreview.net/forum?id=fSTD6NFIW_b), `iclr2021`

#### Invariant Risk Minimization

- [DAIR: Data Augmented Invariant Regularization](https://arxiv.org/pdf/2110.11205.pdf), Oct. 21 2021.
- [IRM---when it works and when it doesn't: A test case of natural language inference](https://openreview.net/forum?id=KtvHbjCF4v), `nips2021`

### OOD Detection

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
- [Can multi-label classification networks know what they don’t know?](https://arxiv.org/pdf/2109.14162.pdf), Sep. 2021. `ood detection`
- [Provable Guarantees for Understanding Out-of-distribution Detection](https://arxiv.org/pdf/2112.00787.pdf), Dec. 1 2021.

### Robustness and Adversarial Examples/Training

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

#### Attacks

- [Adversarial Preprocessing: Understanding and Preventing Image-Scaling Attacks in Machine Learning](https://www.usenix.org/system/files/sec20fall_quiring_prepub.pdf).
- [Manipulating SGD with Data Ordering Attacks](https://arxiv.org/pdf/2104.09667.pdf), Jun. 5 2021.

### Dataset Shift

- [On Label Shift in Domain Adaptation via Wasserstein Distance](https://arxiv.org/pdf/2110.15520.pdf), Oct. 29 2021.
- [An Information-theoretic Approach to Distribution Shifts](https://arxiv.org/pdf/2106.03783.pdf), Nov. 1 2021. `nips2021`
- [An opinioin from Cho about distributional robustness](https://twitter.com/kchonyc/status/1455619054786519045), Nov. 3 2021.

### Certified Robustness

- [SAFER: A Structure-free Approach for Certified Robustness to Adversarial Word Substitutions](https://aclanthology.org/2020.acl-main.317.pdf), `acl2021`
- [Certifiable Robustness and Robust Training for Graph Convolutional Networks](https://arxiv.org/pdf/1906.12269.pdf), `kdd2019`
- [Collective Robustness Certificates: Exploiting Interdependence in Graph Neural Networks](https://openreview.net/forum?id=ULQdiUTHe3y), `iclr2021`

#### Randomness transformation defenses

- [Demystifying the Adversarial Robustness of Random Transformation Defenses](https://openreview.net/pdf?id=p4SrFydwO5), Dec. 16 2021.
