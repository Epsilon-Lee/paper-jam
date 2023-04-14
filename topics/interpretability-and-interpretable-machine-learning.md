
# Interpretability and Interpretable Machine Learning

- [Definition of interpretability](#definition-of-interpretability)
- [Philosophy of interpretability](#philosophy-of-interpretability)
- [Theory of interpretability](#theory-of-interpretability)
- [Interpretation methods](#interpretation-methods)
  - [Feature attribution methods](#feature-attribution-methods)
  - [Critics of feature importance](#critics-of-feature-importance)
  - [Beyond feature attribution](#beyond-feature-attribution)
  - [Dataset attribution methods](#dataset-attribution-methods)
    - [Critics of influence function](#critics-of-influence-function)
- [Visualization methods](#visualization-methods)
- [Representation comparison](#representation-comparison)
- [Probing methods](#probing-methods)
- [Evaluation](#evaluation)
- [Transparent model](#transparent-model)
  - [Model editing](#model-editing)
  - [Model debugging](#model-debugging)
- [Analysis](#analysis)
- [Toolkits](#toolkits)


## Definition of interpretability

- [The Mythos of Model Interpretability](https://arxiv.org/pdf/1606.03490.pdf), 2016.
- [Machine Learning Interpretability: A Survey on Methods and Metrics](https://www.mdpi.com/2079-9292/8/8/832), 2019.
- [Interpretable machine learning: definitions, methods, and applications](https://arxiv.org/pdf/1901.04592.pdf), 2019.
- [Interpretable Machine Learning: A Guide for Making Black Box Models Explainable](https://christophm.github.io/interpretable-ml-book/), 2019.
  - "It is difficult to (mathematically) define interpretability. A non-mathematical definition of interpretability that I like by Miller 2017 is: **interpretability is the degree to which a human can understand the cause of decision**", and another one by Been Kim is "**interpretability is the degree to which a human can consistently predict the model's result**"
- [Consistent Counterfactuals for Deep Models](https://arxiv.org/pdf/2110.03109.pdf), `evaluation of interpretability` `iclr2022` submitted.

## Philosophy of interpretability

- [Seamful XAI: Operationalizing Seamful Design in Explainable AI](https://arxiv.org/pdf/2211.06753.pdf), Nov. 12 2022.
- [Towards Formal Approximated Minimal Explanations of Neural Networks](https://arxiv.org/pdf/2210.13915.pdf), Oct. 25 2022.

## Theory of interpretability

- [Foundations of Symbolic Languages for Model Interpretability](https://proceedings.neurips.cc/paper/2021/file/60cb558c40e4f18479664069d9642d5a-Paper.pdf), `nips2021`.
- [Model Interpretability through the Lens of Computational Complexity](https://arxiv.org/pdf/2010.12265.pdf), Nov. 12 2020.

## Interpretation methods

- [A Benchmark for Interpretability Methods in Deep Neural Networks](https://arxiv.org/pdf/1806.10758.pdf), `nips2019` `remove and retrain`
- [Evaluating the Faithfulness of Importance Measures in NLP by Recursively Masking Allegedly Important Tokens and Retraining](https://arxiv.org/pdf/2110.08412.pdf), Oct. 15 `recursive remove and retrain`
- [Interpreting Deep Learning Models in Natural Language Processing: A Review](https://arxiv.org/pdf/2110.10470.pdf), Oct. 25 2021.
- [Where do Models go Wrong? Parameter-Space Saliency Maps for Explainability](https://arxiv.org/pdf/2108.01335.pdf), Aug. 3 2021.
- [From global to local MDI variable importances for random forests and when they are Shapley values](https://arxiv.org/pdf/2111.02218.pdf), Nov. 3 2021. `shapley values` `random forest` `nips2021`
- [Natural Language Descriptions of Deep Features](https://openreview.net/forum?id=NudBMY-tzDr), `iclr2022 submit`
- [Beyond Importance Scores: Interpreting Tabular ML by Visualizing Feature Semantics](https://arxiv.org/pdf/2111.05898.pdf), Nov. 10 2021 `nips2021`
- [Improving Deep Learning Interpretability by Saliency Guided Training](https://arxiv.org/pdf/2111.14338.pdf), Nov. 29 2021 `nips2021`.
- [Do Input Gradients Highlight Discriminative Features?](https://arxiv.org/pdf/2102.12781.pdf), Oct. 26 2021. `nips2021`
- [Interpretable Image Classification with Differentiable Prototypes Assignment](https://arxiv.org/pdf/2112.02902.pdf), Dec. 6 2021
- [More Than Words: Towards Better Quality Interpretations of Text Classifiers](https://arxiv.org/pdf/2112.12444.pdf), Dec. 23 2021
- [High Fidelity Visualization of What Your Self-supervised Representation Knows About](https://arxiv.org/pdf/2112.09164.pdf), Dec. 16 2021
- [DeDUCE: Generating Counterfactual Explanations Efficiently](https://arxiv.org/pdf/2111.15639.pdf), Nov. 29 2021
- [Making a (Counterfactual) Difference One Rationale at a Time](https://arxiv.org/pdf/2201.05177.pdf), `nips2021`
- [WHEN LESS IS MORE: SIMPLIFYING INPUTS AIDS NEURAL NETWORK UNDERSTANDING](https://arxiv.org/pdf/2201.05610.pdf), Jan. 14 2022. `medical` Google Brain
- [Explanatory Learning: Beyond Empiricism in Neural Networks](https://openreview.net/pdf?id=46lmrnVBHBL), `iclr2022` rejected
- [Interpreting Language Models with Contrastive Explanations](https://arxiv.org/pdf/2202.10419.pdf), Feb. 21 2022.

### Feature attribution methods

#### Survey

- [Explaining by Removing: A Unified Framework for Model Explanation](https://www.jmlr.org/papers/volume22/20-1316/20-1316.pdf), `jmlr2021`.

#### Methods

- [Learning to Explain: An Information-Theoretic Perspective on Model Interpretation](https://arxiv.org/abs/1802.07814), `icml2018`.
- [The Shapley Taylor Interaction Index](http://proceedings.mlr.press/v119/sundararajan20a/sundararajan20a.pdf), `icml2020`.
- [Understanding Interlocking Dynamics of Cooperative Rationalization](https://arxiv.org/pdf/2110.13880.pdf), Oct. 26 2021.
- [Rationales for Sequential Predictions](https://arxiv.org/pdf/2109.06387.pdf), Keyon Vafa et al. `emnlp2021` `interpretability` `nmt` `combinatorial optimization`
  - *Rationales*: subset of context that ...;
  - combinatorial optimization formulation of rationale finding: the best rationale is the smallest subset of input tokens that could predict the same prediction as the orignal ones;
  - how to measure faithfulness?
- [Partial order: Finding Consensus among Uncertain Feature Attributions](https://arxiv.org/pdf/2110.13369.pdf), Oct. 26 2021.
- [Joint Shapley values: a measure of joint feature importance](https://openreview.net/forum?id=vcUmUvQCloe), `iclr2022 submit`
- [Fast Axiomatic Attribution for Neural Networks](https://arxiv.org/pdf/2111.07668.pdf), Nov. 15 2021. `nips2021`
- [Fine-Grained Neural Network Explanation by Identifying Input Features with Predictive Information](https://arxiv.org/pdf/2110.01471.pdf), `nips2021` `attribution`.
- [DANCE: Enhancing saliency maps using decoys](http://proceedings.mlr.press/v139/lu21b/lu21b.pdf), `icml2021`
- [The explanation game: a formal framework for interpretable machine learning](https://link.springer.com/content/pdf/10.1007/s11229-020-02629-9.pdf), 2021.
- [Rational Shapley Values](https://arxiv.org/pdf/2106.10191.pdf), May 16 2021.
- [Local Explanations via Necessity and Sufficiency: Unifying Theory and Practice](https://proceedings.mlr.press/v161/watson21a/watson21a.pdf), `uai2021`.
- [EXSUM: From Local Explanations to Model Understanding](https://arxiv.org/abs/2205.00130), `naacl2022`.
- [Faith-Shap: The Faithful Shapley Interaction Index](https://arxiv.org/abs/2203.00870), Mar. 9 2022.
- [XAI for Transformers: Better Explanations through Conservative Propagation](https://proceedings.mlr.press/v162/ali22a.html), `icml2022`.
- [Learning with Explanation Constraints](https://arxiv.org/pdf/2303.14496.pdf), Mar. 25 2023.

#### Critics and evaluation of feature importance

- [Interpretation of Neural Networks is Fragile](https://arxiv.org/abs/1710.10547), Oct. 29 2019 `aaai2018`
- [On the (In)fidelity and Sensitivity of Explanations](https://proceedings.neurips.cc/paper/2019/file/a7471fdc77b3435276507cc8f2dc2569-Paper.pdf), `nips2019`
- [Do Feature Attribution Methods Correctly Attribute Features?](https://yilun.scripts.mit.edu/pdf/xaiworkshop2021feature.pdf), `XAI4Debugging@NeurIPS2021`.
- [The Disagreement Problem in Explainable Machine Learning: A Practitioner’s Perspective](https://arxiv.org/pdf/2202.01602.pdf), Feb. 8 2022.
- [Sanity Simulation for Saliency Methods](https://arxiv.org/pdf/2105.06506.pdf), `icml2022`.
- [Use-Case-Grounded Simulations for Explanation Evaluation](https://arxiv.org/pdf/2206.02256.pdf), `nips2022`.
- [Impossibility Theorems for Feature Attribution](https://arxiv.org/pdf/2212.11870.pdf), Dec. 22 2022. `theory`.
- [Use-Case-Grounded Simulations for Explanation Evaluation](https://arxiv.org/pdf/2206.02256.pdf),Aug. 20 2022. `evaluation` `nips2022`.
- [Better Understanding Differences in Attribution Methods via Systematic Evaluations](https://arxiv.org/pdf/2303.11884.pdf), Mar. 21 2023.
- [Using explanations to guide models](https://arxiv.org/pdf/2303.11932.pdf), Mar. 21 2023.

### Feature interaction

- [Detecting Statistical Interactions from Neural Network Weights](https://arxiv.org/pdf/1705.04977.pdf), `iclr2017`.
- [Neural Interaction Transparency (NIT): Disentangling Learned Interactions for Improved Interpretability](https://papers.nips.cc/paper/2018/hash/74378afe5e8b20910cf1f939e57f0480-Abstract.html), `nips2018`.
- [How does This Interaction Affect Me? Interpretable Attribution for Feature Interactions](https://proceedings.neurips.cc/paper/2020/hash/443dec3062d0286986e21dc0631734c9-Abstract.html), `nips2020`.
- [Feature Interaction Interpretability: A Case for Explaining Ad-Recommendation Systems via Neural Interaction Detection](https://openreview.net/forum?id=BkgnhTEtDS), `iclr2020`.
- [Quantifying & Modeling Feature Interactions: An Information Decomposition Framework](https://arxiv.org/pdf/2302.12247.pdf), Feb. 23 2023.

### Dataset attribution methods

- [Representer Point Selection for Explaininig Deep Neural Networks](https://arxiv.org/abs/1811.09720), `nips2018`
- [Input Similarity from the Neural Network Perspective](https://arxiv.org/abs/2102.05262), `nips2019`
- [Understanding black-box predictions via influence functions](http://proceedings.mlr.press/v70/koh17a/koh17a.pdf), Apr. 2017. `icml2017`
- [Towards Efficient Data Valuation Based on the Shapley Value](http://proceedings.mlr.press/v89/jia19a/jia19a.pdf), `aistats2019`
- [Interpreting Black Box Predictions using Fisher Kernels](http://proceedings.mlr.press/v89/khanna19a/khanna19a.pdf), `aistats2019`
- [Data Shapley: Equitable Valuation of Data for Machine Learning](http://proceedings.mlr.press/v97/ghorbani19c.html), `icml2019`
- [On the Accuracy of Influence Functions for Measuring Group Effects](https://arxiv.org/pdf/1905.13289.pdf), `nips2019`
- [Deep Learning Interpretation: Flip Points and Homotopy Methods](http://proceedings.mlr.press/v107/yousefzadeh20a/yousefzadeh20a.pdf), `ml2020`
- [RelatIF: Identifying Explanatory Training Examples via Relative Influence](http://proceedings.mlr.press/v108/barshan20a/barshan20a.pdf), `aistats2020`
- [On Second-Order Group Influence Functions for Black-Box Predictions](http://proceedings.mlr.press/v119/basu20b/basu20b.pdf), `iclr2020`
- [True to the Model or True to the Data?](https://arxiv.org/pdf/2006.16234.pdf), Jun. 29 2020.
- [Approximate Cross-Validation for Structured Models](https://arxiv.org/pdf/2006.12669.pdf), `nips2020`
- [HYDRA: Hypergradient Data Relevance Analysis for Interpreting Deep Neural Networks](https://www.aaai.org/AAAI21Papers/AAAI-8686.ChenY.pdf), `aaai2021`
- [Scaling Up Influence Functions](https://arxiv.org/pdf/2112.03052.pdf), Dec. 6 2021. Google Research `aaai2022`.
- [DIVINE: Diverse Influential Training Points for Data Visualization and Model Refinement](https://arxiv.org/pdf/2107.05978.pdf), Jul. 13 2021.
- [Metadata Archaeology: Unearthing Data Subsets by Leveraging Training Dynamics](https://arxiv.org/pdf/2209.10015.pdf), Sep. 20 2022.
- [Deconstructing Distributions: A Pointwise Framework of Learning](https://arxiv.org/pdf/2202.09931.pdf), Feb. 20 2022.
- [Datamodels: Predicting Predictions from Training Data](https://arxiv.org/abs/2202.00622), Feb. 1 2022.
- [Deep Learning on a Data Diet: Finding Important Examples Early in Training](https://arxiv.org/abs/2107.07075), Jul. 15 2021. `nips2021`
- [Scaling Up Influence Functions](https://www.aaai.org/AAAI22Papers/AAAI-5853.SchioppaA.pdf), `aaai2022` [jax implementation](https://github.com/google-research/jax-influence).
- [FastIF: Scalable Influence Functions for Efficient Model Interpretation and Debugging](https://aclanthology.org/2021.emnlp-main.808/), `emnlp2021` [code](https://github.com/salesforce/fast-influence-functions)
- [Unifying Approaches in Data Subset Selection via Fisher Information and Information-Theoretic Quantities](https://arxiv.org/pdf/2208.00549.pdf), Aug. 1 2022.
- [The Spotlight: A General Method for Discovering Systematic Errors in Deep Learning Models](https://arxiv.org/pdf/2107.00758.pdf), Oct. 16 2021. `FAC 2022`.
- [First is Better Than Last for Training Data Influence](https://arxiv.org/pdf/2202.11844.pdf), Feb. 24 2022. `data-centric`
- [Influence Functions for Sequence Tagging Models](https://arxiv.org/abs/2210.14177), Oct. 25 2022. `emnlp2022`.
- [First is Better Than Last for Language Data Influence](https://arxiv.org/abs/2202.11844), arXiv.v3 Oct. 27 2022. `nips2022`.
- [The Shapley Value in Machine Learning](https://arxiv.org/pdf/2202.05594.pdf), May 26 2022.
- [Beta Shapley: a Unified and Noise-reduced Data Valuation Framework for Machine Learning](https://proceedings.mlr.press/v151/kwon22a/kwon22a.pdf), `aistats2022`.
- [Simfluence: Modeling the Influence of Individual Training Examples by Simulating Training Runs](https://arxiv.org/pdf/2303.08114.pdf), Mar. 14 2023.
- [TRAK: Attributing Model Behavior at Scale](https://arxiv.org/pdf/2303.14186.pdf), Mar. 24 2023.
- [A Note on “Efficient Task-Specific Data Valuation for Nearest Neighbor Algorithms”](https://arxiv.org/pdf/2304.04258.pdf), Apr. 9 2023.

#### Critics of influence function

- [Influence Functions in Deep Learning are Fragile](https://arxiv.org/pdf/2006.14651.pdf), Feb. 10 2021, `iclr2021`
- [If Influence Functions are the Answer, Then What is the Question?](https://arxiv.org/pdf/2209.05364.pdf), arXiv Sep. 12 2022.
- [Understanding Influence Functions and Datamodels via Harmonic Analysis](https://arxiv.org/pdf/2210.01072.pdf), arXiv Oct. 3 2022.
- [Revisiting the fragility of influence functions](https://arxiv.org/pdf/2303.12922.pdf), Mar. 22 2023.

#### Instance-based explanation: prototypes, exemplars

- [Prototype selection for interpretable classification](https://projecteuclid.org/journals/annals-of-applied-statistics/volume-5/issue-4/Prototype-selection-for-interpretable/10.1214/11-AOAS495.pdf), 2011.
- [Examples are not enough, learn to criticize! critism for interpretability](https://proceedings.neurips.cc/paper/2016/file/5680522b8e2bb01943234bce7bf84534-Paper.pdf), `nips2016`.
- [This Looks Like That: Deep Learning for Interpretable Image Recognition](https://proceedings.neurips.cc/paper/2019/file/adf7ee2dcf142b0e11888e72b43fcb75-Paper.pdf), `nips2019`.
- [Interpretable Counterfactual Explanations Guided by Prototypes](https://arxiv.org/pdf/1907.02584.pdf), Feb. 18 2020.

### Counterfactual, contrastive explanation

- [Explanations based on the Missing: Towards Contrastive Explanations with Pertinent Negatives](https://proceedings.neurips.cc/paper/2018/file/c5ff2543b53f4cc0ad3819a36752467b-Paper.pdf), `nips2018`.
- [Counterfactual Explanations for Machine Learning: A Review](https://arxiv.org/pdf/2010.10596.pdf), Oct. 20 2020.
- [FACE: Feasible and Actionable Counterfactual Explanations](https://arxiv.org/pdf/1909.09369.pdf), Feb. 24 2020.
- [DeDUCE: Generating Counterfactual Explanations At Scale](https://xai4debugging.github.io/files/papers/deduce_generating_counterfactu.pdf), `XAI4Debugging workshop`, 2021.
- [Diffusion Visual Counterfactual Explanations](https://arxiv.org/pdf/2210.11841.pdf), Oct. 21 2022. `nips2022`.
- [Counterfactual Generation Under Confounding](https://arxiv.org/abs/2210.12368), Oct. 22 2022.
- [Explaining Model Confidence Using Counterfactuals](https://arxiv.org/pdf/2303.05729.pdf), Mar. 10 2023.
- [Semi-supervised counterfactual explanations](https://arxiv.org/pdf/2303.12634.pdf), Mar. 22 2023.
- [Explaining Groups of Instances Counterfactually for XAI: A Use Case, Algorithm and User Study for Group-Counterfactuals](https://arxiv.org/pdf/2303.09297.pdf), Mar. 16 2023.

#### Leveraging interpretation

/

## Visualization methods

- [Can Neural Nets Learn the Same Model Twice? Investigating Reproducibility and Double Descent from the Decision Boundary Perspective](https://arxiv.org/pdf/2203.08124.pdf), Mar. 15 2022.
- [A Spectral Method for Assessing and Combining Multiple Data Visualizations](https://arxiv.org/pdf/2210.13711.pdf), Oct. 25 2022.
- [Understanding the Evolution of Linear Regions in Deep Reinforcement Learning](https://arxiv.org/pdf/2210.13611.pdf), Oct. 24 2022.
- [Understanding how dimensioin reduction tools work: an empirical approach to deciphering t-SNE, UMAP, TriMap, and PaCMAP for data visualization](https://www.jmlr.org/papers/volume22/20-1061/20-1061.pdf), `jmlr2021`.
- [Improved visualization of high-dimensional data using the distance-to-distance transformation](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010764), Dec. 20 2022. learned this method from [tweet](https://twitter.com/hippopedoid/status/1613096715963211776?cn=ZmxleGlibGVfcmVjcw%3D%3D&refsrc=email).

## Representation comparison

- [Representation Alignment in Neural Networks](https://openreview.net/pdf?id=fLIWMnZ9ij), `tmlr2022`.
- [GULP: a prediction-based metric between representations](https://arxiv.org/pdf/2210.06545.pdf), arXiv Oct. 12 2022. `probing`?
  - This paper introduces a family of distance measures between representations that is explicitly motivated by downstream predictive tasks.
- [On the Versatile Uses of Partial Distance Correlation in Deep Learning](https://arxiv.org/pdf/2207.09684.pdf), Jul. 20 2022.
- [Representational Dissimilarity Metric Spaces for Stochastic Neural Networks](https://openreview.net/forum?id=xjb563TH-GH), `iclr2023`.

## Probing methods

- [How transferable are features in deep neural networks?](https://arxiv.org/abs/1411.1792), `nips2014`. ***The original paper***
- [Understanding Intermediate layers using linear classifier probes](https://openreview.net/pdf?id=HJ4-rAVtl) and [a longer version](https://arxiv.org/pdf/1610.01644.pdf), `iclr2017`.
- [What you can cram into a single vector: Probing sentence embeddings for linguistic properties](https://arxiv.org/abs/1805.01070), May 3 2018.
- [Network Dissection: Quantifying Interpretability of Deep Visual Representations](https://openaccess.thecvf.com/content_cvpr_2017/papers/Bau_Network_Dissection_Quantifying_CVPR_2017_paper.pdf), `cvpr2017`.
- [Interpretability beyond feature attribution: Quantitative testing with concept activation vectors (tcav)](http://proceedings.mlr.press/v80/kim18d/kim18d.pdf), `icml2018`
- [On the Global Optima of Kernelized Adversarial Representation Learning](https://arxiv.org/abs/1910.07423), Oct. 16 2019, `iccv2019`.
- [Intrinsic Probing through Dimension Selection](https://arxiv.org/pdf/2010.02812.pdf), Oct. 6 2020.
- [A Non-Linear Structural Probe](https://arxiv.org/pdf/2105.10185.pdf), May 21 2021.
- [Perturbed Masking: Parameter-free Probing for Analyzing and Interpreting BERT](https://arxiv.org/pdf/2004.14786.pdf), May 28 2021.
- [Probing as Quantifying the Inductive Bias of Pre-trained Representations](https://arxiv.org/pdf/2110.08388.pdf), Oct. 15 2021.
- [Counterfactual Interventions Reveal the Causal Effect of Relative Clause Representations on Agreement Prediction](https://arxiv.org/pdf/2105.06965.pdf), Sep. 15 2021.
- [A Latent-Variable Model for Intrinsic Probing](https://arxiv.org/abs/2201.08214), Jan. 20 2022.
- [ADVERSARIAL CONCEPT ERASURE IN KERNEL SPACE](https://arxiv.org/pdf/2201.12191.pdf), Jan. 28 2022.
- [LINEAR ADVERSARIAL CONCEPT ERASURE](https://arxiv.org/pdf/2201.12091.pdf), Jan. 28 2022.
- [On the data requirements of probing](https://arxiv.org/pdf/2202.12801.pdf), Feb. 25 2022.
- [Causal Abstractions of Neural Networks](https://proceedings.neurips.cc/paper/2021/file/4f5c422f4d49a5a807eda27434231040-Paper.pdf), `nips2021`
- [Probing Classifiers: Promises, Shortcomings, and Advances](https://arxiv.org/abs/2102.12452), `cl2021`.
- [Linear Guardedness and Its Implications](https://arxiv.org/pdf/2210.10012.pdf), Oct. 18 2022.

## Evaluation

- [Paradigm Shift in Natural Language Processing](https://arxiv.org/pdf/2109.12575.pdf), Xipeng Qiu et al. `unify methodology`
- [Finding a Balanced Degree of Automation for Summary Evaluation](https://arxiv.org/pdf/2109.11503.pdf), Mohit Bansal et al. `evaluation`
- [Sample Efficient Model Evaluation](https://arxiv.org/pdf/2109.12043.pdf), David Barber et al. `evaluation`
- [The Curse of Performance Instability in Analysis Datasets: Consequences, Source, and Suggestions](https://aclanthology.org/2020.emnlp-main.659.pdf), `emnlp2020`
- [Better than Average: Paired Evaluation of NLP Systems](https://arxiv.org/pdf/2110.10746.pdf), Oct. 20 2021.
- [“Will You Find These Shortcuts?” A Protocol for Evaluating the Faithfulness of Input Salience Methods for Text Classification](https://arxiv.org/pdf/2111.07367.pdf), Nov. 14 2021, Google, `emnlp2021`
- [Explain, Edit, and Understand: Rethinking User Study Design for Evaluating Model Explanations](https://arxiv.org/pdf/2112.09669.pdf), Dec. 17 2021 `aaai2022`
- [Human Interpretation of Saliency-based Explanation Over Text](https://arxiv.org/abs/2201.11569), Jan. 27 2022
- [Does BERT Learn as Humans Perceive? Understanding Linguistic Styles through Lexica](https://aclanthology.org/2021.emnlp-main.510.pdf), 2021.
- [Diagnosing AI Explanation Methods with Folk Concepts of Behavior](https://arxiv.org/abs/2201.11239), Alon Jacovi et al. Jan. 27 2022.

## Transparent model

- [Gaining Free or Low-Cost Transparency with Interpretable Partial Substitute](https://arxiv.org/pdf/1802.04346.pdf), `icml2019`.
- [Self-explaining deep models with logic rule reasoning](https://www.microsoft.com/en-us/research/uploads/prod/2022/10/CameraReady_NeurIPS22_SELOR.pdf), `nips2022`.

### Rule list model

- [Falling rule lists](http://proceedings.mlr.press/v38/wang15a.pdf), `aistats2015`.
- [Interpretable classifiers using rules and Bayesian analysis: Building a better stroke prediction model](https://projecteuclid.org/journals/annals-of-applied-statistics/volume-9/issue-3/Interpretable-classifiers-using-rules-and-Bayesian-analysis--Building-a/10.1214/15-AOAS848.full), Sep. 2015, Ann. Appl. Stat. 9(3).
- [A bayesian framework for learning rule sets for interpretable classification](https://www.jmlr.org/papers/volume18/16-003/16-003.pdf), `jmlr2017`.
- [Learning certifiably optimal rule lists for categorical data](https://www.jmlr.org/papers/volume18/17-716/17-716.pdf), `jmlr2018`.
- [Scalable Bayesian rule lists](http://proceedings.mlr.press/v70/yang17h/yang17h.pdf), `icml2017`. [long version](https://arxiv.org/pdf/1602.08610.pdf).
- [Globally-Consistent Rule-Based Summary-Explanations for Machine Learning Models: Application to Credit-Risk Evaluation](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3395422), Jun. 12 2019.
- [Causal rule sets for identifying subgroups with enhanced treatment effects](https://arxiv.org/pdf/1710.05426), `informs2022`.

### Model editing

- [Editing a classifier by rewriting its prediction rules](https://arxiv.org/pdf/2112.01008.pdf), Dec. 2 2021.
- [Influence Tuning: Demoting Spurious Correlations via Instance Attribution and Instance-Driven Updates](https://arxiv.org/pdf/2110.03212.pdf), `emnlp2021` with `code`
- [Fast Model Editing at Scale](https://arxiv.org/pdf/2110.11309.pdf), Oct. 21 2021.

### Model debugging

- [Debugging Tests for Model Explanations](https://arxiv.org/pdf/2011.05429.pdf), Nov. 10 2020.

## Analysis

- [Controlled Evaluation of Grammatical Knowledge in Mandarin Chinese Language Models](https://arxiv.org/pdf/2109.11058.pdf), Roger Levy et al. `controlled evaluation` `language model`
- [Can Question Generation Debias Question Answering Models? A Case Study on Question–Context Lexical Overlap](https://arxiv.org/pdf/2109.11256.pdf), `qa`
- [Sorting through the noise: Testing robustness of information processing in pre-trained language models](https://arxiv.org/pdf/2109.12393.pdf), `language model`
- [Patterns of Lexical Ambiguity in Contextualised Language Models](https://arxiv.org/pdf/2109.13032.pdf), `language model`
- [Word Acquisition in Neural Language Models](https://arxiv.org/pdf/2110.02406.pdf), `language model` `learning dynamics`
- [Self-conditioning pre-trained language models](https://arxiv.org/pdf/2110.02802.pdf), `neuron-level analysis`
- [How BPE Affects Memorization in Transformers](https://arxiv.org/pdf/2110.02782.pdf), Facebook AI.
- [Capturing Structural Locality in Non-parametric Language Models](https://arxiv.org/pdf/2110.02870.pdf), CMU data `locality`
- [Understanding How Encoder-Decoder Architectures Attend](https://arxiv.org/pdf/2110.15253.pdf), Oct. 28 2021. `dynamics of attention`
- [The Low-Dimensional Linear Geometry of Contextualized Word Representations](https://arxiv.org/pdf/2105.07109.pdf), Sep. 14 2021.
- [Acquisition of Chess Knowledge in AlphaZero](https://arxiv.org/pdf/2111.09259.pdf), Nov. 27 2021.
- [Learning Bounded Context-Free-Grammar via LSTM and the Transformer: Difference and Explanations](https://arxiv.org/pdf/2112.09174.pdf), Dec. 16 2021. `aaai2022`

## Toolkits

- [imodels](https://github.com/csinva/imodels), Interpretable ML package 🔍 for concise, transparent, and accurate predictive modeling (sklearn-compatible). [paper](https://www.theoj.org/joss-papers/joss.03192/10.21105.joss.03192.pdf). published at May 4 2021.


## Interpretability for traditional and tree models

- [Finding Minimum-Cost Explanations for Predictions made by Tree Ensembles](https://arxiv.org/pdf/2303.09271.pdf), Mar. 16 2023.

### Influence for tree models

- [Influence Functions for CART](https://hal.science/hal-00562039/document), Feb. 2 2011. [presentation](https://www.gdr-mascotnum.fr/media/mascot11poggi.pdf).
- [Finding Influential Training Samples for Gradient Boosted Decision Trees](https://arxiv.org/pdf/1802.06640.pdf), Mar. 2018. `icml2018`.
  - [Machine Unlearning for Random Forests](http://proceedings.mlr.press/v139/brophy21a/brophy21a.pdf), `icml2021`.
- [TREX: Tree-Ensemble Representer-Point Explanations](https://arxiv.org/abs/2009.05530), Sep. 11 2020.
- [Towards Efficient Data Valuation Based on the Shapley Value](http://proceedings.mlr.press/v89/jia19a/jia19a.pdf), `aistats2019`.

## Human-in-the-loop

- [Human Uncertainty in Concept-Based AI Systems](https://arxiv.org/pdf/2303.12872.pdf), Mar. 22 2023.
