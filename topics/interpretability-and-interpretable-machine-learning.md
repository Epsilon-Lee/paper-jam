
# Interpretability and Interpretable Machine Learning

- [Definition of interpretability](#definition-of-interpretability)
- [Philosophy of interpretability](#philosophy-of-interpretability)
- [Theory of interpretability](#theory-of-interpretability)
- [Interpretation methods](#interpretation-methods)
  - [Feature attribution methods](#feature-attribution-methods)
  - [Critics of feature importance](#critics-of-feature-importance)
  - [Dataset attribution methods](#dataset-attribution-methods)
    - [Critics of influence function](#critics-of-influence-function)
  - [Counterfactual, contrastive explanation](#counterfactual-and-contrastive-explanation)
- [Visualization methods](#visualization-methods)
- [Representation comparison](#representation-comparison)
- [Probing methods](#probing-methods)
- [Evaluation](#evaluation)
- [Transparent model](#transparent-model)
  - [Rule learning](#rule-learning)
  - [Model editing](#model-editing)
  - [Model debugging](#model-debugging)
- [Analysis](#analysis)
- [Toolkits](#toolkits)
- [Interpretability for traditional and tree models](#interpretability-for-traditional-and-tree-models)
  - [Visualization of tree models](#visualization-of-tree-models)
  - [Influence for tree models](#influence-for-tree-models)
  - [Uncertainty](#uncertainty)
  - [Adversarials and robustness](#adversarials-and-robustness)





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
- [Is Task-Agnostic Explainable AI a Myth?](https://arxiv.org/pdf/2307.06963.pdf), Jul. 13 2023.

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
- [A Survey on the Robustness of Feature Importance and Counterfactual Explanations](https://arxiv.org/pdf/2111.00358.pdf), Jan. 3 2023.

#### Methods

- [Axiomatic attribution for deep networks](http://proceedings.mlr.press/v70/sundararajan17a/sundararajan17a.pdf), `icml2017`.
- [Learning to Explain: An Information-Theoretic Perspective on Model Interpretation](https://arxiv.org/abs/1802.07814), `icml2018`.
- [Rethinking Cooperative Rationalization: Introspective Extraction and Complement Control](https://arxiv.org/pdf/1910.13294.pdf), Dec. 15 2019. `rationalization`. [code](https://github.com/Gorov/three_player_for_emnlp).
- [The Shapley Taylor Interaction Index](http://proceedings.mlr.press/v119/sundararajan20a/sundararajan20a.pdf), `icml2020`.
- [Understanding Interlocking Dynamics of Cooperative Rationalization](https://arxiv.org/pdf/2110.13880.pdf), Oct. 26 2021.
- [Rationales for Sequential Predictions](https://arxiv.org/pdf/2109.06387.pdf), Keyon Vafa et al. `emnlp2021` `interpretability` `nmt` `combinatorial optimization`
  - *Rationales*: subset of context that ...;
  - combinatorial optimization formulation of rationale finding: the best rationale is the smallest subset of input tokens that could predict the same prediction as the orignal ones;
  - how to measure faithfulness?
- [Partial order: Finding Consensus among Uncertain Feature Attributions](https://arxiv.org/pdf/2110.13369.pdf), Oct. 26 2021.
- [Joint Shapley values: a measure of joint feature importance](https://openreview.net/forum?id=vcUmUvQCloe), `iclr2022 submit`.
- [The Out-of-Distribution Problem in Explainability and Search Methods for Feature Importance Explanations](https://arxiv.org/abs/2106.00786), `nips2021`.
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
- [Distributing Synergy Functions: Unifying Game-Theoretic Interaction Methods for Machine-Learning Explainability](https://arxiv.org/pdf/2305.03100.pdf), May 4 2023.
- [Are Machine Rationales (Not) Useful to Humans? Measuring and Improving Human Utility of Free-Text Rationales](https://arxiv.org/pdf/2305.07095.pdf), May 11 2023.
- [The Weighted Möbius Score: A Unified Framework for Feature Attribution](https://arxiv.org/pdf/2305.09204.pdf), May 16 2023.
- [Token-wise Decomposition of Autoregressive Language Model Hidden States for Analyzing Model Predictions](https://arxiv.org/pdf/2305.10614.pdf), May 17 2023. `acl2023`.
- [Efficient Shapley Values Estimation by Amortization for Text Classification](https://arxiv.org/pdf/2305.19998.pdf), May 31 2023.
- [Integrated Decision Gradients: Compute Your Attributions Where the Model Makes Its Decision](https://arxiv.org/pdf/2305.20052.pdf), May 31 2023.
  - saturation effect.
- [DecompX: Explaining Transformers Decisions by Propagating Token Decomposition](https://arxiv.org/pdf/2306.02873.pdf), Jun. 5 2023. `acl2023`.
- [Towards Trustworthy Explanation: On Causal Rationalization](https://arxiv.org/pdf/2306.14115.pdf), Jun. 25 2023. `icml2023`.
  - from association-based attribution to excluding spurious but important features.
- [PWSHAP: A path-wise explanation model for targeted variables](https://arxiv.org/pdf/2306.14672.pdf), Jun. 26 2023.
- [On Formal Feature Attribution and Its Approximation](https://arxiv.org/pdf/2307.03380.pdf), Jul. 7 2023. [code](https://github.com/ffattr/ffa).
- [Generalizing Backpropagation for Gradient-Based Interpretability](https://arxiv.org/pdf/2307.03056.pdf), Jul. 6 2023. [code](https://github.com/kdu4108/semiring-backprop-exps).
- [MDI+: A Flexible Random Forest-Based Feature Importance Framework](https://arxiv.org/pdf/2307.01932.pdf), Jul. 4 2023. [imodels](https://github.com/csinva/imodels).
- [Shapley Sets: feature attribution via recursive function decomposition](https://arxiv.org/pdf/2307.01777.pdf), Jul. 4 2023.
- [Gradient strikes back: How filtering out high frequencies improves explanations](https://arxiv.org/pdf/2307.09829.pdf), Jul. 18 2023.
- [Verifiable feature attributions: A bridge between post hoc explainabilty and inherent interpretability](https://arxiv.org/pdf/2307.15007.pdf), Jul. 27 2023.
- [An Efficient Shapley Value Computation for the Naive Bayes Classifier](https://arxiv.org/pdf/2307.16718.pdf), Jul. 31 2023.
- [Signature Activation: A Sparse Signal View for Holistic Saliency](https://arxiv.org/pdf/2309.11443.pdf), Sep. 20 2023.
- [Towards attributions of input variables in a coalition](https://arxiv.org/pdf/2309.13411.pdf), Sep. 23 2023.
- [SPADE: Sparsity-guided debugging for deep neural networks](https://arxiv.org/pdf/2310.04519.pdf), Oct. 6 2023.

#### Critics and evaluation of feature importance

- [Interpretation of Neural Networks is Fragile](https://arxiv.org/abs/1710.10547), Oct. 29 2019 `aaai2018`
- [On the (In)fidelity and Sensitivity of Explanations](https://proceedings.neurips.cc/paper/2019/file/a7471fdc77b3435276507cc8f2dc2569-Paper.pdf), `nips2019`
- [Do Feature Attribution Methods Correctly Attribute Features?](https://yilun.scripts.mit.edu/pdf/xaiworkshop2021feature.pdf), `XAI4Debugging@NeurIPS2021`.
- [On Locality of Local Explanation Models](https://proceedings.neurips.cc/paper/2021/file/995665640dc319973d3173a74a03860c-Paper.pdf), `nips2021`.
- [The Disagreement Problem in Explainable Machine Learning: A Practitioner’s Perspective](https://arxiv.org/pdf/2202.01602.pdf), Feb. 8 2022.
- [Sanity Simulation for Saliency Methods](https://arxiv.org/pdf/2105.06506.pdf), `icml2022`.
- [Use-Case-Grounded Simulations for Explanation Evaluation](https://arxiv.org/pdf/2206.02256.pdf), `nips2022`.
- [Impossibility Theorems for Feature Attribution](https://arxiv.org/pdf/2212.11870.pdf), Dec. 22 2022. `theory`.
- [Use-Case-Grounded Simulations for Explanation Evaluation](https://arxiv.org/pdf/2206.02256.pdf),Aug. 20 2022. `evaluation` `nips2022`.
- [Framework for evaluating faithfulness of local explanations](https://proceedings.mlr.press/v162/dasgupta22a/dasgupta22a.pdf), `icml2022`.
- [Which Explanation Should I Choose? A Function Approximation Perspective to Characterizing Post Hoc Explanations](https://arxiv.org/pdf/2206.01254.pdf), `nips2022`.
- [Better Understanding Differences in Attribution Methods via Systematic Evaluations](https://arxiv.org/pdf/2303.11884.pdf), Mar. 21 2023.
- [Using explanations to guide models](https://arxiv.org/pdf/2303.11932.pdf), Mar. 21 2023.
- [Theoretical Behavior of XAI Methods in the Presence of Suppressor Variables](https://arxiv.org/pdf/2306.01464.pdf), Jun. 2 2023. `icml2023`. `causality`.
  - non-class related suppressor feature might have non-zero importance.
- [Don’t trust your eyes: on the (un)reliability of feature visualizations](https://arxiv.org/pdf/2306.04719.pdf), Jun. 7 2023.
- [On the Robustness of Removal-Based Feature Attributions](https://arxiv.org/pdf/2306.07462.pdf), Jun. 12 2023.
- [Consistent Explanations in the Face of Model Indeterminacy via Ensembling](https://arxiv.org/pdf/2306.06193.pdf)， Jun. 13 2023.
- [Four Axiomatic Characterizations of the Integrated Gradients Attribution Method](https://arxiv.org/pdf/2306.13753.pdf), Jun. 23 2023. `IG`.
- [Evaluating the overall sensitivity of saliency-based explanation methods.](https://arxiv.org/ftp/arxiv/papers/2306/2306.13682.pdf)， Jun. 28 2023. `sensitivity` `faithfulness`.
- [Stability guarantees for feature attributions with multiplicative smoothing](https://arxiv.org/pdf/2307.05902.pdf), Jul. 12 2023.
- [Harmonizing Feature Attributions Across Deep Learning Architectures: Enhancing Interpretability and Consistency](https://arxiv.org/pdf/2307.02150.pdf), Jul. 5 2023.
- [Fixing confirmation bias in feature attribution methods via semantic search](https://arxiv.org/pdf/2307.00897.pdf), Jul. 3 2023.
- [The future of human-centric eXplainable Artificial Intelligence (XAI) is not post-hoc explanations](https://arxiv.org/pdf/2307.00364.pdf), Jul. 1 2023.
- [Confident Feature Ranking](https://arxiv.org/pdf/2307.15361.pdf), Jul. 28 2023.
- [A Dual-Perspective Approach to Evaluating Feature Attribution Methods](https://arxiv.org/pdf/2308.08949.pdf), Aug. 17 2023.
- [On Gradient-like Explanation under a Black-box Setting: When Black-box Explanations Become as Good as White-box](https://arxiv.org/pdf/2308.09381.pdf), Aug. 18 2023.
- [Predictability and Comprehensibility in Post-Hoc XAI Methods: A User-Centered Analysis](https://arxiv.org/pdf/2309.11987.pdf), Sep. 21 2023.
- [AttributionLab: Faithfulness of feature attribution under controllable environments](https://arxiv.org/pdf/2310.06514.pdf), Oct. 10 2023.

### Feature interaction

- [Detecting Statistical Interactions from Neural Network Weights](https://arxiv.org/pdf/1705.04977.pdf), `iclr2017`.
- [Neural Interaction Transparency (NIT): Disentangling Learned Interactions for Improved Interpretability](https://papers.nips.cc/paper/2018/hash/74378afe5e8b20910cf1f939e57f0480-Abstract.html), `nips2018`.
- [How does This Interaction Affect Me? Interpretable Attribution for Feature Interactions](https://proceedings.neurips.cc/paper/2020/hash/443dec3062d0286986e21dc0631734c9-Abstract.html), `nips2020`.
- [Feature Interaction Interpretability: A Case for Explaining Ad-Recommendation Systems via Neural Interaction Detection](https://openreview.net/forum?id=BkgnhTEtDS), `iclr2020`.
- [Quantifying & Modeling Feature Interactions: An Information Decomposition Framework](https://arxiv.org/pdf/2302.12247.pdf), Feb. 23 2023.
- [Asymmetric feature interaction for interpreting model predictions](https://arxiv.org/pdf/2305.07224.pdf), May 12 2023.
- [Explaining the Model and Feature Dependencies by Decomposition of the Shapley Value](https://arxiv.org/pdf/2306.10880.pdf), Jun. 19 2023.
- [Feature Selection: A perspective on inter-attribute cooperation.](https://arxiv.org/pdf/2306.16559.pdf), Jun. 28 2023. 

### Training data attribution (TDA) methods

#### Survey

- [Training Data Influence Analysis and Estimation: A Survey](https://arxiv.org/pdf/2212.04612.pdf), Dec. 9 2022.

#### Methods

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
- [Introduction to Core-sets: an Updated Survey](https://arxiv.org/pdf/2011.09384.pdf), Nov. 18 2022.
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
- [Deep Learning on a Data Diet: Finding Important Examples Early in Training](https://proceedings.neurips.cc/paper/2021/hash/ac56f8fe9eea3e4a365f29f0f1957c55-Abstract.html), `nips2021`.
- [First is Better Than Last for Training Data Influence](https://arxiv.org/pdf/2202.11844.pdf), Feb. 24 2022. `data-centric`
- [Influence Functions for Sequence Tagging Models](https://arxiv.org/abs/2210.14177), Oct. 25 2022. `emnlp2022`.
- [First is Better Than Last for Language Data Influence](https://arxiv.org/abs/2202.11844), arXiv.v3 Oct. 27 2022. `nips2022`.
- [The Shapley Value in Machine Learning](https://arxiv.org/pdf/2202.05594.pdf), May 26 2022.
- [Beta Shapley: a Unified and Noise-reduced Data Valuation Framework for Machine Learning](https://proceedings.mlr.press/v151/kwon22a/kwon22a.pdf), `aistats2022`.
- [Simfluence: Modeling the Influence of Individual Training Examples by Simulating Training Runs](https://arxiv.org/pdf/2303.08114.pdf), Mar. 14 2023.
- [TRAK: Attributing Model Behavior at Scale](https://arxiv.org/pdf/2303.14186.pdf), Mar. 24 2023.
- [A Note on “Efficient Task-Specific Data Valuation for Nearest Neighbor Algorithms”](https://arxiv.org/pdf/2304.04258.pdf), Apr. 9 2023.
- [Measuring the Effect of Training Data on Deep Learning Predictions via Randomized Experiments](https://arxiv.org/pdf/2206.10013.pdf), Jun. 20 2022.
- [Data-OOB: Out-of-bag Estimate as a Simple and Efficient Data Value](https://arxiv.org/pdf/2304.07718.pdf), Apr. 16 2023.
- [Fairness-aware data valuation for supervised learning](https://arxiv.org/pdf/2303.16963.pdf), Mar. 29 2023.
- [Identifying a Training-Set Attack's Target Using Renormalized Influence Estimation](https://arxiv.org/pdf/2201.10055.pdf), Sep. 5 2022.
- [Data Valuation Without Training of a Model](https://openreview.net/forum?id=XIzO8zr-WbM), `iclr2023`.
- [DAVINZ: Data Valuation using Deep Neural Networks at Initialization](https://proceedings.mlr.press/v162/wu22j/wu22j.pdf), `icml2022`.
- [Class based Influence Functions for Error Detection](https://arxiv.org/pdf/2305.01384.pdf), May 2 2023.
- [On Influence Functions, Classification Influence, Relative Influence, Memorization and Generalization](https://arxiv.org/pdf/2305.16094.pdf), May 25 2023.
- [Theoretical and Practical Perspectives on what Influence Functions Do](https://arxiv.org/pdf/2305.16971.pdf), May 26 2023.
- [Shapley Based Residual Decomposition for Instance Analysis](https://arxiv.org/pdf/2305.18818.pdf), May 30 2023.
- [A Bayesian Perspective On Training Data Attribution](https://arxiv.org/pdf/2305.19765.pdf), May 31 2023.
- [Representater point selectioin for explaining regularized high-dimensional models](https://arxiv.org/pdf/2305.20002.pdf), May 31 2023. `recommender system`.
- [Shapley Value on Probabilistic Classifiers](https://arxiv.org/pdf/2306.07171.pdf), Jun. 12 2023.
- [Evaluating Data Attribution for Text-to-Image Models](https://arxiv.org/pdf/2306.09345.pdf), Jun. 15 2023. [code](https://arxiv.org/pdf/2306.09345.pdf).
- [OpenDataVal: a Unified Benchmark for Data Valuation](https://arxiv.org/pdf/2306.10577.pdf), Jun. 18 2023.
- [2D-Shapley: A Framework for Fragmented Data Valuation](https://arxiv.org/pdf/2306.10473.pdf), Jun. 18 2023.
- [A Model-free Closeness-of-influence Test for Features in Supervised Learning](https://arxiv.org/pdf/2306.11855.pdf), Jun. 20 203.
- [Data Selection for Fine-tuning Large Language Models Using Transferred Shapley Values](https://arxiv.org/pdf/2306.10165.pdf), Jun. 16 2023. `application of data shapley`.
- [Variance reduced shapley value estimation for trustworthy data valuation](https://www.sciencedirect.com/science/article/pii/S0305054823001697), Jun. 21 2023.
- [Tools for Verifying Neural Models' Training Data](https://arxiv.org/abs/2307.00682), Jul. 2 2023.
- [Threshold KNN-Shapley: A Linear-Time and Privacy-Friendly Approach to Data Valuation](https://arxiv.org/pdf/2308.15709.pdf), Aug. 30 2023.
- [Representer Point Selection for Explaining Regularized High-dimensional Models](https://proceedings.mlr.press/v202/tsai23a/tsai23a.pdf), `icml2023`.
  - can the method be used for tree ensemble models?
- [Anchor Points: Benchmarking Models with Much Fewer Examples](https://arxiv.org/pdf/2309.08638.pdf), Sep. 14 2023.
- [Shapley Based Residual Decomposition for Instance Analysis](https://proceedings.mlr.press/v202/liu23b/liu23b.pdf), `icml2023`.
- [Data Banzhaf: A Robust Data Valuation Framework for Machine Learning](https://proceedings.mlr.press/v206/wang23e/wang23e.pdf), `aistats2023`.
- [Understanding deep gradient leakage via inversion influence functions](https://arxiv.org/pdf/2309.13016.pdf), Sep. 22 2023.
  - [Deep Leakage from Gradients](https://arxiv.org/pdf/1906.08935.pdf), Dec. 19 2019. `nips2019`.
- [What Learned Representations and Influence Functions Can Tell Us About Adversarial Examples](https://arxiv.org/pdf/2309.10916.pdf), Sep. 21 2023.
- [NPEFF: Non-negative per-example fisher factorization](https://arxiv.org/pdf/2310.04649.pdf), Oct. 7 2023. [code](https://github.com/mmatena/npeff_ref).
- [Composable coresets for determinant maximization: greedy is almost optimal](https://browse.arxiv.org/pdf/2309.15286.pdf), Sep. 26 2023.
- [Accelerated Shapley Value Approximation for Data Evaluation](https://arxiv.org/pdf/2311.05346.pdf), Nov. 9 2023.
- [On Training Data Influence of GPT Models](https://aclanthology.org/2024.emnlp-main.183.pdf), ACL 2024. [code](https://github.com/ernie-research/gptfluence).
- [Towards tracing factual knowledge in language models back to the training data](https://arxiv.org/pdf/2205.11482), Oct. 25 2022.
- [Simfluence: Modeling the influence of individual training examples by simulating training runs](https://arxiv.org/pdf/2303.08114), Mar. 14 2023.
- [Scalable influence and fact tracing for large language model pretraining](https://arxiv.org/pdf/2410.17413), Oct. 22 2024.

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

#### Codes for TDA to reuse

- [quanda](https://github.com/dilyabareeva/quanda). [paper](https://arxiv.org/pdf/2410.07158).
- [dattri](https://github.com/TRAIS-Lab/dattri). [paper](https://arxiv.org/pdf/2410.04555).
- [simple-data-attribution](https://github.com/vasusingla/simple-data-attribution). [paper](https://arxiv.org/abs/2311.03386).
- [bayesian-tda](https://github.com/ElisaNguyen/bayesian-tda). [paper](https://arxiv.org/pdf/2305.19765), NeurIPS 2023.
- [instance_attribution_NLP]. [paper](https://arxiv.org/abs/2104.04128).
- [influence-function-analysis](https://github.com/xhan77/influence-function-analysis/tree/master). [paper](https://arxiv.org/abs/2005.06676), ACL 2020.

### Counterfactual and contrastive explanation

- [Explanations based on the Missing: Towards Contrastive Explanations with Pertinent Negatives](https://proceedings.neurips.cc/paper/2018/file/c5ff2543b53f4cc0ad3819a36752467b-Paper.pdf), `nips2018`.
- [Learning Model-Agnostic Counterfactual Explanations for Tabular Data](https://arxiv.org/pdf/1910.09398.pdf), May 3 2020.
- [Counterfactual Explanations for Machine Learning: A Review](https://arxiv.org/pdf/2010.10596.pdf), Oct. 20 2020.
- [FACE: Feasible and Actionable Counterfactual Explanations](https://arxiv.org/pdf/1909.09369.pdf), Feb. 24 2020.
- [DeDUCE: Generating Counterfactual Explanations At Scale](https://xai4debugging.github.io/files/papers/deduce_generating_counterfactu.pdf), `XAI4Debugging workshop`, 2021.
- [Robust Counterfactual Explanations for Tree-Based Ensembles](https://arxiv.org/pdf/2207.02739.pdf), Jul. 17 2022. `icml2022`.
- [Diffusion Visual Counterfactual Explanations](https://arxiv.org/pdf/2210.11841.pdf), Oct. 21 2022. `nips2022`.
- [Counterfactual Generation Under Confounding](https://arxiv.org/abs/2210.12368), Oct. 22 2022.
- [Explaining Model Confidence Using Counterfactuals](https://arxiv.org/pdf/2303.05729.pdf), Mar. 10 2023.
- [Semi-supervised counterfactual explanations](https://arxiv.org/pdf/2303.12634.pdf), Mar. 22 2023.
- [Explaining Groups of Instances Counterfactually for XAI: A Use Case, Algorithm and User Study for Group-Counterfactuals](https://arxiv.org/pdf/2303.09297.pdf), Mar. 16 2023.
- [Ensemble of counterfactual explainers](https://arxiv.org/pdf/2308.15194.pdf), Aug. 29 2023.
- [Learning to Counterfactually Explain Recommendations](https://arxiv.org/pdf/2211.09752.pdf), Feb. 8 2023. `counterfactual` `interpretability`.
- [Counterfactual explanations via locally-guided sequential algorithmic recourse](https://arxiv.org/pdf/2309.04211.pdf), Sep. 8 2023. `counterfactual` `algorithmic recourse`.
- [Flexible and Robust Counterfactual Explanations with Minimal Satisfiable Perturbations](https://arxiv.org/pdf/2309.04676.pdf), Sep. 9 2023.

## Visualization methods

- [Can Neural Nets Learn the Same Model Twice? Investigating Reproducibility and Double Descent from the Decision Boundary Perspective](https://arxiv.org/pdf/2203.08124.pdf), Mar. 15 2022.
- [A Spectral Method for Assessing and Combining Multiple Data Visualizations](https://arxiv.org/pdf/2210.13711.pdf), Oct. 25 2022.
- [Understanding the Evolution of Linear Regions in Deep Reinforcement Learning](https://arxiv.org/pdf/2210.13611.pdf), Oct. 24 2022.
- [Understanding how dimensioin reduction tools work: an empirical approach to deciphering t-SNE, UMAP, TriMap, and PaCMAP for data visualization](https://www.jmlr.org/papers/volume22/20-1061/20-1061.pdf), `jmlr2021`.
- [Improved visualization of high-dimensional data using the distance-to-distance transformation](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010764), Dec. 20 2022. learned this method from [tweet](https://twitter.com/hippopedoid/status/1613096715963211776?cn=ZmxleGlibGVfcmVjcw%3D%3D&refsrc=email).

## Representation comparison

- [Similarity of Neural Network Representations Revisited](https://proceedings.mlr.press/v97/kornblith19a/kornblith19a.pdf), `icml2019`.
- [Revisiting model stitching to compare neural representations](https://proceedings.neurips.cc/paper/2021/file/01ded4259d101feb739b06c399e9cd9c-Paper.pdf), `nips2021`.
- [Representation Alignment in Neural Networks](https://openreview.net/pdf?id=fLIWMnZ9ij), `tmlr2022`.
- [GULP: a prediction-based metric between representations](https://arxiv.org/pdf/2210.06545.pdf), arXiv Oct. 12 2022. `probing`?
  - This paper introduces a family of distance measures between representations that is explicitly motivated by downstream predictive tasks.
- [On the Versatile Uses of Partial Distance Correlation in Deep Learning](https://arxiv.org/pdf/2207.09684.pdf), Jul. 20 2022.
- [Representational Dissimilarity Metric Spaces for Stochastic Neural Networks](https://openreview.net/forum?id=xjb563TH-GH), `iclr2023`.
- [Pointwise Representational Similarity](https://arxiv.org/pdf/2305.19294.pdf), May 30 2023. `nips2023 submitted`.

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
- [Interventional Probing in High Dimensions: An NLI Case Study](https://arxiv.org/pdf/2304.10346.pdf), Apr. 20 2023.
- [A Geometric Notion of Causal Probing](https://arxiv.org/pdf/2307.15054.pdf), Jul. 27 2023.

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

- [Fast, Accurate, and Simple Models for Tabular Data via Augmented Distillation](https://arxiv.org/pdf/2006.14284.pdf), Jun. 25 2020.
- [Gaining Free or Low-Cost Transparency with Interpretable Partial Substitute](https://arxiv.org/pdf/1802.04346.pdf), `icml2019`.
- [Self-explaining deep models with logic rule reasoning](https://www.microsoft.com/en-us/research/uploads/prod/2022/10/CameraReady_NeurIPS22_SELOR.pdf), `nips2022`.
- [Probabilistic Concept Bottleneck Models](https://arxiv.org/pdf/2306.01574.pdf), Jun. 2 2023.
- [Explainable AI using expressive Boolean formulas](https://arxiv.org/pdf/2306.03976.pdf), Jun. 6 2023.
- [Tilted Sparse Additive Models](https://fengxianghe.github.io/paper/wang2023tilted.pdf), `icml2023`.
- [Weighted Automata Extraction and Explanation of Recurrent Neural Networks for Natural Language Tasks](https://arxiv.org/pdf/2306.14040.pdf), Jun. 24 2023. [code](https://github.com/weizeming/Extract_WFA_from_RNN_for_NL).
- [Route, interpret, repeat: Blurring the line between post-hoc explainability and interpretable models](https://arxiv.org/pdf/2307.05350.pdf), Jul. 7 2023.
  - _"we propose beginning with a flexible BlackBox model and gradually carving out a mixture of interpretable models and a residual network"_
- [Learning by Self-Explaining](https://arxiv.org/pdf/2309.08395.pdf), Sep. 15 2023.

### Rule learning

#### Rule extraction from NNs

- [Using Neural Network Rule Extraction and Decision Tables for Credit-Risk Evaluation](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=9dc6ca547ec5caf449540655892ee66f5e0fd64c), 2003.
- [Rule extraction: using neural networks on neural networks](https://jcst.ict.ac.cn/en/article/pdf/preview/1004.pdf), 2004. `Zhi-Hua Zhou`.
- [Rule Extraction from Recurrent Neural Networks: A Taxonomy and Review](https://www.diva-portal.org/smash/get/diva2:2402/FULLTEXT01.pdf), 2005.
- [Rule Extraction Algorithm for Deep Neural Networks: A Review](https://arxiv.org/ftp/arxiv/papers/1610/1610.05267.pdf), 2016.
- [DeepRED: rule extraction from deep neural networks](https://link.springer.com/chapter/10.1007/978-3-319-46307-0_29), 2016. [slides](https://pdfs.semanticscholar.org/d41e/d85d5808addd0320b55cbd28415e15687854.pdf).
- [Learning Accurate and Interpretable Decision Rule Sets from Neural Networks](https://arxiv.org/abs/2103.02826), `aaai2021`. [code](https://github.com/Joeyonng/decision-rules-network).
- [Efficient Decompositional Rule Extraction for Deep Neural Networks](https://arxiv.org/pdf/2111.12628.pdf). [code: REMIX](https://github.com/mateoespinosa/remix).

#### Logic-based model

- [Interpretable and Explainable Logical Policies via Neurally Guided Symbolic Abstraction](https://arxiv.org/pdf/2306.01439.pdf), Jun. 2 2023. `neuro-symbolic` `rl`.
- [Learning Reliable Logical Rules with SATNet](https://browse.arxiv.org/pdf/2310.02133.pdf), Oct. 3 2023. `nips2023`.

#### Rule list model

- [Falling rule lists](http://proceedings.mlr.press/v38/wang15a.pdf), `aistats2015`.
- [Interpretable classifiers using rules and Bayesian analysis: Building a better stroke prediction model](https://projecteuclid.org/journals/annals-of-applied-statistics/volume-9/issue-3/Interpretable-classifiers-using-rules-and-Bayesian-analysis--Building-a/10.1214/15-AOAS848.full), Sep. 2015, Ann. Appl. Stat. 9(3).
- [A bayesian framework for learning rule sets for interpretable classification](https://www.jmlr.org/papers/volume18/16-003/16-003.pdf), `jmlr2017`.
- [Learning certifiably optimal rule lists for categorical data](https://www.jmlr.org/papers/volume18/17-716/17-716.pdf), `jmlr2018`.
- [Scalable Bayesian rule lists](http://proceedings.mlr.press/v70/yang17h/yang17h.pdf), `icml2017`. [long version](https://arxiv.org/pdf/1602.08610.pdf).
- [Globally-Consistent Rule-Based Summary-Explanations for Machine Learning Models: Application to Credit-Risk Evaluation](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3395422), Jun. 12 2019.
- [Compressed Rule Ensemble Learning](https://proceedings.mlr.press/v151/nalenz22a/nalenz22a.pdf), `aistats2022`.
- [Causal rule sets for identifying subgroups with enhanced treatment effects](https://arxiv.org/pdf/1710.05426), `informs2022`.
- [Fire: An Optimization Approach for Fast Interpretable Rule Extraction](https://arxiv.org/pdf/2306.07432.pdf), Jun. 12 2023.

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
- [PiML](https://github.com/SelfExplainML/PiML-Toolbox), PiML (Python Interpretable Machine Learning) toolbox for model development & diagnostics. [paper](https://arxiv.org/pdf/2305.04214.pdf).

## Interpretability for traditional and tree models

- [Interpreting Tree Ensembles with inTrees](https://arxiv.org/pdf/1408.5456.pdf), Aug. 23 2014. [R package](https://github.com/softwaredeng/inTrees).
- [Making Tree Ensembles Interpretable: A Bayesian Model Selection Approach](https://proceedings.mlr.press/v84/hara18a/hara18a.pdf), `aistats2018`. [code](https://github.com/sato9hara/defragTrees).
- [Born-again tree ensembles](https://arxiv.org/abs/2003.11132), Aug. 2020. [slides](https://icml.cc/media/Slides/icml/2020/virtual(no-parent)-16-13-00UTC-6682-born-again_tree.pdf).
- [Generalized and Scalable Optimal Sparse Decision Trees](https://arxiv.org/abs/2006.08690), Jun. 15 2020. `icml2020`.
- [Neural Prototype Trees for Interpretable Fine-grained Image Recognition](https://openaccess.thecvf.com/content/CVPR2021/papers/Nauta_Neural_Prototype_Trees_for_Interpretable_Fine-Grained_Image_Recognition_CVPR_2021_paper.pdf), `cvpr2021`.
- [A comparison among interpretative proposals for Random Forests](https://www.sciencedirect.com/science/article/pii/S2666827021000475), Dec. 15 2021.
- [Approximating XGBoost with an interpretable decision tree](https://dl.acm.org/doi/abs/10.1016/j.ins.2021.05.055), 2021. [code](https://github.com/sagyome/XGBoostTreeApproximator).
- [TE2Rules: Explaining Tree Ensembles using Rules](https://arxiv.org/pdf/2206.14359.pdf), Aug. 25 2023. [code](https://github.com/linkedin/TE2Rules).
- [Finding Minimum-Cost Explanations for Predictions made by Tree Ensembles](https://arxiv.org/pdf/2303.09271.pdf), Mar. 16 2023.
- [Abstract Interpretation of Decision Tree Ensemble Classifiers](https://ojs.aaai.org/index.php/AAAI/article/view/5998), `aaai2020`.

### Feature importance

- [Improved feature importance computation for tree models based on the Banzhaf value](https://proceedings.mlr.press/v180/karczmarz22a.html), `uai2023`.

### Visualization of tree models

- [iForest: Interpreting Random Forests via Visual Analytics](https://ieeexplore.ieee.org/abstract/document/8454906), `tvcg2019`.

### Influence for tree models

- [Influence Functions for CART](https://hal.science/hal-00562039/document), Feb. 2 2011. [presentation](https://www.gdr-mascotnum.fr/media/mascot11poggi.pdf).
- [Finding Influential Training Samples for Gradient Boosted Decision Trees](https://arxiv.org/pdf/1802.06640.pdf), Mar. 2018. `icml2018`. [code](https://github.com/bsharchilev/influence_boosting).
- [Towards Efficient Data Valuation Based on the Shapley Value](http://proceedings.mlr.press/v89/jia19a/jia19a.pdf), `aistats2019`.
  - [Machine Unlearning for Random Forests](http://proceedings.mlr.press/v139/brophy21a/brophy21a.pdf), `icml2021`.
- [Tree space prototypes: another look at making tree ensembles interpretable](https://arxiv.org/pdf/1611.07115.pdf), `fods2020`. [code](https://github.com/shftan/tree_ensemble_distance).
- [TREX: Tree-Ensemble Representer-Point Explanations](https://arxiv.org/abs/2009.05530), Sep. 11 2020. [code](https://github.com/jjbrophy47/trex).
- [Group’s Influence Value in Logistic Regression Model and Gradient Boosting Model](https://link.springer.com/chapter/10.1007/978-981-16-2377-6_66), 2021.
- [Adapting and Evaluating Influence-Estimation Methods for Gradient-Boosted Decision Trees](https://arxiv.org/pdf/2205.00359.pdf), Apr. 30 2022, [jmlr version](https://www.jmlr.org/papers/volume24/22-0449/22-0449.pdf), `jmlr2023`. [code](https://github.com/jjbrophy47/tree_influence).

### Uncertainty

- [DAW RF: Data AWare Random Forests](https://github.com/jjbrophy47/daw), 2022.
- [Instance-Based Uncertainty Estimation for Gradient-Boosted Regression Trees](https://arxiv.org/abs/2205.11412), May 23 2022. [code](https://github.com/jjbrophy47/ibug). `uncertainty estimation`.

### Adversarials and robustness

- [Robust Decision Trees Against Adversarial Examples](https://proceedings.mlr.press/v97/chen19m/chen19m.pdf), `icml2019`.
- [Provably Robust Boosted Decision Stumps and Trees against Adversarial Attacks](https://proceedings.neurips.cc/paper_files/paper/2019/file/4206e38996fae4028a26d43b24f68d32-Paper.pdf), `nips2019`.
- [Robustness Verification of Tree-based Models](https://proceedings.neurips.cc/paper_files/paper/2019/file/cd9508fdaa5c1390e9cc329001cf1459-Paper.pdf), `nips2019`.
- [Defending Against Adversarial Attacks Using Random Forest](https://openaccess.thecvf.com/content_CVPRW_2019/papers/CV-COPS/Ding_Defending_Against_Adversarial_Attacks_Using_Random_Forest_CVPRW_2019_paper.pdf), `cvpr2019`.
- [An Efficient Adversarial Attack for Tree Ensembles](https://proceedings.neurips.cc/paper/2020/file/ba3e9b6a519cfddc560b5d53210df1bd-Paper.pdf), `nips2020`.
- [Certifying Robustness to Programmable Data Bias in Decision Trees](https://proceedings.neurips.cc/paper_files/paper/2021/file/dcf531edc9b229acfe0f4b87e1e278dd-Paper.pdf), `nips2021`.
- [Not all datasets are born equal: On heterogeneous tabular data and adversarial examples](https://arxiv.org/abs/2010.03180), `knowledge-base system 2022`.

## Human-in-the-loop

- [Human Uncertainty in Concept-Based AI Systems](https://arxiv.org/pdf/2303.12872.pdf), Mar. 22 2023.




