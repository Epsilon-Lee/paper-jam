
# Interpretability and Interpretable Machine Learning

- [Definition of interpretability](#definition-of-interpretability)
- [Interpretation methods](#interpretation-methods)
  - [Feature attribution methods](#feature-attribution-methods)
  - [Critics of feature importance](#critics-of-feature-importance)
  - [Beyond feature attribution](#beyond-feature-attribution)
  - [Dataset attribution methods](#dataset-attribution-methods)
    - [Critics of influence function](#critics-of-influence-function)
- [Analysis](#analysis)
- [Probing methods](#probing-methods)
- [Evaluation](#evaluation)
- [Transparent model](#transparent-model)
  - [Model editing](#model-editing)
  - [Model debugging](#model-debugging)

## Definition of interpretability

- [The Mythos of Model Interpretability](https://arxiv.org/pdf/1606.03490.pdf), 2016.
- [Machine Learning Interpretability: A Survey on Methods and Metrics](https://www.mdpi.com/2079-9292/8/8/832), 2019.
- [Interpretable machine learning: definitions, methods, and applications](https://arxiv.org/pdf/1901.04592.pdf), 2019.
- [Interpretable Machine Learning: A Guide for Making Black Box Models Explainable](https://christophm.github.io/interpretable-ml-book/), 2019.
  - "It is difficult to (mathematically) define interpretability. A non-mathematical definition of interpretability that I like by Miller 2017 is: **interpretability is the degree to which a human can understand the cause of decision**", and another one by Been Kim is "**interpretability is the degree to which a human can consistently predict the model's result**"
- [Consistent Counterfactuals for Deep Models](https://arxiv.org/pdf/2110.03109.pdf), `evaluation of interpretability` `iclr2022` submitted.

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

### Critics of feature importance

- [Interpretation of Neural Networks is Fragile](https://arxiv.org/abs/1710.10547), Oct. 29 2019 `aaai2018`
- [On the (In)fidelity and Sensitivity of Explanations](https://proceedings.neurips.cc/paper/2019/file/a7471fdc77b3435276507cc8f2dc2569-Paper.pdf), `nips2019`

### Beyond feature attribution

- [Interpretability beyond feature attribution: Quantitative testing with concept activation vectors (tcav)](http://proceedings.mlr.press/v80/kim18d/kim18d.pdf), `icml2018`

### Dataset attribution methods

- [Understanding black-box predictions via influence functions](http://proceedings.mlr.press/v70/koh17a/koh17a.pdf), Apr. 2017. `icml2017`
- [Towards Efficient Data Valuation Based on the Shapley Value](http://proceedings.mlr.press/v89/jia19a/jia19a.pdf), `aistats2019`
- [Interpreting Black Box Predictions using Fisher Kernels](http://proceedings.mlr.press/v89/khanna19a/khanna19a.pdf), `aistats2019`
- [Data Shapley: Equitable Valuation of Data for Machine Learning](http://proceedings.mlr.press/v97/ghorbani19c.html), `icml2019`
- [On the Accuracy of Influence Functions for Measuring Group Effects](https://arxiv.org/pdf/1905.13289.pdf), `nips2019`
- [Deep Learning Interpretation: Flip Points and Homotopy Methods](http://proceedings.mlr.press/v107/yousefzadeh20a/yousefzadeh20a.pdf), `ml2020`
- [RelatIF: Identifying Explanatory Training Examples via Relative Influence](http://proceedings.mlr.press/v108/barshan20a/barshan20a.pdf), `aistats2020`
- [On Second-Order Group Influence Functions for Black-Box Predictions](http://proceedings.mlr.press/v119/basu20b/basu20b.pdf), `iclr2020`
- [Approximate Cross-Validation for Structured Models](https://arxiv.org/pdf/2006.12669.pdf), `nips2020`
- [HYDRA: Hypergradient Data Relevance Analysis for Interpreting Deep Neural Networks](https://www.aaai.org/AAAI21Papers/AAAI-8686.ChenY.pdf), `aaai2021`
- [Scaling Up Influence Functions](https://arxiv.org/pdf/2112.03052.pdf), Dec. 6 2021. Google Research `aaai2022`.
- [DIVINE: Diverse Influential Training Points for Data Visualization and Model Refinement](https://arxiv.org/pdf/2107.05978.pdf), Jul. 13 2021.
- [Deconstructing Distributions: A Pointwise Framework of Learning](https://arxiv.org/pdf/2202.09931.pdf), Feb. 20 2022.
- [Datamodels: Predicting Predictions from Training Data](https://arxiv.org/abs/2202.00622), Feb. 1 2022.

#### Critics of influence function

- [Influence Functions in Deep Learning are Fragile](https://arxiv.org/pdf/2006.14651.pdf), Feb. 10 2021, `iclr2021`


#### Leveraging interpretation

/


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

## Probing methods

- [Probing as Quantifying the Inductive Bias of Pre-trained Representations](https://arxiv.org/pdf/2110.08388.pdf), Oct. 15 2021.
- [Counterfactual Interventions Reveal the Causal Effect of Relative Clause Representations on Agreement Prediction](https://arxiv.org/pdf/2105.06965.pdf), Sep. 15 2021.
- [A Latent-Variable Model for Intrinsic Probing](https://arxiv.org/abs/2201.08214), Jan. 20 2022.
- [ADVERSARIAL CONCEPT ERASURE IN KERNEL SPACE](https://arxiv.org/pdf/2201.12191.pdf), Jan. 28 2022.
- [LINEAR ADVERSARIAL CONCEPT ERASURE](https://arxiv.org/pdf/2201.12091.pdf), Jan. 28 2022.
- [On the data requirements of probing](https://arxiv.org/pdf/2202.12801.pdf), Feb. 25 2022.


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

### Model editing

- [Editing a classifier by rewriting its prediction rules](https://arxiv.org/pdf/2112.01008.pdf), Dec. 2 2021.
- [Influence Tuning: Demoting Spurious Correlations via Instance Attribution and Instance-Driven Updates](https://arxiv.org/pdf/2110.03212.pdf), `emnlp2021` with `code`
- [Fast Model Editing at Scale](https://arxiv.org/pdf/2110.11309.pdf), Oct. 21 2021.

### Model debugging

- [Debugging Tests for Model Explanations](https://arxiv.org/pdf/2011.05429.pdf), Nov. 10 2020.
