## Compositionality from/for Machine Learning

### Measuring compositionality

- [Why Fodor and Pylyshyn Were Wrong: A Simplest Refutation](https://uh.edu/~garson/Chalmers.PDF), `compositionality` 1990s.
- [A benchmark for systematic generalization in grounded language understanding](https://arxiv.org/pdf/2003.05161), `icml` 2020.
- [Measuring Compositional Generalization: A comprehensive method on realistic data](https://arxiv.org/pdf/1912.09713.pdf), ICLR 2020, Google Brain.
  - They propose a realistic benchmark for evaluating compositional generalization on question answering task that 1) **maximizes compound divergence**, 2) **mininizes atom divergence**
- 🤍 [Are Representations Built from the Ground Up? An Empirical Examination of Local Composition in Language Models](https://arxiv.org/pdf/2210.03575.pdf), Oct. 22 2022.
  - Empirical investigation into composition of phrase from smaller parts (words, short phrases)
  - They find that LMs may not accurately distinguish between compositional and non-compositional phrases
  - This might also motivate my current (10/25/2022) proposed methods for fraud address detection
- [Break It Down: Evidence for Structural Compositionality in Neural Networks](https://arxiv.org/abs/2301.10884), Jan. 2 2023.

### Compositionality and Neural Machine Translation

- [On Compositionality in Neural Machine Translation](https://arxiv.org/pdf/1911.01497.pdf), 2019.
  - Discuss two manifestations of compostionality in NMT:
    - Productivity  - the ability of the model to extend its predictions beyond the observed length in training data
    - Systematicity - the ability of the model to systematically combine known parts and rules
  - They find that `inadequate temporal processing` in the form of poor encoder representations is a bottleneck for both productivity and systematicity.
  - They also propose simple pretraining mechanism which leads to significant BLEU improvement.
- [On Compositional Generalization of Neural Machine Translation](https://aclanthology.org/2021.acl-long.368.pdf), `acl` 2021.
  - *"our compositional generalization test set consists of 2,160 novel compounds, with up to 5 atoms and 7 words [...] generalization ability can be evaluated based on compound translation error rate"*
  - **Problem Definition**: *atoms*, primitive elements in the train set whereas *compounds* are obtained by composing *atoms*; due to hardness of generating sentence-level compounds, the authors constrain compounds to syntactic constituents, and define atoms as basic semantic components in constituents, and assign randomly multiple sentential contexts for investigating each compounds.
  - **Test a model**: train Transformer on standard benchmark, e.g. WMT17 En-Zh, test it on the constructed dataset, to see its performance on those compounds within the test sentence.
  - **Analysis**: investigating factors - compound frequency, compound length, atom frequency, atom co-occurrence, linguistic factors, external context. 
- [The paradox of the compositionality of natural language: a neural machine translation case study](https://arxiv.org/pdf/2108.05885.pdf), 2021.
  - **Question**
  - **Methodology**: testing *systematicity, substitutivity, overgeneralization*, 


### Define compositionality, or generalize compositionally

- [Compositionality Decomposed:  How do Neural Networks Generalise?](https://jair.org/index.php/jair/article/view/11674/26576), JAIR 2020.
  - Collect different definitions of compositionality and translate them into five task-independent tests
    1. can models systematically combine known parts and rules?
    2. can models extends its prediction beyond the lengths in training set?
    3. are the model's composition operations local or global?
    4. can models' prediction be robust to synonym substitution (paraphrasing)?
    5. does the model favor rules or exceptions during training?
- [How Do Neural Sequence Models Generalize? Local and Global Context Cues for Out-of-Distribution Prediction](https://aclanthology.org/2021.emnlp-main.448.pdf), `emnlp2021`.
- [CTL++: Evaluating Generalization on Never-Seen Compositional Patterns of Known Functions, and Compatibility of Neural Representations](https://arxiv.org/pdf/2210.06350.pdf), Oct. 12 2022.

### Impose compositionality on (neural) models

- [On the Realization of Compositionality in Neural Networks](https://www.aclweb.org/anthology/W19-4814), 2019.
- [Compositional Generalization for Primitive Substitutions](https://aclanthology.org/D19-1438.pdf), `emnlp` 2019.
- [Permutation Equivariant Models for Compositional Generalization in Language](https://openreview.net/forum?id=SylVNerFvr), `iclr2020`.
- [Compositional Generalization via Neural-Symbolic Stack Machines](https://papers.nips.cc/paper/2020/file/12b1e42dc0746f22cf361267de07073f-Paper.pdf), `nips` 2020.
- [Compositional generalization in semantic parsing: Pre-training vs. specialized architectures](https://arxiv.org/pdf/2007.08970), 2020.
- [Location Attention for Extrapolation to Longer Sequences](https://www.aclweb.org/anthology/2020.acl-main.39.pdf), `acl` 2020.
- [Compositional Generalization and Natural Language Variation: Can a Semantic Parsing Approach Handle Both?](https://arxiv.org/pdf/2010.12725), 2020.
- [Inducing Transformer's Compositional Generalization Ability via Auxiliary Sequence Prediction Tasks](https://arxiv.org/abs/2109.15256), `emnlp` 2021.
- [Making Transformers Solve Compositional Tasks](https://arxiv.org/pdf/2108.04378.pdf), Google Research 2021.
- [Unlocking Compositional Generalization in Pre-trained Models Using Intermediate Representations](https://arxiv.org/pdf/2104.07478.pdf), 2021.
- [Iterative Decoding for Compositional Generalization in Transformers](https://arxiv.org/abs/2110.04169), `iclr2022` submitted.
- [Skill Induction and Planning with Latent Language](https://arxiv.org/pdf/2110.01517.pdf), `emnlp2021` Jacob Andreas' group.
- [Compositional Generalization in Dependency Parsing](https://arxiv.org/pdf/2110.06843.pdf), Oct. 13 2021.
- [The Neural Data Router: Adaptive Control Flow in Transformers Improves Systematic Generalization](https://arxiv.org/abs/2110.07732), Oct. 2021.
- [Disentangled Sequence to Sequence Learning for Compositional Generalization](https://arxiv.org/pdf/2110.04655.pdf), Oct. 9 2021.
- [LAGr: Labeling Aligned Graphs for Improving Systematic Generalization in Semantic Parsing](https://arxiv.org/pdf/2110.07572.pdf), Oct. 14 2021.
- [Illiterate DALL-E Learns to Compose](https://arxiv.org/pdf/2110.11405.pdf), Oct. 17 2021.
- [Controllable and Compositional Generation with Latent-Space Energy-Based Models](https://arxiv.org/pdf/2110.10873.pdf), Oct. 21 2021. NVIDIA.
- [Unsupervised Learning of Compositional Energy Concepts](https://arxiv.org/pdf/2111.03042.pdf), Nov. 4 2021. `nips2021`
- [ProTo: Program-Guided Transformer for Program-Guided Tasks](https://arxiv.org/pdf/2110.00804.pdf), Oct. 2021. `compositionality`
- [BeliefBank: Adding Memory to a Pre-Trained Language Model for a Systematic Notion of Belief](https://arxiv.org/pdf/2109.14723.pdf), Sep. 2021.
- [Learning to Generalize Compositionally by Transferring Across Semantic Parsing Tasks](https://arxiv.org/pdf/2111.05013.pdf), Nov. 9 2021.
- [Grounded Graph Decoding Improves Compositional Generalization in Question Answering](https://arxiv.org/pdf/2111.03642.pdf), Nov. 5 2021. `semantic parsing`
- [Systematic Generalization with Edge Transformers](https://arxiv.org/abs/2112.00578), Dec. 1 2021 `nips2021`
- [Improving Coherence and Consistency in Neural Sequence Models with Dual-System, Neuro-Symbolic Reasoning](https://cims.nyu.edu/~brenden/papers/NyeEtAl2021NeurIPS.pdf), `nips2021`
- [Learning to Compose Visual Relations](https://arxiv.org/pdf/2111.09297.pdf), Nov. 17 2021 `nips2021`
- [Improving Compositional Generalization with Latent Structure and Data Augmentation](https://arxiv.org/pdf/2112.07610.pdf), Dec. 14 2021.
- [Compositionality as Lexical Symmetry](https://arxiv.org/pdf/2201.12926.pdf), Jan. 30 2022.
- [Improving Systematic Generalization Through Modularity and Augmentation](https://arxiv.org/pdf/2202.10745.pdf), Feb. 22 2022.
- [Compositional Generalization Requires Compositional Parsers](https://arxiv.org/pdf/2202.11937.pdf), Feb. 24 2022. 

#### Data augmentation based approach

- [Learning compositional rules via neural program synthesis](https://arxiv.org/pdf/2003.05562), 2020.
- [Learning data manipulation for augmentation and weighting](https://arxiv.org/pdf/1910.12795), `nips` 2019.
- [Learning to recombine and resample data for compositional generalization](https://arxiv.org/pdf/2010.03706), `iclr` 2020.
- [Good-Enough Compositional Data Augmentation](https://aclanthology.org/2020.acl-main.676.pdf), `acl` 2020.
- [Improving Compositional Generalization in Semantic Parsing](https://arxiv.org/pdf/2010.05647.pdf), 2020.
- [Neural Data Augmentation via Example Extrapolation](https://arxiv.org/pdf/2102.01335.pdf), 2021.
- [Revisiting Iterative Back-Translation from the Perspective of Compositional Generalization](https://arxiv.org/pdf/2012.04276.pdf), `aaai` 2021.
- [Equivariant Transduction through Invariant Alignment](https://arxiv.org/pdf/2209.10926.pdf), `coling2022`. [code](https://github.com/rycolab/equivariant-transduction).

### Compositionality and interpretability

- [L-Shapley and C-Shapley: Efficient Model Interpretation for Structured Data](https://arxiv.org/pdf/1808.02610.pdf), 2018.
- [Compositional explanations of neurons](https://arxiv.org/pdf/2006.14032), `nips` 2020.
- [Attention vs Non-attention for a Shapley-based Explanation Method](https://aclanthology.org/2021.deelio-1.13.pdf), `naacl` 2021.
- [Syntactic Perturbations Reveal Representational Correlates of Hierarchical Phrase Structure in Pretrained Language Models](https://arxiv.org/pdf/2104.07578.pdf), 2021.
- [Compositional generalization in semantic parsing with pretrained transformers](https://arxiv.org/pdf/2109.15101.pdf), 2021 `pretraining`.
- [Can Transformers Jump Around Right in Natural Language? Assessing Performance Transfer from SCAN](https://arxiv.org/pdf/2107.01366.pdf), 2021 `analysis`
- [Symbolic Brittleness in Sequence Models: on Systematic Generalization in Symbolic Mathematics](https://arxiv.org/pdf/2109.13986.pdf), 2021.
- [Systematic Generalization on gSCAN: What is Nearly Solved and What is Next?](https://arxiv.org/pdf/2109.12243.pdf), Fei Sha et al. `compositionality` `systematicality`

### Compositionality for generation

- [Reduce, Reuse, Recycle: Compositional Generation with Energy-Based Diffusion Models and MCMC](https://arxiv.org/pdf/2302.11552.pdf), Feb. 22 2023. `diffusion model`

## Systematic Generalization

- [What underlies rapid learning and systematic generalization in humans?](https://stanford.edu/~jlmcc/papers/NamMcC21RapidLearningGeneralizationInHumansArxiv.pdf), Jul. 10 2021.

### Learning systemacity

- [Systematic Generalization with Group Invariant Predictions](https://openreview.net/pdf?id=b9PoimzZFJ), `iclr2021`
  - "we consider situations where the presence of dominant simpler correlations with the target variable in a training set can cause an SGD-traind neural network to be less reliant on more persistently correlating complex features."
- [Unobserved Local Structures Make Compositional Generalization Hard](https://arxiv.org/pdf/2201.05899.pdf), Jan. 15 2022.
