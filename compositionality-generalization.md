### Measuring compositionality

- [A benchmark for systematic generalization in grounded language understanding](https://arxiv.org/pdf/2003.05161), `icml` 2020.
- [Measuring Compositional Generalization: A comprehensive method on realistic data](https://arxiv.org/pdf/1912.09713.pdf), ICLR 2020, Google Brain.
  - They propose a realistic benchmark for evaluating compositional generalization on question answering task that 1) **maximizes compound divergence**, 2) **mininizes atom divergence**


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

- [Compositionality Decomposed:  How doNeural Networks Generalise?](https://jair.org/index.php/jair/article/view/11674/26576), JAIR 2020.
  - Collect different definitions of compositionality and translate them into five task-independent tests
    1. can models systematically combine known parts and rules?
    2. can models extends its prediction beyond the lengths in training set?
    3. are the model's composition operations local or global?
    4. can models' prediction be robust to synonym substitution (paraphrasing)?
    5. does the model favor rules or exceptions during training?


### Impose compositionality on (neural) models

- [On the Realization of Compositionality in Neural Networks](https://www.aclweb.org/anthology/W19-4814), 2019.
- [Compositional Generalization for Primitive Substitutions](https://aclanthology.org/D19-1438.pdf), `emnlp` 2019.
- [Compositional Generalization via Neural-Symbolic Stack Machines](https://papers.nips.cc/paper/2020/file/12b1e42dc0746f22cf361267de07073f-Paper.pdf), `nips` 2020.
- [Compositional generalization in semantic parsing: Pre-training vs. specialized architectures](https://arxiv.org/pdf/2007.08970), 2020.
- [Location Attention for Extrapolation to Longer Sequences](https://www.aclweb.org/anthology/2020.acl-main.39.pdf), `acl` 2020.
- [Compositional Generalization and Natural Language Variation: Can a Semantic Parsing Approach Handle Both?](https://arxiv.org/pdf/2010.12725), 2020.
- [Inducing Transformer's Compositional Generalization Ability via Auxiliary Sequence Prediction Tasks](https://arxiv.org/abs/2109.15256), `emnlp` 2021.
- [Making Transformers Solve Compositional Tasks](https://arxiv.org/pdf/2108.04378.pdf), Google Research 2021.
- [Unlocking Compositional Generalization in Pre-trained Models Using Intermediate Representations](https://arxiv.org/pdf/2104.07478.pdf), 2021.

#### Data augmentation based approach

- [Learning compositional rules via neural program synthesis](https://arxiv.org/pdf/2003.05562), 2020.
- [Learning data manipulation for augmentation and weighting](https://arxiv.org/pdf/1910.12795), `nips` 2019.
- [Learning to recombine and resample data for compositional generalization](https://arxiv.org/pdf/2010.03706), `iclr` 2020.
- [Good-Enough Compositional Data Augmentation](https://aclanthology.org/2020.acl-main.676.pdf), `acl` 2020.
- [Improving Compositional Generalization in Semantic Parsing](https://arxiv.org/pdf/2010.05647.pdf), 2020.
- [Neural Data Augmentation via Example Extrapolation](https://arxiv.org/pdf/2102.01335.pdf), 2021.
- [Revisiting Iterative Back-Translation from the Perspective of Compositional Generalization](https://arxiv.org/pdf/2012.04276.pdf), `aaai` 2021.

### Compositionality and interpretability

- [L-Shapley and C-Shapley: Efficient Model Interpretation for Structured Data](https://arxiv.org/pdf/1808.02610.pdf), 2018.
- [Compositional explanations of neurons](https://arxiv.org/pdf/2006.14032), `nips` 2020.
- [Attention vs Non-attention for a Shapley-based Explanation Method](https://aclanthology.org/2021.deelio-1.13.pdf), `naacl` 2021.
- [Syntactic Perturbations Reveal Representational Correlates of Hierarchical Phrase Structure in Pretrained Language Models](https://arxiv.org/pdf/2104.07578.pdf), 2021.
