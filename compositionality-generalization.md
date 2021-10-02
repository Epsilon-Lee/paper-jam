
### Compositionality and Neural Machine Translation

- [On Compositionality in Neural Machine Translation](https://arxiv.org/pdf/1911.01497.pdf), 2019.
  - Discuss two manifestations of compostionality in NMT:
    - Productivity  - the ability of the model to extend its predictions beyond the observed length in training data
    - Systematicity - the ability of the model to systematically combine known parts and rules
  - They find that `inadequate temporal processing` in the form of poor encoder representations is a bottleneck for both productivity and systematicity.
  - They also propose simple pretraining mechanism which leads to significant BLEU improvement.
- [On Compositional Generalization of Neural Machine Translation](https://aclanthology.org/2021.acl-long.368.pdf), `acl` 2021.
- [The paradox of the compositionality of natural language: a neural machine translation case study](https://arxiv.org/pdf/2108.05885.pdf), 2021.


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
- [Location Attention for Extrapolation to Longer Sequences](https://www.aclweb.org/anthology/2020.acl-main.39.pdf), `acl` 2020.
- [Inducing Transformer's Compositional Generalization Ability via Auxiliary Sequence Prediction Tasks](https://arxiv.org/abs/2109.15256), `emnlp` 2021.

### Compositionality and interpretability

- [Attention vs Non-attention for a Shapley-based Explanation Method](https://aclanthology.org/2021.deelio-1.13.pdf), `naacl` 2021.
