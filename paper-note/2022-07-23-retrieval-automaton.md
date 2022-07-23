
- [Neuro-Symbolic Language Modeling with Automaton-augmented Retrieval](https://proceedings.mlr.press/v162/alon22a/alon22a.pdf), `icml2022`.

Previous challenge of retrieval-based language model - computationally costly datastore search, *"which can be performed as frequently as every time step"*.
The proposed `retomaton` can approximate the datastore search by
1. saving pointers between consecutive datastore entries
2. clustering of entries into "states"
This can result into a weighted finite automaton built on top of the datastore in an unsupervised manner. Thus, the benefits are:
- the construction of such automaton can be imposed beyond the training set to any other text storages of other domains;
- PPL reduces about 1.85;
- saving about 83% kNN search over kNN-LM without hurting PPL.

### kNN-LM

$k$NN-LM

