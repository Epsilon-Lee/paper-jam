
_[Init. 11.21.2022]_

This discussion is motivated by the lunch talk with a colleague.
I argue that self-attention is probably the ***end of architecture search*** by human engineering due to its (might-be) proved advantages over feedforward, recurrent and convolutional neural networks [1, 2].
However, my colleague thinks that there _"must"_ be some neural archs that can be more like the human neural system which gives birth to human intelligence.
To be specific, the connection of human neural system is a more complex than current forward-directed acyclic graph of computation, though might not have the same number of neurons as current scaled large neural networks (NNs) [3].
So how about acyclic computational graph?

Before reading about recent two works on the modular inductive bias of NNs [4, 5] (though more works are related [6, 7]), the first question that strikes myself is:
- What are the possible strengths of cyclic computation graphs compared to acyclic graphs?

To conduct a thought experiment, I can think about recurrent NNs as constructing a cyclic graph.
However, when unfolding it along the time axis, it is sharing parameters across different timesteps or depths of layers with each layer received a new input $x_t$.
So the key characteristics of recurrent NNs might be parameter sharing and this is already exploited by self-attention, where the projection matrices $Q, K, V$ are shared across different timesteps or depths as well.
So recurrent NN is not cyclic in terms of NN architecture.
<!-- So the **real cyclic computation graph** might be defined as _containing cyclic paths with different link parameters instead of shared parameters_. -->
So the real cyclic computation graph is defined as a computation graph with cycle in at least one input to output path, where the cycle cannot be removed by unfolding.
This definition reminds me of Hopfield NNs [8].

_[Added 11.22.2022]_

Today I read a professor's [tweet](https://twitter.com/DimitrisPapail/status/1593664990103363584) about the future arch of LLMs.
He pointed to two papers [9, 10] about the Turing-completeness of Transformers.
Weight-sharing can lead to recurrence structure which is implemented in Universal Transformers [9].

_[Added 12.2.2022]_

Anirudh Goyal together with his PhD advisor Yoshua Bengio have written a survey on inductive biases for deep learning of high-level cognition [11].
And I witness that recently Bengio's works on future architectures aim to investigate some key contents from [11], e.g. [4, 12, 13].

### Refs

[1] [Unveiling Transformers with LEGO: a synthetic reasoning task](https://arxiv.org/pdf/2206.04301), Jun. 9 2022.

[2] [In-context Learning and Induction Heads](https://arxiv.org/ftp/arxiv/papers/2209/2209.11895.pdf), Mar. 8 2022.

[3] [Transcending Scaling Laws with 0.1% Extra Compute](https://arxiv.org/abs/2210.11399), Oct. 2022.

[4] [Is a Modular Architecture Enough?](https://arxiv.org/abs/2206.02713), Jun. 6 2022.

[5] [Are Neural Nets Modular? Inspecting Functional Modularity Through Differentiable Weight Masks](https://arxiv.org/abs/2010.02066), Oct. 5 2020.

[6] [On the Binding Problem in Artificial Neural Networks](https://arxiv.org/abs/2012.05208), Dec. 9 2020.

[7] [Linear transformers are secretly fast weight programmers](http://proceedings.mlr.press/v139/schlag21a/schlag21a.pdf), `icml2021`.

[8] [Hopfield networks is all you need](https://arxiv.org/pdf/2008.02217.pdf), 2020.

[9] [Universal Transformers](https://openreview.net/forum?id=HyzdRiR9Y7&noteId=HyxfZDmCk4&noteId=rkginvfklN), 2019.

[10] [Transformers Learn Shortcuts to Automata](https://arxiv.org/pdf/2210.10749.pdf), Oct. 19 2022.

[11] [Inductive biases for deep learning of high-level cognition](https://royalsocietypublishing.org/doi/pdf/10.1098/rspa.2021.0068), Accepted on Sep. 13 2022.

[12] [Neural attentive circuits](https://arxiv.org/pdf/2210.08031), `nips2022`.

[13] [Synergy between disentanglement and sparsity: a multi-task learning perspective](https://arxiv.org/pdf/2211.14666.pdf), Nov. 26 2022.
