
This small article aims at explaining the meaning of some recent terminologies, such as `model editing` or `algorithmic recourse` or `test-time fast adaptation` et al.

- :white_heart: [A survey of algorithmic recourse: contrastive explanations and consequential recommendations](https://arxiv.org/pdf/2010.04050.pdf), Mar. 1 2021.

In the above paper, the authors define `algorithmic recourse` as: ***providing explanations and recommendations to individuals who are unfavorably treated by automated decision-making systems***.
In their introduction, the authors give an example with a loan-granting situation where a female software engineer was rejected the loan.
In this scenario, she might ask the following questions:

- Q1. Why was I rejected the loan?
- Q2. What can I do in order to get the loan in the future? (Counterfactual)

There are many aspects of the research with `algorithmic recourse`, one that I am intereted most is:

- designing models that not only have high objective accuracy but also ***afford the individual with explanations and recommendations to favourably change their situation***

> **Inspiration.** This makes me thinking about situations in neural machine translation, where the input is not translated good enough by the model; if the model could ask the user to (slightly) adjust the input a little bit, e.g. lexical simlification, partial paraphrasing, sentence order adjustment et al. so that the model can handle the input as well as possible.
