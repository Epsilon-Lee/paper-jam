
# Shocking moments of `ODQA`

This log gives a summary of my impression on big events/improvements on `ODQA`. Several judgements are drawn only via comparing performance of retriever (e.g. recall@20) and reader (exact match), so be critical about my comments here!

I am shocked by the results of the following paper on NQ (EM: 52.3 for qa pair retriever > 47.8 of [colBERT-based ODQA](https://arxiv.org/pdf/2007.00814.pdf))

- [PAQ: 65 Million Probably-Asked Questions and What You Can Do With Them](https://arxiv.org/pdf/2102.07033.pdf), Feb. 13 2021.

The power must lie in the PAQ dataset which is a large-scale synthetic qa pairs (65M). As said in the abstract of the above paper:
- *We find that PAQ preempts and caches test questions, enabling RePAQ to match the accuracy of recent retrieve-and-read models, whilst being significantly faster*

> To be clear, closed-book QA and QA-pair retriever are two models that do not require a reader to read a retrieved passage.
> They only rely on a single generative model (given q, that generates a autoregressively), or on nearest-neighbor search to find qa pairs in store with which the a pairt can be directly used as answer.

The biggest question I am feeling about after reading the abstract of PAQ is:

- ***Have our current evaluation benchmarks been broken?*** 

To answer the above question, I think we should understand how the PAQ paper creates the 65M qa-pair dataset.

Several properties of PAQ are listed below:
- PAQ is automatically constructed using a question generation model and Wikipedia;
- questions are generated such that they are likely to appear in ODQA datasets.

The astonishing result of RePAQ makes think about why this memorizing appoach can be on par with or even better than retrieve-and-read paradigm. This leads to a very hard question in question-answering research:
- How does the model generalize? Or what can a model leverage for its generalization?
