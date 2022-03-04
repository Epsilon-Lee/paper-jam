
# Shocking moments of `ODQA`

This log gives a summary of my impression on big events/improvements on `ODQA`. Several judgements are drawn only via comparing performance of retriever (e.g. recall@20) and reader (exact match), so be critical about my comments here!

I am shocked by the results of the following paper on NQ (**EM**: **52.3** for qa pair retriever > **47.8** of [colBERT-based ODQA](https://arxiv.org/pdf/2007.00814.pdf))

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
- **How does the model generalize? Or what can a model leverage for its generalization?**

In analogy of a human student who works hard for perform excellent in exams, she could have two ways to generalize to new questions:
1. *practice with a huge amount of questions and memorize how to work out their answers, so that in examinations*, she can find that most of the questions in the exam are previously encountered;
2. *practice with systematic categories of questions and learn about how to solve certain category inductively*, so that in examinations, when encountered with question that belongs to certain previously studied category, she could solve it as well.

### How about using the PAQ qa pairs to finetune state-of-the-art retrieve-and-read models?

It is natural to use the 65M qa pairs for finetuning **both retriever and reader** of certain retrieve-and-read model.
In the original [DPR](https://arxiv.org/pdf/2004.04906.pdf) paper, the qa pairs used to finetune the retriever and reader modules are listed here in the table below.

| dataset | #train |
| ----    | ----   |
| NQ      | 79168/58880  |
| TQ      | 78785/60413  |
| WQ      | 3417/2474    |
| cTREC   | 1353/1125    |
| SQuAD   | 78713/70096  |

With a total number of about (58880+60413+2474+1125+70096=)192,988 << 65,000,000. (65M/192988~=336). So, as a guess:
- finetuning on this 65M (336 times larger) qa-pair datasets **can** bring **large improments** over original DPR.

![image](https://user-images.githubusercontent.com/7335618/144694538-52e1f744-1fc8-4c2e-b602-c9ac068fde3b.png)

And the above figure extracted from the original DPR paper also demostrates the *scaling law* of increasing **number of training qa pairs**.
That is, if we draw vertical lines on this figure, we can find that there is a large jump moving from 1k to 10k training examples, and also a smaller but nonnegligible jump moving from 10k to 40k training examples.

The following paper also from Facebook AI starts to use PAQ for finetuning the original DPR checkpoint. And the results of retriever finetuning is shown in the following table from this paper.
> Note that, since this paper focus on pretraining of DPR, it does not experiment with the reader module and compare final EM scores. 

- [Domain-matched Pre-training Tasks for Dense Retrieval](https://arxiv.org/pdf/2107.13602.pdf), Jul. 28 2021.

![image](https://user-images.githubusercontent.com/7335618/144693287-d2d26e52-e1ec-4594-8c69-662a096d7b82.png)


### How PAQ generate and filter qa pairs?

