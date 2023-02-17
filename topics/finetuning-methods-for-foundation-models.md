
# Few-shot and fine-tuning methods for foundation models

- [How to Fine-Tune BERT for Text Classification?](https://arxiv.org/pdf/1905.05583.pdf), arXiv.v3 Feb. 5 2020.
- [Don’t Stop Pretraining: Adapt Language Models to Domains and Tasks](https://arxiv.org/pdf/2004.10964.pdf), arXiv.v5 May 5 2020.
- [Recall and learn: Fine-tuning deep pretrained language models with less forgetting](https://arxiv.org/abs/2004.12651), `emnlp2018`.
- [Mixout: Effective regularization to finetune large-scale pretrained language models](https://arxiv.org/abs/1909.11299), `iclr2020`.
- [Smart: Robust and efficient fine-tuning for pre trained natural language models through principled regularized optimization](https://arxiv.org/abs/1911.03437), `acl2020`.
- [Domain Adversarial Fine-Tuning as an Effective Regularizer](https://arxiv.org/abs/2009.13366), [github](https://github.com/GeorgeVern/AFTERV1.0), `emnlp2020`.
- [Better fine-tuning by reducing representational collapse](https://arxiv.org/abs/2008.03156), `iclr2021`.
- [NoisyTune: A Little Noise Can Help You Finetune Pretrained Language Models Better](https://arxiv.org/pdf/2202.12024.pdf), `acl2022`.
- [Raise a child in large language model: Towards effective and generalizable fine-tuning](https://arxiv.org/abs/2109.05687), `emnlp2022`.
- [AD-DROP: Attribution-Driven Dropout for Robust Language Model Fine-Tuning](https://arxiv.org/pdf/2210.05883.pdf), `nips2022`.
- [ROSE: Robust Selective Fine-tuning for Pre-trained Language Models](https://arxiv.org/pdf/2210.09658.pdf), `emnlp2022`.
- [Surgical Fine-Tuning Improves Adaptation to Distribution Shifts](https://arxiv.org/pdf/2210.11466.pdf), Oct. 20 2022.
- [Finetuning language models via epistemic neural networks](https://arxiv.org/pdf/2211.01568.pdf), Nov. 2 2022.
- [Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time](https://proceedings.mlr.press/v162/wortsman22a/wortsman22a.pdf), `icml2022`.
- [Finetune like you pretrain: Improved finetuning of zero-shot vision models](https://arxiv.org/pdf/2212.00638.pdf), Dec. 1 2022. `vision`
- [PEST: Combining Parameter-Efficient Fine-Tuning with Self-Training and Co-Training](https://neurips2022-enlsp.github.io/papers/paper_27.pdf), `nips2022`.
- [SubTuning: Efficient Finetuning for Multi-Task Learning](https://arxiv.org/pdf/2302.06354.pdf), Feb. 14 2023.

## Few-shot methods

- [Improving Few-Shot Generalization by Exploring and Exploiting Auxiliary Data](https://arxiv.org/pdf/2302.00674.pdf), Feb. 1 2023. [tweet](https://github.com/alon-albalak/FLAD).
  - _"the use of a small training set makes it difficult to avoid overfitting"_
  - proposes a training paradigm that assuems access to auxiliary data, aka FLAD (few-shot learning with auxiliary data)
  - _"finding that the combination of exploration and exploitation is crucial"_
  - **challenges of FLAD**: increased algorithmic and computational complexity, _"incorporating auxiliary data during training introduces a large space of design choices for FLAD algorithms (e.g. how and when to train on auxiliary data)"_ ✋Is FLAD similar to multi-task learning?
  - From manually designing the curriculum of learning on large quantities of auxiliary data to delegating such choices to an algorithm, however this further introduces algorithmic complexity, motivating the search for efficient methods as the quantity of auxiliary data grows
  - desiderata of FLAD:
    - makes no assumption on available auxiliary data a-priori (in-domain, on-task, quality, quantity, etc.)
    - continuously updates belief on importance of auxiliary data, and
    - adds minimal memory and computational overhead.
