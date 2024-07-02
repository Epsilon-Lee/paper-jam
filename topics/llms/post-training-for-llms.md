
# Post-training in bert era

## Few-shot and fine-tuning methods

- [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/pdf/1902.00751), Jun. 13 2019.
- [How to Fine-Tune BERT for Text Classification?](https://arxiv.org/pdf/1905.05583.pdf), arXiv.v3 Feb. 5 2020.
- [Don’t Stop Pretraining: Adapt Language Models to Domains and Tasks](https://arxiv.org/pdf/2004.10964.pdf), arXiv.v5 May 5 2020.
- [Recall and learn: Fine-tuning deep pretrained language models with less forgetting](https://arxiv.org/abs/2004.12651), `emnlp2018`.
- [On the stability of fine-tuning bert: Misconceptions, explainations, and strong baselines](https://openreview.net/pdf?id=nzpLWnVAyah), ICLR 2021.
- [Mixout: Effective regularization to finetune large-scale pretrained language models](https://arxiv.org/abs/1909.11299), `iclr2020`.
- [Smart: Robust and efficient fine-tuning for pre trained natural language models through principled regularized optimization](https://arxiv.org/abs/1911.03437), `acl2020`.
- [Better fine-tuning by reducing representational collapse](https://arxiv.org/abs/2008.03156), `iclr2021`.
- [NoisyTune: A Little Noise Can Help You Finetune Pretrained Language Models Better](https://arxiv.org/pdf/2202.12024.pdf), `acl2022`.
- [Raise a child in large language model: Towards effective and generalizable fine-tuning](https://arxiv.org/abs/2109.05687), `emnlp2022`.
- [Surgical Fine-Tuning Improves Adaptation to Distribution Shifts](https://arxiv.org/pdf/2210.11466.pdf), Oct. 20 2022.
- [Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time](https://proceedings.mlr.press/v162/wortsman22a/wortsman22a.pdf), `icml2022`.
- [Finetune like you pretrain: Improved finetuning of zero-shot vision models](https://arxiv.org/pdf/2212.00638.pdf), Dec. 1 2022. `vision`

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





# Post-training in llms era

## Surveys

- [Large Language Model Alignment: A Survey](https://arxiv.org/pdf/2309.15025), Sep. 26 2023.
- [Aligning Large Language Models with Human: A Survey](https://arxiv.org/pdf/2307.12966), Jul. 24 2023.
- [Instruction Tuning for Large Language Models: A Survey](https://arxiv.org/pdf/2308.10792), Mar. 14 2024.
- [Towards Bidirectional Human-AI Alignment: A Systematic Review for Clarifications, Framework, and Future Directions](https://arxiv.org/pdf/2406.09264), Jun. 17 2024.

## Instruction-tuning (aka. supervised fine-tuning)

- [Parameter-efficient fine-tuning of large-scale pre-trained language models](https://www.nature.com/articles/s42256-023-00626-4.pdf), Nature Machine Intelligence 2023.
- [Towards a unified view of parameter-efficient transfer learning](https://arxiv.org/pdf/2405.14838), Feb. 2 2022.
- [Fine-Tuning Language Models with Just Forward Passes](https://proceedings.neurips.cc/paper_files/paper/2023/file/a627810151be4d13f907ac898ff7e948-Paper-Conference.pdf), NeurIPS 2023.
- [NefTune: Noisy Embeddings Improve Instruction Tuning](https://arxiv.org/pdf/2310.05914), Oct. 10 2023.  
- [Bitune: Bidirectional Instruction-Tuning](https://arxiv.org/pdf/2405.14862), May 23 2024.
- [RE-Adapt: Reverse Engineered Adaptationof Large Language Models](https://arxiv.org/pdf/2405.15007), May 23 2024.
- [Mixture-of-Subspaces in Low-Rank Adaptation](https://arxiv.org/pdf/2406.11909), Jun. 16 2024. `alignment`.
- [BPO: Supercharging Online Preference Learning by Adhering to the Proximity of Behavior LLM](https://arxiv.org/pdf/2406.12168), Jun. 18 2024. `alignment`.

## Peft

- [Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning](https://arxiv.org/pdf/2205.05638), Aug. 26 2022.
- [Choice of PEFT Technique in Continual Learning: Prompt Tuning is Not All You Need](https://arxiv.org/pdf/2406.03216), Jun. 5 2024.

## Preference learning/alignment

- [Scheming AIs Will AIs fake alignment during training in order to get power?](https://arxiv.org/pdf/2311.08379), Nov. 27 2023.
- [Preference Ranking Optimization for Human Alignment](https://arxiv.org/pdf/2306.17492), Feb. 27 2024.
- [Quantifying the Gain in Weak-to-Strong Generalization](https://arxiv.org/pdf/2405.15116), May 24 2024.
- [Value Augmented Sampling for Language Model Alignment and Personalization](https://arxiv.org/pdf/2405.14578v1), May 23 2024.
- [Self-Exploring Language Models: Active Preference Elicitation for Online Alignment](https://arxiv.org/pdf/2405.19332), May 29 2024.
- [Steering without side effects: Improving post-deployment control of language models](https://arxiv.org/pdf/2406.15518), Jun. 21 2024.
- [Unpacking DPO and PPO: Disentangling Best Practices for Learning from Preference Feedback](https://arxiv.org/pdf/2406.09279), Jun. 13 2024.
- [Model Alignment as Prospect Theoretic Optimization](https://openreview.net/pdf?id=iUwHnoENnl), ICML 2024.
- [It Takes Two: On the Seamlessness between Reward and Policy Model in RLHF](https://arxiv.org/abs/2406.07971), Jul. 24 2024.



