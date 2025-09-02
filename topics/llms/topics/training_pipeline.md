
> The tricks and trade of training LLMs, namely, pre-training, post-training, peft, preference alignment etc.

### Optimization

- [Deconstructing What Makes a Good Optimizer for Language Models](https://arxiv.org/pdf/2407.07972), Jul. 10 2024.
- [Narrowing the Focus: Learned Optimizers for Pretrained Models](https://arxiv.org/pdf/2408.09310), Aug. 21 2024.
- [Power scheduler: A batch size and token number agnostic learning rate scheduler](https://arxiv.org/pdf/2408.13359), Aug. 23 2024.
- [SOAP: Improving and stabilizing Shampoo using Adam](https://arxiv.org/abs/2409.11321), Sep. 17 2024.
- [Analyzing & reducing the need for learning rate warmup in GPT training](https://arxiv.org/pdf/2410.23922), Oct. 31 2024.
- [How does critical batch size scale in pre-training?](https://arxiv.org/pdf/2410.21676), Oct. 29 2024. [code](https://github.com/hlzhang109/critical-batch-size).
- [What do learning dynamics reveal about generalization in LLM reasoning](https://arxiv.org/pdf/2411.07681), Nov. 12 2024. [code](https://github.com/katiekang1998/reasoning_generalization).
- [MARS: Unleashing the Power of Variance Reduction for Training Large Models](https://arxiv.org/pdf/2411.10438), Nov. 15 2024.
- [Methods of improving LLM training stability](https://arxiv.org/pdf/2410.16682v1), Oct. 22 2024.
- [Optimizing large language model training using fp4 quanitzation](https://arxiv.org/abs/2501.17116), Jan. 28 2025.
- [The surprising agreement between convex optimization theory and learning-rate scheduling for large model training](https://arxiv.org/abs/2501.18965), Jan. 31 2025.
- [A review of deepseek models' key innovative techniques](https://arxiv.org/pdf/2503.11486), Mar. 14 2025.
- [ASGO: Adaptive structured gradient optimization](https://arxiv.org/pdf/2503.20762), Mar. 26 2025.
- [Understanding the learning dynamics of LoRA: A gradient flow perspective on low-rank adaptation in matrix factorization](https://arxiv.org/pdf/2503.06982), Mar. 10 2025.
- [Dion: Distributed Orthonormalized Updates](https://arxiv.org/pdf/2504.05295), May 21 2025. [code](https://github.com/microsoft/dion).

### Pre-training

- [Analyzing & Eliminating Learning Rate Warmup in GPT Pre-Training](https://openreview.net/pdf?id=RveSp5oESA), 2024. `learning dynamics`.
- [Training Trajectories of Language Models Across Scales](https://arxiv.org/pdf/2212.09803), May 30 2023. `training dynamics`.
- [Local to Global: Learning Dynamics and Effect of Initialization for Transformers](https://arxiv.org/pdf/2406.03072), Jun. 5 2024. `training dynamics`.
- [Investigating the Pre-Training Dynamics of In-Context Learning: Task Recognition vs. Task Learning](https://arxiv.org/pdf/2406.14022), Jun. 20 2024. `learning dynamics`.
- [Towards a Theoretical Understanding of the ‘Reversal Curse’ via Training Dynamics](https://arxiv.org/pdf/2405.04669), May 7 2024.
- [Phase Transitions in the Output Distribution of Large Language Models](https://arxiv.org/pdf/2405.17088), May 27 2024. `training dynamics`.
- [Memorization Without Overfitting: Analyzing the Training Dynamics of Large Language Models](https://proceedings.neurips.cc/paper_files/paper/2022/file/fa0509f4dab6807e2cb465715bf2d249-Paper-Conference.pdf), NeurIPS 2022.
- [How Do Large Language Models Acquire Factual Knowledge During Pretraining?](https://arxiv.org/abs/2406.11813), Jun. 17 2024. `interpretability`.
- [How to Train Data-Efficient LLMs](https://arxiv.org/pdf/2402.09668), Feb. 15 2024.
- [Simple and Scalable Strategies to Continually Pre-train Large Language Models](https://arxiv.org/pdf/2403.08763), Mar. 26 2024.
- [Metadata conditioning accelerates language model pre-training](https://arxiv.org/pdf/2501.01956), Jan. 3 2025. [code](https://github.com/princeton-pli/MeCo).
  - _"MeCo first provides metadata (e.g. URLs like en.wikipedia.org alongside the text during training and later uses a cooldown phase with only the standard text, theirby enabling the model to function normally even without metadata.)"_
- [LLM pretraining with continuous concepts](https://arxiv.org/pdf/2502.08524), Feb. 12 2025. [code](https://github.com/facebookresearch/RAM/tree/main/projects/cocomix).
- [Reasoning to learn from latent thoughts](https://arxiv.org/pdf/2503.18866), Mar. 24 2025. [code](https://arxiv.org/pdf/2503.18866).
- [Gemini pretraining](https://vladfeinberg.com/assets/2025-04-24-princeton-talk.pdf), Apr. 24 2025. `slides`.
- [Model merging in pre-training of large language models](https://arxiv.org/pdf/2505.12082), May 17 2025.
- [Language models scale reliably with over-training and on downstream tasks](https://openreview.net/forum?id=iZeQBqJamf), ICLR 2025.
- [Model merging in pre-training of large language models](https://arxiv.org/pdf/2505.12082), May 22 2025.

### Continual pre-training, or mid-training

- [Continual learning for large language models: A survey](https://arxiv.org/pdf/2402.01364), Feb. 7 2024.
- [Continual learning of large language models: A comprehensive survey](https://arxiv.org/pdf/2404.16789), Nov. 25 2024.
- [Investigating continual pretraining in large language models: Insights and implications](https://arxiv.org/pdf/2402.17400), Feb. 12 2025. `TMLR`.

### Peft techniques

- [Parameter-efficient fine-tuning of large-scale pre-trained language models](https://www.nature.com/articles/s42256-023-00626-4.pdf), Nature Machine Intelligence 2023.
- [Towards a unified view of parameter-efficient transfer learning](https://arxiv.org/pdf/2405.14838), Feb. 2 2022.
- [Fine-Tuning Language Models with Just Forward Passes](https://proceedings.neurips.cc/paper_files/paper/2023/file/a627810151be4d13f907ac898ff7e948-Paper-Conference.pdf), NeurIPS 2023.
- [LoRA-Pro: Are Low-Rank Adapters Properly Optimized?](https://arxiv.org/pdf/2407.18242), Jul. 25 2024.
- [QLoRA: Efficient finetuning of quantized LLMs](https://arxiv.org/pdf/2305.14314), May 23 2023.
- [LISA: Layerwise Importance Sampling for Memory-Efficient Large Language Model Fine-Tuning](https://arxiv.org/pdf/2403.17919), May 25 2024.
- [Conditional LoRA Parameter Generation](https://arxiv.org/pdf/2408.01415), Aug. 2024.
- [Fast Forward Low-Rank Training](https://arxiv.org/pdf/2409.04206), Sep. 6  2024.
  - _"In a Fast Forward stage, we **repeat** the most recent optimizer step until the loss stops improving on a tiny validation set."_
  - _"By alternating between regular optimization steps and Fast Forward stages, Fast Forward provides up to an87% reduction in FLOPs and up to an 81% reduction in train time over standard SGD with Adam."_
- [Choice of PEFT Technique in Continual Learning: Prompt Tuning is Not All You Need](https://arxiv.org/pdf/2406.03216), Jun. 5 2024.
- [Pay attention to small weights](https://arxiv.org/pdf/2506.21374), Jun. 26 2025.

### Supervised fine-tuning, instruction-tuning and following

- [AI capabilities can be significantly improved without expensive retraining](https://arxiv.org/pdf/2312.07413), Dec. 12 2023. `icml2023`.
- [3-in-1: 2D Rotary Adaptation for Efficient Finetuning, Efficient Batching and Composability](https://arxiv.org/pdf/2409.00119), Aug. 28 2024.
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
- [Instruction Tuning for Large Language Models: A Survey](https://arxiv.org/pdf/2308.10792), Mar. 14 2024.
- [Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning](https://arxiv.org/pdf/2205.05638), Aug. 26 2022.
- [The Impact of Initialization on LoRA Finetuning Dynamics](https://arxiv.org/pdf/2406.08447), Jun. 12 2024. `lora`.
- [LoRAMoE: Alleviate World Knowledge Forgetting in Large Language Models via MoE-Style Plugin](https://arxiv.org/abs/2312.09979), Dec. 15 2023. `knowledge forgetting`.
- [LoRA Learns Less and Forgets Less](https://arxiv.org/pdf/2405.09673v1), May 15 2024.
- [Orthogonal Finetuning for Direct Preference Optimization](https://arxiv.org/pdf/2409.14836), Sep. 23 2024.
- [ReFT: Representation Finetuning for Language Models](https://arxiv.org/abs/2404.03592), Apr. 4 2024.
- [ALLoRA: Adaptive learning rate mitigates LoRA fatal flaws](https://arxiv.org/pdf/2410.09692), Oct. 13 2024.
- [NefTune: Noisy Embeddings Improve Instruction Tuning](https://arxiv.org/pdf/2310.05914), Oct. 10 2023.  
- [Bitune: Bidirectional Instruction-Tuning](https://arxiv.org/pdf/2405.14862), May 23 2024.
- [RE-Adapt: Reverse Engineered Adaptationof Large Language Models](https://arxiv.org/pdf/2405.15007), May 23 2024.
- [Mixture-of-Subspaces in Low-Rank Adaptation](https://arxiv.org/pdf/2406.11909), Jun. 16 2024. `alignment`.
- [Does instruction tuning reduce diversity? A case study using code generation](https://openreview.net/pdf?id=40uDwtrbd3), submitted to ICLR 2025.
- [Demystifying Instruction Mixing for Fine-tuning Large Language Models](https://aclanthology.org/2024.acl-srw.15.pdf), ACL 2024. [code](https://github.com/Reason-Wang/InstructLLM).
- [Instruction pretraining: Language models are supervised multitask learners](https://arxiv.org/pdf/2406.14491), Nov. 28 2024.
- [Predicting emergent capabilitiew by finetuning](https://arxiv.org/pdf/2411.16035), Nov. 25 2024.
- [Overtrained language models are harder to fine-tune](https://openreview.net/pdf?id=H2SbfCYsgn), ICLR 2025.
- [On the generalization of language models from in-context learning and finetuning: a controlled study](https://arxiv.org/pdf/2505.00661), May 1 2025.
- [Meeseeks: An iterative benchmark evaluating LLMs multi-turn instruction-following ability](https://arxiv.org/pdf/2504.21625), Apr. 30 2025.
- [How much knowledge can you pack into a lora adapter without harming LLM?](https://arxiv.org/pdf/2502.14502), Mar. 24 2025. [code](https://github.com/AIRI-Institute/knowledge-packing).
- [LoRA Training Provably Converges to a Low-Rank Global Minimum or It Fails Loudly (But it Probably Won’t Fail)](https://arxiv.org/pdf/2502.09376), Jun. 3 2025.
- [DoMIX: An efficient framework for exploiting domain knowledge in fine-tuning](https://arxiv.org/pdf/2507.02302), Jul. 3 2025. [code](https://github.com/dohoonkim-ai/DoMIX).

### Alignment, preference alignment, reward models

- [Large Language Model Alignment: A Survey](https://arxiv.org/pdf/2309.15025), Sep. 26 2023.
- [Aligning Large Language Models with Human: A Survey](https://arxiv.org/pdf/2307.12966), Jul. 24 2023.
- [Towards Bidirectional Human-AI Alignment: A Systematic Review for Clarifications, Framework, and Future Directions](https://arxiv.org/pdf/2406.09264), Jun. 17 2024.
- [Alignment of language agents](https://arxiv.org/pdf/2103.14659), Mar. 26 2021.
- [Goal misgeneralization: Why correct specifications aren't enough for correct goals](https://arxiv.org/pdf/2210.01790), Nov. 2 2022.
  - _"an AI system may pursue an undesired goal even when the specification is correct, in the case of goal misgeneralization"_
  - How does this topic correlate with reward hacking? A more broad sense of alignment faking?
- [Goal misgeneralization in deep reinforcement learning](https://proceedings.mlr.press/v162/langosco22a/langosco22a.pdf), ICML 2022.
  - _"Goal misgeneralization occurs when an RL agent retains its capabilities out-of-distribution yet pursues the wrong goal"_
  - "We formalize this distinction between capability and goal generalization"
- [BPO: Supercharging Online Preference Learning by Adhering to the Proximity of Behavior LLM](https://arxiv.org/pdf/2406.12168), Jun. 18 2024. `alignment`.
- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/pdf/2305.18290), Jul. 29 2024.
- [UltraFeedback: Boosting language models with scaled AI feedback](https://arxiv.org/pdf/2310.01377), Jul. 16 2024.
- [Dissecting Human and LLM Preferences](https://arxiv.org/pdf/2402.11296), Feb. 17 2024.
- [The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions](https://arxiv.org/pdf/2404.13208), Apr. 19 2024. `system prompt`.
- [RL on Incorrect Synthetic Data Scales the Efficiency of LLM Math Reasoning by Eight-Fold](https://arxiv.org/pdf/2406.14532), Jun. 20 2024. `post-training`.
- [Measuring memorization in RLHF for code completion](https://arxiv.org/pdf/2406.11715), Jun. 17 2024.
- [WARP: On the Benefits of Weight Averaged Rewarded Policies](https://arxiv.org/abs/2406.16768), Jun. 24 2024. [tweet](https://x.com/ramealexandre/status/1805525340699185493). `post-training`.
- [Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms](https://arxiv.org/pdf/2406.02900), Jun. 5 2024. `scaling law` `post-training`.
- [New Desiderata for Direct Preference Optimization](https://arxiv.org/pdf/2407.09072), Jul. 12 2024.
- [Weak-to-Strong Reasoning](https://arxiv.org/pdf/2407.13647), Jul. 18 2024.
- [Understanding Reference Policies in Direct Preference Optimization](https://arxiv.org/pdf/2407.13709), Jul. 18 2024.
- [Learning from Naturally Occurring Feedback](https://arxiv.org/pdf/2407.10944), Jul. 15 2024.
- [Fine-tuning a "good" model with ppo](https://twitter.com/natolambert/status/1815412187617517612), Jul. 22 2024. `tweet`.
- [Conditional language policy: A general framework for steerable multi-objective finetuning](https://arxiv.org/pdf/2407.15762), Jul. 22 2024.
- [The Hitchhiker’s Guide to Human Alignment with *PO](https://arxiv.org/pdf/2407.15229), Jul. 21 2024.
- [Conditioned Language Policy: A General Framework for Steerable Multi-Objective Finetuning](https://arxiv.org/pdf/2407.15762), Jul. 22 2024.
- [BOND: Aligning LLMs with Best-of-N Distillation](https://arxiv.org/pdf/2407.14622), Jul. 19 2024.
- [A comprehensive survey of llm alignment techniques: rlhf, rlaif, ppo, dpo and more](https://arxiv.org/pdf/2407.16216), Jul. 23 2024.
- [Rule based rewards for language model safety](https://cdn.openai.com/rule-based-rewards-for-language-model-safety.pdf), Jul. 2024.
- [The Alignment Problem from a Deep Learning Perspective](https://arxiv.org/pdf/2209.00626), Mar. 19 2024.
- [A Gradient Analysis Framework for Rewarding Good and Penalizing Bad Examples in Language Models](https://arxiv.org/pdf/2408.16751), Aug. 29 2024.
- [Alignment of Diffusion Models: Fundamentals, Challenges, and Future](https://arxiv.org/pdf/2409.07253), Sep. 12 2024.
- [Semi-Supervised Reward Modeling via Iterative Self-Training](https://arxiv.org/pdf/2409.06903), Sep. 10 2024.
- [Programming Refusal with Conditional Activation Steering](https://arxiv.org/pdf/2409.05907), Sep. 6 2024.
- [On the limited generalization capability of the implicit reward model induced by direct preference optimization](https://arxiv.org/pdf/2409.03650), Sep. 5 2024.
- [Does Alignment Tuning Really Break LLMs’ Internal Confidence?](https://arxiv.org/pdf/2409.00352), Aug. 31 2024.
- [Towards Data-Centric RLHF: Simple Metrics for Preference Dataset Comparison](https://www.arxiv.org/pdf/2409.09603), Sep. 15 2024.
- [Interpreting Learned Feedback Patterns in Large Language Models](https://arxiv.org/pdf/2310.08164), Aug. 19 2024.
- [The N Implementation Details of RLHF with PPO](https://iclr-blogposts.github.io/2024/blog/the-n-implementation-details-of-rlhf-with-ppo/), May 7 2024.
- [Automated rewards via LLM-generated progress functions](https://arxiv.org/pdf/2410.09187), Oct. 11 2024.
- [Nudging: Inference-time alignment via model collaboration](https://arxiv.org/pdf/2410.09300), Oct. 15 2024. [code](https://github.com/fywalter/nudging).
- [Alignment Between the Decision-Making Logic of LLMs and Human Cognition: A Case Study on Legal LLMs](https://arxiv.org/pdf/2410.09083), Oct. 6 2024.
- [Instructional segment embedding: Improving LLM safety with instruction hierarchy](https://arxiv.org/pdf/2410.09102), Oct. 9 2024.
- [Reducing the scope of language models with circuit breakers](https://arxiv.org/pdf/2410.21597), Oct. 28 2024.
- [Steering language model refusal with sparse autoencoders](https://arxiv.org/pdf/2411.11296), Nov. 18 2024.
- [Pluralistic Alignment Over Time](https://arxiv.org/pdf/2411.10654), Nov. 2024.
- [Dataset Reset Policy Optimization for RLHF](https://arxiv.org/pdf/2404.08495), Apr. 16 2024. [code](https://github.com/Cornell-RL/drpo).
- [Learning Loss Landscapes in Preference Optimization](https://arxiv.org/pdf/2411.06568), Nov. 10 2024.
- [ALMA: Alignment with minimal annotation](https://arxiv.org/pdf/2412.04305), Dec. 5 2024.
- [Easy-to-Hard Generalization: Scalable Alignment Beyond Human Supervision](https://arxiv.org/abs/2403.09472#), Mar. 14 2024.
- [A Systematic Examination of Preference Learning through the Lens of Instruction-Following](https://arxiv.org/pdf/2412.15282), Dec. 18 2024.
- [A theory of appropriateness with applications to generative artificial intelligence](https://arxiv.org/pdf/2412.19010), Dec. 26 2024.
- [Fundamental limitations of alignment in large language models](https://arxiv.org/pdf/2304.11082), Jun. 3 2024.
- [Process Reinforcement through Implicit Rewards](https://curvy-check-498.notion.site/Process-Reinforcement-through-Implicit-Rewards-15f4fcb9c42180f1b498cc9b2eaf896f), [code](https://github.com/PRIME-RL/PRIME).
- [HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/pdf/2409.19256v2), Oct. 2 2024. [verl](https://github.com/volcengine/verl).
- [Scheming AIs Will AIs fake alignment during training in order to get power?](https://arxiv.org/pdf/2311.08379), Nov. 27 2023.
- [Preference Ranking Optimization for Human Alignment](https://arxiv.org/pdf/2306.17492), Feb. 27 2024.
- [Quantifying the Gain in Weak-to-Strong Generalization](https://arxiv.org/pdf/2405.15116), May 24 2024.
- [Value Augmented Sampling for Language Model Alignment and Personalization](https://arxiv.org/pdf/2405.14578v1), May 23 2024.
- [Self-Exploring Language Models: Active Preference Elicitation for Online Alignment](https://arxiv.org/pdf/2405.19332), May 29 2024.
- [Steering without side effects: Improving post-deployment control of language models](https://arxiv.org/pdf/2406.15518), Jun. 21 2024.
- [Unpacking DPO and PPO: Disentangling Best Practices for Learning from Preference Feedback](https://arxiv.org/pdf/2406.09279), Jun. 13 2024.
- [Model Alignment as Prospect Theoretic Optimization](https://openreview.net/pdf?id=iUwHnoENnl), ICML 2024.
- [It Takes Two: On the Seamlessness between Reward and Policy Model in RLHF](https://arxiv.org/abs/2406.07971), Jul. 24 2024.
- [The energy loss phenomenon in rlhf: A new perspective on mitigating reward hacking](https://arxiv.org/pdf/2501.19358), Jan. 31 2025.
- [Statistical impossibility and possibility of alignment LLMs with human preferences: From Condorcet paradomx to Nash equilibrium](https://arxiv.org/pdf/2503.10990), Mar. 14 2025.
  - _"with a focus on the probabilistic representation of human preferences and the preservation of diverse preferences in aligned LLMs"_
  - _"prove that this condition holds with high probability under the probabilistic preference model, thereby highlighting the statistical possibiity of preserving minority preferences without explicit regularization in aligning LLMs"_
- [A survey of direct preference optimization](https://arxiv.org/pdf/2503.11701), Mar. 12 2025.
- [Spread preference annotation: Direct preference judgement for efficient LLM alignment](https://arxiv.org/pdf/2406.04412), Mar. 4 2025.
- [On a connection between imitation learning and rlhf](https://arxiv.org/pdf/2503.05079), Mar. 7 2025. [code](https://github.com/tengxiao1/DIL).
- [Societal alignment frameworks can improve LLM alignment](https://arxiv.org/pdf/2503.00069), Feb. 27 2025.
- [Learning a canonical basis of human preferences from binary settings](https://arxiv.org/pdf/2503.24150), Mar. 31 2025. [code](https://github.com/kailas-v/HumanPreferencesBasis).
- [Better estimation of the KL divergence between language models](https://arxiv.org/pdf/2504.10637), Apr. 14 2025. [code](https://github.com/rycolab/kl-rb).
- [Understanding the logic of direct preference alignment through logic](https://arxiv.org/pdf/2412.17696), Dec. 2024.
- [Self-supervised alignment with mutual information learning to follow principles without preference labels](https://arxiv.org/pdf/2404.14313), May 21 2024. [code](https://github.com/janphilippfranken/sami).
- [Do LLMs recognize your preferences? Evaluating personalized preference following in LLMs](https://arxiv.org/pdf/2502.09597), Feb. 13 2025. [code](https://github.com/amazon-science/PrefEval).
- [Scaling laws for scalable oversight](https://arxiv.org/pdf/2504.18530), May 9 2025.
- [SimPER: A minimalist approach to preference alignment without hyperparameters](https://arxiv.org/pdf/2502.00883), Feb. 20 2025. [code](https://github.com/tengxiao1/SimPER).

### Post-training for bert

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
- [All roads lead to likelihood: The value of reinforcement learning in fine-tuning](https://arxiv.org/pdf/2503.01067), Mar. 3 2025.


