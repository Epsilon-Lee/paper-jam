
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

### Peft techniques

- [LoRA Learns Less and Forgets Less](https://arxiv.org/pdf/2405.09673v1), May 15 2024.
- [QLoRA: Efficient finetuning of quantized LLMs](https://arxiv.org/pdf/2305.14314), May 23 2023.
- [The Impact of Initialization on LoRA Finetuning Dynamics](https://arxiv.org/pdf/2406.08447), Jun. 12 2024. `lora`.
- [LoRAMoE: Alleviate World Knowledge Forgetting in Large Language Models via MoE-Style Plugin](https://arxiv.org/abs/2312.09979), Dec. 15 2023. `knowledge forgetting`.
- [LISA: Layerwise Importance Sampling for Memory-Efficient Large Language Model Fine-Tuning](https://arxiv.org/pdf/2403.17919), May 25 2024.
- [LoRA-Pro: Are Low-Rank Adapters Properly Optimized?](https://arxiv.org/pdf/2407.18242), Jul. 25 2024.
- [Conditional LoRA Parameter Generation](https://arxiv.org/pdf/2408.01415), Aug. 2024.
- [Fast Forward Low-Rank Training](https://arxiv.org/pdf/2409.04206), Sep. 6  2024.
  - _"In a Fast Forward stage, we **repeat** the most recent optimizer step until the loss stops improving on a tiny validation set."_
  - _"By alternating between regular optimization steps and Fast Forward stages, Fast Forward provides up to an87% reduction in FLOPs and up to an 81% reduction in train time over standard SGD with Adam."_
- [3-in-1: 2D Rotary Adaptation for Efficient Finetuning, Efficient Batching and Composability](https://arxiv.org/pdf/2409.00119), Aug. 28 2024.
- [Orthogonal Finetuning for Direct Preference Optimization](https://arxiv.org/pdf/2409.14836), Sep. 23 2024.
- [Fast Forwarding Low-Rank Training](https://arxiv.org/pdf/2409.04206), Sep. 6 2024.
- [ReFT: Representation Finetuning for Language Models](https://arxiv.org/abs/2404.03592), Apr. 4 2024.
- [ALLoRA: Adaptive learning rate mitigates LoRA fatal flaws](https://arxiv.org/pdf/2410.09692), Oct. 13 2024.

### Alignment, preference alignment, reward models

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


