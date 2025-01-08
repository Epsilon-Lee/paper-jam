
> The science of endowing model with new & updated knowledge.

### Continual learning (cpt, cft), knowledge updating, model editing, model merging

- [Intrinsic Dimensionality Explains The Effectiveness of Language Model Fine-Tuning](https://arxiv.org/pdf/2012.13255), Dec. 22 2020.
- [Covert Malicious Finetuning: Challenges in Safeguarding LLM Adaptation](https://arxiv.org/pdf/2406.20053), Jun. 28 2024. `post-training`.
- [Connecting the Dots: LLMs can Infer and VerbalizeLatent Structure from Disparate Training Data](https://arxiv.org/pdf/2406.14546), Jun. 20 2024. `post-training`.
- [Exploring Design Choices for Building Language-Specific LLMs](https://arxiv.org/pdf/2406.14670), Jun. 20 2024. `post-training for multilinguality`.
- [70B-parameter large language models in Japanese medical question-answering](https://arxiv.org/pdf/2406.14882), Jun. 21 2024. `post-training prompt design`.
- [Towards Understanding Multi-Task Learning (Generalization) of LLMs via Detecting and Exploring Task-Specific Neurons](https://arxiv.org/pdf/2407.06488), Jul. 9 2024. `task-specific subnet in llms`.
- [Raw Text is All you Need: Knowledge-intensive Multi-turn Instruction Tuning for Large Language Model](https://arxiv.org/pdf/2407.03040), Jul. 3 2024. `dialogue data curation`.
- [Enhancing Translation Accuracy of Large Language Models through Continual Pre-Training on Parallel Data](https://arxiv.org/pdf/2407.03145), Jul. 3 2024. `post-training for mt`.
- [Learning dynamics of llm finetuning](https://www.arxiv.org/pdf/2407.10490), Jul. 15 2024. [github](https://github.com/Joshua-Ren/Learning_dynamics_LLM).
- [Prover-verifier games improves legibility of llm outputs](https://arxiv.org/pdf/2407.13692), Jul. 18 2024.
- [INSTRUCT-SKILLMIX: A Powerful Pipeline for LLM Instruction Tuning](https://arxiv.org/pdf/2408.14774), Aug. 27 2024.
- [How Do Your Code LLMs Perform? Empowering Code Instruction Tuning with High-Quality Data](https://arxiv.org/pdf/2409.03810), Sep. 5 2024.
- [The Representation Landscape of Few-Shot Learning and Fine-Tuning in Large Language Models](https://arxiv.org/pdf/2409.03662), Sep. 7 2024.
- [Instruction following without instruction tuning](https://arxiv.org/pdf/2409.14254), Sep. 21 2024. [tweet](https://x.com/johnhewtt/status/1838605168579121599).
- [LLM Continue Pretrain](https://zhuanlan.zhihu.com/p/707751901), Jul. 10 2024. [MiniCPM](https://shengdinghu.notion.site/MiniCPM-c805a17c5c8046398914e47f0542095a#73df646223b54e22957bdd926b41fc0e).
- [Domain Adaptation of Llama3-70B-Instruct through Continual Pre-Training and Model Merging: A Comprehensive Evaluation](https://arxiv.org/pdf/2406.14971), Jun. 21 2024. `continual pre-training`.
- [Understanding catastrophic forgetting in language models via implicit inference](https://openreview.net/pdf?id=VrHiF2hsrm), ICLR 2024. `sft and forgetting`.
  - _"We hypothesize that language models implicitly infer the task of the prompt and then fine-tuning skews this inference towards tasks in the fine-tuning distribution."_
- [Empirical influence functions to understand the logic of fine-tuning](https://arxiv.org/pdf/2406.00509), Jun. 1 2024. `influence function` and `post-training`.
- [An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning](https://arxiv.org/abs/2308.08747), Aug. 17 2023. `post-training`.
- [Scaling Laws for Forgetting When Fine-Tuning Large Language Models](https://arxiv.org/pdf/2401.05605v1), Jan. 11 2024. `forgetting`.
- [Efficient Continual Pre-training by Mitigating the Stability Gap](https://arxiv.org/pdf/2406.14833), Jun. 21 2024. `post-training` `cpt`.
- [Gradient-Mask Tuning Elevates the Upper Limits of LLM Performance](https://arxiv.org/pdf/2406.15330), Jun. 21 2024. `other post-training method`.
- [Can LLMs Learn by Teaching? A Preliminary Study](https://arxiv.org/pdf/2406.14629), Jun. 20 2024.
- [Skywork-Math: Data Scaling Laws for Mathematical Reasoning in Large Language Models — The Story Goes On](https://arxiv.org/pdf/2407.08348), Jul. 11 2024.
- [Mitigating catasthrophic forgetting in language transfer via model merging](https://arxiv.org/pdf/2407.08699), Jul. 11 2024.
  - What are the differences between EMA and the proposed BAM method?
- [Mix-CPT: A domain adaptation framework via decoupling knowledge learning and format alignment](https://arxiv.org/pdf/2407.10804), Jul. 15 2024.
  - _*we revise this process and propose a new domain adaptation framework including domain knowledge learning and general format alignment,*_
- [Fine-Tuning and Prompt Optimization: Two Great Steps that Work Better Together](https://arxiv.org/pdf/2407.10930), Jul. 15 2024.
- [InstructAV: Instruction Fine-tuning Large Language Models for Authorship Verification](https://arxiv.org/pdf/2407.12882), Jul. 16 2024.
  - How to construct examples for domain-specific training?
- [DART-Math: Difficulty-Aware Rejection Tuning for Mathematical Problem-Solving](https://arxiv.org/pdf/2407.13690), Jun. 18 2024.
- [DDK: Distilling Domain Knowledge for Efficient Large Language Models](https://arxiv.org/pdf/2407.16154), Jul. 23 2024.
- [Towards effective and efficient continual pre-training of large language models](https://arxiv.org/pdf/2407.18743), Jul. 26 2024.
- [Lawma: The power of specialization for legal tasks](https://arxiv.org/pdf/2407.16615), Jul. 23 2024. [github](https://github.com/socialfoundations/lawma).
  - _"We then demostrate that a highly fine-tuned Llama 3 model vastly outperforms GPT-4 on almost all tasks. [...] find that larger models respond better to fine-tuning than smaller models. A few ten to hundreds of examples suffice to achieve high classification accuracy."_
- [Leveraging the Power of LLMs: A Fine-Tuning Approach for High-Quality Aspect-Based Summarization](https://arxiv.org/pdf/2408.02584), Aug. 5 2024.
- [Re-TASK: Revisiting LLM Tasks from Capability, Skill, and Knowledge Perspectives](https://arxiv.org/pdf/2408.06904), Aug. 13 2024.
- [Instruction Tuning for Large Language Models: A Survey](https://arxiv.org/pdf/2308.10792), Mar. 14 2024. `survey`.
- [LLM Pruning and Distillation in Practice: The Minitron Approach](https://arxiv.org/pdf/2408.11796), Aug. 26 2024.
- [Simple and Scalable Strategies to Continually Pre-train Large Language Models](https://arxiv.org/pdf/2403.08763), Sep. 4 2024.
- [Synthetic continued pretraining](https://arxiv.org/pdf/2409.07431), Sep. 11 2024.
- [Large Language Models Encode Clinical Knowledge](https://arxiv.org/pdf/2212.13138), Dec. 26 2022.
- [Editing models with task arithmic](https://arxiv.org/pdf/2212.04089), Mar. 31 2023. `task vector` `model merging`.
  - _"A task vector specifies in the weight space of a pre-trained model, such that movement in that direction improves performance on the task. We build task vectors by subtracting the weights of a pre-trained model from the weights of the same model after fine-tuning on a task"_
- [Measuring and Modifying Factual Knowledge in Large Language Models](https://arxiv.org/pdf/2306.06264), Jun. 9 2023. `knowledge measurement`.
- [Do LLMs Understand Social Knowledge? Evaluating the Sociability of Large Language Models with the SOCKET Benchmark](https://arxiv.org/pdf/2305.14938), Dec. 7 2023. `knowledge measurement`.
- [Fundamental Problems With Model Editing: How Should Rational Belief Revision Work in LLMs?](https://arxiv.org/pdf/2406.19354), Jun. 27 2024. `model editing`.
- [Studying Large Language Model Behaviors Under Realistic Knowledge Conflicts](https://arxiv.org/pdf/2404.16032), Apr. 24 2024. `knowledge conflicts`.
- [Understanding Finetuning for Factual Knowledge Extraction](https://arxiv.org/abs/2406.14785), Jun. 20 2024. [tweet](https://x.com/gaurav_ghosal/status/1806365312620589496). `post-training` `knowledge forgetting`.
  - A very similar paper is [Establishing Knowledge Preference in Language Models](https://arxiv.org/pdf/2407.13048), Jul. 17 2024.
  - Another similar one is [Large Language Models as Reliable Knowledge Bases?](https://arxiv.org/pdf/2407.13578), Jul. 18 2024.
- [Knowledge Conflicts for LLMs: A Survey](https://arxiv.org/pdf/2403.08319), Jun. 22 2024. `survey` `knowledge conflicts`.
- [How to Precisely Update Large Language Models Knowledge While Avoiding Catastrophic Forgetting](https://www.cambridge.org/engage/api-gateway/coe/assets/orp/resource/item/667bb159c9c6a5c07a689cfa/original/how-to-precisely-update-large-language-models-knowledge-while-avoiding-catastrophic-forgetting.pdf), 2024.
- [Knowledge Overshadowing Causes Amalgamated Hallucination in Large Language Models](https://arxiv.org/pdf/2407.08039), Jul. 10 2024. `hallucination`.
- [Lynx: An Open Source Hallucination Evaluation Model](https://arxiv.org/pdf/2407.08488), Jul. 11 2024.
- [Towards understanding factual knowledge of large language models](https://openreview.net/pdf?id=9OevMUdods), ICLR 2024.
- [Knowledge Card: Filling LLMs' Knowledge Gaps with Plug-in Specialized Language Models](https://openreview.net/pdf?id=WbWtOYIzIK), ICLR 2024.
- [Knowledge Mechanisms in Large Language Models: A Survey and Perspective](https://arxiv.org/pdf/2407.15017), Jul. 22 2024.
- [MoFO: Momentum-Filtered Optimizer for Mitigating Forgetting in LLM Fine-Tuning](https://arxiv.org/pdf/2407.20999), Jul. 30 2024.
- [Knowledge in Superposition: Unveiling the Failures of Lifelong Knowledge Editing for Large Language Models](https://arxiv.org/pdf/2408.07413), Aug. 14 2024.
- [Training Language Models on the Knowledge Graph: Insights on Hallucinations and Their Detectability](https://arxiv.org/pdf/2408.07852v1), Aug. 14 2024.
- [Alternate Preference Optimization for Unlearning Factual Knowledge in Large Language Models](https://arxiv.org/pdf/2409.13474), Sep. 20 2024.
- [ClashEval: Quantifying the tug-of-war between an LLM’s internal prior and external evidence](https://arxiv.org/pdf/2404.10198), Jun. 10 2024.
- [Large language model validity via enhanced conformal prediction methods](https://arxiv.org/pdf/2406.09714), Jun. 14 2024. [youtube](https://www.youtube.com/watch?v=fsgyllS43KY).
  - [Can Unconfident LLM Annotations Be Used for Confident Conclusions?](https://arxiv.org/pdf/2408.15204), Aug. 2024.
- [Merge to Learn: Efficiently Adding Skills to Language Models with Model Merging](https://arxiv.org/pdf/2410.12937), Oct. 16 2024.
- [RAFT: Adapting Language Model to Domain Specific RAG](https://arxiv.org/pdf/2403.10131), Jun. 5 2024.
- [Is Parameter Collision Hindering Continual Learning in LLMs?](https://arxiv.org/pdf/2410.10179), Oct. 14 2024.
- [Context-parameteric inversion: Why instruction finetuning may not actually improve context reliance](https://arxiv.org/pdf/2410.10796), Oct. 14 2024. [code](https://github.com/locuslab/context-parametric-inversion).
- [Toward General Instruction-Following Alignment for Retrieval-Augmented Generation](https://arxiv.org/pdf/2410.09584), Oct. 12 2024. [code](https://github.com/dongguanting/FollowRAG).
- [Synthetic Knowledge Ingestion: Towards Knowledge Refinement and Injection for Enhancing Large Language Models](https://arxiv.org/pdf/2410.09629), Oct. 12 2024.
- [STACKFEED: Structured textual actor-critic knowledge base editing with feedback](https://arxiv.org/pdf/2410.10584), Oct. 14 2024.
- [Mix data or merge models? Optimizing for diverse multi-task learning](https://arxiv.org/pdf/2410.10801), Oct. 14 2024.
- [MTL-LoRA: Low-Rank Adaptation for Multi-Task Learning](https://arxiv.org/pdf/2410.09437), Oct. 15 2024. 
- [Open Domain Question Answering with Conflicting Contexts](https://arxiv.org/pdf/2410.12311), Oct. 18 2024.
- [Understanding finetuning for factual knowledge extraction from language models](https://arxiv.org/pdf/2301.11293), Jan. 26 2023.
- [Knowledge Circuits in Pretrained Transformers](https://arxiv.org/pdf/2405.17969), Oct. 16 2024. [code](https://github.com/zjunlp/KnowledgeCircuits/tree/main).
- [Head-to-Tail: How Knowledgeable are Large Language Models (LLMs)? A.K.A. Will LLMs Replace Knowledge Graphs?](https://arxiv.org/pdf/2308.10168), Apr. 3 2024.
- [Continual memorization of factoids in large language models](https://arxiv.org/pdf/2411.07175), Nov. 11 2024. [code](https://github.com/princeton-nlp/continual-factoid-memorization).
- [Velocitune: A Velocity-based Dynamic Domain Reweighting Method for
Continual Pre-training](https://arxiv.org/pdf/2411.14318), Nov. 21 2024.
- [Source-Aware Training Enables Knowledge Attribution in Language Models](https://arxiv.org/pdf/2404.01019), Aug. 13 2024. [code](https://github.com/mukhal/intrinsic-source-citation).
- [Do I know this entity? Knowledge awareness and hallucinations in language models](https://arxiv.org/pdf/2411.14257), Nov. 21 2024.
- [In Search of the Long-Tail: Systematic Generation of Long-Tail Inferential Knowledge via Logical Rule Guided Search](https://aclanthology.org/2024.emnlp-main.140.pdf), EMNLP 2024.
- [Knowledge Graphs, Large Language Models, and Hallucinations: An NLP Perspective](https://arxiv.org/pdf/2411.14258), Nov. 21 2024.
- [Filter-then-Generate: Large Language Models with Structure-Text Adapter for Knowledge Graph Completion](https://arxiv.org/abs/2412.09094), Dec. 12 2024.
- [Extractive Structures Learned in Pretraining Enable Generalization on
Finetuned Facts](https://arxiv.org/pdf/2412.04614), Dec. 5 2024.

#### RAG and knowledge graph

- [Unsupervised Information Refinement Training of Large Language Models for Retrieval-Augmented Generation](https://arxiv.org/pdf/2402.18150v1), Feb. 28 2024. `post-training` `unsupervised` `rag`.
- [LLMs Know What They Need: Leveraging a Missing Information Guided Framework to Empower Retrieval-Augmented Generation](https://arxiv.org/pdf/2404.14043v1), Apr. 22 2024.
- [Evaluation of RAG Metrics for Question Answering in the Telecom Domain](https://arxiv.org/pdf/2407.12873), Jul. 15 2024.
  - application in vertical-domain
- [Great Memory, Shallow Reasoning: Limits of kNN-LMs](https://arxiv.org/pdf/2408.11815), Aug. 21 2024. [code](https://arxiv.org/pdf/2408.11815). [code](https://github.com/GSYfate/knnlm-limits).
- [Graph Retrieval-Augmented Generation: A Survey](https://arxiv.org/pdf/2408.08921), Aug. 15 2024.
- [Grounding by trying: LLMs with reinforcement learning-enhanced retrieval](https://arxiv.org/pdf/2410.23214), Oct. 31 2024. [code](https://github.com/sher222/LeReT).
- [BRIGHT: A realistic and challenging benchmark for reasoning-intensitve retrieval](https://arxiv.org/pdf/2407.12883), Oct. 24 2024.
- [Likelihood as a Performance Gauge for Retrieval-Augmented Generation](https://arxiv.org/pdf/2411.07773), Nov. 12 2024. [code](https://github.com/lyutyuh/poptimizer).
- [Drowning in Documents: Consequences of Scaling Reranker Inference](https://arxiv.org/pdf/2411.11767), Nov. 18 2024.
- [OpenScholar: Synthesizing scientific literature with retrieval-augmented LMs](https://arxiv.org/pdf/2411.14199), Nov. 21 2024. [code](https://github.com/AkariAsai/OpenScholar).
- [Retrieval-Augmented Generation for Domain-Specific Question Answering: A Case Study on Pittsburgh and CMU](https://arxiv.org/pdf/2411.13691), Nov. 20 2024.
- [Filter-then-generate: Large language models with structure-text adapter for knowledge graph completion](https://arxiv.org/pdf/2412.09094), Dec. 12 2024.


