
### LLM as a judge

- [Judging llm-as-a-judge with mt-bench and chatbot](https://arxiv.org/abs/2306.05685), Jun. 9 2023.
- [Benchmarking cognitive biases in large language models as evaluators](https://arxiv.org/pdf/2309.17012), Sep. 25 2024. [code](https://github.com/minnesotanlp/cobbler).
- [Preference leakage: A contamination problem in LLM-as-a-judge](https://arxiv.org/pdf/2502.01534), Feb. 3 2025.
- [Investigating non-transitivity in LLM-as-a-judge](https://arxiv.org/pdf/2502.14074), Feb. 19 2025.
- [Crowd comparative reasoning: Unlocking comprehensive evaluation for LLM-as-a-judge](https://arxiv.org/pdf/2502.12501), Feb. 18 2025.
- [LLM Juries for Evaluation](https://www.comet.com/site/blog/llm-juries-for-evaluation/), `blogpost`.
- [Does context matter? ContextJudgeBench for evaluating LLM-based judges in contextual settings](), Mar. 19 2025. [code](https://github.com/SalesforceAIResearch/ContextualJudgeBench).
- [Fantastic LLMs for preference data annotation and how to (not) find them](https://arxiv.org/pdf/2411.02481v1), Nov. 4 2024.
- [Replacing judges with juries: Evaluting LLM generations with a penel of diverse models](https://arxiv.org/abs/2404.18796), Apr. 29 2024.
- [Limits to scalable evaluation at the frontier: LLM as Judge won't beat twice the data](https://arxiv.org/pdf/2410.13341), Feb. 11 2025.
  - _"Our main results shows that when the judge is no more accurate than the evaluated model, no debiasing method can decrease the required amount of ground truth labels by more than half."_
- [On scalable oversight with weak LLMs judging strong LLMs](https://proceedings.neurips.cc/paper_files/paper/2024/file/899511e37a8e01e1bd6f6f1d377cc250-Paper-Conference.pdf), NeurIPS 2024.
- [Evaluating language model agency through negotiations](https://arxiv.org/abs/2401.04536), Jan. 9 2024.
- [Replacing judges with juries: Evaluating LLM generations with a panel of diverse models](https://arxiv.org/pdf/2404.18796), May 1 2024.
- [Trust or escalate: LLM judges with provable guarantees for human agreement](https://arxiv.org/pdf/2407.18370), Jul. 25 2024. [code](https://github.com/jaehunjung1/cascaded-selective-evaluation).
- [TRACT: Regression-aware fine-tuning meets chain-of-thought reasoning for LLM-as-a-judge](https://arxiv.org/pdf/2503.04381), Mar. 6 2025. [code](https://github.com/d223302/TRACT).
- [Learning to plan & reason for evaluation with thinking-LLM-as-a-judge](https://arxiv.org/pdf/2501.18099), Jan. 30 2025.
- [An LLM-as-judge won't save the product - fixing your process will](https://eugeneyan.com/writing/eval-process/), 2025.
- [Judging LLMs on a simplex](https://arxiv.org/pdf/2505.21972), May 28 2025.
  - _"These results underscore the importance of taking a more holistic approach to uncertainty quantification when using LLMs as judges."_
- [Feedback friction: LLMs struggle to fully incorporate external feedback](https://arxiv.org/pdf/2506.11930), Jun. 13 2025.
- [Bridging human and LLM judgements: Understanding and narrowing the gap](https://arxiv.org/pdf/2508.12792), Aug. 18 2025.
- [Justice or prejudice? Quantifying biases in LLM-as-a-Judge](https://arxiv.org/pdf/2410.02736), Oct. 4 2024. [code](https://github.com/llm-judge-bias/llm-judge-bias.github.io/).
- [Reverse engineering human preferences with reinforcement learning](https://arxiv.org/abs/2505.15795), May 21 2025. `LLM-as-a-judge`.
- [No free labels: Limitations of LLM-as-a-judge without human grounding](https://arxiv.org/pdf/2503.05061), Mar. 7 2025. [data](https://huggingface.co/collections/kensho/no-free-labels-67ca139c3943728b3be887a6).
- [Large language models are inconsistent and biased evaluators](https://arxiv.org/pdf/2405.01724), May 2 2024.
- [JudgeLM: Fine-tuning large language models are scalable judges](https://arxiv.org/pdf/2310.17631v2), Mar. 1 2025. [code](https://github.com/baaivision/JudgeLM).
- [Analyzing uncertainty of LLM-as-a-judge: Interval evaluations with conformal prediction](https://arxiv.org/abs/2509.18658), Sep. 23 2025. [code](https://github.com/BruceSheng1202/Analyzing_Uncertainty_of_LLM-as-a-Judge).
- [Towards scalable oversight with collaborative multi-agent debate in error detection](https://arxiv.org/pdf/2510.20963), Oct. 23 2025.

#### Self-critique, self-verification

- [Uncertainty estimation for language reward models](https://arxiv.org/pdf/2203.07472), Mar. 14 2022.
- [Self-critiquing models for assisting human evaluators](https://arxiv.org/pdf/2206.05802), Jun. 14 2022.
- [Teaching language models to support answers with verified quotes](https://arxiv.org/pdf/2203.11147), Mar. 21 2022.
- [Enabling large language models to generate text with citations](https://arxiv.org/pdf/2305.14627), Oct. 31 2023. [code](https://github.com/princeton-nlp/ALCE).
- [Chain-of-verification reduces hallucination in large language models](https://arxiv.org/pdf/2309.11495), Sep. 25 2023.
- [Self iterative label refinement via robust unlabeled learning](https://arxiv.org/pdf/2502.12565), Feb. 18 2025.

#### Survey papers

- [A survey on LLM-as-a-judge](https://arxiv.org/pdf/2411.15594), Nov. 2024.
- [From generation to judgement: Opportunities and challenges of LLM-as-a-judge](https://arxiv.org/pdf/2411.16594), Nov. 2024.
- [LLMs-as-judges: A comprehensive survey on llm-based evaluation methods](https://arxiv.org/pdf/2412.05579), Dec. 10 2024.
- [Automatically correcting large language models: Surveying the landscape of diverse automated correction strategies](https://arxiv.org/abs/2308.03188), TACL 2024.
- [Verdict: A library for scaling judge-time compute](https://arxiv.org/pdf/2502.18018), Feb. 25 2025.

#### Fine-tuned judge, debate, collaborate

- [Fine-tuning language models to find agreement among humans with diverse preferences](https://openreview.net/pdf?id=G5ADoRKiTyJ), NeurIPS 2022.
- [The goldilocks of pragmatic understanding: Fine-tuning strategy matters for implicature resolution by LLMs](https://proceedings.neurips.cc/paper_files/paper/2023/file/4241fec6e94221526b0a9b24828bb774-Paper-Conference.pdf), NeurIPS 2023.
- [Training language models to win debates with self-play improves judge accuracy](https://arxiv.org/pdf/2409.16636), Sep. 2024.
- [Melting Pot Context: Charting the future of generalized cooperative intelligence](https://proceedings.neurips.cc/paper_files/paper/2024/file/1d3ea22480873b389a3365d711eb1e91-Paper-Datasets_and_Benchmarks_Track.pdf), NeurIPS 2024.
- [Improving factuality and reasoning in language models through multiagent debate](https://openreview.net/pdf?id=zj7YuTE4t8), ICML 2024. [code](https://composable-models.github.io/llm_debate/).
- [Cooperation, competition, and maliciousness: LLM-stakeholders interactive negotiation](https://proceedings.neurips.cc/paper_files/paper/2024/file/984dd3db213db2d1454a163b65b84d08-Paper-Datasets_and_Benchmarks_Track.pdf), NeurIPS 2024.
- [How well can LLMs negotiate? NegotiationAreana platform and analysis](https://arxiv.org/abs/2402.05863), Feb. 8 2024.
- [Debating with more persuasive LLMs leads to more truthful answers](https://arxiv.org/pdf/2402.06782), Jun. 25 2024.
- [Training language models to win debates with self-play improves judge accuracy](https://arxiv.org/pdf/2409.16636v1), Sep. 25 2024. [code](https://github.com/samuelarnesen/nyu-debate-modeling).
- [LLM-deliberation: Evaluating LLMs with interactive multi-agent negotiation game](https://openreview.net/forum?id=cfL8zApofK), ICLR 2024.
- [Large language model agents can coordinate beyond human scale](https://arxiv.org/pdf/2409.02822), Dec. 22 2024.
- [Evaluating language model agency through negotiations](https://arxiv.org/abs/2401.04536), Jan. 9 2024.
- [Multi-agent consensus seeking via large language models](https://arxiv.org/pdf/2310.20151), Jan. 21 2025.
- [Great models think alike and this undermines AI oversight](https://arxiv.org/pdf/2502.04313), Feb. 6 2025. [code](https://github.com/model-similarity/lm-similarity).
- [AI debate aids assessment of controversial claims](https://arxiv.org/pdf/2506.02175), Jun. 2 2025. [code](https://github.com/salman-lui/ai-debate).

### LLM-as-an-annotator

- [Large language models as annotators: Enhancing generalization of NLP models at minimal cost](https://arxiv.org/pdf/2306.15766), Jun. 27 2023.
- [LLMaAA: Making large language models as active annotators](https://arxiv.org/pdf/2310.19596), Oct. 31 2023. [code](https://github.com/ridiculouz/LLMAAA).
- [Best practices for text annotation with large language models](https://arxiv.org/pdf/2402.05129), Feb. 2024.
- [MEGAnno+: A human-LLM collaborative annotation system](https://arxiv.org/pdf/2402.18050), Feb. 28 2024. [code](https://github.com/megagonlabs/meganno-client).
- [The effectiveness of LLMs as annotators: A comparative overview and empirical analysis of direct representation](https://aclanthology.org/2024.nlperspectives-1.11.pdf), 2024.
- [The promises and pitfalls of LLM annotations in dataset labeling: A case study on media bias detection](https://aclanthology.org/2025.findings-naacl.75.pdf), NAACL 2025. [code](https://github.com/Media-Bias-Group/llm-annotations-annomatic).
- [Can unconfident LLM annotations be used for confident conclusions?](https://arxiv.org/pdf/2408.15204), Feb. 8 2025. [code](https://github.com/kristinagligoric/confidence-driven-inference).
- [Can reasoning help large language models capture human annotator disagreement?](https://arxiv.org/pdf/2506.19467), Aug. 4 2025. [code](https://github.com/EdisonNi-hku/Disagreement_Prediction).
- [Large language model hacking: Quantifying the hidden risks of using LLMs for text annotation](https://arxiv.org/pdf/2509.08825), Sep. 10 2025.


