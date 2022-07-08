
[Chain of Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/pdf/2201.11903.pdf), Jun. 13 2022.

- This paper is an ongoing idea of **prompting** that utilizes a multi-stage LM generation process to finally get the prediction.
- They prove empirically that, this multi-stage prompting methods can behave greater than single prompting on complex reasoning tasks.
- **Proposed method**: _chain of thought prompting_, i.e. where a few chain of thought demonstrations are provided as exemplars in prompting.
- **Tasks tested on**: arithmetic, commonsense, and symbolic reasoning tasks.

The motivation is so natural, but this paper cites several previous papers that solidate the evidence of multi-step reasoning helps prediction of complex tasks.
As always, a work built on others' shoulders.

---

<img width="891" alt="image" src="https://user-images.githubusercontent.com/7335618/177905725-701f799b-5989-481e-a955-501d84e84c2a.png">

The tasks evaluated with chain of thought are clearly demo in the above image.

---

<img width="824" alt="image" src="https://user-images.githubusercontent.com/7335618/177900631-31054a60-ffc3-445e-95c6-6fec64723dda.png">

In this table, it seems that GPT-3 (175B) is doing quite well on arithmetic reasoning tasks compared to much bigger model PaLM (540, but a seemingly comparabe-sized model, LaMDA (137B), seems to be less effective.
I wonder how does the model scaling law fit into this context. Or do the factors involved in training (dataset, learning schedule and model selection rules) matter more than model scale?

---
