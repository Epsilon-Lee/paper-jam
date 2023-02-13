
# Toolkits to Learn From

- [AI](#ai)
- [Annotation](#annotation)
- [Production tools](#production-tools)

## AI

- [weaviate](https://github.com/semi-technologies/semantic-search-through-wikipedia-with-weaviate), Semantic search through a vectorized Wikipedia (SentenceBERT) with the Weaviate vector search engine.

## Annotation

- [Rubrix](https://rubrix.readthedocs.io/en/stable/), annotation tools for visualization and quick model construction for a specific task.
- [brat rapid annotation tool](https://brat.nlplab.org/index.html).

## Production tools

- [pdfminer.six](https://github.com/pdfminer/pdfminer.six).
  - Community maintained fork of pdfminer - we fathom PDF.
- [pdfplumber](https://github.com/jsvine/pdfplumber).
  - Plumb a PDF for detailed information about each char, rectangle, line, et cetera — and easily extract text and tables.

---

# Codebases to Learn From

- [Data Structure](#data-structure)
- [System](#system)
- [Data Science](#data-science)

## Data Structure

- [google/pygtrie](https://github.com/google/pygtrie), python library implementation of a trie data structure.

## System

- [Hermit: Deterministic Linux for Controlled Testing and Software Bug-finding](https://github.com/facebookexperimental/hermit).
- [codon](https://github.com/exaloop/codon), A high-performance, zero-overhead, extensible Python compiler using LLVM.

## Data Science

### Traditional ML and DL framework

- [xgboost](https://github.com/dmlc/xgboost).
  - Scalable, Portable and Distributed Gradient Boosting (GBDT, GBRT or GBM) Library, for Python, R, Java, Scala, C++ and more. Runs on single machine, Hadoop, Spark, Dask, Flink and DataFlow.
- [Vopal Wabbit](https://github.com/VowpalWabbit/vowpal_wabbit).
  - Vowpal Wabbit is a machine learning system which pushes the frontier of machine learning with techniques such as online, hashing, allreduce, reductions, learning2search, active, and interactive learning.
- [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn), Lightning fast C++/CUDA neural network framework.
- [oneflow](https://github.com/Oneflow-Inc/oneflow), OneFlow is a deep learning framework designed to be user-friendly, scalable and efficient. `c++`.

### Prototype model

- [nanoGPT](https://github.com/karpathy/nanoGPT), the simplest, fastest repo for training/finetuning medium-sized GPTs by Andrew Karpathy.
- [minGPT](https://github.com/karpathy/minGPT), a minimal pytorch re-implementation of the OpenAI GPT (Generative Pretrained Transformer) training.
- [PERSIA](https://github.com/PersiaML/Persia), High performance distributed framework for training deep learning recommendation models based on PyTorch.
- [GFPGAN](https://github.com/TencentARC/GFPGAN), GFPGAN aims at developing Practical Algorithms for Real-world Face Restoration.

### Training + Modeling

- [fairseq](https://github.com/facebookresearch/fairseq), Facebook AI Research Sequence-to-Sequence Toolkit written in Python.
- [transformers](https://github.com/huggingface/transformers), 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX.
- [alpha](https://github.com/alpa-projects/alpa), training and serving large-scale neural networks. `jax`.
  - Alpa is a system for training and serving large-scale neural networks.
  - Scaling neural networks to hundreds of billions of parameters has enabled dramatic breakthroughs such as GPT-3, but training and serving these large-scale neural networks require complicated distributed system techniques. Alpa aims to automate large-scale distributed training and serving with just a few lines of code.
  - The key features of Alpa include:
    - 💻 Automatic Parallelization. Alpa automatically parallelizes users' single-device code on distributed clusters with data, operator, and pipeline parallelism.
    - 🚀 Excellent Performance. Alpa achieves linear scaling on training models with billions of parameters on distributed clusters.
    - ✨ Tight Integration with Machine Learning Ecosystem. Alpa is backed by open-source, high-performance, and production-ready libraries such as Jax, XLA, and Ray.
  - Research paper: [Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning](https://arxiv.org/abs/2201.12023), Jan. 28 2022.

### Training framework

- [lightning](https://github.com/Lightning-AI/lightning), build and train pytorch models and connect them to the ML lifecycle using lightning app templates, without handling DIY infrastructure, cost management, scaling, and other headaches.
- [trlx](https://github.com/CarperAI/trlx), a repo for distributed training of language models with RLFH. `distributed training` | `rlfh`.

### MLOps and Model serving

- [mltrace](https://github.com/loglabs/mltrace), Coarse-grained lineage and tracing for machine learning pipelines. https://mltrace.readthedocs.io/en/latest/.
- [Petals](https://github.com/bigscience-workshop/petals), Run 100B+ language models at home, BitTorrent-style. Fine-tuning and inference up to 10x faster than offloading.

### Optimization toolbox

- [cvxpy](https://github.com/cvxpy/cvxpy), A Python-embedded modeling language for convex optimization problems.

### LLMs and toolkits around

**QA**

- [primeqa](https://github.com/primeqa/primeqa), The prime repository for state-of-the-art Multilingual Question Answering research and development.
- [ColBERT: state-of-the-art neural search](https://github.com/stanford-futuredata/ColBERT).
  - (1/29/23) We have merged a new index updater feature and support for additional Hugging Face models! These are in beta so please give us feedback as you try them out.
  - (1/24/23) If you're looking for the DSP framework for composing ColBERTv2 and LLMs, it's at: https://github.com/stanfordnlp/dsp

**Toolbox**

- [langchain](https://github.com/hwchase17/langchain), building applications with LLMs through composibility.
  - [Next.js frontend for LangChain Chat](https://github.com/zahidkhawaja/langchain-chat-nextjs).
- [gpt_index](https://github.com/jerryjliu/gpt_index), an index created by GPT to organize external information and answer queries.
- [pyChatGPT](https://github.com/terry3041/pyChatGPT), An unofficial Python wrapper for OpenAI's ChatGPT API.

**Tokenizers**

- [tiktoken](https://github.com/openai/tiktoken), tiktoken is a fast BPE tokenizer for use with OpenAI's models. `rust` | `python`.
- [huggingface tokenizers](https://github.com/huggingface/tokenizers), fast state-of-the-art tokenizers optimized for research and production. `rust` | `python`.

**Evaluation**

- [helm](https://github.com/stanford-crfm/helm), Holistic Evaluation of Language Models (HELM), a framework to increase the transparency of language models (https://arxiv.org/abs/2211.09110).

### XAI

**Visualization**

- [WassersteinTSNE](https://github.com/fsvbach/WassersteinTSNE), wasserstein version of the t-sne algorithm.
- [ManimML](https://github.com/helblazer811/ManimML), ManimML is a project focused on providing animations and visualizations of common machine learning concepts with the Manim Community Library.
