
Elements of tuning deep neural network (nn) is dazziling. It is actually an art to me. This text is dedicated to the art of making deep models really work and overfit to train distribution or even generalize to test and ood distribution.

According to my current understanding of nn, there are several important elements one should consider, some of them are called hyperparameters, and others tricks and techniques. Roughly speaking, I make the following categorization.
- hyperparams
  - training epoch
  - batch size
  - learning rate
  - optimizer
  - learning rate schedule
- inductive bias (so-called architecture)
- regularizations
  - dropout
  - weight decay
- scaling up
- normalizations
  - feature normalization
  - layer normalization
  - batch normalization
- curriculum learning
  - self-paced learning
- robust learning
- architecture search
- tricks
  - weight averaging
 
### Curriculum learning

- [A survey on curriculum learning](https://arxiv.org/pdf/2010.13166.pdf), Mar. 25 2021.
- [Curriculum learning: a survey](https://arxiv.org/pdf/2101.10382.pdf), Apr. 11 2022.
- [awesome-curriculum-learning](https://github.com/Openning07/awesome-curriculum-learning).

#### Methods

- [A curriculum learning method for improved noise robustness in automatic speech recognition](https://arxiv.org/pdf/1606.06864.pdf), 2016.
- [Dynamic Curriculum Learning for Imbalanced Data Classification](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Dynamic_Curriculum_Learning_for_Imbalanced_Data_Classification_ICCV_2019_paper.pdf), `iccv2019`.
- [Robust curriculum learning: from clean label detection to noisy label self-correction](https://openreview.net/pdf?id=lmTWnm3coJJ), `iclr2021`.

### Robust learning

- [Learning from Noisy Labels with Deep Neural Networks: A Survey](https://arxiv.org/pdf/2007.08199.pdf), Mar. 10 2022.

#### Methods

- [MentorNet: Learning Data-Driven Curriculum for Very Deep Neural Networks on Corrupted Labels](https://proceedings.mlr.press/v80/jiang18c/jiang18c.pdf), `icml2018`.
- [Co-teaching: Robust training of deep neural networks with extremely noisy labels](https://proceedings.neurips.cc/paper_files/paper/2018/file/a19744e268754fb0148b017647355b7b-Paper.pdf), `nips2018`.
- [Meta-Weight-Net: Learning an Explicit Mapping For Sample Weighting](https://proceedings.neurips.cc/paper_files/paper/2019/file/e58cc5ca94270acaceed13bc82dfedf7-Paper.pdf), `nips2019`.
- [Dividemix: learning with noisy labels as semi-supervised learning](https://arxiv.org/pdf/2002.07394.pdf), `iclr2020`.
- [Symmetric cross-entropy for robust learning with noisy labels](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Symmetric_Cross_Entropy_for_Robust_Learning_With_Noisy_Labels_ICCV_2019_paper.pdf), `iccv2019`.




