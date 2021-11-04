# ElasticBERT

![ElasticBERT-gif](https://github.com/fastnlp/ElasticBERT/blob/main/pics/elasticBERT.gif)

This repository contains finetuning code and checkpoints for **ElasticBERT**.

[**Towards Efficient NLP: A Standard Evaluation and A Strong Baseline**](https://arxiv.org/pdf/2110.07038.pdf)

Xiangyang Liu, Tianxiang Sun, Junliang He, Lingling Wu, Xinyu Zhang, Hao Jiang, Zhao Cao, Xuanjing Huang, Xipeng Qiu

## Requirements

We recommend using Anaconda for setting up the environment of experiments:

```bash
conda create -n elasticbert python=3.8.8
conda activate elasticbert
conda install pytorch==1.8.1 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install -r requirements.txt
```

## Pre-trained Models

We provide the pre-trained weights of **ElasticBERT-BASE** and **ElasticBERT-LARGE**, which can be directly used in Huggingface-Transformers.

- **`ElasticBERT-BASE`**: 12 layers, 12 Heads and 768 Hidden Size.
- **`ElasticBERT-LARGE`**: 24 layers, 16 Heads and 1024 Hidden Size.

The pre-trained weights can be downloaded here.
| Model | `MODEL_NAME`|
| --- | --- |
| **`ElasticBERT-BASE`**   | [fnlp/elasticbert-base](https://huggingface.co/fnlp/elasticbert-base) | 
| **`ElasticBERT-LARGE`**   | [fnlp/elasticbert-large](https://huggingface.co/fnlp/elasticbert-large) |


## Downstream task datasets

The GLUE task datasets can be downloaded from the [**GLUE leaderboard**](https://gluebenchmark.com/tasks)

The ELUE task datasets can be downloaded from the [**ELUE leaderboard**](http://eluebenchmark.fastnlp.top/#/landing)


## Finetuning in static usage

We provide the finetuning code for both GLUE tasks and ELUE tasks in static usage on **ElasticBERT**. 

For GLUE:

```bash
cd finetune-static
bash finetune_glue.sh
```

For ELUE:

```bash
cd finetune-static
bash finetune_elue.sh
```

## Finetuning in dynamic usage

We provide finetuning code to apply two kind of early exiting methods on **ElasticBERT**. 

For early exit using entropy criterion:

```bash
cd finetune-dynamic
bash finetune_elue_entropy.sh
```

For early exit using patience criterion:

```bash
cd finetune-dynamic
bash finetune_elue_patience.sh
```

**Please see our paper for more details!**

## Contact

If you have any problems, raise an issue or contact [Xiangyang Liu](mailto:palladiozt@gmail.com)

## Citation

If you find this repo helpful, we'd appreciate it a lot if you can cite the corresponding paper:

```
@article{liu2021elasticbert,
  author    = {Xiangyang Liu and
               Tianxiang Sun and
               Junliang He and
               Lingling Wu and
               Xinyu Zhang and
               Hao Jiang and
               Zhao Cao and
               Xuanjing Huang and
               Xipeng Qiu},
  title     = {Towards Efficient {NLP:} {A} Standard Evaluation and {A} Strong Baseline},
  journal   = {CoRR},
  volume    = {abs/2110.07038},
  year      = {2021},
  url       = {https://arxiv.org/abs/2110.07038},
  eprinttype = {arXiv},
  eprint    = {2110.07038},
  timestamp = {Fri, 22 Oct 2021 13:33:09 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2110-07038.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
