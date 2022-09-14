# ElasticBERT

<div align=center><img width="167" height="320" src="https://github.com/fastnlp/ElasticBERT/blob/main/pics/elasticBERT.gif"/></div>
<div align=center><img width="556" height="161" src="https://github.com/fastnlp/ElasticBERT/blob/main/pics/pic.png"/></div>

- **`ElasticBERT-Chinese-BASE`**: ElasticBERT-Chinese has been uploaded to [huggingface model hub](https://huggingface.co/fnlp/elasticbert-chinese-base). Welcome to download and use it.

This repository contains finetuning code and checkpoints for **ElasticBERT**.

[**Towards Efficient NLP: A Standard Evaluation and A Strong Baseline**](https://aclanthology.org/2022.naacl-main.240/)

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
@inproceedings{liu-etal-2022-towards-efficient,
    title = "Towards Efficient {NLP}: A Standard Evaluation and A Strong Baseline",
    author = "Liu, Xiangyang  and
      Sun, Tianxiang  and
      He, Junliang  and
      Wu, Jiawen  and
      Wu, Lingling  and
      Zhang, Xinyu  and
      Jiang, Hao  and
      Cao, Zhao  and
      Huang, Xuanjing  and
      Qiu, Xipeng",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.naacl-main.240",
    pages = "3288--3303",
}
```
