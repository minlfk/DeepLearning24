# Whitewashed - ETHZ CIL Text Classification 2024 

##### Table of Contents  
[Introduction](#Intro) \
[Setup](#Setup) \
[Data](#Data)  \
[Experiments](#Experiments)  
[Contact](#Contact)  



## Introduction

This is the repository of our project for Deep Learning, Fall semester 2024-2025.
In this repo you will find the source code, the project paper, as well as indications on how to run experiments.

All the source code can be found under the folder `src`. \
The data used can be found under `src\data`. \
The embedding methods can be found under `src\embeddings`. \
The models used can be found under `src\models`.

## Setup

We suppose you have `conda` installed on your machine.
You can create an environment with all the necessary dependencies with:
```
conda env create -f environment.yaml
```
You can then activate the environment with:
```
conda activate cil_twitter
```


## Data
We first suppose you already have the Twitter dataset as it was given to us. Moreover, you can obtain the Glove embeddings at [GloVe](https://nlp.stanford.edu/projects/glove/). You should download only the Wikipedia and Twitter pre-trained word vectors. The data organization should look like this

```console
├── data
│   ├── glove_twitter
│   │   ├── glove.twitter.27B.25d.txt
│   │   ├── glove.twitter.27B.50d.txt
│   │   ├── glove.twitter.27B.100d.txt
│   │   ├── glove.twitter.27B.200d.txt  
│   │   │   
└───└── glove_wiki
│   │   ├── glove.6B.50d.txt
│   │   ├── glove.6B.100d.txt
│   │   ├── glove.6B.200d.txt
│   │   ├── glove.6B.300d.txt
│   │   │   
└───└── twitter-datasets
        ├── test_data.txt
        ├── train_neg_full.txt
        ├── train_neg.txt
        ├── train_pos_full.txt
        ├── train_pos.txt
```


## Experiments

For the baseline models, you can run the experiments from [Baselines notebook](src/baselines_notebook.ipynb) (and run the relevant cells depending on which models / embeddings you want to use). \
For the BERT model, you can run the experiments from [Bert notebook](src/run_bert.ipynb). We recommend running the experiments for BERT on Google Colab or a cluster with strong GPU capabilities (we used the Nvidia L4 GPU), and not on a local machine.
\
Note: you need to run all your experiments from `src`. 


## Contact
You can contact the authors at:
- sjerad@student.ethz.ch
- eukwak@student.ethz.ch
- mmultani@student.ethz.ch
- lluong@student.ethz.ch
