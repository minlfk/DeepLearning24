# Deep Learning 24-25: Efficient Pluralistic alignment with RAG

##### Table of Contents  
[Introduction](#Intro) \
[Setup](#Setup) \
[Experiments](#Experiments)  
[Contact](#Contact)  



## Introduction

This is the repository of our project for Deep Learning, Fall semester 2024-2025.
In this repo you will find the source code, the project paper, as well as indications on how to run experiments.

All the source code can be found under the folder `src`. \
The entire pipeline for experiments on Moral Stories: `src\moral_stories.ipynb`. \
The data used for Moral Choice can be found under: `src\data`. \
The entire pipeline for experiments on Moral Choice: `src\moral_choice_rag.ipynb`. \

## Setup

We suppose you have `conda` installed on your machine.
You can create an environment with all the necessary dependencies with:
```
conda env create -f environment.yaml
```
You can then activate the environment with:
```
conda activate dl_env
```


## Experiments

For experiments using Moral Stories, you can run the notebook [Moral Stories](src\moral_stories.ipynb). \
For experiments using Moral Choice, you can run the notebook [Moral Choice](src\moral_stories_rag.ipynb). \
We recommend running the experiments for BERT on Google Colab or a cluster with strong GPU capabilities (we used the Nvidia L4 GPU), and not on a local machine.
\
Note: You need to run all your experiments from `src`. 


## Contact
You can contact the authors at:
- sjerad@student.ethz.ch
- eukwak@student.ethz.ch
- cmichel@student.ethz.ch
- hdevimeux@student.ethz.ch
