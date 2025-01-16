<h1 align="center">
   Deep Learning FS24: <br/>
  Efficient Pluralistic alignment with RAG
</h1>

### Table of Contents  
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

#### Conda

If you have `conda` installed on your machine.
You can create an environment with all the necessary dependencies with:
```
conda env create -f environment.yaml
```
You can then activate the environment with:
```
conda activate dl_env
```

#### pip

If you do NOT have `conda` installed on your machine.
```
pip install -r requirements.txt
```


## Experiments

For experiments using Moral Stories, you can run the notebook [Moral Stories](src\moral_stories.ipynb). \
For experiments using Moral Choice, you can run the notebook [Moral Choice](src\moral_stories_rag.ipynb). \

Note: You need to run all your experiments from `src`. 


## Contact
You can contact the authors at:
- sjerad@student.ethz.ch
- eukwak@student.ethz.ch
- cmichel@student.ethz.ch
- hdevimeux@student.ethz.ch
