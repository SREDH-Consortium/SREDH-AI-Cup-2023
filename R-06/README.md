# AI Cup Source Code Documentation

## Project Title
AI CUP 2023 Fall Competition

## Description
Privacy and Medical Data Standardization Competition: Decoding Clinical Cases, Letting Data Tell the Story Competition.

We employed a Longformer model [76] in combination with conditional random fields (CRF). Longformer, designed to handle long texts up to 4096 tokens, was chosen to capture the extensive context present in EHRs, a critical feature for accurately identifying SHI. Several pre-processing steps were applied, including tokenization and annotation verification, and we employed a sliding window technique to manage texts exceeding the token limit, ensuring continuity across segments. For model training, we incorporated a fully connected dense layer with 768 units after the Longformer’s transformer output, followed by a CRF layer to improve sequence labelling. The dense layer allowed us to capture rich contextual information, while the CRF layer enhanced the model’s ability to recognize patterns where certain SHI entities tend to appear consecutively. This approach builds on our earlier work in biomedical named entity recognition tasks that emphasize the importance of large contexts to capture the nuances of sensitive information [77]. In the evaluation phase, we used macro-F to assess model performance, ensuring that the model performed well across different PHI categories. Furthermore, to enhance robustness, we implemented a 5-fold ensemble learning approach, where predictions from five distinct models were aggregated using a voting system. This method significantly improved the accuracy and stability of our results, resulting in a final macro F1-score of 0.878 on the testing set. Following the initial model implementation, we integrated a rule-based normalization strategy to address date-related SHIs. Using regular expressions and word-to-number conversion, we normalized textual numbers and date formats. Non-standard entries, such as "several years" or incomplete dates (e.g., "12/18"), were marked as to maintain data integrity.

## Team Information
  - Team name: TEAM_3879(Codalab account: Xiuyu223)
  - Private leaderboard: 0.864657 / Rank 8
  - Members: 
    - [侯秀瑜(Sally)](https://github.com/Xiuyu223)  
    - [曾繁斌(Royce)](https://github.com/trueroyce) 
    - [柯函君(Chloe)](https://github.com/hanchunkk)
    - [Danang Wijaya](https://github.com/danangwijaya750/)

## Table of Contents

- [AI Cup Source Code Documentation](#ai-cup-source-code-documentation)
  - [Project Title](#project-title)
  - [Description](#description)
  - [Team Information](#team-information)
  - [Table of Contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Train and evaluate Task 1 Model](#train-and-evaluate-task-1-model)
  - [Task 1 Inference](#task-1-inference)
  - [Task 2 Inference from Task 1](#task-2-inference-from-task-1)
  - [Others Code](#others-code)

## Prerequisites
    torch == 2.1.1
    transformers == 4.35.2
    datasets == 2.15.0
    torch-crf == 0.7.2 

## Installation
    $ git clone https://github.com/danangwijaya750/AI-CUP-2023-Fall.git
    $ cd AI-CUP-2023-Fall
    $ pip install -r requirements.txt
    

## Train and evaluate Task 1 Model
  Train and Evaluate source code for Task 1  :
  - [Task1_CRF_training_single.ipynb](https://github.com/danangwijaya750/AI-CUP-2023-Fall/blob/master/src/Task1_CRF_training_single.ipynb)
  - [Task1_CRF_training_new.ipynb](https://github.com/danangwijaya750/AI-CUP-2023-Fall/blob/master/src/Task1_CRF_training_new.ipynb)

## Task 1 Inference
  Inference for Task 1 model :
  - [Task1_CRF_inference_single.ipynb](https://github.com/danangwijaya750/AI-CUP-2023-Fall/blob/master/src/Task1_CRF_inference_single.ipynb)

## Task 2 Inference from Task 1
   Task 2 Solution code :
  - [Task2.ipynb](https://github.com/danangwijaya750/AI-CUP-2023-Fall/blob/master/src/Task2.ipynb)
  
## Others Code 
  - [Result anlysis for Task 1](https://github.com/danangwijaya750/AI-CUP-2023-Fall/blob/master/src/Result_analysis.ipynb)
  - [Find Problem from predicted Task 1](https://github.com/danangwijaya750/AI-CUP-2023-Fall/blob/master/src/Find_problem.ipynb)
  - [Final Voting from all predicted models](https://github.com/danangwijaya750/AI-CUP-2023-Fall/blob/master/src/Final_voting.ipynb)