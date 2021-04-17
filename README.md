# Cross-Lingual and Multilingual Word Alignment
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-orange?style=for-the-badge&logo=pytorch"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black -l 300" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge"></a>

This project provide an API to perform word alignment.<br>
The list of languages supported depends on the transformer architecture used.
## How to use
See the main.py file to run an example
```python3
sentence1 = "Today I went to the supermarket to buy apples".split()
sentence2 = "Oggi io sono andato al supermercato a comprare le mele".split()
BERT_NAME = "bert-base-multilingual-cased"
wa = WordAlignment(BERT_NAME, BERT_NAME)
_, decoded = wa.get_alignment(sentence1, sentence2, calculate_decode=True)
for (sentence1_w, sentence2_w) in decoded:
    print(sentence1_w, sentence2_w)
```
The output
```
Today           ---> Oggi
I               ---> io
went            ---> andato
to              ---> al
the             ---> al
supermarket     ---> supermercato
to              ---> a
buy             ---> comprare
apples          ---> mele
```
## How to install 
The Word Alignment is fully compatible with NVIDIA CUDA.<br>
To use CUDA you have to install the cuda version of Torch-Scatter lib, I made a simple script to automate it
```sh
bash cuda_install_requirements.sh
```
N.B.: The CUDA installation of Torch-Scatter require minutes to be compiled.
### Dependencies
- Python3
- Torch
- Transformers
- Torch-Scatter


# Authors

* **Andrea Bacciu**  - [Github](https://github.com/andreabac3)

