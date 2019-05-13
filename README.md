# Spectral Approximate Inference
python & matlab codes used for experiments in "Spectral Approximate Inference" (ICML 2019)

## Run experiment
Run evaluate.py with python3
```
python3 evaluation.py
```
This will compute partition function errors of belief propagation, mean-field approximation, mini-bucket elimination and our spectral approximate inference for pairwise binary models on complete graph of 20 vertices among a range of edge coupling strengths.

## Folder descriptions
#### gm

This folder contains classes related with general graphical models and pairwise binary models.

#### inference

This folder contains inference algorithms for estimating the partition function (belief propagation, mean-field approximation, mini-bucket elimination and our spectral approximate inference).

#### mat

This folder contains datasets of pairwise binary graphical models.

#### matlab_code

This folder contains codes for running semi-definite programming solver used for our spectral approximate inference
