# Spectral Approximate Inference
python & matlab codes used for experiments in "Spectral Approximate Inference" (ICML 2019)

## Run experiment
Run evaluate.py with python3
```
python3 evaluate.py
```
By default, this will compute partition function errors of belief propagation, mean-field approximation, mini-bucket elimination and our spectral approximate inference for pairwise binary models on complete graph of 20 vertices among a range of edge coupling strengths, except for running semi-definite programming of our spectral approximate inference.

To run semi-definite programming, install CVX from http://cvxr.com/cvx/ and run 'compute_sdp_time.m' in matlab_code folder using MATLAB.

## Folder descriptions
#### gm

This folder contains python classes related with general graphical models and pairwise binary models.

#### inference

This folder contains python codes for inference algorithms for estimating the partition function (belief propagation, mean-field approximation, mini-bucket elimination and our spectral approximate inference).

#### mat

This folder contains matlab datasets of pairwise binary graphical models.

#### matlab_code

This folder contains matlab codes for running semi-definite programming solver used for our spectral approximate inference. Before running 'compute_sdp_time.m', install CVX from http://cvxr.com/cvx/
