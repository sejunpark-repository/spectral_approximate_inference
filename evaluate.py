import numpy as np
import csv
import time
import sys
import argparse
import os
import random
from copy import copy

from inference.bucket_elimination import BucketElimination
from inference.mean_field_ising import MeanFieldIsing
from inference.belief_propagation_ising import BeliefPropagationIsing
from inference.rank1_ising import Rank1Ising
from inference.mini_bucket_renormalization import MiniBucketRenormalization
from read_mat import ReadMat

algorithms = ['BP200', 'MF1000', 'MBE-10', 'rank1']

results = {}
time_results = {}

num_sample = 100




exp_type = '20complete'
dirname = 'results/' + exp_type
file_name = 'result.csv'
time_file_name = 'time_result_nosdp.csv'

if not os.path.exists(dirname):
    os.makedirs(dirname)

mat = ReadMat('mat/'+exp_type +'_dataset')
cw_range = mat.get_cwrange()
cw_len = cw_range.shape[1];
print(cw_len)


for alg_name in algorithms:
    results[alg_name] = np.zeros([cw_len, num_sample])
    time_results[alg_name] = np.zeros([cw_len, num_sample])

with open('./{}/{}'.format(dirname,file_name), 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['{}'.format(' ')] + [alg_name for alg_name in algorithms])

with open('./{}/{}'.format(dirname,time_file_name), 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(
        ['{}'.format(' ')] + [alg_name for alg_name in algorithms])


for cw in range(cw_len):
    for sample in range(num_sample):
        print(cw, sample)
        gm = mat.mat2gm(cw,sample)
        im = mat.mat2im(cw,sample)
        V, s = mat.get_sdp_eig(cw, sample)
        true_logZ = mat.logZ[cw,sample]

        t = time.time()
        bp = BeliefPropagationIsing(im)
        results['BP200'][cw, sample] = np.abs(bp.run(maxIter=200) - true_logZ)
        time_results['BP200'][cw, sample] = time.time() - t

        t = time.time()
        mf = MeanFieldIsing(im)
        results['MF1000'][cw, sample] = np.abs(mf.run(maxIter=1000) - true_logZ)
        time_results['MF1000'][cw, sample] = time.time() - t

        t = time.time()
        mbe = MiniBucketRenormalization(gm, mbound=10)
        results['MBE-10'][cw, sample] = mbe.get_logZ() - true_logZ
        time_results['MBE-10'][cw, sample] = time.time() - t


        t = time.time()
        rank = Rank1Ising(im, V, s)
        results['rank1'][cw, sample] = rank.run_rank1() - true_logZ
        time_results['rank1'][cw, sample] = time.time() - t
        time_results['rank1'][cw, sample] += mat.get_sdp_time(cw, sample)


        with open('./{}/{}'.format(dirname,file_name), 'a', newline = '') as f:
            writer = csv.writer(f)
            writer.writerow(['{}'.format(cw_range[0,cw])] + [results[alg_name][cw,sample] for alg_name in algorithms])

        with open('./{}/{}'.format(dirname,time_file_name), 'a', newline = '') as f:
            writer = csv.writer(f)
            writer.writerow(['{}'.format(cw_range[0,cw])] + [time_results[alg_name][cw,sample] for alg_name in algorithms])

with open('./{}/{}'.format(dirname,file_name), 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['overall'])

for cw in range(cw_len):
    with open('./{}/{}'.format(dirname,file_name), 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['{}'.format(cw_range[0, cw])] + [np.mean(np.abs(results[alg_name][cw, :])) for alg_name in algorithms])

with open('./{}/{}'.format(dirname,time_file_name), 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['overall'])

for cw in range(cw_len):
    with open('./{}/{}'.format(dirname,time_file_name), 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['{}'.format(cw_range[0, cw])] + [np.mean(time_results[alg_name][cw, :]) for alg_name in algorithms])