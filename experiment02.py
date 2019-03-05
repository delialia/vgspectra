# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# AUTHOR: Delia Fano Yela
# DATE:  February 2019
# CONTACT: d.fanoyela@qmul.ac.uk
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# WARNING: this script uses multiprocessing => CPU intensive
# you can change the number of processes bellow if you wish


# Compare all vocal tracks with all the mixture (for all the dataset)
import os
from itertools import chain
from librosa import load
import multiprocessing as mp
import random
import sklearn.metrics
import pandas as pd
import librosa
import numpy as np
from visibility_algorithms import nvg_dc
import sys


def processFile(subdir, filename):
    # Get the path for each mixture and its corresponding vocal stem
    mix_path = os.path.join(subdir, filename)
    vox_path = os.path.join(subdir.replace("Mixtures", "Sources"), "vocals.wav")

    # Load the files:
    mixture, sr = load(mix_path, sr=44100, mono=True)
    vocals, sr  = load(vox_path, sr=44100, mono=True)

    vocals[np.where(vocals < 0.001)[0]] = 0 # gated by audible level  (in theory) - -60dB (0.01 for -40dB)

    # STFT
    M = abs(librosa.core.stft(mixture, n_fft=2046, hop_length=1024, win_length=None, window='hann'))
    V = abs(librosa.core.stft(vocals,  n_fft=2046, hop_length=1024, win_length=None, window='hann'))


    # Index of frames vocal activation:
    Ev = np.sum(V**2, axis=0) / V.shape[1]

    Emin = np.min(Ev[np.where(Ev>0.)[0]]) # get the minimum excluding zeros
    Emax = np.max(Ev[np.where(Ev>0.)[0]])

    lamda = (Emax - Emin) / Emax

    th = (1 - lamda) * Emax + lamda * Emin # "Approach for Energy-Based Voice Detector with Adaptive Scaling Factor", K. Sakhnov and al., 36:4, IJCS_36_4_16, 2009

    vox_frames  = np.where( Ev > th )[0]
    other_frames = np.delete(range(V.shape[1]), vox_frames)

    # Stack them in a matrix:  corresponding mixture frames, other frames
    A = np.hstack((V[:,vox_frames], M[:,vox_frames], M[:,other_frames]))

    # To reduce computation time, focus on precise band (<10KHz)
    A = A[:500,:]

    # Calculate natural visibility graph(NVg) of each spectrum in A (i.e. columns) and its degree
    Nf = A.shape[0]
    Na = A.shape[1]
    freq_bins = range(Nf)
    K = np.empty([Nf,0]) # Degree matrix
    P = np.empty([Nf,0]) # Degree distribution matrix

    for col in xrange(Na):
        NVg_edges = nvg_dc(series = A[:,col].tolist() , timeLine = freq_bins , left = 0, right = Nf)

        # Adjacency matrix from the natural visibility edges (i.e. connections):
        Adj = np.zeros((Nf, Nf))

        for edge in NVg_edges:
            Adj[edge] = 1
            Adj[edge[-1::-1]] = 1  # NVg is an undirected graph so the Adjacency matrix is symmetric

        # Degree from adjancecy matrix:
        NVg_degree = np.sum(Adj, axis = 0)

        # Degree distribution
        NVg_dist = np.bincount(NVg_degree.astype(int), minlength = Nf).astype('float64') / Nf

        # Store results
        K = np.hstack((K, NVg_degree[:,None]))
        P = np.hstack((P, NVg_dist[:,None]))

    # Vocal frames
    Lv = len(vox_frames)
    frames2check = range(Lv)  # only the first 100 ones are the clean vocals

    # Distance analysis:
    # Euclidean
    dAe = sklearn.metrics.pairwise_distances(np.transpose(A[:,:Lv]),np.transpose(A[:,Lv:]), metric='euclidean')
    dKe = sklearn.metrics.pairwise_distances(np.transpose(K[:,:Lv]),np.transpose(K[:,Lv:]), metric='euclidean')
    dPe = sklearn.metrics.pairwise_distances(np.transpose(P[:,:Lv]),np.transpose(P[:,Lv:]), metric='euclidean')
    # Cosine
    dAc = sklearn.metrics.pairwise_distances(np.transpose(A[:,:Lv]),np.transpose(A[:,Lv:]), metric='cosine')
    dKc = sklearn.metrics.pairwise_distances(np.transpose(K[:,:Lv]),np.transpose(K[:,Lv:]), metric='cosine')
    dPc = sklearn.metrics.pairwise_distances(np.transpose(P[:,:Lv]),np.transpose(P[:,Lv:]), metric='cosine')



    # Sort the distances in ascending order and keep the location
    dAe_s = np.argsort(dAe, axis = 1)
    dKe_s = np.argsort(dKe, axis = 1)
    dPe_s = np.argsort(dPe, axis = 1)

    dAc_s = np.argsort(dAc, axis = 1)
    dKc_s = np.argsort(dKc, axis = 1)
    dPc_s = np.argsort(dPc, axis = 1)


    # Find rank of the wanted neighbour (the correspondent mixture frame of the vocal clean frame)
    retrieved_rank_Ae = np.array([np.where( dAe_s[i,:] == i )[0][0] for i in frames2check ])
    retrieved_rank_Ae[retrieved_rank_Ae == 0] = 1 # in the case where the vocals are alone in the mix

    retrieved_rank_Ke = np.array([np.where( dKe_s[i,:] == i )[0][0] for i in frames2check ])
    retrieved_rank_Ke[retrieved_rank_Ke == 0] = 1

    retrieved_rank_Pe = np.array([np.where( dPe_s[i,:] == i )[0][0] for i in frames2check ])
    retrieved_rank_Pe[retrieved_rank_Pe == 0] = 1

    retrieved_rank_Ac = np.array([np.where( dAc_s[i,:] == i)[0][0] for i in frames2check ])
    retrieved_rank_Ac[retrieved_rank_Ac == 0] = 1

    retrieved_rank_Kc = np.array([np.where( dKc_s[i,:] == i)[0][0] for i in frames2check ])
    retrieved_rank_Kc[retrieved_rank_Kc == 0] = 1

    retrieved_rank_Pc = np.array([np.where( dPc_s[i,:] == i)[0][0] for i in frames2check ])
    retrieved_rank_Pc[retrieved_rank_Pc == 0] = 1


    # Mean reciprocal rank
    Ae_mrr = np.mean(1./retrieved_rank_Ae)
    Ke_mrr = np.mean(1./retrieved_rank_Ke)
    Pe_mrr = np.mean(1./retrieved_rank_Pe)

    Ac_mrr = np.mean(1./retrieved_rank_Ac)
    Kc_mrr = np.mean(1./retrieved_rank_Kc)
    Pc_mrr = np.mean(1./retrieved_rank_Pc)

    # Store results:
    df_mrr = pd.DataFrame([[Ac_mrr,Ae_mrr,Kc_mrr,Ke_mrr,Pc_mrr,Pe_mrr]], columns=['Ac', 'Ae', 'Kc' ,'Ke','Pc','Pe'])
    df_mrr.to_csv ('results04_Test/df_mrr_%s.csv' % os.path.basename(subdir), index = None, header=True)


----------------------------------------------------------------------------------------------------------
# Path to the audio dataset:
dir = "AUDIO/DSD100/Mixtures/Test" # <----------------------- SET PATH TO DATASET

# Setup multiprocessing
procs = []

for subdir, dirs, files in os.walk(dir) :
    for filename in files:
        print "Processing %s" % subdir
        proc = mp.Process(target=processFile, args=(subdir, filename))
        proc.start()
        procs.append(proc)

for pr in procs:
    pr.join()

print("All done!")
