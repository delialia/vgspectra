# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# AUTHOR: Delia Fano Yela
# DATE:  February 2019
# CONTACT: d.fanoyela@qmul.ac.uk
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Import the libraries that help for this experiment,
# however, none are strictly necessary here
import os
import librosa
from librosa import load
import numpy as np
import pandas as pd
# import seaborn as sns
import sklearn.metrics
import time
import sys
#from tabulate import tabulate
# Import the Divide & Conquer natural visibility graph implementation (necessary)
from visibility_algorithms import nvg_dc

# Testing parameters:
dir = "AUDIO/synth_dataset/" # Define the path to the synthesised test data
snr_values = [-24, -12, -6, -3, 0, 3, 6]
metrics = ['euclidean', 'cosine']

# Define the divisions for the audio file corresponding to the following MIDI notes:
# [A2,B2,C3,D3,E3,F3,G3,A3,B3,C4,D4,E4,F4,G4 ]
L = 22000               # length of a note in samples
div = range(0,2*L,L) + range(573000,(12*L + 573000), L)

# INITS ------------------------------------------------------------------------
# Max frequency is sampling rate 44100, and the FFT size is 16384; so the frequency increase is 44100/16384 = 2.7 Hz
# The highest note in the dataset is C5, 523.25Hz. If we want to include 10 harmonics we need 523.25 * 10 / 2.7 bins of
# the FFT, 1943.974 bins, rounding up to 2000.
Nf = 16384          # FFT size in samples
N = 2000            # number of bins of interest in the FFT
mu, sigma = 0, 1    # mean and standard deviation for gaussian noise


# Pandas DataFrame that will contain the results
df_mrr = pd.DataFrame({'MRR':[], 'ftype':[], 'dtype':[], 'SNR':[]})

for dis in metrics:
    print "/n Distance Metric: ", dis

    for snr in snr_values:
        print "/n SNR:", snr
        # Start processing--------------------------------------------------------------
        A = np.empty([N,0]) # pair column : abs(FT(cleannote)), impair colum : abs(FT(noisynote))
        K = np.empty([N,0]) # pair column : degree(FT(cleannote)), impair colum : degree(FT(noisynote))
        P = np.empty([N,0]) # pair column : degreedistribution(FT(cleannote)), impair colum : degree(FT(noisynote))

        for subdir, dirs, files in os.walk(dir):
            for i, filename in enumerate(files):
                signal, fs = load(os.path.join(subdir, filename), sr=44100, mono=True)

                sys.stdout.write('\r')
                # the exact output you're looking for:
                sys.stdout.write("[%s%s] %d%%" % ('='*i, ' '*((len(files)-1)-i), 100.*i/(len(files)-1)))
                sys.stdout.flush()
                time.sleep(0.25)

                for index in div:
                    # TIME DOMAIN ---------------------------------------------------
                    s = signal[index:index + Nf]
                    n = np.random.normal(mu, sigma, Nf)
                    #sfc = snr_scaling_factor( signal = s, noise = n, SNRdB = snr )
                    sfc = np.sqrt(np.sum(s**2)/np.sum(n**2))*10**(-snr/20.)
                    m = s + sfc*n

                    # FFT DOMAIN ---------------------------------------------------
                    sa = np.abs(np.fft.fft(np.array(s)))
                    ma = np.abs(np.fft.fft(np.array(m)))

                    # Crop FFT to relevant part:
                    sa = sa[:N]
                    ma = ma[:N]

                    # NVG DOMAIN ---------------------------------------------------
                    sn = nvg_dc(series = sa.tolist() , timeLine = range(N), left = 0, right = N)
                    mn = nvg_dc(series = ma.tolist() , timeLine = range(N), left = 0, right = N)
                    # Adjacency matrix from the horizontal connections:
                    Adj_s = np.zeros((N, N))
                    for el in sn:
                        Adj_s[el] = 1
                        Adj_s[el[-1::-1]] = 1

                    Adj_m = np.zeros((N, N))
                    for el in mn:
                        Adj_m[el] = 1
                        Adj_m[el[-1::-1]] = 1

                    # Degree from adjancecy matrix:
                    sk = np.sum(Adj_s, axis = 0)
                    mk = np.sum(Adj_m, axis = 0)

                    # Degree distribution
                    sp = np.bincount(sk.astype(int), minlength = N).astype('float64') / N
                    mp = np.bincount(mk.astype(int), minlength = N).astype('float64') / N

                    # Stack in processing matrix -----------------------------------
                    A = np.hstack((A,sa[:,None],ma[:,None]))
                    K = np.hstack((K,sk[:,None],mk[:,None]))
                    P = np.hstack((P,sp[:,None],mp[:,None]))


        # SELF-SIMILARITY MATRIX -----------------------------------
        dA = sklearn.metrics.pairwise_distances(np.transpose(A), metric=dis)
        dK = sklearn.metrics.pairwise_distances(np.transpose(K), metric=dis)
        dP = sklearn.metrics.pairwise_distances(np.transpose(P), metric=dis)
        # Sort -----------------------------------------------------
        dA_s = np.argsort(dA, axis = 1)
        dK_s = np.argsort(dK, axis = 1)
        dP_s = np.argsort(dP, axis = 1)
        # MEAN RECIPROCAL RANK -------------------------------------
        noisy = range(1,dA.shape[0],2)
        clean = range(0,dA.shape[0],2)

        retrieved_rank_A = [np.where(dA_s[clean[i],:] == noisy[i])[0][0] for i in range(0,len(clean)) ]
        retrieved_rank_K = [np.where(dK_s[clean[i],:] == noisy[i])[0][0] for i in range(0,len(clean)) ]
        retrieved_rank_P = [np.where(dP_s[clean[i],:] == noisy[i])[0][0] for i in range(0,len(clean)) ]

        mrrA = np.mean(1./np.array(retrieved_rank_A))
        mrrK = np.mean(1./np.array(retrieved_rank_K))
        mrrP = np.mean(1./np.array(retrieved_rank_P))

        print "\nmean reciprocal rank A: ", np.mean(1./np.array(retrieved_rank_A))
        print "mean reciprocal rank K: ", np.mean(1./np.array(retrieved_rank_K))


        # Store ----------------------------------------------------
        df_mrr = df_mrr.append(pd.DataFrame([[mrrA, 'spectrum', dis, snr]], columns = ['MRR', 'ftype', 'dtype', 'SNR']), ignore_index=True)#,sort=True)
        df_mrr = df_mrr.append(pd.DataFrame([[mrrK, 'degree', dis, snr]], columns = ['MRR', 'ftype', 'dtype', 'SNR']), ignore_index=True)#,sort=True)
        df_mrr = df_mrr.append(pd.DataFrame([[mrrP, 'degree distribution', dis, snr]], columns = ['MRR', 'ftype', 'dtype', 'SNR']), ignore_index=True)#,sort=True)


df_mrr.to_csv ('df_mrr_exp01.csv', index = None, header=True)
