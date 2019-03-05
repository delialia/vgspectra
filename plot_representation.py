from librosa import load
import librosa
import numpy as np
from visibility_algorithms import nvg_dc
import librosa.display
import matplotlib.pyplot as plt

mix_path = "AUDIO/DSDTEST/Mixtures/Dev/051 - AM Contra - Heart Peripheral/mixture.wav"

mixture, sr = load(mix_path, sr=44100, mono=True)

M = abs(librosa.core.stft(mixture, n_fft=2046, hop_length=1024, win_length=None, window='hann'))

M = M[:500,:]

# Calculate natural visibility graph(NVg) of each spectrum in A (i.e. columns) and its degree
Nf = M.shape[0]
Na = M.shape[1]
freq_bins = range(Nf)
K = np.empty([Nf,0]) # Degree matrix


for col in xrange(Na):
    NVg_edges = nvg_dc(series = M[:,col].tolist() , timeLine = freq_bins , left = 0, right = Na)

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

# Normalise
Mn = M/np.max(M)
Kn = K/np.max(K)

Mn = Mn[:300,:]
Kn = Kn[:300,:]

mil = 47
fig, ax = plt.subplots(1,2, figsize=(10,5))
plt.subplot(1,2,1)
librosa.display.specshow(Mn**0.6, sr = 44100,  hop_length =1024 , x_axis = 'time', cmap = 'Greys')#, y_axis = 'linear') #librosa.amplitude_to_db(M,ref=np.max)
plt.yticks(np.arange(0,6*mil, mil), np.arange(0,6000,1000) )
plt.ylim([0, 6*mil])
plt.ylabel("Frequency (Hz)", fontname = 'serif')
plt.xlabel("Time (s)", fontname = 'serif')
plt.title('A. Spectrogram', fontname = 'serif')
plt.subplot(1,2,2)
librosa.display.specshow(Kn**0.6, sr = 44100, hop_length =1024, x_axis = 'time',cmap = 'Greys')#, y_axis = 'linear' )
plt.ylim([0, 6*mil])
plt.xlabel("Time (s)", fontname = 'serif')
plt.title('B. Spectral Visibility Graph Degree', fontname = 'serif')
plt.tight_layout()
plt.savefig("plot_representation_6k.png")
plt.show()
