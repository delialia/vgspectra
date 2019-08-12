# vgspectra
Spectral Visibility Graphs

This code accompanies the paper "Spectral Visibility Graphs: Application to Similarity of Harmonic Signals" by Delia Fano Yela, Dan Stowell and Mark Sandler, where we introduce the visibility graph for audio spectra and propose a novel representation for audio analysis: the spectral visibility graph degree.
Such representation inherently captures the harmonic content of the signal whilst being resilient to broadband noise. We present experiments demonstrating its utility to measure robust similarity between harmonic signals in real and synthesised audio data.
Preprint can be found at http://arxiv.org/abs/1903.01976  


--> NEWS : Paper accepted and will be presented at EUSIPCO in A Coruña on Tuesday 3rd of September 2019 - join us! <--


In the paper we present two experiments demonstrating the utility of the proposed representation of audio signals for harmonic similarity measure.

Experiment 01 for synthesised audio data:
 - Audio data used: AUDIO/synth_dataset
 - Script to run experiment: experiment01.py
 - Script to plot the results (Figure 3 in the paper): plot_exp01.py

Experiment 02 for real audio data:
- Audio data used: DSD100 dataset available at https://sigsep.github.io/datasets/dsd100.html
- Script to run experiment: experiment02.py
- Script to plot the results (Figure 4 in the paper): plot_exp02.py


Other:
- plot_representation.py : Script to plot Figure 2 of the paper showing an example of a spectrogram and its corresponding spectral visibility graph degree proposed representation. The sample audio can be found in AUDIO/sample_used_plot_representation

- visibility_algorithms.py : our implementations of the different visibility graphs algorithms.

- figures : folder containing the images used in the paper

- results_experiments : folder containing the results (in form .csv) obtained from the experiments scripts and used by the plotting scripts.

------------------------------------------------------------------------------
AUTHOR: Delia Fano Yela  
DATE: March 2019
CONTACT: d.fanoyela@qmul.ac.uk  
