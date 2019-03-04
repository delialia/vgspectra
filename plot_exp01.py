# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# AUTHOR: Delia Fano Yela
# DATE:  February 2019
# CONTACT: d.fanoyela@qmul.ac.uk
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# ------------------------------------------------------------------------------
# PLOT EXPERIMENT 01
# ------------------------------------------------------------------------------
# Import the results from experiment 01
df_mrr = pd.read_csv('results_experiments/df_mrr_exp01.csv')

# Set the layout for the plots in seaborn
sns.set(style="whitegrid", font = 'serif')
sns.set_context("paper",  font_scale=1.5, rc={"lines.linewidth": 1.0})


# Point plot
g = sns.catplot( x = "SNR", y='MRR', hue = "ftype", data=df_mrr, col = "dtype", legend = False, kind = "point",
                markers=['d','o','s'], linestyles=[":", "-", "-."],
                size=8, aspect=1)

# Set the limits
plt.ylim((-0.1, 1.1))

# Remove top and left spines, set labels and titles at certain fontsizes
ax = g.axes
for i in xrange(ax.shape[0]):
    for j in xrange(ax.shape[1]):
        ax[i,j].spines['right'].set_visible(False)
        ax[i,j].spines['top'].set_visible(False)
        ax[i,j].yaxis.set_ticks_position('left')
        ax[i,j].xaxis.set_ticks_position('bottom')
        ax[i,j].set_xlabel(' SNR in dB', fontsize=12)
        if j == 0:
            ax[i,j].set_ylabel('MRR', fontsize=12)
            ax[i,j].set_title('Euclidean Distance', fontsize=14)
        else:
            ax[i,j].set_ylabel('')
            ax[i,j].set_title('Cosine Distance', fontsize=14)

# Plot it
plt.suptitle("EXPERIMENT 01", fontsize=16, fontweight="bold")
plt.legend(scatterpoints=1, loc='best')
plt.savefig('exp01.png')
plt.show()
