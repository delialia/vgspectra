# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# AUTHOR: Delia Fano Yela
# DATE:  February 2019
# CONTACT: d.fanoyela@qmul.ac.uk
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# PLOT EXPERIMENT 02
# ------------------------------------------------------------------------------
# Import the results experiment 02 for DSD100 dataset
df_mrr = pd.read_csv('results_experiments/df_mrr_DSD100.csv')

# Rearrange the data frame  so it's easier to use in seaborn
df_mrr['song'] = range(100)
df_mrr = pd.melt(df_mrr, id_vars=['song'], value_vars=['Ae', 'Ke', 'Pe', 'Ac', 'Kc', 'Pc'])
labels = ["Euclidean Distance"] *100*3 + ["Cosine Distance"]*100*3
df_mrr['type'] = labels


df_mrr = df_mrr.replace("Ae", "spectrum")
df_mrr = df_mrr.replace("Ac", "spectrum")

df_mrr = df_mrr.replace("Ke", "degree")
df_mrr = df_mrr.replace("Kc", "degree")

df_mrr = df_mrr.replace("Pe", "degree distribution")
df_mrr = df_mrr.replace("Pc", "degree distribution")


# Set the layout for the plots in seaborn
sns.set(style="whitegrid", font = 'serif')
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 1.0})

# Boxplot
g = sns.catplot( x = "type", y='value', hue = "variable", data=df_mrr, kind = 'box',
                palette="deep", legend = False,
                size=8, aspect=1)

# Remove top and left spines, set labels and titles at certain fontsizes
ax = g.axes
for i in xrange(ax.shape[0]):
    for j in xrange(ax.shape[1]):
        ax[i,j].spines['right'].set_visible(False)
        ax[i,j].spines['top'].set_visible(False)
        ax[i,j].yaxis.set_ticks_position('left')
        ax[i,j].xaxis.set_ticks_position('bottom')
        ax[i,j].set_xlabel('')
        ax[i,j].set_ylabel('MRR', fontsize=14)

# Plot it
plt.suptitle("EXPERIMENT 02", fontsize=16, fontweight="bold", va = 'center')
plt.legend(loc='best')
plt.savefig('exp02.png')
plt.show()
