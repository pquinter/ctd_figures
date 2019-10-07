from matplotlib import pyplot as plt
import pandas as pd
from utils import ecdftools, plot
from joblib import Parallel, delayed

# get sleuth data
sleuth = pd.read_csv('../data/sleuth_all.csv').dropna(axis=0)
# get unique coefficient id, as they occur in multiple models
sleuth['cid'] = sleuth['coef'] + sleuth['model']
# get q-value cumulative distribution by model and coefficient
qvalecdf_series = Parallel(n_jobs=6)(delayed(ecdftools.compute_qvalecdf)(model, coef, _df)
                       for (model, coef), _df in sleuth.groupby(['model','coef']))
qvalecdf = pd.concat(qvalecdf_series)
# get back cid
qvalecdf['cid'] = qvalecdf['coef'] + qvalecdf['model']

###############################################################################
# Heatmap of q-value distribution for galactose, truncation and interaction coef
###############################################################################

coefs = ['galmain','yQC7main','yQC7:galmain']
coef_names = ['Galactose','Truncation','Interaction']
ax = plot.qvalecdf_hmap(qvalecdf, coefs, coef_names=coef_names)
plt.savefig('../output/Fig2_RNAseqHmap.svg')

###############################################################################
# Lollipop plot of number of transcripts at q-value<0.1
###############################################################################

ax = plot.coef_stemplot(qvalecdf, coefs, qval_thresh=0.1, coef_names=coef_names)
plt.savefig('../output/Fig2_RNAseqLolipop.svg')

###############################################################################
# Scatter plot of interaction vs galactose coefficients
###############################################################################
plt.ioff()
ax = plot.scatter_coef(sleuth, 'galmain', 'yQC7:galmain', alpha=0.2)
# Get GAL4 gene targets
gal4targets = pd.read_csv('../data/gal4_targetgenes_ORegAnno20160119.csv')
gal4targets = sleuth[sleuth.target_id.isin(gal4targets.Gene_ID.values)]
# plot on top with different color
plot.scatter_coef(gal4targets, 'galmain', 'yQC7:galmain', auto_ref=False,
                                        color='#f85f68', alpha=0.8, ax=ax)
ax.set(ylabel='Interaction', xlabel='Galactose')
plt.savefig('../output/Fig2_RNAseqScatter.svg')
