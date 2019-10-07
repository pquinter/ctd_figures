from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from utils import plot

###############################################################################
# Growth curves for strains with 26, 14, 12 and 10 CTDr
###############################################################################

od = pd.read_csv('../data/growthOD_merged_tidy.csv')
strains = ['TL47', 'yQC5','yQC6','yQC62', 'yQC63', 'yQC64']
# make short palette
ctdr = [26,14,12,10,9,8]
colors_ctd, patches_ctd = plot.get_palette(ctdr)
fig, ax = plt.subplots(figsize=(14, 8))
ax = plot.growth_curve(od, strains=strains, high=1.5, ax=ax, colors=colors_ctd)
ax.set(xlim=(-200, 1000), ylim=(0.15, 1.6), xticks=(np.arange(0, 1100, 200)))
plt.tight_layout()
plt.savefig('../output/Fig2_growth_.svg')

###############################################################################
# Doubling times for strains with 26, 14, 12 and 10 CTDr
###############################################################################

growth = pd.read_csv('../data/growthrates_merged.csv')
fig, ax = plt.subplots(figsize=(10,8))
ax = plot.stripplot_errbars('inverse max df', 'strain', 'stdev', strains,
                    growth, ax=ax, colors=[colors_ctd[c] for c in colors_ctd])
ax.set(xlabel='Doubling Time (min)', ylabel='CTD repeats')
# assign intelligible strain names
strain_labels = ['26','14','12','10','9','8']
plt.yticks(plt.yticks()[0], strain_labels)
plt.tight_layout()
plt.savefig('../output/Fig2_growthrates.png')

###############################################################################
# RPB1 nuclear fluorescence
###############################################################################
rpb1fluor = pd.read_csv('../data/mScarRPB1_05212019_quantiles.csv')
fig, ax = plt.subplots(figsize=(14, 8))
ax = plot.ecdfbystrain('0.95_quant_nc', rpb1fluor, ax=ax)
ax.set(xlabel='Nuclear Fluorescence (a.u.)', ylabel='ECDF', xlim=(100, 1250))
plt.tight_layout()
plt.savefig('../output/Fig2_rpb1fluor_.svg')
