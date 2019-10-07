from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from utils import ecdftools, plot
import seaborn as sns

###############################################################################
# Fraction of active cells from smFISH
###############################################################################

freq_df = pd.read_csv('../data/smFISH_GAL10_GAL3_FracActive.csv')
freq_df.strain = freq_df.strain.str.replace(r'hCTD\w*','hCTD')

order = ['TL47','wCTD','hCTD']
colors = {s:c for s,c in zip(order,['#326976', '#5e7669','#9e2b56'])}

for gene, _freq_df in freq_df.groupby('gene'):
    fig, ax = plt.subplots(figsize=(9,8))
    sns.stripplot(y='strain', x='frac_active', data=_freq_df,
        ax=ax, alpha=0.5, order=order, palette=colors, size=10)
    sns.pointplot(y='strain', x='frac_active', data=_freq_df, palette=colors,
        order=order, ax=ax, alpha=1, size=20, join=False, ci=99)
    # below two lines to draw points over strip plot
    plt.setp(ax.lines, zorder=100)
    plt.setp(ax.collections, zorder=100, label="")
    plt.legend([], frameon=False)
    ax.set(xlabel='Active cells fraction', ylabel='CTD repeats', xticks=np.arange(0,1.1, 0.2))
    strain_labels = ['scCTD','ceCTD','hsCTD']
    plt.yticks(plt.yticks()[0], strain_labels)
    plt.tight_layout()
    plt.title(gene.upper())
    plt.savefig('../output/Fig7_FracActive_{}.svg'.format(gene))

###############################################################################
# TS intensity ECDF from smFISH
###############################################################################
parts = pd.read_csv('../data/smFISH_GAL10_GAL3_TS.csv')
parts = parts[(parts.strain.isin(order))&(parts.date==9132019)]
parts['mass'] = parts.groupby(['date','gene']).mass.transform(lambda x: (x-np.min(x))/(np.max(x)-np.min(x)))
parts.strain = parts.strain.str.replace(r'hCTD\w*','hCTD')
parts = parts[parts.strain.isin(['TL47','hCTD','wCTD'])]
for gene, _parts in parts.groupby('gene'):
    fig, ax = plt.subplots(figsize=(9, 8))
    ax = plot.ecdfbystrain('mass', _parts, groupby='strain', colors=colors, ax=ax, formal=True, line_kws={'alpha':1, 'rasterized':True})
    ax.set(xlabel='TS Fluorescence (a.u.)', ylabel='ECDF', title=gene.upper(),
            xlim=((-0.05,0.6) if gene=='gal3' else (-0.05, 0.6)))
    plt.tight_layout()
    plt.savefig('../output/Fig7_ECDF_{}.svg'.format(gene))

###############################################################################
# Cell intensity ECDF from smFISH with polydT probes
###############################################################################
polydt = pd.read_csv('../data/smFISH_polydT.csv')
#polydt = pd.read_csv('../data/smFISH_polydT_bgsubtracted.csv')
# Discard likely broken cells by dapi
polydt['mean_intensity'] = polydt.intensity_sum_cell/polydt.area_cell

im_mean = polydt.groupby(['im_name','channel']).mean_intensity.mean().reset_index()
ims_thresh = im_mean[(im_mean.channel==2)&(im_mean.mean_intensity>1300)].im_name.values
#ims_thresh = im_mean[(im_mean.channel==2)&(im_mean.mean_intensity<0.009)].im_name.values
#bad_ims = ['09182019_hCTD_17', '09182019_hCTD_19', '09182019_hCTD_20',
#       '09182019_hCTD_21', '09182019_hCTD_22', '09182019_hCTD_23', '09182019_hCTD_24']
#ims_thresh = im_mean[~(im_mean.im_name.isin(bad_ims))].im_name.values
# scale range
for val in ['mean_intensity','intensity_sum_cell']:
    polydt[val] = (polydt[val]-polydt[val].min())/(polydt[val].max()-polydt[val].min())*100
# filter index for polydT and intact cells
filt = (polydt.channel==0)&(polydt.im_name.isin(ims_thresh))

# Total mRNA
fig, ax = plt.subplots(figsize=(9, 8))
plot.ecdfbystrain('intensity_sum_cell', polydt[filt], groupby='strain', colors=colors, ax=ax, formal=True, line_kws={'alpha':1, 'rasterized':True})
ax.set(xlabel='Total mRNA fluorescence (a.u.)', ylabel='ECDF')
plt.tight_layout()
plt.legend(handles=plot.get_patches(colors))
plt.savefig('../output/Fig7_ECDF_totalmRNA.svg')

# mRNA Concentration
fig, ax = plt.subplots(figsize=(9, 8))
plot.ecdfbystrain('mean_intensity', polydt[filt], groupby='strain', colors=colors, ax=ax, formal=True, line_kws={'alpha':1, 'rasterized':True})
ax.set(xlabel='Mean mRNA fluorescence (a.u.)', ylabel='ECDF')
plt.tight_layout()
plt.legend(handles=plot.get_patches(colors))
plt.savefig('../output/Fig7_ECDF_meantotalmRNA.svg')

###############################################################################
# ECDF of cell sizes
###############################################################################
# convert area to microns
micr2ppx = 0.072**2
polydt['area_cell_um2'] = polydt.area_cell.values*micr2ppx
fig, ax = plt.subplots(figsize=(9, 8))
plot.ecdfbystrain('area_cell_um2', polydt.drop_duplicates(['im_name','label']), groupby='strain', colors=colors, ax=ax, formal=True, line_kws={'alpha':1, 'rasterized':True})
ax.set(xlabel=r'Cell area ($\mu$m$^2$)', ylabel='ECDF', xticks=np.arange(0,110, 15), xlim=(0,70))
plt.tight_layout()
plt.legend(handles=plot.get_patches(colors))
plt.savefig('../output/Fig7_ECDF_cellsize.svg')

###############################################################################
# Cell size vs intensity
###############################################################################
fig, ax = plt.subplots(figsize=(9, 8))
polydt[filt].groupby('strain').apply(lambda x: plt.scatter(x.mean_intensity, x.area_cell_um2, alpha=0.3, color=colors[x.name], s=10))
ax.set(ylabel=r'Cell area ($\mu$m$^2$)', xlabel='Mean mRNA fluorescence (a.u.)', ylim=(5,60))
plt.tight_layout()
plt.legend(handles=plot.get_patches(colors))
plt.savefig('../output/Fig7_scatter_cellsizefluor.svg')
