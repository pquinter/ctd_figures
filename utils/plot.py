from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib import cm
import matplotlib.patches as mpatches
from matplotlib import gridspec
from utils import ecdftools
import seaborn as sns
import corner

# style and colors ===========================================================

# matplotlib style
rc= {'axes.edgecolor': '0.3', 'axes.labelcolor': '0.3', 'text.color':'0.3',
    'axes.spines.top': 'False','axes.spines.right': 'False',
    'xtick.color': '0.3', 'ytick.color': '0.3', 'font.size':'35',
    'savefig.bbox':'tight', 'savefig.transparent':'True', 'savefig.dpi':'500'}
for param in rc:
    mpl.rcParams[param] = rc[param]

# scatter style
scatter_kws = {"s": 50, 'alpha':0.3,'rasterized':True}
# line style
line_kws = {'alpha':0.3,'rasterized':True}
CTDr_dict = {'yQC21':26,'yQC22':14,'yQC23':12}

def get_patches(color_dict):
    """
    Generate patches for figure legend
    """
    patches = [mpatches.Patch(color=color_dict[l], label=l) for l in color_dict]
    return patches

def get_palette(groups, palette='GnBu_d'):
    """
    Generate dict of color palette and corresponding legend patches
    Using seaborn color palettes
    """
    color_dict = {g:c for g,c in zip(groups, sns.color_palette(palette, len(groups)))}
    patches = get_patches(color_dict)
    return color_dict, patches

# CTD color palette
ctdr = ['26','14','12','10','9','8']
ctdr = [26,14,12,10,9,8]
colors_ctd, patches_ctd = get_palette(ctdr)

# Self recruitment palette
# composite of two cubehelix palettes, one for FUS and one for TAF variants

selfr_pal = {var:c for (var,c) in zip(
    ['TL47pQC99', 'TL47pQC1192E', 'TL47pQC1195D', 'TL47pQC119S2', #FUS variants
    'TL47pQC115', 'TL47pQC1202H','TL47pQC1203K'], #TAF variants
    sns.diverging_palette(220, 20, n=10))}
selfr_pal.update({var:c for (var,c) in zip(\
                ['yQC21', 'TL47pQC121', 'TL47pQC116'],#control, 13r and 10r CTD
                ['#4D4D4D','#DA84A6','#8A236E'])})


# scatter functions ===========================================================

def scatter(x, y, df, ax=None, color='#326976', scatter_kws=scatter_kws):
    """
    Convenience function to plot scatter from dataframe with default mpl style
    """
    if ax is None: fig, ax = plt.subplots(figsize=(14,7))
    ax.scatter(df[x].values, df[y].values, color=color, **scatter_kws)
    return ax

def selfrecruit_boxp(df, x='num_spots', y='frac_cells', hue='strain', 
    order=None, labels=None,
    median=None, xlabel='Number of Spots', ylabel='Fraction of Cells',
    palette=selfr_pal, boxkwargs={'height':12, 'aspect':18/10},
    median_kwargs={'linestyle':'--', 'linewidth':2, 'alpha':0.3}):
    """
    Boxplot of number of spots by cell fraction for self-recruitment assay
    """
    fig = sns.catplot(x=x, y=y, hue=hue, kind='box', legend=False,
            palette=palette, hue_order=order, data=df,
            **boxkwargs)

    # Plot median
    # easier to use boxplot func for horizontal line, hide everything but median
    if median:
        sns.boxplot(x=x, y=y, whis=0, fliersize=0, linewidth=0,
            data=df[df.strain==median], color=palette[median],
            medianprops=median_kwargs, boxprops={'alpha':0}, ax=fig.ax)
    fig.ax.set(xlabel=xlabel, ylabel=ylabel,
            xlim=(-0.5, 5.5), xticks=np.arange(0,6))
    fig.ax.set_xticklabels(np.arange(1,7))

    if labels and order:
        patches = [mpatches.Patch(color=palette[l], label=name) for l,name in zip(order, labels)]
    else:
        patches = [mpatches.Patch(color=palette[l], label=l) for l in palette]
    fig.ax.legend(handles=patches, ncol=len(patches))
    return fig.ax

# ECDF functions ==============================================================

def plot_ecdf(arr, formal=0, ax=None, label='', color='#326976', scatter_kws=scatter_kws, line_kws=line_kws):#alpha=0.3, formal=0, label='', ylabel='ECDF', xlabel='', color='b', title='', rasterized=True, lw=None):
    """
    Convenience function to plot ecdf with default mpl style
    """
    if ax==None: fig, ax = plt.subplots(figsize=(14,7))
    if formal:
        ax.plot(*ecdftools.ecdf(arr, conventional=formal), label=label, color=color, **line_kws)
    else:
        ax.scatter(*ecdftools.ecdf(arr, conventional=formal), label=label, color=color, **scatter_kws)
    return ax


def ecdf_ci(x, df, ax=None, no_bs=1000, ci=99, plot_median=True, color='#326976',
        ci_alpha=0.3, med_alpha=0.8, label='', **ecdfkwargs):
    """
    Plot ECDF with bootstrapped confidence interval
    """
    if ax is None: fig, ax = plt.subplots()

    if ci:
        # generate bootstrap samples
        bs_samples = ecdftools.bs_samples(df, x, no_bs)
        # compute ecdfs from bs samples
        bs_ecdf = ecdftools.ecdfs_par(bs_samples)
        # get ECDF confidence intervals
        quants, ci_high, ci_low = ecdftools.ecdf_ci(bs_ecdf, ci=ci)
        # plot intervals
        ax.fill_betweenx(quants, ci_high, ci_low, color=color,
                alpha=ci_alpha, rasterized=True)

    if plot_median:
        # get median and bootstrapped CI if available
        median = np.median(df[x])
        try:
            # get median index
            med_ix = np.argmin((np.abs(quants - 0.5)))
            err = np.array([[median-ci_low[med_ix]],   # lower error
                            [ci_high[med_ix]-median]]) # higher error
        except NameError: err = None
        ax.errorbar(median, 1.05, xerr=err, color=color, fmt='.',
                markersize=20, alpha=med_alpha, elinewidth=3, rasterized=True)

    plot_ecdf(df[x], ax=ax, color=color, label=label, **ecdfkwargs)
    return ax

def ecdfbystrain(x, df, groupby='CTDr', ax=None, strains='all', plot_median=True,
        ci=99, no_bs=1000, ci_alpha=0.3, med_alpha=0.8, colors=colors_ctd, patches=patches_ctd,
        **ecdfkwargs):
    """
    Plot ECDFs by strain
    x: str
        name of column to plot
    """
    if ax is None: fig, ax = plt.subplots()
    if strains=='all': strains = df[groupby].unique()

    for s, group in df[df[groupby].isin(strains)].groupby(groupby):
        ecdf_ci(x, group, ax=ax, no_bs=no_bs, ci=ci, plot_median=plot_median,
                color=colors[s], ci_alpha=ci_alpha, med_alpha=med_alpha, 
                label=s, **ecdfkwargs)

    # filter legend patches
    patches = [p for p in patches if p.get_label() in [str(s) for s in strains]]
    ax.legend(handles=patches)
    return ax

# RNAseq ===============================================================

def qvalecdf_hmap(df, coefs, coef_names=None, ax=None):
    """
    Plot cumulative distribution of q-values as a heatmap

    df: DataFrame
        with q-value thresholds, number of transcripts and coefficients
        must have column names: ['cid','qval_thresh','no_transcripts']
    coefs: iterable
        list of coefficients to plot
    """
    # select coefficients
    df_sel = df[df.cid.isin(coefs)]
    # pivot data into heatmap format
    df_hmap = df_sel.pivot(index='cid', columns='qval_thresh', values='no_transcripts')
    # sort
    sorter= dict(zip(coefs,range(len(coefs))))
    df_hmap['sorter'] = df_hmap.index.map(sorter)
    df_hmap.sort_values('sorter', inplace=True)

    # plot selected
    if ax is None: fig, ax = plt.subplots(figsize=(22,2*len(coefs)))
    sns.heatmap(df_hmap, yticklabels=df_hmap.index, xticklabels=99, ax=ax,
            cmap='viridis', rasterized=True, cbar_kws={'label':'Transcripts'})

    # format xticks
    x_format = ax.xaxis.get_major_formatter()
    # make q-value axis str to avoid extremely long floats
    x_format.seq = ["{:0.1f}".format(float(s)) for s in x_format.seq]
    ax.xaxis.set_major_formatter(x_format)
    # horizontal xticks
    plt.xticks(rotation=0)
    ax.set(xlabel='q-value threshold', ylabel='Coefficient')
    # hide blue strip: no genes after cum fraction>1.0!
    ax.set_xlim(0, 999)
    plt.tight_layout()
    # assign intelligible coefficient names if provided
    if coef_names is not None: ax.set_yticklabels(coef_names)
    return ax

def coef_stemplot(df, coefs, coef_names=None, qval_thresh=0.1, color='#326976', 
        orient='v', ax=None):
    """
    Plot number of transcripts at q-value thresh per coefficient
    """
    # select coefficients
    df_sel = df[df.cid.isin(coefs)]
    # get number of transcripts at q-value threshold for each coefficient
    transcripts = df_sel[df_sel.qval_thresh<qval_thresh].sort_values('qval_thresh').groupby('cid').tail(1)
    # sort
    sorter= dict(zip(coefs,range(len(coefs))))
    transcripts['sorter'] = transcripts.cid.map(sorter)
    transcripts.sort_values('sorter', inplace=True)
    # plot
    if ax is None and orient=='v': fig, ax = plt.subplots(figsize=(len(coefs)*2, 8))
    elif ax is None and orient=='h': fig, ax = plt.subplots(figsize=(12,len(coefs)*2))
    if orient=='v':
        ax.vlines(transcripts.cid, 0, transcripts.no_transcripts, colors=color)
        ax.scatter(transcripts.cid, transcripts.no_transcripts, s=150, color=color)
        ax.set(ylabel=r'Transcripts at q<{0:0.1f}'.format(qval_thresh), ylim=0)
        ax.margins(0.1)
        plt.xticks(rotation=60)
        if coef_names is not None: ax.set_xticklabels(coef_names)
    elif orient=='h':
        # sort again to have up-down order
        if isinstance(color, list): color=color[::-1]
        transcripts.sort_values('sorter', ascending=False, inplace=True)
        ax.hlines(transcripts.cid, 0, transcripts.no_transcripts, colors=color)
        ax.scatter(transcripts.no_transcripts, transcripts.cid, s=150, color=color)
        ax.set(xlabel=r'Transcripts at q<{0:0.1f}'.format(qval_thresh), xlim=0)
        ax.margins(0.1)
        if coef_names is not None: ax.set_yticklabels(coef_names[::-1])
    plt.tight_layout()
    return ax

def scatter_coef(df, x_coef, y_coef, ax=None, auto_ref=True, alpha=0.2,
    color='#326976', color_autoref='#99003d',qvaloffset=1e-100, **scatter_kwargs):
    """
    Scatter plot of coefficients
    df: DataFrame
        Must contain columns ['cid', 'target_id','b']
    x_coef, y_coef: str
        name of coefficients to plot
    auto_ref: bool
        whether to plot diagonal of x_coef vs x_coef with appropriate sign from correlation
    qvaloffset: float
        small number to add qval to prevent infinity from np.log(0)
    """

    if ax is None: fig, ax = plt.subplots(figsize=(12,10))

    # split and merge to make sure they are ordered
    xx = df[df.cid==x_coef]
    yy = df[df.cid==y_coef]
    merge = pd.merge(xx, yy, on='target_id')

    # scatter with size inversely proportional to log of q-value
    ax.scatter(merge.b_x.values, merge.b_y.values,
                s=-np.log(merge.qval_y.values+qvaloffset),
                rasterized=True, alpha=alpha, c=color, **scatter_kwargs)
    if auto_ref:
        # get correlation and use sign to plot reference
        corr = np.corrcoef(merge.b_x.values, merge.b_y.values)[0,1]
        print('Pearson correlation = {0:.2f}'.format(corr))
        ax.scatter(xx.b.values, np.sign(corr) * xx.b.values,
            s=1, rasterized=True, alpha=0.5, c=color_autoref)
        ax.legend(['Pearson correlation = {0:.2f}'.format(corr)], fontsize=20)
    ax.axhline(0, ls='--', alpha=0.1, color='k')
    ax.axvline(0, ls='--', alpha=0.1, color='k')
    ax.set(xlabel=x_coef, ylabel=y_coef)
    plt.tight_layout()
    return ax

# Growth curves ===============================================================
def filtOD(x, low=0.15, high=0.8):
    """
    get exponential region of growth curve, between 'low' and 'high' ODs
    """
    od = x.od.values
    time = x.Time.values
    # mask values in between allowed OD range
    mask = (od>low) & (od<high)
    if np.sum(mask)==0: return 0,0
    # create masked array
    masked_od = np.ma.array(od, mask=np.logical_not(mask))
    # get longest contigous array
    contigs = np.ma.flatnotmasked_contiguous(masked_od)
    # measure array lengths
    contigs_len = [s.stop - s.start for s in contigs]
    # get longest
    contig_ind = contigs[np.argmax(contigs_len)]
    od = od[contig_ind].astype(float)
    time = time[contig_ind].astype(int)
    # subtract initial time
    time -= time[0]
    return time, od

def growth_curve(curve_tidy, strains='all', groupby='CTDr', plot='range',
        colors=colors_ctd, alpha=0.8, range_alpha=0.3, low=0.15, high=0.8,
        ax=None, figsize=(12,8)):
    """
    Plot growth curves by strain
    """
    if strains=='all': strains = curve_tidy.strain.unique()
    if ax is None: fig, ax = plt.subplots(figsize=figsize)
    # filter strains
    curve_tidy = curve_tidy[curve_tidy.strain.isin(strains)]
    for strain, curve in curve_tidy.groupby(groupby):
        # filter OD to specified range
        curves_filt = [filtOD(x[1], low=low, high=high) for x in curve.groupby('well')]

        # plot every replicate
        if plot=='all':
            [ax.plot(_time, _curve, alpha=alpha, rasterized=True, label=strain,
                color=colors[strain]) for (_time, _curve) in curves_filt]

        # plot mean and replicate range only
        elif plot=='range':
            time, curves_filt = curves_filt[0][0], [c[1] for c in curves_filt]
            # get time of shortest curve
            time = time[:min([len(c) for c in curves_filt])]
            # get mean and range
            odlow = [np.min(_ods) for _ods in zip(*curves_filt)]
            odhigh = [np.max(_ods) for _ods in zip(*curves_filt)]
            odmean = [np.mean(_ods) for _ods in zip(*curves_filt)]
            ax.fill_between(time, odhigh, odlow, alpha=range_alpha,
                    rasterized=True, color=colors[strain])
            ax.plot(time, odmean, alpha=alpha, rasterized=True, label=strain,
                    color=colors[strain])

    patches = [mpatches.Patch(color=colors[s], label=s) for s in curve_tidy[groupby].unique()]
    ax.legend(handles=patches, title=groupby)
    ax.set(xlabel='Time (min)', ylabel='OD$_{600}$')
    plt.tight_layout()

    return ax

def _stripplot_errbars(strip_data, i, ax, jlim=0.1, yerr=None, xerr=None, **kwargs):
    """ Hack to plot errorbars with jitter """
    from scipy import stats
    jitterer = stats.uniform(-jlim, jlim * 2).rvs
    cat_pos = np.ones(strip_data.size) * i
    cat_pos += jitterer(len(strip_data))
    ax.errorbar(strip_data, cat_pos, yerr, xerr, **kwargs)

def stripplot_errbars(x, y, err, order, df, errax='xerr', colors='#326976', 
        ax=None, plot_kws={'fmt':'o', 'alpha':0.8, 'ms':8, 'elinewidth':2,'rasterized':True}):
    """
    Plot horizontal stripplot with custom errorbars for each point from dataframe
    """
    if ax is None: fig, ax = plt.subplots()
    if isinstance(colors, str): colors=[colors]*len(order)
    # create sorting index; needed because of duplicate labels in groups
    sort_ix = {strain:ix for ix, strain in enumerate(order)}
    # plot with seaborn to add labels and hide with alpha; do NOT rasterize,
    # otherwise cannot save as vector graphics
    sns.stripplot(x=x, y=y, data=df, order=order, alpha=0, ax=ax)
    # plot with by group with errorbars
    for strain, group in df.groupby(y):
        # Get xvalue from sorting index, skip if not included
        try: yval = sort_ix[strain]
        except KeyError: continue
        _stripplot_errbars(group[x].values, yval, ax, color=colors[yval],
                                xerr=group[err], **plot_kws)
    return ax

# Spot classification ===========================================================

def plot2dDecisionFunc(clf, xs, ys, colors=('#326976','#da6363'), labels=(True, False),
        xlabel='Correlation with Ideal Spot', ylabel='Intensity',
        plot_data_contour=True, plot_scatter=False,
        scatter_alpha=0.1, figsize=(12, 10)):
    """
    Plot decision surface of classifier with 2D points on top
    """
    # transpose data if necessary
    if len(xs)>len(xs.T): xs = xs.T

    # make grid
    xx, yy = np.meshgrid(
            np.linspace(np.min(xs[0]), np.max(xs[0]), 100),
            np.linspace(np.min(xs[1]), np.max(xs[1]), 100))
    # get decision function
    if hasattr(clf, 'decision_function'):
        z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    z = z.reshape(xx.shape)

    # put things in dataframe
    data = pd.DataFrame()
    data[xlabel] = xs[0]
    data[ylabel] = xs[1]
    data['y'] = ys

    colors = dict(zip(labels, colors))
    # make the base figure for corner plot
    ndim = len(xs)
    fig, axes = plt.subplots(ndim, ndim, figsize=figsize)

    if not plot_data_contour: axes[1,0].clear()

    if hasattr(clf, 'decision_function'):
        # plot decision boundary
        axes[1,0].contour(xx, yy, z, levels=[0], linewidths=2, colors='#FFC300')
    else:
        # or probability distribution
        cs = axes[1,0].contourf(xx, yy, z, cmap='viridis')
    handles = [mpatches.Patch(color=colors[l], label=l) for l in labels]
    axes[0,1].legend(handles=handles, loc='lower left')

    # plot data with corner
    data.groupby('y').apply(lambda x: corner.corner(x, color=colors[x.name], 
        hist_kwargs={'density':True}, fig=fig, rasterized=True))
    # plot data on top
    if plot_scatter:
        data.groupby('y').apply(lambda x: axes[1,0].scatter(x[xlabel],
        x[ylabel], alpha=scatter_alpha, color=colors[x.name], rasterized=True))

    # add colorbar to countourf. Must be done after corner, or it will complain
    if hasattr(clf, 'predict_proba'):
        fig.colorbar(cs, ax=axes[1,0], ticks=np.linspace(0,1,5))
    plt.tight_layout()
    return axes
