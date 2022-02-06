import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import ScalarFormatter
import scipy.stats


TITLESIZE = 28
LABELSIZE = 24
FIGSIZE   = (10,6)


def change_legend(ax, new_loc, fontsize, titlesize, **kws):
    '''funzione per modificare posizione e font size della legenda generata da seaborn'''
    old_legend = ax.legend_
    handles = old_legend.legendHandles
    labels = [t.get_text() for t in old_legend.get_texts()]
    title = old_legend.get_title().get_text()
    ax.legend(handles, labels, loc=new_loc, title=title, 
              fontsize=fontsize, title_fontsize=titlesize, 
              frameon = True, fancybox = False, framealpha = 0.5, **kws)

    return



# -------- FULL DATASETS


def plot_full_dataset(df, features, bins):
    """histogram of two features"""
    
    dt_bins     = bins[0]
    theta_bins  = bins[1]
    
    fig, ax = plt.subplots(ncols=2, figsize=(14,6), sharey=True)
    
    ax[0].set_title("time box",           fontsize=TITLESIZE)
    ax[0].set_xlabel("drift time (ns)",   fontsize=LABELSIZE)
    ax[0].set_ylabel("counts",            fontsize=LABELSIZE)
    
    ax[1].set_title("theta distribution", fontsize=TITLESIZE)
    ax[1].set_xlabel("theta (deg)",       fontsize=LABELSIZE)
    # ax[1].set_ylabel("counts",            fontsize=LABELSIZE)
    
    ax[0].set_xlim(dt_bins[0], dt_bins[-1])
    ax[1].set_xlim(theta_bins[0], theta_bins[-1])
    
    # exponential y ticks
    ax[0].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax[0].ticklabel_format(axis = "y", style = "sci", scilimits = (0,0))
    ax[0].yaxis.get_offset_text().set_fontsize(22)
    ax[1].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax[1].ticklabel_format(axis = "y", style = "sci", scilimits = (0,0))
    ax[1].yaxis.get_offset_text().set_fontsize(22)
    
    ax[0].tick_params(axis="both", which="major", labelsize=22, length=5)
    ax[1].tick_params(axis="both", which="major", labelsize=22, length=5)
    
    # drift time distribution
    sns.histplot(
        data      = df,
        x         = features[0],
        bins      = dt_bins,
        stat      = "count",
        element   = "step",
        fill      = True,
        color     = "#aadeff",
        edgecolor = "#009cff",
        linewidth = 3,
        label     = "full dataset",
        ax        = ax[0]
    )

    # theta distribution
    sns.histplot(
        data      = df,
        x         = features[1],
        bins      = theta_bins,
        stat      = "count",
        element   = "step",
        fill      = True,
        color     = "#aadeff",
        edgecolor = "#009cff",
        linewidth = 3,
        label     = "full dataset",
        ax        = ax[1]
    )
    
        
    return fig, ax
    

    
def plot_full_scatter(df, features):
    """scatterplot and profile plot of two features"""
    
    # perform a linear regression using scipy.stats.linregress
    reg_results = scipy.stats.linregress(df[features[0]], df[features[1]])

    # bin the x data
    bins = np.linspace(-200, 400, 12)
    # we add a column to our dataframe so we can use pandas.DataFrame.groupby()
    df["bin"] = np.digitize(df[features[0]], bins=bins)

    # bin centers
    x    = 0.5 * (bins[1:] + bins[:-1])
    # mean of y data inside each bin
    y    = df.groupby("bin")[features[1]].mean()
    # std of y data inside each bin
    erry = df.groupby("bin")[features[1]].std()
    
    xgrid = np.linspace(df[features[0]].min(), df[features[0]].max(), 300)
    
    fig, ax = plt.subplots(figsize=(10,8))
    
    ax.set_title("scatter theta vs drift time")
    ax.set_xlabel("drift time (ns)")
    ax.set_ylabel("theta (deg)")

    ax.scatter(
        data      = df,
        x         = features[0],
        y         = features[1],
        color     = "#aadeff",
        edgecolor = "#009cff",
        alpha     = 0.7
    )
    
    ax.errorbar(
        x, 
        y, 
        yerr=erry, 
        color="black",
        ecolor ="black", 
        elinewidth=2,
        capsize=2,
        capthick=2,
        marker="o", 
        linestyle="",
        alpha=1,
        label = "profile", 
    )
    
    ax.plot(
        xgrid,    
        reg_results.intercept + reg_results.slope*xgrid, 
        color = "#ff6300", 
        linestyle = "dashed", 
        linewidth = 3, 
        label="reg"
    )
    
    
    df.drop(labels="bin", axis=1, inplace=True)
    
    return fig, ax




def plot_full_dataset_2(ref, data, features, bins, weights):
    """histogram of two features"""
    
    dt_bins     = bins[0]
    theta_bins  = bins[1]
    weightsRef  = weights[0]
    weightsData = weights[1]
    
    fig, ax = plt.subplots(ncols=2, figsize=(16,8), sharey=True)
    
    ax[0].set_title("time box",           fontsize=TITLESIZE)
    ax[0].set_xlabel("drift time (ns)",   fontsize=LABELSIZE)
    ax[0].set_ylabel("counts",            fontsize=LABELSIZE)
    
    ax[1].set_title("theta distribution", fontsize=TITLESIZE)
    ax[1].set_xlabel("theta (deg)",       fontsize=LABELSIZE)
    # ax[1].set_ylabel("counts",            fontsize=LABELSIZE)
    
    ax[0].set_xlim(dt_bins[0], dt_bins[-1])
    ax[1].set_xlim(theta_bins[0], theta_bins[-1])
    
    # exponential y ticks
    ax[0].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax[0].ticklabel_format(axis = "y", style = "sci", scilimits = (0,0))
    ax[0].yaxis.get_offset_text().set_fontsize(22)
    ax[1].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax[1].ticklabel_format(axis = "y", style = "sci", scilimits = (0,0))
    ax[1].yaxis.get_offset_text().set_fontsize(22)
    
    ax[0].tick_params(axis="both", which="major", labelsize=22, length=5)
    ax[1].tick_params(axis="both", which="major", labelsize=22, length=5)
    
    
    
    ax[0].hist(
        ref[features[0]], 
        bins=dt_bins, 
        weights=weightsRef,
        density=False,
        histtype="stepfilled", 
        linewidth=3,
        edgecolor="#009cff", 
        facecolor="#aadeff", 
        alpha=1, 
        label="reference"
    )
    
    ax[0].hist(
        data[features[0]], 
        bins=dt_bins, 
        weights=weightsData,
        density=False,
        histtype="stepfilled", 
        linewidth=3,
        edgecolor="#ff6300", 
        facecolor="none", 
        alpha=1, 
        label="data"
    )
    


    ax[1].hist(
        ref[features[1]], 
        bins=theta_bins, 
        weights=weightsRef,
        density=False,
        histtype="stepfilled", 
        linewidth=3,
        edgecolor="#009cff", 
        facecolor="#aadeff", 
        alpha=1, 
        label="reference"
    )
    
    ax[1].hist(
        data[features[1]], 
        bins=theta_bins, 
        weights=weightsData,
        density=False,
        histtype="stepfilled", 
        linewidth=3,
        edgecolor="#ff6300", 
        facecolor="none", 
        alpha=1, 
        label="data"
    )

    # legend    
    ax[0].legend()
    ax[1].legend()
    change_legend(ax=ax[0], new_loc="upper right", fontsize=LABELSIZE, titlesize=0)
    change_legend(ax=ax[1], new_loc="upper right", fontsize=LABELSIZE, titlesize=0)
        
    return fig, ax



# ------- NPLM Datasets


def plot_ref_data_1(ref, data, features, bins, weights):
    """plot reference and data distributions of two features"""
    
    dt_bins     = bins[0]
    theta_bins  = bins[1]
    weightsRef  = weights[0]
    weightsData = weights[1]
    
    fig, ax = plt.subplots(ncols=2, figsize=(18,8), sharey=True)
    
    ax[0].set_title("time box",           fontsize=TITLESIZE)
    ax[0].set_xlabel("drift time (ns)",   fontsize=LABELSIZE)
    ax[0].set_ylabel("counts",            fontsize=LABELSIZE)
    
    ax[1].set_title("theta distribution", fontsize=TITLESIZE)
    ax[1].set_xlabel("theta (deg)",       fontsize=LABELSIZE)
    
    ax[0].set_xlim(dt_bins[0], dt_bins[-1])
    ax[1].set_xlim(theta_bins[0], theta_bins[-1])
    
    ax[0].hist(
        ref[features[0]], 
        bins=dt_bins, 
        weights=weightsRef,
        density=False,
        histtype="stepfilled", 
        linewidth=3,
        edgecolor="#009cff", 
        facecolor="#aadeff", 
        alpha=1, 
        label="reference"
    )
    
    ax[0].hist(
        data[features[0]], 
        bins=dt_bins, 
        weights=weightsData,
        density=False,
        histtype="stepfilled", 
        linewidth=3,
        edgecolor="#ff6300", 
        facecolor="none", 
        alpha=1, 
        label="data"
    )
    


    ax[1].hist(
        ref[features[1]], 
        bins=theta_bins, 
        weights=weightsRef,
        density=False,
        histtype="stepfilled", 
        linewidth=3,
        edgecolor="#009cff", 
        facecolor="#aadeff", 
        alpha=1, 
        label="reference"
    )
    
    ax[1].hist(
        data[features[1]], 
        bins=theta_bins, 
        weights=weightsData,
        density=False,
        histtype="stepfilled", 
        linewidth=3,
        edgecolor="#ff6300", 
        facecolor="none", 
        alpha=1, 
        label="data"
    )
    
    # exponential y ticks
    ax[0].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax[0].ticklabel_format(axis = "y", style = "sci", scilimits = (0,0))
    ax[0].yaxis.get_offset_text().set_fontsize(22)
    ax[1].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax[1].ticklabel_format(axis = "y", style = "sci", scilimits = (0,0))
    ax[1].yaxis.get_offset_text().set_fontsize(22)
    
    ax[0].tick_params(axis="both", which="major", labelsize=22, length=5)
    ax[1].tick_params(axis="both", which="major", labelsize=22, length=5)
    
    ax[0].legend()
    ax[1].legend()
    change_legend(ax=ax[0], new_loc="upper right", fontsize=LABELSIZE, titlesize=0)
    change_legend(ax=ax[1], new_loc="upper right", fontsize=LABELSIZE, titlesize=0)
    
    fig.tight_layout()
    
    return fig, ax