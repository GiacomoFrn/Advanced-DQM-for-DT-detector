import os
import h5py
import glob
import json
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FormatStrFormatter

from NPLM.PLOTutils import *
from NPLM.ANALYSISutils import *
from NPLM.NNutils import *

FIG_SIZE = (12,7)


def change_legend(ax, new_loc, fontsize, titlesize, **kws):
    """function to easily cutomize legends"""

    old_legend = ax.legend_
    handles = old_legend.legendHandles
    labels = [t.get_text() for t in old_legend.get_texts()]
    title = old_legend.get_title().get_text()

    ax.legend(handles[::-1], labels[::-1], loc=new_loc, title=title, 
              fontsize=fontsize, title_fontsize=titlesize, 
              frameon = True, fancybox = False, framealpha = 0.5, **kws)


def smoothen(tvalues_check, threshold=3):
    """function that fixes weird history trends"""
    thrs = threshold
    tvalues_check = tvalues_check.copy()
    for toy in tvalues_check:
        for i, epoch_check in enumerate(toy):
            if not i:
                continue
            if i>=toy.shape[0]-1:
                break
            if np.abs(toy[i]-toy[i-1]) > thrs and (np.abs(toy[i]-toy[i+1]) > thrs or np.abs(toy[i]-toy[i+2]) > thrs):
                toy[i] = (toy[i-1]+toy[i+1])/2
                i+=2
        for i in range(toy.shape[0]-10, toy.shape[0]-1):
            if np.abs(toy[i]-toy[-1])> thrs or np.abs(toy[i]-toy[i+1])> thrs:
                toy[i] = (toy[i-1] + toy[-1])/2
                    
    return tvalues_check
    
    
def t_median(t_list, dof):
    """function that computes the median of the t_list and its pvalue"""
    
    # calcolo la mediana della lista
    median_t = np.median(t_list)
    print(f"\nMedian t distribution: {median_t:.5f}")
    median_chi2 = scipy.stats.chi2.median(df=dof)
    print(f"Median chi2 (ndf={dof}): {median_chi2:.5f}")
    
    print()
    
    # calcolo il p-value della lista 
    p_list = np.sum([1/(len(t_list)) for x in t_list if x>median_t])
    print(
        f"Median p-value: {p_list :.4f}\
        Median significance: {scipy.stats.norm.ppf(1-p_list):.4f}\
        from t list"
    )
    print(
        f"Median p-value: {scipy.stats.chi2.sf(median_t, df=dof):.4f}\
        Median significance: {scipy.stats.norm.ppf(1-scipy.stats.chi2.sf(median_t, df=dof)):.4f}\
        from chi2 distribution" 
    ) 


def collect_weights(DIR_IN, suffix='weights'):
    parameters = {}
    init = False
    for fileIN in glob.glob("%s/*_%s.h5" %(DIR_IN, suffix)):
        f = h5py.File(fileIN, 'r')
        for j in f:
            for k in f.get(j):
                for m in f.get(j).get(k):
                    if not init:
                        parameters[k+'_'+m[0]]= np.expand_dims(np.array(f.get(j).get(k).get(m)), axis=0)
                    else:
                        parameters[k+'_'+m[0]]= np.concatenate((parameters[k+'_'+m[0]], np.expand_dims(np.array(f.get(j).get(k).get(m)), axis=0)), axis=0)
        f.close()
        init=True
        
    return parameters


def build_model(jsonfile):
    with open(jsonfile, 'r') as jsonfile:
        config_json = json.load(jsonfile)
    correction = config_json["correction"]
    NU_S, NUR_S, NU0_S, SIGMA_S = [0], [0], [0], [0]
    NU_N, NUR_N, NU0_N, SIGMA_N = 0, 0, 0, 0
    shape_dictionary_list = []
    #### training time                
    total_epochs_tau   = config_json["epochs_tau"]
    patience_tau       = config_json["patience_tau"]
    total_epochs_delta = config_json["epochs_delta"]
    patience_delta     = config_json["patience_delta"]

    #### architecture                
    BSMweight_clipping = config_json["BSMweight_clipping"]
    BSMarchitecture    = config_json["BSMarchitecture"]
    inputsize          = BSMarchitecture[0]
    BSMdf              = compute_df(input_size=BSMarchitecture[0], hidden_layers=BSMarchitecture[1:-1])
    tau = imperfect_model(
        input_shape=(None, inputsize),
        NU_S=NU_S, NUR_S=NUR_S, NU0_S=NU0_S, SIGMA_S=SIGMA_S, 
        NU_N=NU_N, NUR_N=NUR_N, NU0_N=NU0_N, SIGMA_N=SIGMA_N,
        correction=correction, shape_dictionary_list=shape_dictionary_list,
        BSMarchitecture=BSMarchitecture, BSMweight_clipping=BSMweight_clipping, train_f=True, train_nu=False
    )
    
    return tau


def plot_1distribution(t, df, xmin=None, xmax=None, nbins=10, wclip=0, save=False, save_path=None, file_name=None):
    """
    Plot the histogram of a test statistics sample (t) and the target chi2 distribution. 
    The median and the error on the median are calculated in order to calculate the median Z-score and its error.
    
    t:  (numpy array shape (None,))
    df: (int) chi2 degrees of freedom
    """
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    XMIN = 0
    if xmin:
        XMIN = xmin
    if t.max() >= 3*df:
        XMAX = t.max() + t.min()
    else:
        XMAX = 3*df
    if xmax:
        XMAX = xmax
              
    # binning
    hist, bin_edges = np.histogram(t, density=True, bins=nbins)
    
    binswidth = bin_edges[1]-bin_edges[0]
    bincenters = 0.5 * (bin_edges[1:]+bin_edges[:-1])
    err = np.sqrt(hist/(t.shape[0]*binswidth))
    
    # stat
    Z_obs     = scipy.stats.norm.ppf(scipy.stats.chi2.cdf(np.median(t), df))
    t_obs_err = 1.2533*np.std(t)*1./np.sqrt(t.shape[0])
    Z_obs_p   = scipy.stats.norm.ppf(scipy.stats.chi2.cdf(np.median(t)+t_obs_err, df))
    Z_obs_m   = scipy.stats.norm.ppf(scipy.stats.chi2.cdf(np.median(t)-t_obs_err, df))
    
    # ks
    KS, KS_pval = scipy.stats.kstest(
        rvs=t,
        cdf="chi2",
        args=(df,0,1)
    )
    
    
    # legend text
    label  = f"W_clip = {wclip}\nSample size = {t.shape[0]}\nMedian = {np.median(t):.2f}\nStd = {np.std(t):.2f}" 
    label += f"\nZ = {Z_obs:.2f} (+{Z_obs_p-Z_obs:.2f}/-{Z_obs-Z_obs_m:.2f})"
    if KS_pval < 10-3:
        label += f"\nKS pvalue = {KS_pval:.2e}"
    else:
        label += f"\nKS pvalue = {KS_pval:.2f}"
    

    sns.histplot(
        x=bin_edges[:-1], 
        weights=hist, 
        bins=bin_edges,
        stat="count", 
        element="bars", 
        linewidth=2,
        fill=True, 
        color="lightblue", 
        #color="#aadeff", 
        edgecolor="#2c7fb8",
        #edgecolor="#009cff", 
        ax=ax, 
        label=label
    )

    ax.errorbar(bincenters, hist, yerr=err, color="#2c7fb8", linewidth=2, marker="o", ls="")
   
    # plot reference chi2
    x = np.linspace(scipy.stats.chi2.ppf(0.0001, df), scipy.stats.chi2.ppf(0.9999, df), 100)
    ax.plot(
        x, 
        scipy.stats.chi2.pdf(x, df),
        "midnightblue", 
        lw=5, 
        alpha=0.8, 
        label=r"Target $\chi^2$(ndf="+str(df)+")",
        zorder=10
    )
    
    
    ax.legend()
    change_legend(ax=ax, new_loc="upper right", fontsize=14, titlesize=0)
    
    ax.set_title(f"Test statistic distribution", fontsize = 22)
    ax.set_xlabel("t", fontsize = 18)
    ax.set_ylabel(r"p(t | $\mathcal{R}$)", fontsize = 18)
    ax.set_xlim(XMIN, XMAX)
    
    ax.tick_params(axis = "both", which = "major", labelsize = 14, direction = "out", length = 5)
    
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0,0))
    ax.yaxis.get_offset_text().set_fontsize(14)

    fig.tight_layout()
    
    if save:
        if not save_path: print("argument save_path is not defined. The figure will not be saved.")
        else:
            if not file_name: file_name = "1distribution"
            else: file_name += "_1distribution"
            fig.savefig(save_path+file_name+".png", dpi = 300, facecolor="white")
    plt.show()
    return

    
def plot_percentiles(tvalues_check, df, patience=1000, wclip=None, ymax=None, ymin=None, save=False, save_path=None, file_name=None, smooth=None):
    """
    The funcion creates the plot of the evolution in the epochs of the [2.5%, 25%, 50%, 75%, 97.5%] quantiles of the toy sample distribution.
    The percentile lines for the target chi2 distribution are shown as a reference.
    
    patience:      (int) interval between two check points (epochs).
    tvalues_check: (numpy array shape (N_toys, N_check_points)) array of t=-2*loss
    df:            (int) chi2 degrees of freedom
    """
    
    if smooth:
        tvalues_check = smoothen(tvalues_check)
    
    N_CHECKS = tvalues_check.shape[1]
    EPOCH_CHECK = [patience*(i+1) for i in range(N_CHECKS)]
    XMIN = 0
    XMAX = N_CHECKS*patience
    YMIN = 0
    
    if ymin:
        YMIN = ymin
    if tvalues_check[:,-1].max() >= 3*df:
        YMAX = tvalues_check[:,-1].max() + YMIN
    else:
        YMAX = 3*df
    if ymax:
        YMAX = ymax
            
    color_list = ["seagreen", "mediumseagreen", "lightseagreen", "#2c7fb8", "midnightblue"]
    # color_list = ["#00b32a", "#00c282", "#00D2FF", "#009cff", "#005e99"]
    quantile_list   = [0.05,0.25,0.50,0.75,0.95]
    quantile_labels = ["5%", "25%", "50%", "75%", "95%"]
    
    th_quantile_position = [scipy.stats.chi2.ppf(i, df=df) for i in quantile_list]
    t_quantile = np.quantile(tvalues_check, quantile_list, axis=0)
    
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)

    for i in range(len(quantile_list)):
        ax.plot(
            EPOCH_CHECK, 
            t_quantile[i][:], 
            color = color_list[i], 
            linestyle="solid", 
            linewidth = 3, 
            label = quantile_labels[i]
        )
        ax.hlines(
            y=th_quantile_position[i], 
            xmin = XMIN, 
            xmax = N_CHECKS*patience, 
            color = color_list[i], 
            linestyle="dashed", 
            linewidth = 3, 
            alpha = 0.5, 
            label = "theoretical " + quantile_labels[i]
        )
        #ax.text(
        #    N_CHECKS*patience*1.05, 
        #    th_quantile_position[i], 
        #    quantile_labels[i], 
        #    horizontalalignment="left", 
        #    verticalalignment="center", 
        #    color=color_list[i],
        #    fontsize=22,
        #    transform=ax.transData
        #)
        #
    ax.legend(ncol=2, loc="lower left", fontsize=14, bbox_to_anchor=(1, 0.5))
    
    ax.set_title(f"Percentiles evolution", fontsize = 22)
    ax.set_xlabel("training epochs", fontsize = 18)
    ax.set_ylabel(r"t", fontsize = 18)
    ax.tick_params(axis = "both", which = "major", labelsize = 14, direction = "out", length = 5)
    
    #plt.setp(ax.get_xticklabels()[-1], visible=False)
    
    fig.tight_layout()
    
    if save:
        if not save_path: print("argument save_path is not defined. The figure will not be saved.")
        else:
            if not file_name: file_name = "percentiles"
            else: file_name += "_percentiles"
            fig.savefig(save_path+file_name+".png", dpi = 300, facecolor="white")
        
    plt.show()
    return


def plot_reco(
    df,
    data,
    weight_data,
    ref,
    weight_ref,
    tau_obs,
    tau_ref,
    features,
    bins_code,
    xlabel_code, 
    toy,
    save=False, save_path=None, file_name=None
):
    """function that plots data reconstrution"""
    
    Zscore=scipy.stats.norm.ppf(scipy.stats.chi2.cdf(tau_obs, df))
    
    fig, ax = plt.subplots(
        nrows=2, 
        ncols=len(features), 
        figsize=(18,10),
        sharey="row",
        sharex="col"
    )
    
    fig.suptitle(f"t_obs={tau_obs:.2f}    Z={Zscore:.2f}", fontsize = 26)
    recorow  = 0
    ratiorow = 1
        
    for figcol, feature in enumerate(features):

        
        bins = bins_code[feature]
        x = 0.5*(bins[1:]+bins[:-1])
        
        ################################### RECO
        hR = ax[recorow][figcol].hist(
            ref[feature],
            weights=weight_ref,
            bins=bins,
            histtype="stepfilled", 
            linewidth=3,
            edgecolor="none", 
            facecolor="#aadeff", 
            label="REFERENCE",  
            zorder=1
        )
        
        hD = ax[recorow][figcol].hist(
            data[feature],
            weights=weight_data,
            bins=bins,
            histtype="step",
            linewidth=2,
            color="#009cff",
            label="DATA", 
            zorder=2
        )  
        
        hN = ax[recorow][figcol].hist(
            ref[feature],
            weights=np.exp(tau_ref[:, 0])*weight_ref,
            bins=bins,
            histtype="step",
            linewidth=2,
            color="#45bf55",
            label="RECO",  
            zorder=3
        )
        
        ax[recorow][figcol].legend()
        change_legend(ax=ax[recorow][figcol], new_loc="upper right", fontsize=14, titlesize=0)
    
        #ax[recorow][figcol].set_xlabel(xlabel_code[feature], fontsize = 18)
        ax[recorow][0].set_ylabel("counts", fontsize = 18)
        ax[recorow][figcol].set_xlim(bins[0], bins[-1])

        ax[recorow][figcol].tick_params(axis = "both", which = "major", labelsize = 14, direction = "out", length = 5)

        ax[recorow][figcol].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax[recorow][figcol].ticklabel_format(axis = 'y', style = 'sci', scilimits = (0,0))
        ax[recorow][figcol].yaxis.get_offset_text().set_fontsize(14)
        
        
        
        ################################### RATIO
        ax[ratiorow][figcol].plot(
            x, 
            (hD[0])/(hR[0]),
            linewidth=3,
            color="#009cff", 
            alpha=1, 
            label="DATA / REF"
        )
        ax[ratiorow][figcol].plot(
            x, 
            (hN[0])/(hR[0]),
            linewidth=8,
            color="#45bf55", 
            alpha=1, 
            label="RECO / REF"
        )
        
        ax[ratiorow][figcol].legend()
        change_legend(ax=ax[ratiorow][figcol], new_loc="upper right", fontsize=14, titlesize=0)
    
        ax[ratiorow][figcol].set_xlabel(xlabel_code[feature], fontsize = 18)
        ax[ratiorow][0].set_ylabel(r"$n\,(x\,|\,\mathcal{D})\,\,/\,\,n\,(x\,|\,\mathcal{R})$", fontsize = 18)
        ax[ratiorow][figcol].set_xlim(bins[0], bins[-1])

        ax[ratiorow][figcol].tick_params(axis = "both", which = "major", labelsize = 14, direction = "out", length = 5)

        ax[ratiorow][figcol].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax[ratiorow][figcol].ticklabel_format(axis = 'y', style = 'sci', scilimits = (0,0))
        ax[ratiorow][figcol].yaxis.get_offset_text().set_fontsize(14)
          
        
    fig.tight_layout()
    
    if save:
        if not save_path: print("argument save_path is not defined. The figure will not be saved.")
        else:
            if not file_name: file_name = f"reco{toy}"
            else: file_name += f"_reco{toy}"
            fig.savefig(save_path+file_name+".png", dpi = 300, facecolor="white")
    plt.show()