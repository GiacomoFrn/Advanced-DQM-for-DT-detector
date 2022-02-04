import os
import h5py
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FormatStrFormatter


def change_legend(ax, new_loc, fontsize, titlesize, **kws):
    '''funzione per modificare posizione e font size della legenda generata da seaborn'''

    old_legend = ax.legend_
    handles = old_legend.legendHandles
    labels = [t.get_text() for t in old_legend.get_texts()]
    title = old_legend.get_title().get_text()

    ax.legend(handles[::-1], labels[::-1], loc=new_loc, title=title, 
              fontsize=fontsize, title_fontsize=titlesize, 
              frameon = True, fancybox = False, framealpha = 0.5, **kws)

    return
    
    
def t_median(t_list, dof):
    '''calcolo la mediana per un rapido controllo di compatibilità'''
    
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
    
    return


def plot_1distribution(t, df, xmin=None, xmax=None, nbins=10, wclip=0, save=False, save_path=None, file_name=None):
    '''
    Plot the histogram of a test statistics sample (t) and the target chi2 distribution. 
    The median and the error on the median are calculated in order to calculate the median Z-score and its error.
    
    t:  (numpy array shape (None,))
    df: (int) chi2 degrees of freedom
    '''
    fig, ax = plt.subplots(figsize=(10, 6))
    
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
        stat='count', 
        element='bars', 
        linewidth=2,
        fill=True, 
        color='lightblue', 
        #color='#aadeff', 
        edgecolor='#2c7fb8',
        #edgecolor='#009cff', 
        ax=ax, 
        label=label
    )

    ax.errorbar(bincenters, hist, yerr=err, color='#2c7fb8', linewidth=2, marker='o', ls='')
   
    # plot reference chi2
    x = np.linspace(scipy.stats.chi2.ppf(0.0001, df), scipy.stats.chi2.ppf(0.9999, df), 100)
    ax.plot(
        x, 
        scipy.stats.chi2.pdf(x, df),
        'midnightblue', 
        lw=5, 
        alpha=0.8, 
        label=r'Target $\chi^2$(ndf='+str(df)+')',
        zorder=10
    )
    
    
    ax.legend()
    change_legend(ax=ax, new_loc="upper right", fontsize=14, titlesize=0)
    
    ax.set_title(f'Test statistic distribution', fontsize = 22)
    ax.set_xlabel('t', fontsize = 18)
    ax.set_ylabel(r'p(t | $\mathcal{R}$)', fontsize = 18)
    ax.set_xlim(XMIN, XMAX)
    
    ax.tick_params(axis = 'both', which = 'major', labelsize = 14, direction = 'out', length = 5)
    
    if save:
        if not save_path: print('argument save_path is not defined. The figure will not be saved.')
        else:
            if not file_name: file_name = '1distribution'
            else: file_name += '_1distribution'
            fig.savefig(save_path+file_name+'.png', dpi = 300, facecolor='white')
    plt.show()
    return


    
def plot_percentiles(tvalues_check, df, patience=1000, wclip=None, ymax=None, ymin=None, save=False, save_path=None, file_name=None, smooth=None):
    '''
    The funcion creates the plot of the evolution in the epochs of the [2.5%, 25%, 50%, 75%, 97.5%] quantiles of the toy sample distribution.
    The percentile lines for the target chi2 distribution are shown as a reference.
    
    patience:      (int) interval between two check points (epochs).
    tvalues_check: (numpy array shape (N_toys, N_check_points)) array of t=-2*loss
    df:            (int) chi2 degrees of freedom
    '''
    
    thrs = 3
    
    tvalues_check = tvalues_check.copy()
    if smooth:
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
    
    N_CHECKS = tvalues_check.shape[1]
    EPOCH_CHECK = [patience*(i+1) for i in range(N_CHECKS)]
    XMIN = 0
    XMAX = N_CHECKS*patience*1.2
    YMIN = 0
    if ymin:
        YMIN = ymin
    if tvalues_check[:,-1].max() >= 3*df:
        YMAX = tvalues_check[:,-1].max() + YMIN
    else:
        YMAX = 3*df
    if ymax:
        YMAX = ymax
            
    color_list = ['seagreen', 'mediumseagreen', 'lightseagreen', '#2c7fb8', 'midnightblue']
    # color_list = ['#00b32a', '#00c282', '#00D2FF', '#009cff', '#005e99']
    quantile_list   = [0.05,0.25,0.50,0.75,0.95]
    quantile_labels = ["5%", "25%", "50%", "75%", "95%"]
    
    th_quantile_position = [scipy.stats.chi2.ppf(i, df=df) for i in quantile_list]
    t_quantile = np.quantile(tvalues_check, quantile_list, axis=0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)

    for i in range(len(quantile_list)):
        ax.plot(
            EPOCH_CHECK, 
            t_quantile[i][:], 
            color = color_list[i], 
            linestyle='solid', 
            linewidth = 3, 
            label = format(quantile_list[i], '1.2f')
        )
        ax.hlines(
            y=th_quantile_position[i], 
            xmin = XMIN, 
            xmax = N_CHECKS*patience, 
            color = color_list[i], 
            linestyle='dashed', 
            linewidth = 3, 
            alpha = 0.5, 
            label = 'theoretical ' + format(quantile_list[i], '1.2f')
        )
        ax.text(
            N_CHECKS*patience*1.05, 
            th_quantile_position[i], 
            quantile_labels[i], 
            horizontalalignment='left', 
            verticalalignment='center', 
            color=color_list[i],
            fontsize=22,
            transform=ax.transData
        )
        
    # ax.legend(ncol=2, loc="upper right", fontsize=14)
    
    ax.set_title(f'Percentiles evolution', fontsize = 22)
    ax.set_xlabel('training epochs', fontsize = 18)
    ax.set_ylabel(r't', fontsize = 18)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 14, direction = 'out', length = 5)
    plt.setp(ax.get_xticklabels()[-1], visible=False)
    
    if save:
        if not save_path: print('argument save_path is not defined. The figure will not be saved.')
        else:
            if not file_name: file_name = 'percentiles'
            else: file_name += '_percentiles'
            fig.savefig(save_path+file_name+'.png', dpi = 300, facecolor='white')
        
    plt.show()
    return
  

    
def plot_median(tvalues_check, df, patience=1000, wclip=None, ymax=None, ymin=None, save=False, save_path=None, file_name=None, smooth=None):
        
    thrs = 3

    tvalues_check = tvalues_check.copy()
    if smooth:
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
        YMAX = 2*df
    if ymax:
        YMAX = ymax
    
    
    median_history = np.median(tvalues_check, axis=0)
    th_median = scipy.stats.chi2.median(df=df)
    
    median_pval = scipy.stats.chi2.sf(median_history, df=df)
    final_pval = median_pval[-1]
    median_Z = scipy.stats.norm.ppf(1-median_pval)
    final_Z = median_Z[-1]
    
    th_std = scipy.stats.chi2.std(df=df)
    d      = median_history[-1] - th_median
    d_std  = d / th_std
    
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)
    
    label  = f"final median = {median_history[-1]:.3f}\n"
    label += f"p-value = {median_pval:.3f}\n"
    label += f"Z ="
    
    ax.plot(
        EPOCH_CHECK,
        median_history,
        color="#2c7fb8",
        linestyle='solid', 
        linewidth = 3, 
        # label = ,
    )
    
    ax.hlines(
        y=th_median, 
        xmin = XMIN, 
        xmax = XMAX, 
        color = 'midnightblue', 
        linestyle='dashed', 
        linewidth = 3, 
        alpha = 0.5, 
        label = f'theoretical median: {th_median:.3f}'
    )
    
    ax.legend(loc="upper right", fontsize=14)

    ax.set_title(f'Median evolution', fontsize = 22)
    ax.set_xlabel('training epochs', fontsize = 18)
    ax.set_ylabel(r'$\tilde{t}$', fontsize = 18)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 14, direction = 'out', length = 5)
    
    
    if save:
        if not save_path: print('argument save_path is not defined. The figure will not be saved.')
        else:
            if not file_name: file_name = 'percentiles'
            else: file_name += '_percentiles'
            fig.savefig(save_path+file_name+'.png', dpi = 300, facecolor='white')
        
    plt.show()
    return
        
        
        
    
#     def plotMedianHistory(self):
#         '''andamento della mediana'''
        
#         self.median_history = np.median(self.t_list_history, axis=0)
        
#         th_median = scipy.stats.chi2.median(df=self.dof)
        
#         XMIN = 0
#         XMAX = self.epochs
        
#         YMIN = 0
#         if max(self.median_history) >= 3*self.dof:
#             YMAX = max(self.median_history) + min(self.median_history) 
#         elif max(self.median_history) < 3*self.dof:
#             YMAX = 3*self.dof
            
#         XLIM = [XMIN, XMAX]
#         YLIM = [YMIN, YMAX]
        
#         fig, ax = plt.subplots(figsize=(12,7))
        
#         x_tics = np.array(range(self.epochs))
#         x_tics = x_tics[x_tics % self.check_point_t == 0]
        
        
#         ax.plot(x_tics[:],self.median_history[:], color='#009cff', linestyle='solid', linewidth=3, alpha=1, 
#                 label=f'median final value: {self.median_history[-1]:.3f}')
        
#         ax.hlines(y=th_median, xmin = XMIN, xmax = XMAX, 
#                       color = '#FF0000', linestyle='dashed', linewidth = 3, alpha = 0.5, 
#                     label = f'theoretical median: {th_median:.3f}')
        
#         self.plotterLayout(ax=ax, xlimits=XLIM, ylimits=YLIM, title='median history', titlefont=18, xlabel='training epoch', ylabel='median', labelfont=16)
        
#         ax.legend()
#         self.change_legend(ax=ax, new_loc="upper right", fontsize=14, titlesize=16)
    
#         fig.tight_layout()
#         if self.save_flag:
#             fig.savefig(self.plotOutPath()+'_median_history.png', dpi = 300, facecolor='white')
#         plt.show()
#         return
    
    
#     def plotMedianPval(self):
#         '''andamento del pvalue della mediana'''
        
#         self.median_pval = scipy.stats.chi2.sf(self.median_history[:], df=self.dof)
        
#         XMIN = 0
#         XMAX = self.epochs
#         YMIN = 0
#         YMAX = 1.2
        
#         XLIM = [XMIN, XMAX]
# #         YLIM = [YMIN, YMAX]

#         fig, ax = plt.subplots(figsize=(12,7))

#         x_tics = np.array(range(self.epochs))
#         x_tics = x_tics[x_tics % self.check_point_t == 0]
# #         y_tics = np.array( np.arange(0, 1.1, 0.1) )
        
#         ax.plot(x_tics[10:], self.median_pval[10:], color='#009cff', linestyle='solid', linewidth=3, alpha=1, 
#                 label=f'median p-val final value: {self.median_pval[-1]:.3f}')
        
#         self.plotterLayout(ax=ax, xlimits=XLIM, title='median p-value evolution', titlefont=18, xlabel='training epoch', ylabel='p-value', labelfont=16)
# #         ax.set_yticks(y_tics)
        
#         ax.legend()
#         self.change_legend(ax=ax, new_loc="upper right", fontsize=14, titlesize=16)
        
#         fig.tight_layout()
#         if self.save_flag:
#             fig.savefig(self.plotOutPath()+'_median_pvalue.png', dpi = 300, facecolor='white')
#         plt.show()
#         return
    
    
#     def plotMedianZ(self):
#         '''andamento della significanza della mediana'''
        
#         self.median_Z = np.abs(scipy.stats.norm.ppf(1-self.median_pval[:]))
        
#         XMIN = 0
#         XMAX = self.epochs
#         YMIN = 0
#         YMAX = 1.2
        
#         XLIM = [XMIN, XMAX]
# #         YLIM = [YMIN, YMAX]

#         fig, ax = plt.subplots(figsize=(12,7))

#         x_tics = np.array(range(self.epochs))
#         x_tics = x_tics[x_tics % self.check_point_t == 0]
# #         y_tics = np.array( np.arange(0, 1.1, 0.1) )
        
#         ax.plot(x_tics[10:], self.median_Z[10:], color='#009cff', linestyle='solid', linewidth=3, alpha=1, 
#                 label=f'median Z final value: {self.median_Z[-1]:.3f}')
        
#         self.plotterLayout(ax=ax, xlimits=XLIM, title='median significance evolution', titlefont=18, xlabel='training epoch', ylabel='Z', labelfont=16)
# #         ax.set_yticks(y_tics)
        
#         ax.legend()
#         self.change_legend(ax=ax, new_loc="upper right", fontsize=14, titlesize=16)
        
#         fig.tight_layout()
#         if self.save_flag:
#             fig.savefig(self.plotOutPath()+'_median_significance.png', dpi = 300, facecolor='white')
#         plt.show()
#         return
    
    
#     def ks_test_evo(self):
        
#         self.D_history = []
#         self.D_pval_history = []
#         for i in range(self.t_list_history.shape[1]):
#             self.D, self.Dpval = scipy.stats.kstest(
#                 rvs=self.t_list_history[:,i],
#                 cdf="chi2",
#                 args=(10, 0, 1)
#             )
#             self.D_history.append(self.D)
#             self.D_pval_history.append(self.Dpval)
        
#         self.D_history = np.array(self.D_history)
#         self.D_pval_history = np.array(self.D_pval_history)
        
#         XMIN = 0
#         XMAX = self.epochs
#         YMIN = 0
#         YMAX = 1.2
        
#         XLIM = [XMIN, XMAX]
        
#         fig, ax = plt.subplots(figsize=(12,7))

#         x_tics = np.array(range(self.epochs))
#         x_tics = x_tics[x_tics % self.check_point_t == 0]
        
#         ax.plot(x_tics[10:], self.D_history[10:], color='#009cff', linestyle='solid', linewidth=3, alpha=1, 
#                 label=f'D statistic final value: {self.D:.3f}')
        
#         self.plotterLayout(ax=ax, xlimits=XLIM, title='D statistic evolution', titlefont=18, xlabel='training epoch', ylabel='D', labelfont=16)
        
#         ax.legend()
#         self.change_legend(ax=ax, new_loc="upper right", fontsize=14, titlesize=16)
        
#         fig.tight_layout()
#         if self.save_flag:
#             fig.savefig(self.plotOutPath()+'_D_evolution.png', dpi = 300, facecolor='white')
#         plt.show()
        
#         fig, ax = plt.subplots(figsize=(12,7))

       
        
#         ax.plot(x_tics[10:], self.D_pval_history[10:], color='#009cff', linestyle='solid', linewidth=3, alpha=1, 
#                 label=f'KS final pvalue: {self.Dpval:.3f}')
        
#         self.plotterLayout(ax=ax, xlimits=XLIM, title='KS p-value evolution', titlefont=18, xlabel='training epoch', ylabel='KS pval', labelfont=16)
        
#         ax.legend()
#         self.change_legend(ax=ax, new_loc="upper right", fontsize=14, titlesize=16)
        
#         fig.tight_layout()
#         if self.save_flag:
#             fig.savefig(self.plotOutPath()+'_KSp_evolution.png', dpi = 300, facecolor='white')
#         plt.show()
        
#         return
