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
    
    # legend text
    label  = f"W_clip = {wclip}\nSample size = {t.shape[0]}\nMedian = {np.median(t):.2f}\nStd = {np.std(t):.2f}" 
    label += f"\nZ = {Z_obs:.2f} (+{Z_obs_p-Z_obs:.2f}/-{Z_obs-Z_obs_m:.2f})"
    

    sns.histplot(
        x=bin_edges[:-1], weights=hist, bins=bin_edges,
        stat='count', element='bars', linewidth=2,
        fill=True, color='#aadeff', edgecolor='#009cff', 
        ax=ax, label=label
    )

    ax.errorbar(bincenters, hist, yerr=err, color='#009cff', linewidth=2, marker='o', ls='')
   
    # plot reference chi2
    x = np.linspace(scipy.stats.chi2.ppf(0.0001, df), scipy.stats.chi2.ppf(0.9999, df), 100)
    ax.plot(x, scipy.stats.chi2.pdf(x, df),'midnightblue', lw=5, alpha=0.8, label=r'Target $\chi^2$(ndf='+str(df)+')')
    
    
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
            fig.savefig(save_path+file_name+'.pdf', dpi = 300, facecolor='white')
    plt.show()
    return
    
  
    
    
    
    
# ###########################################################################    
#     def thesisPlot(self):
#         fig, ax = plt.subplots(nrows=2, figsize=(14,16))
        
#          # gestione del plot range
#         XMIN = 0
#         if max(self.t_list) >= 3*self.dof:
#             XMAX = max(self.t_list) + min(self.t_list) 
#         elif max(self.t_list) < 3*self.dof:
#             XMAX = 3*self.dof
            
#         # creo la griglia lungo x
#         XGRID = np.linspace(XMIN, XMAX, 500)
        
#         # numero di bin da utilizzare
#         BINS = self.bins
        
        
#         hist, bin_edges = np.histogram(self.t_list, density=True, bins=BINS)

#         binswidth = bin_edges[1]-bin_edges[0]
#         central_points = [
#             (bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)
#         ]
#         err = np.sqrt(hist/(self.t_list.shape[0]*binswidth))

#         ax[0].plot(
#             XGRID, scipy.stats.chi2.pdf(XGRID, df=self.dof), 
#             color='#005e99', linestyle='solid', linewidth=7, alpha=0.6, 
#             label=r'Target $\chi^{2}_{10}$'
#         )

#         sns.histplot(
#             x=bin_edges[:-1], weights=hist, bins=bin_edges,
#             stat='density', element='bars', linewidth=2,
#             fill=True, color='#aadeff', edgecolor='#009cff', 
#             ax=ax[0]
#         )

#         ax[0].errorbar(central_points, hist, yerr=err, color='#009cff', linewidth=2, marker='o', ls='')

#         ax[0].set_title(f'Test statistic distribution', fontsize = 32)
#         ax[0].set_xlabel('t', fontsize = 56)
#         ax[0].set_ylabel(r'p(t | $\mathcal{R}$)', fontsize = 56)

#         ax[0].set_xlim(XMIN, XMAX)
#         ax[0].set_ylim(0, 0.11)

#         ax[0].tick_params(axis = 'both', which = 'major', labelsize = 48, direction = 'out', length = 5)
# #         ax[0].yaxis.get_offset_text().set_fontsize(48)
# #         ax[0].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
# #         ax[0].ticklabel_format(axis = 'y', style = 'sci', scilimits = (0,0))

#         ax[0].legend()
#         self.change_legend(ax=ax[0], new_loc="upper right", fontsize=44, titlesize=0)
        
#         XMIN = 0
#         XMAX = self.epochs
        
#         YMIN = 0
#         YMAX = 24
            
            
#         XLIM = [XMIN, XMAX]
#         YLIM = [YMIN, YMAX]
        
#         # ticks
#         x_tics = np.arange(0, self.epochs, self.check_point_t)
# #         print(x_tics)
        
#         # quantili
#         quantile_list = [0.05,0.25,0.50,0.75,0.95]
#         quantile_labels = ["5%", "25%", "50%", "75%", "95%"]
#         color_list = ['#00b32a', '#00c282', '#00D2FF', '#009cff', '#005e99']
# #         color_list = list(reversed(color_list))
        
#         # quantili teorici
#         th_quantile_position = [scipy.stats.chi2.ppf(i, df=self.dof) for i in quantile_list]
        
#         # quantili distribuzione
#         t_quantile = np.quantile(self.t_list_history[:], quantile_list, axis=0)
        
#         # ciclo per plottare i 5 quantili 
#         for i in range(len(quantile_list)):
#             ax[1].plot(x_tics[:], t_quantile[i][:], 
#                     color = color_list[i], linestyle='solid', linewidth = 3, 
#                     label = format(quantile_list[i], '1.2f'))
# #             ax.plot(x_tics[-1], th_quantile_position[i], marker='X', markersize = 15,
# #                     color = color_list[i])
#             ax[1].hlines(y=th_quantile_position[i], xmin = XMIN, xmax = XMAX, 
#                       color = color_list[i], linestyle='dashed', linewidth = 3, alpha = 0.5, 
#                     label = 'theoretical ' + format(quantile_list[i], '1.2f'))
#             ax[1].text(
#                 210000, th_quantile_position[i], 
#                 quantile_labels[i], 
#                 horizontalalignment='left', verticalalignment='center', 
#                 color=color_list[i],
#                 fontsize=44,
#                 transform=ax[1].transData)
        
#         # plot layout method
#         self.plotterLayout(ax=ax[1], xlimits=XLIM, ylimits=YLIM, title='', titlefont=32, xlabel='training epochs', ylabel='t', labelfont=56)
#         start, end = ax[1].get_xlim()
#         ax[1].xaxis.set_ticks(np.arange(start, 300000, 50000))
#         plt.setp(ax[1].get_xticklabels()[-1], visible=False)
#         plt.setp(ax[1].get_xticklabels()[-2], visible=False)
#         plt.setp(ax[1].get_xticklabels()[-4], visible=False)
#         plt.setp(ax[1].get_xticklabels()[-3], visible=False)
#         ax[1].tick_params(axis = 'both', which = 'major', labelsize = 48, direction = 'out', length = 5)
        
# #         ax[1].set_title("Quantiles evolution", fontsize=32)
#         ax[1].set_xlabel('training epochs', fontsize = 56)
#         ax[1].set_ylabel('t', fontsize = 56)
#         fig.tight_layout()
#         if self.save_flag:
#             fig.savefig(f"/lustre/cmswork/nlai/PLOTS/DRIFT_TIME/thesis/both_{self.wclip}.pdf", dpi = 300, facecolor='white')
#         plt.show()
#         return
# ###########################################################################    
    
#     def plotTdist(self):
#         '''grafico della distribuzione dei t'''
        
#         fig, ax = plt.subplots(figsize=(14,8))
        
#          # gestione del plot range
#         XMIN = 0
#         if max(self.t_list) >= 3*self.dof:
#             XMAX = max(self.t_list) + min(self.t_list) 
#         elif max(self.t_list) < 3*self.dof:
#             XMAX = 3*self.dof
            
#         # creo la griglia lungo x
#         XGRID = np.linspace(XMIN, XMAX, 500)
        
#         # numero di bin da utilizzare
#         BINS = self.bins
        
        
#         hist, bin_edges = np.histogram(self.t_list, density=True, bins=BINS)

#         binswidth = bin_edges[1]-bin_edges[0]
#         central_points = [
#             (bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)
#         ]
#         err = np.sqrt(hist/(self.t_list.shape[0]*binswidth))

#         ax.plot(
#             XGRID, scipy.stats.chi2.pdf(XGRID, df=self.dof), 
#             color='#005e99', linestyle='solid', linewidth=9, alpha=0.6, 
#             label=r'Target $\chi^{2}_{10}$'
#         )

#         sns.histplot(
#             x=bin_edges[:-1], weights=hist, bins=bin_edges,
#             stat='density', element='bars', linewidth=4,
#             fill=True, color='#aadeff', edgecolor='#009cff', 
#             ax=ax
#         )

#         ax.errorbar(central_points, hist, yerr=err, color='#009cff', linewidth=4, marker='o', ls='')
        
#         ax.text(0.6, 0.6, r"$W_{clip} = $" + f"{round(float(self.wclip))}", fontsize=38, transform=ax.transAxes)

#         ax.set_title(f'Test statistic distribution', fontsize = 46)
#         ax.set_xlabel('t', fontsize = 46)
#         ax.set_ylabel(r'p(t | $\mathcal{R}$)', fontsize = 46)

#         ax.set_xlim(XMIN, XMAX)
# #         ax.set_ylim(0, 0.11)
#         plt.setp(ax.get_yticklabels()[0], visible=False)
#         plt.setp(ax.get_yticklabels()[1], visible=False)
#         plt.setp(ax.get_yticklabels()[2], visible=False)

        
#         ax.tick_params(axis = 'both', which = 'major', labelsize = 38, direction = 'out', length = 5)
#         ax.yaxis.get_offset_text().set_fontsize(24)
#         ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
#         ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0,0))
# #         ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))


#         ax.legend()
#         fig.tight_layout()
#         self.change_legend(ax=ax, new_loc="upper right", fontsize=38, titlesize=0)
#         if self.save_flag:
#             fig.savefig(f"/lustre/cmswork/nlai/thesis/PLOTS/DRIFT_TIME/thesis/a_distribution_{self.wclip}.pdf", dpi = 300, facecolor='white')
        
#         plt.show()
        

#         return 


    
#     def plotQuantilesEvo(self):
#         '''grafico dell'evoluzione dei quantili della distribuzione dei t'''
        
#         # gestione del plot range
#         XMIN = 0
#         XMAX = self.epochs
        
#         YMIN = 0
#         YMAX = 23
            
            
#         XLIM = [XMIN, XMAX]
#         YLIM = [YMIN, YMAX]
        
#         # creo figura e axes
#         fig, ax = plt.subplots(figsize=(14,8))
        
#         # ticks
#         x_tics = np.arange(0, self.epochs, self.check_point_t)
# #         print(x_tics)
        
#         # quantili
#         quantile_list = [0.05,0.25,0.50,0.75,0.95]
#         quantile_labels = ["5%", "25%", "50%", "75%", "95%"]
#         color_list = ['#00b32a', '#00c282', '#00D2FF', '#009cff', '#005e99']
# #         color_list = list(reversed(color_list))
        
#         # quantili teorici
#         th_quantile_position = [scipy.stats.chi2.ppf(i, df=self.dof) for i in quantile_list]
        
#         # quantili distribuzione
#         t_quantile = np.quantile(self.t_list_history[:], quantile_list, axis=0)
        
#         # ciclo per plottare i 5 quantili 
#         for i in range(len(quantile_list)):
#             ax.plot(x_tics[:], t_quantile[i][:], 
#                     color = color_list[i], linestyle='solid', linewidth = 6, 
#                     label = format(quantile_list[i], '1.2f'))
# #             ax.plot(x_tics[-1], th_quantile_position[i], marker='X', markersize = 15,
# #                     color = color_list[i])
#             ax.hlines(y=th_quantile_position[i], xmin = XMIN, xmax = XMAX, 
#                       color = color_list[i], linestyle='dashed', linewidth = 4, alpha = 0.6, 
#                     label = 'theoretical ' + format(quantile_list[i], '1.2f'))
#             ax.text(
#                 210000, th_quantile_position[i], 
#                 quantile_labels[i], 
#                 horizontalalignment='left', verticalalignment='center', 
#                 color=color_list[i],
#                 fontsize=32,
#                 transform=ax.transData)
        
#         # plot layout method
#         self.plotterLayout(ax=ax, xlimits=XLIM, ylimits=YLIM, title="", titlefont=34, xlabel='training epochs', ylabel='t', labelfont=32)
#         ax.tick_params(axis = 'both', which = 'major', labelsize = 32, direction = 'out', length = 5)
#         start, end = ax.get_xlim()
#         ax.xaxis.set_ticks(np.arange(start, 300000, 50000))
#         plt.setp(ax.get_xticklabels()[-1], visible=False)
#         plt.setp(ax.get_yticklabels()[0], visible=False)
# #         plt.setp(ax.get_xticklabels()[-2], visible=False)
# #         plt.setp(ax.get_xticklabels()[-4], visible=False)
# #         plt.setp(ax.get_xticklabels()[-3], visible=False)
        
#         ax.set_title("Quantiles evolution", fontsize=34)
#         ax.set_xlabel('training epochs', fontsize = 32)
#         ax.set_ylabel('t', fontsize = 32)

#         fig.tight_layout()
#         if self.save_flag:
#             fig.savefig(f"/lustre/cmswork/nlai/thesis/PLOTS/DRIFT_TIME/thesis/b_quantiles_{self.wclip}.pdf", dpi = 300, facecolor='white')
#         plt.show()
#         return
    
    
    
    
    
    
    
    
    
    
    
    
    
#     def plotThistory(self):
#         '''grafico della storia dei t'''
        
        
#         # gestione del plot range
#         XMIN = 0
#         XMAX = self.epochs
        
#         YMIN = 0
#         if max(self.t_list) >= 3*self.dof:
#             YMAX = max(self.t_list) + min(self.t_list) 
#         elif max(self.t_list) < 3*self.dof:
#             YMAX = 3*self.dof
            
#         XLIM = [XMIN, XMAX]
#         YLIM = [YMIN, YMAX]
        
#         # creo figura e axes
#         fig, ax = plt.subplots(figsize=(12,7))
        
#         # ticks dell'asse x
#         x_tics = np.array(range(self.epochs))
#         x_tics = x_tics[x_tics % self.check_point_t == 0]
        
#         # ciclo per plottare la storia di tutti i toys
#         for i in range(len(self.t_list_history)):
#             ax.plot(x_tics[1:], self.t_list_history[i][1:])
        
#         # plotter layout method
#         self.plotterLayout(ax=ax, xlimits=XLIM, ylimits=YLIM, title='t history', titlefont=18, xlabel='training epoch', ylabel='t', labelfont=16)
        
#         fig.tight_layout()
#         if self.save_flag:
#             fig.savefig(self.plotOutPath()+'_history.png', dpi = 300, facecolor='white')
#         plt.show()
#         return
    
#     def plotChi2History(self):
#         '''grafico andamento del chi2 per epoche'''
        
#         self.t_chi2_history = []
#         bin_number = self.bins 
        
#         for counter in range(10, self.t_list_history.shape[1]): 
#             # binning della distribuzione dei t per ogni checkpoint
#             t_hist, binedges = np.histogram(self.t_list_history[:, counter], bins=bin_number, density=False) 
#             # cerco il centro di ciascun bin
#             bincenters = (binedges[:-1] + binedges[1:]) / 2
#             # calcolo la larghezza dei bin
#             bin_width = binedges[1]-binedges[0]
#             # calcolo l'area dell'istogramma
#             area_hist = bin_width*self.toys
#             # calcolo l'altezza teorica di ciascun bin
#             th_bins=[]
#             for i in range(len(bincenters)):
#                 area = scipy.integrate.quad(lambda x: (area_hist*scipy.stats.chi2.pdf(x, df=self.dof)), binedges[i], binedges[i+1])[0]
#                 th_bins.append(area/(bin_width))
#             # calcolo il chi2
#             self.t_chi2_history.append( np.sum( (t_hist-th_bins)**2/th_bins, axis=0 ) ) 
            
#         XMIN = 0
#         XMAX = self.epochs
        
#         XLIM = [XMIN, XMAX]

#         fig, ax = plt.subplots(figsize=(12,7))

#         x_tics = np.array(range(self.epochs))
#         x_tics = x_tics[x_tics % self.check_point_t == 0][10:]

        
#         ax.plot(x_tics[:], self.t_chi2_history[:], color='#009cff', linestyle='solid', linewidth=3, alpha=1, 
#                 label=fr'$\chi^2$ final value: {self.t_chi2_history[-1]:.3f}')
        
#         self.plotterLayout(ax=ax, xlimits=XLIM, title=r'$\chi^2$ evolution', titlefont=18, xlabel='training epoch', ylabel=r'$\chi^2$', labelfont=16)
#         ax.set_yscale('log')
        
#         ax.legend()
#         self.change_legend(ax=ax, new_loc="upper right", fontsize=14, titlesize=16)
        
# #         fig.tight_layout()
#         if self.save_flag:
#             fig.savefig(self.plotOutPath()+'_chi2.png', dpi = 300, facecolor='white')
#         plt.show()

#         return
    
    
#     def plotChi2Compatibility(self):
#         '''andamento compatibilità del chi2 con i gradi di libertà per epoche'''
        
#         # calcolo la compatibilità come chi_nu/nu
#         self.t_chi2_compatibility = np.array(self.t_chi2_history[:])/(self.bins-1)
    
#         XMIN = 0
#         XMAX = self.epochs
        
#         XLIM = [XMIN, XMAX]

#         fig, ax = plt.subplots(figsize=(12,7))

#         x_tics = np.array(range(self.epochs))
#         x_tics = x_tics[x_tics % self.check_point_t == 0][10:]
        
#         ax.plot(x_tics[:], self.t_chi2_compatibility[:], color='#009cff', linestyle='solid', linewidth=3, alpha=1, 
#                 label=fr'$\chi^2 / \nu$ final value: {self.t_chi2_compatibility[-1]:.3f}'
#                )
        
#         self.plotterLayout(ax=ax, xlimits=XLIM, title='compatibility evolution', titlefont=18, xlabel='training epoch', ylabel=r'$\chi^2 / \nu$', labelfont=16)
#         ax.set_yscale('log')
        
#         ax.legend()
#         self.change_legend(ax=ax, new_loc="upper right", fontsize=14, titlesize=16)
        
#         fig.tight_layout()
#         if self.save_flag:
#             fig.savefig(self.plotOutPath()+'_compatibility.png', dpi = 300, facecolor='white')
#         plt.show()
        
#         return
    
#     def plotChi2Compatibility2(self):
#         '''andamento compatibilità del chi2 con i gradi di libertà per epoche -- metodo 2'''
        
#         # calcolo la compatibilità come chi_nu/nu
#         self.t_chi2_compatibility =  np.abs( np.array(self.t_chi2_history[:]) - (self.bins-1) )  / np.sqrt(2*(self.bins-1))
    
#         XMIN = 0
#         XMAX = self.epochs
        
#         XLIM = [XMIN, XMAX]

#         fig, ax = plt.subplots(figsize=(12,7))

#         x_tics = np.array(range(self.epochs))
#         x_tics = x_tics[x_tics % self.check_point_t == 0][10:]
        
#         ax.plot(x_tics[:], self.t_chi2_compatibility[:], color='#009cff', linestyle='solid', linewidth=3, alpha=1, 
#                 label=fr'$\lambda$ final value: {self.t_chi2_compatibility[-1]:.3f}'
#                )
        
#         self.plotterLayout(ax=ax, xlimits=XLIM, title='compatibility evolution - version 2', titlefont=18, xlabel='training epoch', ylabel=r'$\lambda$', labelfont=16)
#         ax.set_yscale('log')
        
#         ax.legend()
#         self.change_legend(ax=ax, new_loc="upper right", fontsize=14, titlesize=16)
        
#         fig.tight_layout()
#         if self.save_flag:
#             fig.savefig(self.plotOutPath()+'_compatibility2.png', dpi = 300, facecolor='white')
#         plt.show()
        
#         return
    
    
#     def plotPValHistory(self):
#         '''andamento del pvalue del chi2 per epoche'''
        
#         self.pvalue_history = scipy.stats.chi2.sf(self.t_chi2_history[:], df=self.bins-1) 

#         XMIN = 0
#         XMAX = self.epochs
#         YMIN = 0
#         YMAX = 1.2
        
#         XLIM = [XMIN, XMAX]
# #         YLIM = [YMIN, YMAX]

#         fig, ax = plt.subplots(figsize=(12,7))

#         x_tics = np.array(range(self.epochs))
#         x_tics = x_tics[x_tics % self.check_point_t == 0][10:]
# #         y_tics = np.array( np.arange(0, 1.1, 0.1) )
        
#         ax.plot(x_tics[:], self.pvalue_history[:], color='#009cff', linestyle='solid', linewidth=3, alpha=1, 
#                 label=f'p-val final value: {self.pvalue_history[-1]:.3f}')
        
#         self.plotterLayout(ax=ax, xlimits=XLIM, title='p-value evolution', titlefont=18, xlabel='training epoch', ylabel='p-value', labelfont=16)
# #         ax.set_yticks(y_tics)
        
#         ax.legend()
#         self.change_legend(ax=ax, new_loc="upper right", fontsize=14, titlesize=16)
        
#         fig.tight_layout()
#         if self.save_flag:
#             fig.savefig(self.plotOutPath()+'_pvalue.png', dpi = 300, facecolor='white')
#         plt.show()
        
#         return
    
    
#     def plotSignificanceHistory(self):
#         '''andamento della significanza del chi2 per epoche'''
        
#         self.significance_history=np.abs(scipy.stats.norm.ppf(1-self.pvalue_history[:]))

#         XMIN = 0
#         XMAX = self.epochs
# #         YMIN = 0
# #         YMAX = max(self.significance_history)+min(self.significance_history)

#         XLIM = [XMIN, XMAX]

#         fig, ax = plt.subplots(figsize=(12,7))

#         x_tics = np.array(range(self.epochs))
#         x_tics = x_tics[x_tics % self.check_point_t == 0][10:]
# #         y_tics = np.array( np.arange(0, 1.1, 0.1) )
        
#         ax.plot(x_tics[:], self.significance_history[:], color='#009cff', linestyle='solid', linewidth=3, alpha=1, 
#                 label=f'Z final value: {self.significance_history[-1]:.3f}')
        
#         self.plotterLayout(ax=ax, xlimits=XLIM, title='significance evolution', titlefont=18, xlabel='training epoch', ylabel='Z', labelfont=16)
        
#         ax.legend()
#         self.change_legend(ax=ax, new_loc="upper right", fontsize=14, titlesize=16)
        
#         fig.tight_layout()
#         if self.save_flag:
#             fig.savefig(self.plotOutPath()+'_significance.png', dpi = 300, facecolor='white')
#         plt.show()
        
#         return
    
    
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
