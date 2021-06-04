#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
to plot validation performance between datasets
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
print ("imported modules")

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def box_plt(depth, val_hr, metric, ax):
    anom, err = metric[0], metric[1]
    print("")
    print("**** BOX plot_errs_amongData_boxs:", depth, val_hr, anom, err)
    fpath="./pltdata/errs"+anom+"_point_"+depth+"H06.dat"
    print(" >>> loading err data from:", fpath)

    sel_data =["ml","era5","gleam"]
    sel_label=["SoMo.ml*","ERA5","GLEAM","ESA-CCI"]

    if depth=="layer0": sel_data +=["esa"]

    load=pd.read_csv(fpath, header=0, index_col=None, parse_dates=True, na_values=-9999.)
    df_errs=load.loc[:,["arid","temp","region"]+[x+err for x in sel_data]].copy()

    print(" **plot_ranks:", depth, val_hr, anom, err)
    print(df_errs.describe())
    print("")

    print(" | col | failuers| mean| median| len|")
    err_median, err_vals=list(), list()
    err_mean = list()

    for col in sel_data:

       imsi=df_errs[[col+err]].dropna()

       _median=round(np.nanmedian(imsi[col+err]),3)
       _mean  =round(np.nanmean(imsi[col+err]),3)
       _fail  =len( imsi[col+err][imsi[col+err]<0])

       if (col=="esa")&(depth!="layer0"): #make empty error for esa in deeper layers
          if err in ["_rmse","_bias","_ub_rmse"]:  _mean, _median= 99,  99
          else:                                    _mean, _median=-99, -99

       print(col+err, _fail, _mean, _median, len(imsi))
       if col=="ml": vline = _median #vertial line

       err_mean.append(_mean)
       err_median.append(_median)
       err_vals.append(imsi.values)

    ##### ordered by err_median
    ranked    =[x for _,x in sorted(zip(err_median, err_vals))]
    new_lables=[x for _,x in sorted(zip(err_median, sel_label))]
    #####

    if err in ["_rmse"]:
       ranked=ranked[::-1]
       new_lables=new_lables[::-1]

    print("")
    print(" >> new yy_lables", new_lables)
    print("")

    # plot box by rank
    n=len(err_median)

    pltdata=[]
    print(".to check.")
    for i in range(n):
       #print(new_lables[i], len(ranked[i]), ranked[i].shape, type(ranked[i]))
       pltdata.append(ranked[i].flatten())
       print(new_lables[i], pltdata[-1].shape)
    print("")


    if depth in ["layer1","layer2"]:
        bplot=ax.boxplot([[np.nan]*10]+pltdata, patch_artist=True,\
                         positions=range(n+1), whis=[20,80], widths=[0.4]*(n+1), \
                         vert=False, showfliers=False, showmeans=True, \
                         meanprops={"marker":"^","markerfacecolor":"k", "markeredgecolor":"none"})

        # fill with colors
        k=0
        new_lables=["ESA-CCI"]+new_lables
        for patch in bplot['boxes']:
           if    k ==new_lables.index("ERA5"):       col='#ff9999'
           elif  k ==new_lables.index("GLEAM"):      col='#ff9999'
           elif  k ==new_lables.index("SoMo.ml*"):   col='b'
           elif  k ==new_lables.index("ESA-CCI"):
               k+=1
               continue #due to dum data (at esa-position)
           else: stop

           if k == new_lables.index("SoMo.ml*"): patch.set(facecolor=col, alpha=.7, linewidth=1)
           else:   patch.set(facecolor=col, linewidth=1)

           k+=1
        plt.yticks(range(n+1), new_lables, fontsize=8, rotation=315)

    else:
        bplot=ax.boxplot(pltdata, patch_artist=True,\
                         positions=range(n), whis=[20,80], widths=[0.4]*n, \
                         vert=False, showfliers=False, showmeans=True, \
                         meanprops={"marker":"^","markerfacecolor":"k", "markeredgecolor":"none"})

        # fill with colors
        k=0
        for patch in bplot['boxes']:

           if    k ==new_lables.index("ERA5"):       col='#ff9999'
           elif  k ==new_lables.index("GLEAM"):      col='#ff9999'
           elif  k ==new_lables.index("SoMo.ml*"):   col='b'
           elif  k ==new_lables.index("ESA-CCI"):    col='violet'
           else: stop

           if k ==new_lables.index("SoMo.ml*"): patch.set(facecolor=col, alpha=.7, linewidth=1)
           else:   patch.set(facecolor=col, linewidth=1)

           k+=1
        plt.yticks(range(n), new_lables, fontsize=8, rotation=315)

    for median in bplot['medians']:
        median.set(color='k', linewidth=2)

    ax.yaxis.tick_right()


    if err in ["_rmse"]:
       plt.xlim(0,2.5)
       xxticks=np.arange(0.5,2.6,.5)
       text_x=2.5*.2
    if err in ["_corr"]:
       plt.xlim(0,1)
       xxticks=np.arange(0.2,1.1,0.2)
       text_x=1*.2

    if depth!="layer0": #deeper layers no esa-cci
       plt.text(text_x, -.1, "not applicable", fontsize=7, color="grey")
    if depth=="layer2":
       plt.xticks(xxticks, [round(x,1) for x in xxticks], fontsize=8)
    else:
       plt.xticks(xxticks, ["" for x in xxticks], fontsize=8)

    print("xxticks:", xxticks)
    plt.axvline(vline, c="k", lw=0.7, linestyle=":")
    print("done.")


def custom_div_cmap(numcolors=11, name='custom_div_cmap',
                    mincol='blue', midcol='white', maxcol='red'):
    """ Create a custom diverging colormap with three colors

    Default is blue to white to red with 11 colors.  Colors can be specified
    in any way understandable by matplotlib.colors.ColorConverter.to_rgb()
    """

    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(name=name,
                                             colors =[mincol, midcol, maxcol],
                                             N=numcolors)
    return cmap


def add_cbar(ax, err=""):

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    if err=="_rmse":
        cbar= plt.colorbar(cs, cax=cax, extend='max')
        cbar.set_clim(0,2)
        cbar.set_ticks([0, 1., 2.])
        cbar.set_ticklabels([0, 1.0, 2.0])
    else:
        cbar= plt.colorbar(cs, cax=cax, extend='min')
        cbar.set_clim(0., 1.0)
        cbar.set_ticks([0,0.5,1.0])
        cbar.set_ticklabels([0,0.5,1.0])
    cbar.ax.tick_params(labelsize=8)


def clim_map(depth, val_hr, metric, ax):

    anom, err = metric[0], metric[1]
    print(" \n\n plot_errs_byClim:", depth, val_hr, anom, err)
    fpath="./pltdata/errs"+anom+"_point_"+depth+"H06.dat"
    print(" >>> loading err data from:", fpath)

    load=pd.read_csv(fpath, header=0, index_col=None, parse_dates=True, na_values=-9999.)

    print(" ****** ploting only ML:")
    imsi=load[["arid", "temp", "ml"+err]].dropna()

    arid  = list(imsi["arid"].values)
    temp  = list((imsi["temp"].values)-273.15)
    vals  = list(imsi["ml"+err].values)

    print("")
    print(" >>>", depth, imsi.shape, "len:", len(arid), len(temp), len(vals), round(np.median(vals),2))
    print("temp range:", min(temp), max(temp))
    print("arid range:", min(arid), max(arid))
    print("err  range:", min(vals), max(vals))
    print("")

    vmin, vmax= 0, 1

    #cmap = custom_div_cmap(11, mincol='CornflowerBlue', maxcol='g')
    if err=="_rmse":
        cmap = plt.get_cmap('viridis_r')
        cmap.set_over("grey")

    else:
       cmap = plt.get_cmap('viridis')
       cmap.set_under("darkgrey")

    cs=plt.scatter(arid, temp, marker="o", s=0.7, vmin=vmin, vmax=vmax, c=vals, cmap=cmap)
    ax.set_xscale('log')

    plt.ylim(-11,32)
    plt.xlim(0.1, 80)

    plt.yticks(np.arange(-10,31,10), np.arange(-10,31,10), fontsize=8)
    plt.xticks([0.01,0.1,1,10,100],["","","","",""], fontsize=8)
    return cs


##################################################################
"""
Fig.6 opt: err="_rmse", others=""
Fig.7 opt: err="_corr", anom="_anom31"
"""
##################################################################
layers = ["layer0","layer1","layer2"]
val_hr = 6
err    = "_rmse"
anom   = ""
##################################################################

###############################
fig=plt.figure(figsize=(5, 7))
gs   = gridspec.GridSpec(3, 1, hspace=0.3, wspace=0.25, right=0.87)
gs00 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], width_ratios=[1,1])
gs01 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], width_ratios=[1,1])
gs02 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2], width_ratios=[1,1])
###############################
titles=["Layer1","Layer2","Layer3"]
if err=="_rmse":  err_title, xlabel=" (NRMSE)", "NRMSE"
if err=="_corr":  err_title, xlabel=" (Anom35Corr)", "Corr. Coef."
###############################

i=0
for layer, _gs in zip(layers, [gs00,gs01,gs02]):
   print("")
   print("*****", i, layer)
   print(">>>", err)
   print("")

   ax  = plt.subplot(_gs[0])

   cs=clim_map(layer, val_hr, [anom,err], ax)

   plt.title(titles[i]+err_title, x=0.25, y=1.025, fontsize=9, fontweight='bold')
   plt.ylabel("Temperature ($^\circ$C)", fontsize=8)
   if i==2:
     plt.xlabel("Aridity (Rad$_{net}$/precip)", fontsize=8, labelpad=10)
     plt.xticks([0.01,0.1,1,10,100],[0.01,0.1,1,10,100], fontsize=8)

   add_cbar(ax, err)

   ax  = plt.subplot(_gs[1])
   box_plt(layer, val_hr, [anom,err], ax)
   if i==2: plt.xlabel(xlabel, fontsize=8, labelpad=10)

   i+=1
###############################
print (". . . saving")
plt.savefig("./fig06"+err+".png", dpi=100)
plt.close()
