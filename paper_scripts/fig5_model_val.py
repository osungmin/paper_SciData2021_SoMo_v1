#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
to plot validation performance between datasets
"""
import os
import pandas as pd
import numpy as np
from scipy import stats
#import sub_soilm as sub
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from mpl_toolkits.basemap import Basemap
print ("imported modules")


def load_idxs(depth, val_hr):
    print (">>> plotting scatters:", depth, val_hr)
    idxs=os.listdir("./pltdata/"+depth+"/")
    print (">>> ml data len:", len(idxs))
    return idxs


def load_ml_mitObs(depth, fpath, val_hr):
    #copy ml-val simulation outputs from comp/kfolds/runs_static
    ml=pd.read_csv("./pltdata/"+depth+"/"+fpath,\
                    header=0, index_col=0, parse_dates=True, na_values=-9999.)
    return ml["soilm"].values, ml["ml"].values

    """
    print("!!! to test somo!!!")

    somo=pd.read_csv("./somo_sims/"+depth+"/"+depth+"_"+fpath.split("_")[1]+"_"+fpath.split("_")[2]+"_daily.dat",\
                    header=0, index_col=0, parse_dates=True, na_values=-9999.)
    concat=pd.concat([ml,somo], axis=1, join='inner')
    if len(ml)!=len(concat): stop
    return ml["soilm"].values, concat[depth].values
    """


#for the spatial comparison
def prep_scatt(depth):
    idxs=load_idxs(depth, 6)

    xx_means, ml_means=[], []
    for idx in idxs:
        xx, ml= load_ml_mitObs(depth, idx, 6)
        xx_means.append(np.mean(xx))
        ml_means.append(np.mean(ml))
    print("done. spatial mean ", len(idxs), "=", len(xx_means), len(ml_means))
    return xx_means, ml_means


def plt_scatt(depth, ax):
    xx, ml= prep_scatt(depth)
    print("   xx min-max:", np.min(xx), np.max(xx))
    print("   ml min-max:", np.min(ml), np.max(ml))
    plt.scatter(xx, ml, c="b", alpha=0.8, s=0.1)
    plt.plot(np.linspace(0,1,50), np.linspace(0,1,50), ":", c="k", linewidth=1)
    plt.xlim(0,1)
    plt.ylim(0,1)

    slope, intercept, r, p, stderr = stats.linregress(xx, ml)
    plt.plot(np.linspace(0,1,50), intercept + slope*np.linspace(0,1,50), 'r', label='fitted line')
    print("r=", stats.pearsonr(xx, ml)[0])

    xyticks=[round(x,2) for x in np.arange(0,1.1,0.5)]
    plt.xticks(xyticks, xyticks, fontsize=8)
    plt.yticks(xyticks, xyticks, fontsize=8)

    plt.text(0.1, 0.85, "n="+str(int(len(xx))), fontsize=9)
    plt.text(0.1, 0.75, "r="+str(round(r,2)), fontsize=9)
    plt.ylabel("SoMo.ml* [m$^3$/m$^3$]", fontsize=9)
    ax.set_aspect(1)


#for the distributions
def plt_distri(depth, ax):
    idxs=load_idxs(depth, 6)

    xx, ml=[], []
    for idx in idxs:
        _xx, _ml= load_ml_mitObs(depth, idx, 6)
        xx.append(_xx)
        ml.append(_ml)
    xx = np.array([item for sublist in xx for item in sublist])
    ml = np.array([item for sublist in ml for item in sublist])

    print (" >>>> soilm vs ml:", len(xx), len(ml))
    print ("  min-max xx", np.min(xx), np.max(xx))
    print ("  min-max ml", np.min(ml), np.max(ml))

    bins=np.arange(0,1.01,0.025)
    ax.hist(xx, bins=bins, color='grey', weights=np.zeros_like(xx) + 1. / xx.size, ec='grey', alpha=.8, label="Adj. in situ")
    ax.hist(ml, bins=bins, color='b', weights=np.zeros_like(ml) + 1. / ml.size, ec='b', alpha=.3, label="SoMo.ml*")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(fontsize=8)

    #plt.setp(ax.spines.values(), color='grey')
    #plt.setp([ax.get_xticklines(), ax.get_yticklines()], color='grey')

    xyticks=[round(x,2) for x in np.arange(0,1.1,0.5)]
    plt.xticks(xyticks, xyticks, fontsize=8)
    plt.ylim(0,0.15)
    plt.yticks([0,.1],[0,.1], fontsize=8)
    plt.ylabel("Density", fontsize=9)

def box_plt(depth, val_hr, ax):
    target=pd.read_csv("./pltdata/target_"+depth+"H06.lis",
                       header=0, index_col=0, parse_dates=True, na_values=-9999.)
    regions=["NAmerica", "Europe", "Asia", "Australia", "Africa", "SAmerica"]


    sim_pltdata= []
    obs_pltdata= []
    for region in regions:
        idxs=target[target["region"]==region].copy()

        xx, ml = [], []
        for idx in idxs["idx"]:
            _xx, _ml= load_ml_mitObs(depth, "IDX_"+idx+"_ep15.dat", 6)
            xx.append(_xx)
            ml.append(_ml)
        #######################
        xx = np.array([item for sublist in xx for item in sublist])
        ml = np.array([item for sublist in ml for item in sublist])
        print(">>", region, len(idxs), len(xx), len(ml))
        obs_pltdata.append(xx)
        sim_pltdata.append(ml)


    v1 = ax.violinplot(obs_pltdata,  positions=range(6), widths=0.85,
                       showmeans=False, showextrema=False, showmedians=True)
    v1['cmedians'].set_color('k')
    for b in v1['bodies']:
        m = np.mean(b.get_paths()[0].vertices[:, 0])
        b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
        b.set_color('grey')

    v2 = ax.violinplot(sim_pltdata,  positions=range(6), widths=0.85,
                       showmeans=False, showextrema=False, showmedians=True)

    v2['cmedians'].set_color("b")
    for b in v2['bodies']:
        m = np.mean(b.get_paths()[0].vertices[:, 0])
        b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
        b.set_color("b")


    plt.xticks(range(6), ["","","","","",""], fontsize=8)
    plt.ylim(0,1)
    plt.yticks(np.arange(0,1.1,0.5), np.arange(0,1.1,0.5), fontsize=8)
    plt.ylabel("Soil Moist. [m$^3$/m$^3$]", fontsize=9)

##################################################################
val_hr = 6
layers = ["layer0", "layer1", "layer2"]
##################################################################
fig=plt.figure(figsize=(8, 7.5))
gs  = gridspec.GridSpec(3, 3, hspace=0.25, wspace=0.4)
##################################################################

i=0
for layer in layers:
    print("")
    print(" ********** ", i, layer)
    print(" 1. plotting scatter (spatial comparison) ")

    ax  = plt.subplot(gs[i])
    plt_scatt(layer, ax)
    if i==0:
        plt.text(0.0,1.1, "a) between pixels")
        plt.text(1.4,1.1, "b) all time series")
        plt.text(2.8,1.1, "c) by region")
        plt.text(-0.5, 0.6, "Layer1", weight='bold', rotation=90)
    if i==3: plt.text(-0.5, 0.6, "Layer2", weight='bold', rotation=90)
    if i==6: plt.text(-0.5, 0.6, "Layer3", weight='bold', rotation=90)
    if i==6: plt.xlabel("Adj. in situ [m3/m3]", fontsize=9)


    print(" 2. plotting distribution ")
    ax  = plt.subplot(gs[i+1])
    plt_distri(layer, ax)
    plt.xticks(fontsize=8)
    if i==6: plt.xlabel("Soil Moist. [m3/m3]", fontsize=9)

    print(" 3. regional rel. biases ")
    ax  = plt.subplot(gs[i+2])
    box_plt(layer, val_hr, ax)
    if i==6: plt.xticks(range(6), ["NAmerica", "Europe", "Asia", "Australia", "Africa", "SAmerica"],\
                        fontsize=8, rotation=90)

    i+=3

print (". . . saving")
plt.savefig("./fig5.pdf")
plt.close()

"""
## to change with density scatter plot
    slope, intercept, r, p, stderr = stats.linregress(xx_flat, yy_flat)
    cmap=plt.cm.jet
    cmap.set_bad('white')
    cmap.set_under('white')
    plt.hist2d(xx_flat, yy_flat, (100, 100), vmin=0.0000001, cmap=cmap)
    plt.plot(xx_lins, intercept + slope*xx_lins, 'r', label='fitted line')
    plt.plot( xx_lins,  xx_lins, 'k:')

    cbar=plt.colorbar(cax=cax)
"""
