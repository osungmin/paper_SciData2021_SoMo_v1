#!/usr/bin/env python
import os, os.path
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.lines import Line2D
print ("modules imported")


def load_grid_target(depth, valid_hrs="", target=False):
   targetpath="./pltdata/target_"+depth+"H"+str(valid_hrs).zfill(2)+".lis"
   out = pd.read_csv(targetpath, header=0, index_col=0)

   out["len"] = [x/365. for x in out["len2000"].values]
   out["dep00"], out["dep01"] = out['selected_depth'].str.split('-', 1).str
   out["dep01"] = [ int(x) for x in out.dep01]
   out["dep00"] = [ int(x.split("dep")[1]) for x in out.dep00]
   out["dep"] = [ (x+y)*.5 for x, y in zip(out["dep00"], out["dep01"])]

   return out


def basemap(grids_all, len_l0, len_l1, len_l2, dep_l0, dep_l1, dep_l2):

    print (" >> in basedmap << ")
    print (" >  max data len:", max(len_l0), max(len_l1), max(len_l2))
    print (" >  min data dep:", min(dep_l0), min(dep_l1), min(dep_l2))
    print (" >  max data dep:", max(dep_l0), max(dep_l1), max(dep_l2))
    print ("")

    fig = plt.figure(figsize=(9, 4.5), facecolor='w', edgecolor='k')
    gs  = gridspec.GridSpec(1, 2, width_ratios=[2.5,1])
    gs00 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[0])
    gs01 = gridspec.GridSpecFromSubplotSpec(5, 2, height_ratios=[0.2,1,1,1,0.2], subplot_spec=gs[1], hspace=0.3, wspace=0.3)

    ax = plt.subplot(gs00[0])
    plt.title("(a)", x=0, y=1.123, fontsize=10)

    m = Basemap(projection='robin', lon_0=0,resolution='c', ax=ax)

    m.drawmapboundary(fill_color='white', zorder=-1)
    m.fillcontinents(color='0.95', lake_color='white', zorder=0)

    m.drawcoastlines(color='0.6', linewidth=0.5)
    m.drawcountries(color='0.6', linewidth=0.5)

    m.drawparallels(np.arange(-90.,91.,30.), labels=[1,0,0,1], dashes=[1,1], linewidth=0.25, color='0.5', fontsize=8)
    m.drawmeridians(np.arange(0., 360., 60.), labels=[1,0,0,1], dashes=[1,1], linewidth=0.25, color='0.5', fontsize=8)

    # lon and lat for plot maps
    lons=[float(x.split("_")[1]) for x in grids]
    lats=[float(x.split("_")[0]) for x in grids]

    print ("+ plotting grids !!")
    for lon, lat in zip(lons, lats): ##
        x,y = m(lon, lat)
        if lon>-20: sc = ax.scatter(x,y, marker="x", s=4, alpha=.9, linewidth=.5, color='k')
        else:       sc = ax.scatter(x,y, marker="x", s=3, alpha=0.7, linewidth=.2, color='k')


    print("")
    print("+ adding text")
    x,y = m(25, -48)
    textstr = '\n'.join((
        'Layer1 (  0-10cm): 1,114 px',
        'Layer2 (10-30cm): 1,064 px',
        'Layer3 (30-50cm):    683 px'))

    props = dict(boxstyle='round', facecolor='w', alpha=0.5)
    ax.text(x, y, textstr, fontsize=5.5,
            verticalalignment='top', bbox=props)

    print ("")
    print ("+ plotting data len of layer0")
    ax1 = plt.subplot(gs01[2])
    ax1.text(-4, .69, "(b)", fontsize=10)

    bin_l0,bin_boundary = np.histogram(len_l0,bins=np.arange(0,21,5))

    ax1.bar(range(len(bin_l0)), bin_l0/float(len(len_l0)), color='lightgrey', edgecolor='k', width=.6)
    ax1.tick_params(direction='out', length=2, width=1)
    ax1.tick_params(axis='both', which='major', pad=1)

    plt.xticks(range(len(bin_l0)), [],fontsize=7)
    plt.yticks([0,0.3,0.6],[0,0.3,0.6], fontsize=7)
    print ("***", sum(bin_l0/float(len(len_l0))))
    print (bin_l0/float(len(len_l0)))
    plt.ylabel("Layer1")
    plt.title("data length", fontsize=9)
    plt.ylim(0,0.601)

    ###
    print ("+ plotting depth distri of layer0")
    ax3 = plt.subplot(gs01[3])

    bin_l0,bin_boundary = np.histogram(dep_l0,bins=[0,4.9,5.1,10.1])

    ax3.bar(range(len(bin_l0)), bin_l0/float(len(dep_l0)), color='lightgrey', edgecolor='k', width=0.6)
    ax3.tick_params(direction='out', length=2, width=1)
    ax3.tick_params(axis='both', which='major', pad=1)

    plt.xticks(range(len(bin_l0)), ["<5","5",">5"], fontsize=7)
    plt.yticks([0,0.5,1.0],[0,0.5,1.0], fontsize=7)
    print ("***", sum(bin_l0/float(len(dep_l0))))
    print (bin_l0/float(len(dep_l0)))
    plt.title("meas. depth", fontsize=9)
    plt.ylim(0,1)

    ###
    print ("")
    print ("+ plotting data len of layer1")
    ax2 = plt.subplot(gs01[4])

    bin_l1,bin_boundary = np.histogram(len_l1,bins=np.arange(0,21,5))

    ax2.bar(range(len(bin_l1)), bin_l1/float(len(len_l1)), color='lightgrey', edgecolor='k', width=.6)
    ax2.tick_params(direction='out', length=2, width=1)
    ax2.tick_params(axis='both', which='major', pad=1)

    plt.xticks(range(len(bin_l1)),[],fontsize=7)
    plt.yticks([0,0.3,0.6],[0,0.3,0.6], fontsize=7)
    print ("***", sum(bin_l1/float(len(len_l1))))
    print (bin_l1/float(len(len_l1)))
    plt.ylabel("Layer2")
    plt.ylim(0,0.601)

    print ("+ plotting depth distri of layer1")
    ax4 = plt.subplot(gs01[5])

    bin_l1,bin_boundary = np.histogram(dep_l1,bins=[10,19,21,30.1])

    ax4.bar(range(len(bin_l1)), bin_l1/float(len(dep_l1)), color='lightgrey', edgecolor='k', width=.6)
    ax4.tick_params(direction='out', length=2, width=1)
    ax4.tick_params(axis='both', which='major', pad=1)

    plt.xticks(range(len(bin_l1)), ["<20","20",">20"], fontsize=7)
    plt.yticks([0,0.5,1.0],[0,0.5,1.0], fontsize=7)
    plt.ylim(0,1)
    print ("***", sum(bin_l1/float(len(dep_l1))))
    print (bin_l1/float(len(dep_l1)))


    ###
    print ("")
    print ("+ plotting data len of layer2")
    ax5 = plt.subplot(gs01[6])

    bin_l2,bin_boundary = np.histogram(len_l2,bins=np.arange(0,21,5))

    ax5.bar(range(len(bin_l2)), bin_l2/float(len(len_l2)), color='lightgrey', edgecolor='k', width=.6)
    ax5.tick_params(direction='out', length=2, width=1)
    ax5.tick_params(axis='both', which='major', pad=1)

    plt.xticks(range(len(bin_l2)),["0-5","5-10","10-15","15-20"],fontsize=6, rotation=19)
    plt.yticks([0,0.3,0.6],[0,0.3,0.6], fontsize=7)
    print ("***", sum(bin_l2/float(len(len_l2))))
    print (bin_l2/float(len(len_l2)))
    plt.ylabel("Layer3")
    plt.ylim(0,0.601)
    plt.xlabel("[yr]", fontsize=8)
    ax5.xaxis.set_label_coords(.5, -0.25)

    ###
    print ("+ plotting depth distri of layer2")
    ax6 = plt.subplot(gs01[7])

    bin_l2,bin_boundary = np.histogram(dep_l2,bins=[30,39,41,50.1])

    ax6.bar(range(len(bin_l2)), bin_l2/float(len(dep_l2)), color='lightgrey', edgecolor='k', width=.6)
    ax6.tick_params(direction='out', length=2, width=1)
    ax6.tick_params(axis='both', which='major', pad=1)

    plt.xticks(range(len(bin_l2)), ["<40","40",">40"], fontsize=7)
    plt.yticks([0,0.5,1.0],[0,0.5,1.0], fontsize=7)
    print (sum(bin_l2/float(len(dep_l2))))
    print (bin_l2/float(len(dep_l2)))
    plt.ylim(0,1)
    plt.xlabel("[cm]", fontsize=8)
    ax6.xaxis.set_label_coords(.5, -0.25)

    plotf="./fig02.png"
    plt.savefig(plotf, dpi=300)
    plt.close()
    print ("plot saved:", plotf)

################################################################
print ("\n\n\n ***** START *****")
val_hrs= 6
lon_key, lat_key="lon_d25", "lat_d25"

################################################################ for global map
lis_layer0 = load_grid_target("layer0", valid_hrs=val_hrs, target=True)
lis_layer1 = load_grid_target("layer1", valid_hrs=val_hrs, target=True)
lis_layer2 = load_grid_target("layer2", valid_hrs=val_hrs, target=True)
################################################################
print(" > target len", len(lis_layer0), len(lis_layer1), len(lis_layer2))
print(" > training yrs", np.sum(lis_layer0["len"]), np.sum(lis_layer1["len"]), np.sum(lis_layer2["len"]))
grids = np.unique(list(lis_layer0["idx"].values)+list(lis_layer1["idx"].values)+list(lis_layer2["idx"].values))
print(" ***", len(grids))
print("")
basemap(grids,\
        lis_layer0["len"].values, lis_layer1["len"].values, lis_layer2["len"].values,\
        lis_layer0["dep"].values, lis_layer1["dep"].values, lis_layer2["dep"].values)

print (" basedmap done.")
