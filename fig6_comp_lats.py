#!/usr/bin/env python
import pandas as pd
import numpy as np
import gzip, cPickle
from netCDF4 import Dataset

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from mpl_toolkits.basemap import Basemap
from matplotlib.lines import Line2D
print ("modules imported")


def call_map():
    fig = plt.figure(figsize=(7, 5), facecolor='w', edgecolor='k')
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.05, bottom=0.15)
    gs0 = gridspec.GridSpecFromSubplotSpec(3, 1, height_ratios=[1,1,1], subplot_spec=gs[0])
    gs1 = gridspec.GridSpecFromSubplotSpec(1, 3, width_ratios=[1,1,1], subplot_spec=gs[1])
    return fig, gs0, gs1

def lats_averaging(ave):

   fig, gs0, gs1= call_map()

   print("")
   print(">>>>> map")

   for i in range(3):
       print(" map plotting for ", i)
       ax=fig.add_subplot(gs0[i])

       if i==0: plt.text(-190,110, "(a)", fontsize=8, fontweight="bold")

       m = Basemap(projection='cyl', resolution='c', ax=ax)

       m.drawmapboundary(fill_color='white', zorder=-1)
       m.fillcontinents(color='0.8', lake_color='white', zorder=0)

       m.drawcoastlines(color='0.6', linewidth=0.5)

       m.drawparallels(np.arange(-90.,91.,90.), labels=[1,0,0,1], dashes=[1,1], linewidth=0.25, color='0.5', fontsize=7)
       if i==2: m.drawmeridians(np.arange(-90., 91., 90.), labels=[1,0,0,1], dashes=[1,1], linewidth=0.25, color='0.5', fontsize=7)
       else: m.drawmeridians(np.arange(-90., 91., 90.), labels=[0,0,0,0], dashes=[1,1], linewidth=0.25, color='0.5', fontsize=7)

       lons  = np.arange(-179.875, 180, .25)
       lats  = np.arange(89.875, -90, -.25)

       print("use meshgrid to create 2D arrays")
       lon, lat = np.meshgrid(lons,lats)
       xi, yi = m(lon, lat)

       print("grids ready")
       f = gzip.GzipFile("./pltdata/somo_layer"+str(i)+".dat", "r")
       cs = m.pcolormesh(xi,yi, np.load(f), vmin=0, vmax=0.6, cmap=cm.YlGnBu)

       print("label")
       x,y = m(70, -75)
       ax.text(x, y, "Layer"+str(i+1), fontsize=9)


   print("===================================================")
   print("colorbar")
   cax = fig.add_axes([0.19, 0.08, 0.25, 0.02])
   cbar = plt.colorbar(cs,  orientation='horizontal', extend='max', cax=cax)
   cbar.set_ticks([0,0.2,0.4,0.6])
   cbar.set_ticklabels([0,0.2,0.4,0.6])

   cbar.ax.tick_params(labelsize=7)
   cbar.set_label("SoMo.ml [m$^3$/m$^3$]", fontsize=7)
   cax.tick_params(direction='in', length=1)

   print("")
   print(">>>>> vertival averages")

   f0 = Dataset("./pltdata/GLDASp4_landmask_025d.nc4", mode="r")
   lmask = f0.variables['GLDAS_mask'][0,:,:][::-1] #lat from south => from north
   print(">> landmask loading:", lmask.shape)

   # layer 0 1 2
   pltvars=[["sm","SMsurf","swvl1","layer0"], ["SMroot","swvl2","layer1"], ["SMroot","swvl3","layer2"]]
   cols   =[["violet", "orange","#ff9999",'b'], ["orange", "#ff9999","b"],["orange","#ff9999","b"]]
   depths =["layer0", "layer1", "layer2"]

   for i in range(3):
      print(" plotting layer ", i)
      ax1=fig.add_subplot(gs1[i])
      if i==0: plt.text(0,95.5, "(b)", fontsize=8, fontweight="bold")

      c=0
      for var in pltvars[i]:
          print("")
          print(" >>", var)

          lons  = np.arange(-179.875, 180, .25)
          lats  = np.arange(89.875, -90, -.25)[:600]#to cut Antarctica

          print("  loaded:", var, "lons:", lons.shape, "lats:", lats.shape)

          f = gzip.GzipFile("./pltdata/"+var+"_layer"+str(i)+".dat", "r")
          pltdata = np.load(f)

          #just for beter plotting (somehow swvl has very low numbs at first & last)
          pltdata[pltdata<0.07]=np.nan

          plt.plot(pltdata, np.arange(89.875,-60,-.25), c=cols[i][c], linestyle='-', lw=1, alpha=.7)

          c+=1

      print(">>> plot deatils")

      plt.xlim(0,.4)
      plt.ylim(-90,90)
      ax1.text(0.1, -105, "Layer"+str(i+1), fontsize=9)
      ax1.tick_params(direction='in', length=2)
      ax1.yaxis.tick_right()
      ax1.yaxis.set_label_position("right")
      ax1.xaxis.set_ticks_position("both")
      ax1.yaxis.set_ticks_position("both")

      if i==2: plt.yticks(np.arange(-60,91,30),\
                         ["60$\degree$S","30$\degree$S","0$\degree$","30$\degree$N","60$\degree$N","90$\degree$N"], fontsize=7)
      else:    plt.yticks(np.arange(-60,91,30), [])
      plt.xticks([0,.2,.4], [0,.2,.4], fontsize=7) #x ticks for right
      #plt.grid()


      if i==2:
          legend_elements=[Line2D([0],[0],linestyle="-", color="b",lw=2,label='SoMo.ml'),
                           Line2D([0],[0],linestyle="-", color="#ff9999",alpha=.7, lw=2,label='ERA5'),
                           Line2D([0],[0],linestyle="-", color="orange", alpha=.7, lw=2,label='GLEAM'),
                           Line2D([0],[0],linestyle="-", color="violet", alpha=.7, lw=2, label='ESA-CCI'),
                           Line2D([0],[0],linestyle="None", marker="o", color="grey", ms=3, label='In situ')]

          ax1.legend(handles=legend_elements, loc="lower left", prop={'size': 6.5},
                     bbox_to_anchor=(0., 0., 0.5, 0.5),
                     labelspacing=0.05, frameon=False, facecolor='w')

      #ISMN POINT IN SITU MEASUREMENTS
      print("")
      print(">>>>> ismn plotting:", depths[i], ave)
      ismn=pd.read_csv("./pltdata/ismn_lat_"+depths[i]+"_"+ave+".dat", header=0, index_col=0, na_values=-9999.)
      ismn=ismn.dropna()
      print("")

      for i in range(len(ismn)):
         plt.plot(ismn["soilm"].values[i], ismn.index[i], c="grey", marker="o", alpha=.7, ms=.5)

   print("")
   print("done.")
   plt.savefig("./fig6_"+ave+".png", dpi=300)
   plt.close()
   print("saved.")


####################################
ave="med"
print("\n\n averaging along latitudes:", ave)
nc=lats_averaging(ave)
print("end.")
####################################
