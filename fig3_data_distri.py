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


def clim_map(l_clims, glo_clims):

    print ("+ making plot to show climte regions for training sites")
    print (" >>>", len(l0_clim), len(l1_clim), len(l2_clim))

    titles= ["Layer1 (0-10cm)","Layer2 (10-30cm)","Layer3 (30-50cm)"]

    fig=plt.figure(figsize=(7.5,3))
    gs = gridspec.GridSpec(1, 3, bottom=0.15, wspace=0.1)

    for i in range(3):
      l_clim=l_clims[i]
      print (" >>>> i", len(l_clim))

      ax = plt.subplot(gs[i])

      print(" . . . global")
      for grid in glo_clim.index:
          arid     =glo_clim[glo_clim.index==grid]["arid"]
          temp     =glo_clim[glo_clim.index==grid]["t2m_ave"]
          x, y= arid.values[0], temp.values[0]-273.15
          sc = ax.scatter(x,y, marker="x", linewidth=0.25, s=1.5, alpha=0.3, facecolors='lightgrey', edgecolors='grey')


      ##### plotting layer #####
      n_sh_eu, n_sh_na, n_sh_af, n_sh_sa, n_sh_au, n_sh_asia = [0 for x in range(6)]

      for grid in l_clim.index:

          arid     =l_clim[l_clim.index==grid]["arid"]
          temp     =l_clim[l_clim.index==grid]["t2m_ave"]
          region   =l_clim[l_clim.index==grid]["region"]
          if (len(arid)!=1)|(len(temp)!=1)|(len(region)!=1): stop

          x, y= arid.values[0], temp.values[0]-273.15
          region = region.values[0]

          if (region=="Europe"):
              col="orange"
              n_sh_eu+=1
          elif region=="NAmerica":
              col="k"
              n_sh_na+=1
          elif region=="Africa":
              col="brown"
              n_sh_af+=1
          elif region=="SAmerica":
              col="green"
              n_sh_sa+=1
          elif region=="Australia":
              col="red"
              n_sh_au+=1
          elif region=="Asia":
              col="b"
              n_sh_asia+=1
          else: stop


          sc = ax.scatter(x,y, marker="x", linewidth=0.25, s=2.5, facecolors=col, edgecolors=col)

      print (" + data num. ", n_sh_eu, n_sh_na, n_sh_af, n_sh_sa, n_sh_au, n_sh_asia)



      custom_legend = [Line2D([0], [0], color='k', marker="x", ms=4, markeredgewidth=1,  linewidth=0),
                      Line2D([0], [0], color='orange', marker="x", ms=4, markeredgewidth=1, linewidth=0),
                      Line2D([0], [0], color='blue', marker="x", ms=4, markeredgewidth=1, linewidth=0),
                      Line2D([0], [0], color='red', marker="x", ms=4, markeredgewidth=1, linewidth=0),
                      Line2D([0], [0], color='brown', marker="x", ms=4, markeredgewidth=1, linewidth=0),
                      Line2D([0], [0], color='green', marker="x", ms=4, markeredgewidth=1, linewidth=0),
                      Line2D([0], [0], color='darkgrey', marker="x", ms=4, markeredgewidth=1, linewidth=0)]


      ax.legend(custom_legend, ["NAmerica ("+str(int(n_sh_na))+")",\
                                "Europe ("+str(int(n_sh_eu))+")",\
                                "Asia ("+str(int(n_sh_asia))+")",\
                                "Australia ("+str(int(n_sh_au))+")",\
                                "Africa ("+str(int(n_sh_af))+")",\
                                "SAmerica ("+str(int(n_sh_sa))+")",\
                                "Global*"],\
                                loc='upper left', borderpad=.35, fontsize=6, labelspacing=0.3, framealpha=0.1)


      plt.title(titles[i], fontsize=8)
      plt.ylim(-20,40)
      plt.xlim(0.001, 200)

      ax.set_xscale('log')
      ax.set_aspect(0.75)

      if i==0: plt.yticks(np.arange(-20,41,10), np.arange(-20,41,10), fontsize=7)
      else:    plt.yticks(np.arange(-20,41,10),[])

      plt.xticks([0.01,0.1,1,10,100],[0.01,0.1,1,10,100], fontsize=7)

      if i==0: plt.ylabel('Temperature (C$^\circ$)', fontsize=8)
      plt.xlabel('Aridity (Rad$_{net}$/Precip)', fontsize=8)


    plotf="./Fig03.png"
    plt.savefig(plotf, dpi=300)
    plt.close()
    print ("plot saved:", plotf)


################################################################
print ("\n\n\n ***** START *****")
l0_clim   = pd.read_csv("./pltdata/hydroclim_target_layer0H06.lis", header=0, index_col=0)
l1_clim   = pd.read_csv("./pltdata/hydroclim_target_layer1H06.lis", header=0, index_col=0)
l2_clim   = pd.read_csv("./pltdata/hydroclim_target_layer2H06.lis", header=0, index_col=0)
glo_clim  = pd.read_csv("./pltdata/hydroclim_global_rnd.lis", header=0, index_col=0)#randomly
#########################
print(" > hydroclim len", len(l0_clim), len(l1_clim), len(l2_clim))
print(" > glo_clim:", len(glo_clim))
print("")
print("  target data min-max", np.min(l0_clim["t2m_ave"])-273, np.max(l0_clim["t2m_ave"])-273)
print("              min-max", np.min(l0_clim["arid"]), np.max(l0_clim["arid"]))
print("")
################################################################
clim_map([l0_clim, l1_clim, l2_clim], glo_clim)
################################################################ # for climate map
print ("end.")
