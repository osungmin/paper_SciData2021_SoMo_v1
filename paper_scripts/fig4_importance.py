#!/usr/bin/env python
import pandas as pd
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from mpl_toolkits.basemap import Basemap
from matplotlib.lines import Line2D
print ("modules imported")


def plt_importance(depth, err, ax):
   print("*****")
   print(depth)
   print("*****")

   rn_cols =["log_tp","t2m","snr","ssrd","q","skt"]
   rn_lable=["Precip.","Temp.","Rad$_{net}$","SW$_{down}$","S. Humid.","Surf. Temp."]

   if depth=="layer1":
       rn_cols+=["layer0"]
       rn_lable+=["Layer1"]
   if depth=="layer2":
       rn_cols+=["layer0", "layer1"]
       rn_lable+=["Layer1", "Layer2"]

   load_ohne = pd.read_csv("./pltdata/"+depth+"_shuffled_ohne_input_inportance.dat", header=0, index_col=0)
   ###load_ohne = pd.read_csv("./pltdata/"+depth+"_ohne_input_inportance.dat", header=0, index_col=0)

   score_ohne= np.median(load_ohne[err[1:]].values) #name of err starts with "_"
   print(" >>>>> without statics", round(score_ohne,3))
   print(load_ohne.describe())

   load    = pd.read_csv("./pltdata/"+depth+"_input_inportance"+err+".dat", header=0, index_col=0)
   pltdata = [np.median(load[col].values) for col in rn_cols]
   print (" >>>>> with statics", len(pltdata))
   print(load.describe())

   if err in ["_nse","_corr"]:
       ranked    =[x*-1. for _,x in sorted(zip(pltdata, pltdata))][::-1]
       new_cols  =[x for _,x in sorted(zip(pltdata, rn_lable))][::-1]
   else:
       ranked    =[x for _,x in sorted(zip(pltdata, pltdata))]
       new_cols  =[x for _,x in sorted(zip(pltdata, rn_lable))]

   print("before ordered:", rn_cols)
   print([round(x,3) for x in pltdata])
   print("")
   print("       ranked:", new_cols)
   print([round(x,3) for x in ranked])
   print("static (raw):", round(score_ohne,3))
   print("")

   if err=="_rmse": xtext=0.08*0.1
   if err=="_nse": xtext=0.01
   if err=="_corr": xtext=1.2*0.1

   if depth=="layer0": #+1: ohne +2: two less inputs
      if err in ["_nse","_corr"]:  bplt= ax.barh(range(len(rn_cols)+3), [0,0]+[score_ohne*-1.]+ranked, align='center', color='darkgrey')
      else:   bplt= ax.barh(range(len(rn_cols)+3), [0,0]+[score_ohne]+ranked, align='center', color='darkgrey')

      for i, label in zip(range(len(rn_lable)+3), ["","","static*"]+new_cols):
          plt.text(xtext, i-0.075, label, fontsize=7)

   if depth=="layer1":
       if err in ["_nse","_corr"]:  bplt= ax.barh(range(len(rn_cols)+2), [0]+[score_ohne*-1.]+ranked, align='center', color='darkgrey')
       else:  bplt= ax.barh(range(len(rn_cols)+2), [0]+[score_ohne]+ranked, align='center', color='darkgrey')
       for i, label in zip(range(len(rn_lable)+2), ["", "static*"]+new_cols):
           plt.text(xtext, i-0.075, label, fontsize=7)

   if depth=="layer2":
       if err in ["_nse","_corr"]:  bplt= ax.barh(range(len(rn_cols)+1), [score_ohne*-1.]+ranked, align='center', color='darkgrey')
       else:  bplt= ax.barh(range(len(rn_cols)+1), [score_ohne]+ranked, align='center', color='darkgrey')

       for i, label in zip(range(len(rn_lable)+1), ["static*"]+new_cols):
           plt.text(xtext, i-0.075, label, fontsize=7)


   #bplt[0].set_color('darkgrey')
   if err=="_corr":
       plt.xlim(0,1.2)
       plt.xticks([0,0.6,1.2],[0,0.6,1.2], fontsize=8)
   if err=="_rmse":
       plt.xlim(0,0.08)
       plt.xticks([0,0.05,0.1],[0,0.5,0.1], fontsize=8)
   plt.yticks([])




fig=plt.figure(figsize=(7.5,5))
gs = gridspec.GridSpec(2, 3, bottom=0.1, wspace=0.1)

err="_rmse"

ax  = plt.subplot(gs[0])
plt_importance("layer0", err, ax)
plt.title("Layer 1", fontsize=8)
plt.ylabel("NRMSE", fontsize=9)

ax  = plt.subplot(gs[1])
plt_importance("layer1", err, ax)
plt.title("Layer 2", fontsize=8)

ax  = plt.subplot(gs[2])
plt_importance("layer2", err, ax)
plt.title("Layer 3", fontsize=8)


err="_corr"

ax  = plt.subplot(gs[3])
plt_importance("layer0", err, ax)
plt.ylabel("Corr. Coefficient", fontsize=9)

ax  = plt.subplot(gs[4])
plt_importance("layer1", err, ax)

ax  = plt.subplot(gs[5])
plt_importance("layer2", err, ax)

print (". . . saving")
plt.savefig("./fig4.pdf")
plt.close()
