"""Figure 3 review revision: same observations/size, view styles and descriptive nondominated points."""
from pathlib import Path
import sys,shutil
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0,str(Path(__file__).parent))
import compact_measurements as cm
plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.family':'serif','font.serif':['Times New Roman','Times','STIXGeneral'],'mathtext.fontset':'stix'})
styles={'all':'solid','gt_100':(0,(5,2)),'uplink_only':(0,(2,1.3)),'uplink_only_gt_100':(0,(4,1,1,1))}
fig,axs=plt.subplots(2,2,figsize=(7.16,2.7),sharex='row',sharey='row')
rows=[]
for r,(kq,ylab) in enumerate(cm.KQIS):
 for c,(dev,dname) in enumerate(cm.DEVS):
  ax=axs[r,c];points=[]
  for v,vl in cm.VIEWS:
   df=pd.read_csv(cm.SEL/dev/f'{kq}_{v}.csv',index_col=0)
   for fam,(col,mk,_) in cm.FAM.items():
    pf=df.loc[f'{kq}_{v}_{fam}_performance'];ea=df.loc[f'{kq}_{v}_{fam}_efficiency']
    ax.plot([pf.energy,ea.energy],[pf.score*100,ea.score*100],color=col,lw=.75,alpha=.8,linestyle=styles[v],zorder=1)
    for variant,item in [('pf',pf),('ea',ea)]:
     ax.scatter(item.energy,item.score*100,s=16,marker=mk,facecolor=col if variant=='pf' else 'white',edgecolor='white' if variant=='pf' else col,lw=.4 if variant=='pf' else .9,zorder=3)
     points.append((float(item.energy),float(item.score),v,fam,variant))
  for E,S,v,fam,variant in points:
   nondom=not any(e<=E and s>=S and (e<E or s>S) for e,s,*_ in points)
   rows.append(dict(device=dev,kqi=kq,view=v,family=fam,variant=variant,energy_J=E,score=S,nondominated=nondom))
   if nondom:ax.scatter(E,S*100,s=43,marker='o',facecolors='none',edgecolors='#222222',lw=.65,zorder=4)
  ax.set_xscale('log');ax.grid(True,which='major',lw=.3,color='#d9d8d4');ax.set_axisbelow(True)
  for sp in ('top','right'):ax.spines[sp].set_visible(False)
  if r==0:ax.set_title(dname,fontsize=8)
  if c==0:ax.set_ylabel(ylab)
  if r==1:ax.set_xlabel('Energy for 100,000 predictions (W1) [J]')
  ax.yaxis.set_major_locator(plt.MultipleLocator(10))
axs[1,0].set_ylim(bottom=45)
h=[Line2D([],[],color=col,marker=mk,ls='',ms=4.5,label=lab) for col,mk,lab in cm.FAM.values()]
h += [Line2D([],[],color='#52514e',marker='o',ls='',ms=4.5,label='performance-first'),Line2D([],[],color='#52514e',marker='o',ls='',ms=4.5,mfc='white',label='energy-aware')]
fig.legend(handles=h,loc='upper center',bbox_to_anchor=(.5,1.01),ncol=8,frameon=False,fontsize=7,handletextpad=.2,columnspacing=1)
h2=[Line2D([],[],color='#52514e',ls=styles[v],lw=.9,label=lab) for v,lab in cm.VIEWS]
h2 += [Line2D([],[],color='#222222',marker='o',ls='',mfc='none',ms=6,label='nondominated (W1)')]
fig.legend(handles=h2,loc='upper center',bbox_to_anchor=(.5,.944),ncol=5,frameon=False,fontsize=6.6,handlelength=2.4,handletextpad=.4,columnspacing=1.4)
fig.tight_layout(rect=(0,0,1,1-.39/2.7),pad=.3,w_pad=.8,h_pad=.6)
out=Path(__file__).parents[1]
fig.savefig(out/'figures/offline_pairs.pdf');fig.savefig('/tmp/noms-v13-build/offline-review.png',dpi=200)
pd.DataFrame(rows).to_csv(out/'review_support/figure3_points.csv',index=False)
print('Figure size:',fig.get_size_inches(),'points',len(rows),'nondominated',sum(x['nondominated'] for x in rows))
