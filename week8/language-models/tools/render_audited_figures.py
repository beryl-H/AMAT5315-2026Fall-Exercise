"""Final mechanical rendering; preserve originals and all numerical results."""
from pathlib import Path
import json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path(__file__).resolve().parents[1]
e=r/'evidence'; sys.path.insert(0,str(r))
plt.rcParams.update({'font.size':10,'savefig.bbox':'tight'})
# All pairs, true equal-aspect angles and readable color-mapped legend.
x=json.loads((e/'part1-results.json').read_text())
fig=plt.figure(figsize=(15,9),layout='constrained')
gs=fig.add_gridspec(2,3,height_ratios=[1,.8]); colors=plt.get_cmap('tab10').colors
for j,(name,rel) in enumerate(x['relations'].items()):
 ax=fig.add_subplot(gs[0,j]); pairs=list(rel['table_pair_cosines']); ax.axhline(0,c='0.7',lw=.6)
 for i,pair in enumerate(pairs):
  for score,sign,alpha in [(rel['table_pair_cosines'][pair],1,1),(rel['control_pair_cosines'][pair],-1,.55)]:
   angle=np.arccos(np.clip(score,-1,1)); ax.annotate('',(np.cos(angle),sign*np.sin(angle)),(0,0),arrowprops={'arrowstyle':'->','color':colors[i],'alpha':alpha,'lw':1.5})
 ax.annotate('',(1.08,0),(0,0),arrowprops={'arrowstyle':'->','color':'black','lw':2})
 ax.set(xlim=(-1.15,1.15),ylim=(-1.15,1.15),xlabel='cosine with leave-one-out mean',ylabel='table above / shuffle below')
 ax.set_aspect('equal'); ax.set_title(name.replace('_',' ')+'\nmean table %.3f | shuffle %+.3f'%(rel['table_mean_cosine'],rel['control_seed1730_mean_cosine']))
 ta=fig.add_subplot(gs[1,j]); ta.axis('off')
 tab=ta.table(cellText=[[p,'%.3f'%rel['table_pair_cosines'][p],'%+.3f'%rel['control_pair_cosines'][p]] for p in pairs],colLabels=['Pair (arrow color)','Table','Shuffle'],loc='center',colWidths=[.58,.21,.21]); tab.auto_set_font_size(False); tab.set_fontsize(9); tab.scale(1,1.7)
 for i in range(9): tab[(i+1,0)].get_text().set_color(colors[i])
fig.suptitle('Relations as directions: all pairs, equal x/y scale (shuffle seed 1730)',fontsize=15)
fig.savefig(e/'part1-relations-final.png',dpi=150); plt.close(fig)
# Count ladder and the held-out unseen fractions.
x=json.loads((r/'artifacts/part2/formal-02/results.json').read_text())
(e/'part2-results-audited.json').write_text(json.dumps(x,indent=2)+'\n')
ns=np.arange(1,7); fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
ax.plot(ns,[x['models'][str(n)]['bits_per_char'] for n in ns],'o-',label='held-out bits/character')
ax.axhline(x['models']['uniform']['bits_per_char'],c='gray',ls=':',label='uniform')
ax.set(xlabel='n-gram order',ylabel='bits / original Unicode character',xticks=ns)
a2=ax.twinx(); a2.plot(ns,[100*x['models'][str(n)]['unseen_fraction'] for n in ns],'s--',c='tab:orange',label='unseen held-out n-grams')
a2.set(ylabel='unseen held-out n-grams (%)'); ax.legend(loc='upper left'); a2.legend(loc='upper right')
ax.set_title('Count models: sparsity reverses improvement beyond order 4')
fig.savefig(e/'part2-count-ladder-final.png',dpi=160); plt.close(fig)
# Reconstruct only the fixed showcase; verify every ISR before plotting.
from src.run_part4 import draw_case, recall_part4, picture_R
mask=np.asarray(json.loads((e/'glyph.json').read_text())['mask'],dtype=np.float64); flat=mask.ravel()
p4=json.loads((r/'artifacts/part4/formal-01/results.json').read_text())
fig,axes=plt.subplots(4,4,figsize=(12,12),layout='constrained'); rows,cols=np.indices((32,32)); arr={}
for i,M in enumerate([64,256,1024,4096]):
 K,V=draw_case(flat,M,531504)
 for j,store in enumerate(['outer','relu','relu2','softmax']):
  recall,w=recall_part4(K,V,K[0],store,8.0); R=picture_R(recall,w,flat)
  assert abs(R-p4['showcase_raw_ISR'][f'{store}|M={M}'])<1e-11
  norm=recall/w; arr[f'{store}_{M}']=norm.reshape(32,32); area=np.maximum(0,(norm+1)/2)
  ax=axes[i,j]; color='firebrick' if R<1 else '0.35'
  ax.scatter(cols.ravel(),rows.ravel(),s=area*4,c=color,marker='s',linewidths=0)
  ax.set(xlim=(-1,32),ylim=(32,-1),xticks=[],yticks=[]); ax.set_aspect('equal')
  ax.set_title(f'{store} | M={M}\nshowcase R={R:.3f}',color=color,fontsize=10)
fig.suptitle('User-selected character U+7B97: fixed showcase seed 531504\nRed: ISR < 1. Dot area = 4 max(0, (own-normalized pixel + 1)/2); no upper clipping.',fontsize=12)
fig.savefig(e/'part4-recall-final.png',dpi=160); plt.close(fig)
np.savez(r/'artifacts/part4/formal-01/showcase-normalized-recalls.npz',**arr)
fig,ax=plt.subplots(figsize=(3.2,3.5),layout='constrained')
ax.imshow(mask,cmap='gray_r',vmin=-1,vmax=1,interpolation='nearest'); ax.axis('off'); ax.set_title('U+7B97 | 32 x 32 | +1 ink / -1 background',fontsize=10)
fig.savefig(e/'glyph-preview-final.png',dpi=150); plt.close(fig)
# Scaling curves and residual bias.
b=r/'artifacts/part6/formal-01'; z=json.loads((b/'results.json').read_text()); raw=np.load(b/'filtered-data-and-fit.npz')
N,D,L,th=[raw[k] for k in ['N','D','L','theta']]; E,A,B=np.exp(th[:3]); al,be=th[3:]
pred=lambda n,d:E+A*n**-al+B*d**-be
printed=lambda n,d:1.69+406.4*n**-.34+410.7*d**-.28
fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained'); ax=axes[0]
sc=ax.scatter(N,L,c=np.log10(6*N*D),s=12,alpha=.6); grid=np.geomspace(N.min(),N.max(),400)
for C in [1e19,1e20,1e21]: ax.plot(grid,pred(grid,C/(6*grid)),label=f'C={C:.0e}')
ax.set(xscale='log',xlabel='parameters N',ylabel='final loss',title='240 runs and fitted iso-compute curves'); ax.legend(fontsize=8); fig.colorbar(sc,ax=ax,label='log10 training FLOP')
for ax,pr,title in zip(axes[1:],[pred,printed],['Refit','Printed parameters']):
 rr=np.log(pr(N,D)/L); ax.scatter(np.log10(N),rr,s=10,alpha=.6); ax.axhline(0,c='k',lw=.7); ax.axhline(rr.mean(),c='tab:red',ls='--',label='mean %.5f'%rr.mean()); ax.set(xlabel='log10 parameters N',ylabel='ln predicted loss - ln observed loss',title=title); ax.legend(fontsize=9)
fig.savefig(e/'part6-fit-residuals-final.png',dpi=160); plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,4.4),layout='constrained')
axes[0].scatter(pred(N,D),L,c=np.log10(6*N*D),s=14,alpha=.7)
lims=[min(L.min(),pred(N,D).min()),max(L.max(),pred(N,D).max())]
axes[0].plot(lims,lims,'k--',lw=1); axes[0].set(xlabel='fitted loss',ylabel='measured loss',title='All 240 fitted runs')
axes[1].hist(np.log(pred(N,D)/L),bins=30,alpha=.6,label='refit')
axes[1].hist(np.log(printed(N,D)/L),bins=30,alpha=.6,label='printed')
axes[1].set(xlabel='ln predicted loss - ln observed loss',ylabel='number of runs'); axes[1].legend()
fig.savefig(e/'part6-diagonal-histogram-final.png',dpi=160); plt.close(fig)
C=np.geomspace(1e18,1e25,200); G=(al*A/(be*B))**(1/(al+be)); nn=G*(C/6)**(be/(al+be)); dd=C/(6*nn)
fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
axes[0].loglog(C,nn,label='refit N* parameters'); axes[0].loglog(C,dd,label='refit D* tokens')
gp=(.34*406.4/(.28*410.7))**(1/(.34+.28)); pn=gp*(C/6)**(.28/(.34+.28)); pd=C/(6*pn)
axes[0].loglog(C,pn,'--',c='tab:blue',label='printed N*'); axes[0].loglog(C,pd,'--',c='tab:orange',label='printed D*')
axes[0].scatter([6*70e9*1.4e12]*2,[70e9,1.4e12],c=['tab:blue','tab:orange'],marker='*',s=90,label='Chinchilla (70B, 1.4T)'); axes[0].set(xlabel='training FLOP C',ylabel='compute-optimal allocation'); axes[0].legend(fontsize=8)
ratio=z['optimal_at_C5.76e23']['D_over_N']; ci=z['optimal_ratio_bootstrap_95pct']
axes[1].semilogx(C,dd/nn,label='refit'); axes[1].errorbar([5.76e23],[ratio],yerr=[[ratio-ci[0]],[ci[1]-ratio]],fmt='o',capsize=4,label='400 bootstrap 95% interval'); axes[1].set(xlabel='training FLOP C',ylabel='optimal tokens / parameter',title='At 5.76e23: 17.92 [7.89, 34.11]'); axes[1].legend(fontsize=9)
axes[1].set_xlim(1e18,1e25)
fig.savefig(e/'part6-optimal-allocation-final.png',dpi=160); plt.close(fig)
(e/'part6-results-audited.json').write_text(json.dumps(z,indent=2)+'\n')
print('Rendered Parts 1,2,4,6 from fixed audited numerical results; originals retained.')
