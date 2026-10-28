from pathlib import Path
import json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path(__file__).resolve().parents[1];s=r/'evidence'/'audits';e=r/'evidence'
a=r/'artifacts/part3/memory-formal-01';z=json.loads((a/'sweep-summary.json').read_text());records=[json.loads(t) for t in (a/'progress.jsonl').read_text().splitlines()]
assert len(records)==len({tuple(x['case'][k] for k in ['d','M','seed','store']) for x in records})==630
stores=['outer','ls','relu1','relu2','relu3'];ds=[16,32,64];ms=[2**i for i in range(3,17)];curves={}
for d in ds:
 curves[d]={}
 for st in stores:
  curves[d][st]=[]
  for M in ms:
   med=[x['aggregate']['median'] for x in records if x['case']['d']==d and x['case']['M']==M and x['case']['store']==st]
   assert len(med)==3
   v=float(np.median(med));assert v==z['curves_median_of_3'][str(d)][st][str(M)];curves[d][st].append(v)
def crossing(vals):
 for i in range(len(ms)-1):
  if vals[i]<=1<=vals[i+1]: return float(np.exp(np.interp(0,np.log(vals[i:i+2]),np.log(ms[i:i+2]))))
 return None
caps={d:{st:crossing(curves[d][st]) for st in stores} for d in ds}
for d in ds:
 for st in stores:assert np.isclose(caps[d][st],z['capacity'][str(d)][st],rtol=1e-12)
slopes={st:float(np.polyfit(np.log(ds),np.log([caps[d][st] for d in ds]),1)[0]) for st in stores}
lsmax=max(x['aggregate']['median'] for x in records if x['case']['d']==64 and x['case']['M']<=64 and x['case']['store']=='ls')
gates={'outer_d64_capacity_50_85':50<=caps[64]['outer']<=85,'LS_d64_capacity_100_170':100<=caps[64]['ls']<=170,'LS_d64_R_below_1e-10_up_to64':lsmax<1e-10}
for st in ['outer','ls','relu1']:gates[st+'_capacity_slope_09_11']=abs(slopes[st]-1)<=.1
gates['relu2_capacity_slope_185_215']=abs(slopes['relu2']-2)<=.15;gates['relu3_capacity_slope_gt25']=slopes['relu3']>2.5
# Limited exact diagnostic: independent dense summation on already-defined small cases.
# Recreates the fixed draws, never runs a new sweep or changes seeds.
checks=[]
for d,M in [(16,16),(16,32),(32,32),(32,64),(64,64),(64,128)]:
 for seed in [101,102,103]:
  rg=np.random.default_rng(seed);K=rg.normal(size=(65536,d));K/=np.linalg.norm(K,axis=1,keepdims=True);V=rg.normal(size=(65536,d));V/=np.linalg.norm(V,axis=1,keepdims=True);K,V=K[:M],V[:M]
  H=K@K.T;np.fill_diagonal(H,0);leak=H@V;median=float(np.median(np.linalg.norm(leak,axis=1)))
  saved=next(x['aggregate']['median'] for x in records if x['case']==dict(d=d,M=M,seed=seed,store='outer'))
  assert abs(median-saved)<1e-12;checks.append(dict(d=d,M=M,seed=seed,independent_median=median,saved_median=saved))
individual={}
for seed in [101,102,103]:
 vals=[]
 for d in ds:
  vv=[next(x['aggregate']['median'] for x in records if x['case']==dict(d=d,M=M,seed=seed,store='outer')) for M in ms];vals.append(crossing(vv))
 individual[str(seed)]={'capacities':vals,'slope':float(np.polyfit(np.log(ds),np.log(vals),1)[0])}
out={'status':'PASS' if all(gates.values()) else 'GATE_NOT_MET','capacities':caps,'capacity_slopes':slopes,'LS_max_median_R_d64_Mle64':lsmax,'mandatory_gates':gates,'independent_small_case_checks':checks,'outer_drawwise_diagnostic':individual,'finding':'Outer capacity slope is the measured median-of-three result. Independent direct interference sums agree on 18 small cases; no implementation error demonstrated by this diagnostic. Sampling variability is a hypothesis, not a proven cause. Original fixed seeds, gate and all 630 outcomes retained.','training_repeated':False}
(s/'independent-memory-audit.json').write_text(json.dumps(out,indent=2)+'\n');(e/'part3-memory-audited.json').write_text(json.dumps(out,indent=2)+'\n');(e/'part3-memory-curves.json').write_text(json.dumps(z,indent=2)+'\n')
fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
for st in stores:
 axes[0].loglog(ms,curves[64][st],'o-',ms=3,label=st);axes[1].loglog(ds,[caps[d][st] for d in ds],'o-',label=st+': slope %.3f'%slopes[st])
mm=np.geomspace(8,65536,300);axes[0].loglog(mm,np.sqrt((mm-1)/64),'--',c='0.3',alpha=.6,label='outer RMS expectation sqrt((M-1)/d)');axes[0].axhline(1,c='k',ls=':',lw=.8);axes[0].set(xlabel='stored memories M',ylabel='median own-normalized ISR',ylim=(.002,100),title='d=64; all keys queried; median of 3 draws');axes[0].legend(fontsize=8)
axes[1].set(xlabel='key/value dimension d',ylabel='capacity at ISR=1',title='Measured log-log crossings; no seed changes');axes[1].legend(fontsize=8);fig.savefig(e/'part3-memory-final.png',dpi=160)
print(json.dumps({k:v for k,v in out.items() if k not in ['independent_small_case_checks']},indent=2))
