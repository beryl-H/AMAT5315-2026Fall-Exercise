"""Coordinator's independent scoring/statistical checks of Pi-authored kernels.
No training, model editing, provider call, seed search or threshold changes.
"""
from pathlib import Path
import os,sys,json,hashlib,time,datetime
import numpy as np,jax,jax.numpy as jnp
from scipy.special import logsumexp
from scipy.stats import chi2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path(__file__).resolve().parents[1];s=r/'evidence'/'audits';sys.path.insert(0,str(r));os.chdir(r)
from src.train_driver import load_checkpoint,unflatten_params,load_train_val_streams
from src.transformer_model import forward
from src.mlp_model import forward_all512
from src import sampling as sam
from src.selfconsistency import entropy_zscore,entropy_stats

def write(path,x):path.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def distribution(z,T=1):
 z=np.asarray(z,dtype=np.float64)/T;return np.exp(z-logsumexp(z,axis=-1,keepdims=True))
def check_chi(p,draws):
 # Independent grouping implementation with explicit impossible-support rejection.
 O=np.bincount(draws,minlength=len(p));E=len(draws)*p
 if np.any(O[E==0]>0):return {'p_value':0.,'support_violation':True}
 bins=[[float(O[i]),float(E[i])] for i in np.flatnonzero(E>=5)]
 rare=[float(O[E<5].sum()),float(E[E<5].sum())]
 if rare[1]>0:
  while rare[1]<5 and bins:
   i=int(np.argmin([q[1] for q in bins]));b=bins.pop(i);rare=[rare[0]+b[0],rare[1]+b[1]]
  bins.append(rare)
 df=len(bins)-1
 stat=sum((o-e)**2/e for o,e in bins)
 return {'stat':stat,'df':df,'p_value':float(chi2.sf(stat,df)) if df>0 else 1.,'minimum_expected':min(q[1] for q in bins),'support_violation':False}

def main():
 train,val,meta=load_train_val_streams();e=r/'evidence';a=r/'artifacts/part5/independent-eval-01';a.mkdir(parents=True,exist_ok=False)
 paths={'mlp':r/'artifacts/part3/mlp-formal-01','transformer':r/'artifacts/part5/formal-01'}
 params={};weights={};training={}
 control=json.loads((r/'CONTROL_TRAINING_READY.json').read_text())
 for f,h in control['source_hashes'].items():assert digest(r/'src'/f)==h
 for name,p in paths.items():
  receipt=json.loads((p/'receipt.json').read_text());assert receipt['n_updates']=={'mlp':4000,'transformer':6000}[name]
  assert all(receipt[k] for k in ['completed','reload_bitwise_ok','final_state_finite','all_leaves_float32','optimizer_leaves_float32'])
  logs=[json.loads(t) for t in (p/'train.jsonl').read_text().splitlines()];assert [t['step'] for t in logs]==list(range(1,receipt['n_updates']+1));assert all(t['gnorm']>0 for t in logs)
  for fn,step in [('initial',0),('mid',receipt['n_updates']//2),('final',receipt['n_updates'])]:
   ck=load_checkpoint(p/f'checkpoint-{fn}.npz');assert int(ck['step'])==step;assert all(np.isfinite(v).all() for k,v in ck.items());assert any(k.startswith('m/') for k in ck) and any(k.startswith('v/') for k in ck)
  ck=load_checkpoint(p/'checkpoint-final.npz');params[name]=unflatten_params({k:v for k,v in ck.items() if k.startswith('params/')})['params'];weights[name]=digest(p/'checkpoint-final.npz');training[name]=receipt
 write(a/'settings.json',{'actor':'Codex independent audit using Pi-authored model/sampler kernels','weights_sha256':weights,'source_hashes':control['source_hashes'],'validation_targets':1078995,'chars':2213915,'fixed_sampler_context_rule':'max entropy among first32 prefixes of lengths1..32 of the heldout stream (including initial EOS); chosen before any draws','sampler_draws':100000,'sampler_seed':531505,'story_seed':1848,'entropy_seed':531506,'entropy_sequences':128,'entropy_length':128,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
 result={'weights_sha256':weights,'training':training};gates={}
 # Independent f64 MLP CE from all-512 logits, no training loss reused.
 ml=np.asarray(forward_all512(params['mlp']),dtype=np.float64);mlp_logp=ml-logsumexp(ml,axis=1,keepdims=True);mloss=-mlp_logp[val[:-1],val[1:]];mm=float(mloss.mean());result['mlp']={'nats_per_token':mm,'perplexity':float(np.exp(mm)),'bpc':float(mloss.sum()/np.log(2)/2213915),'targets':len(mloss)}
 gates['MLP_within_001_nats_of_bigram']=abs(mm-3.038191771052508)<=.01
 tr=params['transformer']
 @jax.jit
 def logits_batch(X):return forward(tr,X)[0]
 # Fresh full score. Every target exactly once, f64 logsumexp/accumulation.
 N=len(val)-1;total=0.;possum=np.zeros(64);possumsq=np.zeros(64);poscount=np.zeros(64,int)
 for start in range(0,N,64*64):
  idx=start+np.arange(64)[:,None]*64+np.arange(64)[None,:];mask=idx<N
  X=val[np.minimum(idx,N)];Y=val[np.minimum(idx+1,N)]
  z=np.asarray(logits_batch(jnp.asarray(X,dtype=jnp.int32)),dtype=np.float64);lp=z-logsumexp(z,axis=-1,keepdims=True);loss=-np.take_along_axis(lp,Y[...,None],axis=-1)[...,0]
  total+=float(loss[mask].sum());possum+=(loss*mask).sum(0);possumsq+=(loss**2*mask).sum(0);poscount+=mask.sum(0)
 assert int(poscount.sum())==1078995
 mean=possum/poscount;var=np.maximum((possumsq-possum**2/poscount)/(poscount-1),0);se=np.sqrt(var/poscount)
 result['transformer']={'total_nats':total,'targets':N,'nats_per_token':total/N,'perplexity':float(np.exp(total/N)),'bpc':float(total/np.log(2)/2213915),'position_mean_nats':mean.tolist(),'position_two_SE':(2*se).tolist(),'position_count':poscount.tolist(),'mean_last8':float(mean[-8:].mean())}
 gates['transformer_bpc_below_best_count']=result['transformer']['bpc']<1.3097506075218943;gates['all_positions_context32plus_below_trigram']=bool(np.all(mean[31:]<2.0943404017522624))
 for name in paths:
  old=json.loads((paths[name]/'final-score.json').read_text());new=result[name]['nats_per_token'];assert old['targets']==N;assert abs(old['total_nats']/N-new)<2e-6
 # Eight paired suffix changes: strictly causal zero upper attention and invariance.
 X=np.stack([val[i*64:i*64+64] for i in range(8)]).astype(np.int32);Y=X.copy();Y[:,32:]=(Y[:,32:]+17)%512
 @jax.jit
 def attend(X):return forward(tr,X,causal=True,return_attn=True)
 @jax.jit
 def unmasked(X):return forward(tr,X,causal=False)[0]
 zx,att=attend(jnp.asarray(X));zy,_=attend(jnp.asarray(Y));upper=max(float(np.max(np.abs(np.asarray(q)[:,:,np.triu_indices(64,1)[0],np.triu_indices(64,1)[1]]))) for q in att);delta=float(np.max(abs(np.asarray(zx[:,:32])-np.asarray(zy[:,:32]))));bad=float(np.max(abs(np.asarray(unmasked(jnp.asarray(X)))[:,:32]-np.asarray(unmasked(jnp.asarray(Y)))[:,:32])))
 result['causality']={'pairs':8,'upper_attention_max':upper,'masked_prefix_delta':delta,'unmasked_prefix_delta':bad};gates['causality']=upper==0 and delta==0 and bad>1e-6
 # Fixed context and immutable logit snapshot BEFORE the first sampling draw.
 candidates=np.zeros((32,64),np.int32)
 for i in range(32):candidates[i,:i+1]=val[:i+1]
 zz=np.asarray(logits_batch(jnp.asarray(candidates)))[np.arange(32),np.arange(32)];pps=distribution(zz);ent=-(pps*np.log(pps)).sum(1);selected=int(np.argmax(ent));snap=zz[selected].copy()
 np.savez(a/'sampler-context-before-draws.npz',candidate_logits=zz,selected_logits=snap,context=val[:selected+1],entropies=ent);write(a/'sampler-context-before-draws.json',{'rule':'maximum entropy among first32 fixed held-out prefixes of lengths1..32','selected_index':selected,'context_length':selected+1,'entropies':ent.tolist(),'draws_started':False})
 probs={'temperature08':sam.temperature_dist(snap,.8),'topk20_T08':sam.top_k_dist(snap,20,.8),'topp09_T08':sam.top_p_dist(snap,.9,.8),'minp01_T07':sam.min_p_dist(snap,.1,.7)}
 p08=distribution(snap,.8);order=np.argsort(-p08,kind='stable');k=int(np.searchsorted(np.cumsum(p08[order]),.9))+1
 checks={'temperature08':p08,'topk20_T08':np.where(np.isin(np.arange(512),order[:20]),p08,0),'topp09_T08':np.where(np.isin(np.arange(512),order[:k]),p08,0),'minp01_T07':distribution(snap,.7)}
 checks['minp01_T07']=np.where(checks['minp01_T07']>=.1*checks['minp01_T07'].max(),checks['minp01_T07'],0)
 raw={};stats={};u=np.random.default_rng(531505).random(100000)
 for name,p in probs.items():
  ref=checks[name]/checks[name].sum();assert np.allclose(ref,p,rtol=1e-12,atol=1e-15);draw=sam.sample_dist(p,u);q=check_chi(ref,draw);stats[name]=q;raw[name]=draw;gates['sampler_'+name]=q['p_value']>.001 and q.get('minimum_expected',5)>=5
 cold=sam.sample_dist(sam.temperature_dist(snap,.0001),u);gates['temperature1e-4_argmax']=bool(np.all(cold==np.argmax(snap)));raw['cold']=cold
 bp=sam.top_p_dist_wrong_negative_control(snap,.9,.8);badraw=sam.sample_dist(bp,u);stats['bad_top_p']=check_chi(probs['topp09_T08'],badraw);gates['bad_top_p_rejected']=stats['bad_top_p']['p_value']<.001;raw['bad_top_p']=badraw;np.savez(a/'sampler-raw-draws.npz',**raw);result['samplers']={'selected_context_index':selected,'stats':stats,'argmax_always':gates['temperature1e-4_argmax']}
 # Fixed-length generation. Right padding is causally invisible; crop last64.
 @jax.jit
 def next_logits(X,L):return forward(tr,X)[0][:,L-1,:]
 def one_story(kind,distfn=None):
  rg=np.random.default_rng(1848);hist=[2]
  for _ in range(60):
   if kind=='transformer':
    ctx=hist[-64:];xx=np.zeros((1,64),np.int32);xx[0,:len(ctx)]=ctx;p=distfn(np.asarray(next_logits(jnp.asarray(xx),jnp.int32(len(ctx))))[0])
   elif kind=='uniform':p=np.full(512,1/512)
   else:
    model=models[kind];context=hist[-(model.n-1):] if model.n>1 else [];context=[2]*max(0,model.n-1-len(context))+context;ck=0
    for tok in context:ck=(ck<<9)|int(tok)
    ks=np.full(512,ck,np.uint64);joints=(ks<<np.uint64(9))|np.arange(512,dtype=np.uint64);p=model.logprob_joint(joints,ks)
   hist.append(int(sam.sample_dist(p,[rg.random()])[0]))
  ids=hist[1:];return {'ids':ids,'text':sp.decode([q for q in ids if q!=2]).replace('\ufffd','□')}
 from src.ngrams import NgramCounts
 import sentencepiece as spm
 sp=spm.SentencePieceProcessor(model_file=str(r/'data/tok512.model'));models={name:NgramCounts.fit(train,n,k) for name,n,k in [('unigram',1,.0001),('bigram',2,.01),('trigram',3,.01)]}
 stories={name:one_story(name,lambda z:distribution(z)) for name in ['uniform','unigram','bigram','trigram','transformer']};flavors={name:one_story('transformer',fn) for name,fn in [('temperature08',lambda z:sam.temperature_dist(z,.8)),('topk20_T08',lambda z:sam.top_k_dist(z,20,.8)),('topp09_T08',lambda z:sam.top_p_dist(z,.9,.8)),('minp01_T07',lambda z:sam.min_p_dist(z,.1,.7))]};result['stories']={'seed':1848,'temperature1':stories,'flavors':flavors,'tokens_per_story':60,'EOS':'removed from display only; raw IDs retained'}
 write(a/'scores-and-samplers.json',result|{'mandatory_gates_so_far':gates});print('full scores/causality/samplers/stories saved',flush=True)
 # Entropy experiment: independent sequences are the replication unit.
 entropy={}
 for T in [1.,.7]:
  rg=np.random.default_rng(531506);hist=np.full((128,1),2,np.int32);sur=np.zeros((128,128));H=np.zeros_like(sur)
  for t in range(128):
   ctx=hist[:,-64:];L=ctx.shape[1];xx=np.zeros((128,64),np.int32);xx[:,:L]=ctx;z=np.asarray(next_logits(jnp.asarray(xx),jnp.int32(L)),dtype=np.float64);lp=z-logsumexp(z,axis=1,keepdims=True);p=np.exp(lp);pt=distribution(z,T);cdf=np.cumsum(pt,axis=1);cdf[:,-1]=1;tokens=(rg.random(128)[:,None]>=cdf).sum(1);sur[:,t]=-lp[np.arange(128),tokens];H[:,t]=-(p*lp).sum(1);hist=np.concatenate([hist,tokens[:,None]],axis=1)
  D=(sur-H).mean(1);seD=D.std(ddof=1)/np.sqrt(128);z=float(D.mean()/seD);assert np.isclose(z,entropy_zscore(sur,H),rtol=1e-13);q=entropy_stats(sur,H)|{'z':z,'temperature':T,'seed':531506,'length':128,'mean_D_95pct_normal':[float(D.mean()-1.96*seD),float(D.mean()+1.96*seD)]};entropy[str(T)]=q;np.savez(a/f'entropy-T{T}.npz',tokens=hist[:,1:],surprises=sur,entropies=H,sequence_D=D);print('entropy T',T,'z',z,flush=True)
 result['entropy']=entropy;gates['entropy_T1_absz_le4']=abs(entropy['1.0']['z'])<=4;gates['entropy_T07_absz_gt40']=abs(entropy['0.7']['z'])>40;result['mandatory_gates']=gates;result['all_gates_pass']=all(gates.values());result['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();write(a/'results.json',result);write(e/'neural-results-audited.json',result);write(s/'independent-neural-audit.json',result)
 # Required figures from independent scores and saved training logs.
 p2=json.loads((r/'artifacts/part2/formal-02/results.json').read_text())['models'];names=['uniform','1','2','3','4','5','6'];labels=['uniform','unigram','bigram','trigram','4-gram','5-gram','6-gram'];bp=[p2[n]['bits_per_char'] for n in names]+[result['mlp']['bpc'],result['transformer']['bpc']];fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained');ax.bar(labels+['MLP','Transformer'],bp,color=['0.6']*7+['tab:orange','tab:blue']);ax.set(ylabel='held-out bits / original Unicode character',title='All models: same 1,078,995 targets and 2,213,915 characters');ax.tick_params(axis='x',rotation=25);fig.savefig(e/'part5-model-comparison.png',dpi=160);plt.close(fig)
 fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained');xx=np.arange(1,65);ax.plot(xx,mean,label='Transformer');ax.fill_between(xx,mean-2*se,mean+2*se,alpha=.2,label='two SE over windows');ax.axhline(3.038191771052508,c='tab:orange',ls='--',label='bigram');ax.axhline(2.0943404017522624,c='tab:green',ls='--',label='trigram');ax.set(xlabel='tokens of in-window context read',ylabel='mean held-out nats / token');ax.legend();fig.savefig(e/'part5-context-loss.png',dpi=160);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
 for ax,(name,p) in zip(axes,paths.items()):
  logs=[json.loads(t) for t in (p/'train.jsonl').read_text().splitlines()];ax.plot([t['step'] for t in logs],[t['loss_preupdate'] for t in logs],alpha=.6,lw=.6,label='stochastic training batch');ps=[json.loads(t) for t in (p/'partial-metrics.jsonl').read_text().splitlines()];ax.plot([t['step'] for t in ps],[t['nats_per_token'] for t in ps],'o-',ms=3,label='held-out first128 windows ONLY');ax.set(xlabel='update',ylabel='nats / token',title=name);ax.legend(fontsize=8)
 fig.savefig(e/'neural-training-curves.png',dpi=160);plt.close(fig)
 fig,axes=plt.subplots(2,4,figsize=(12,6),layout='constrained')
 for layer in range(2):
  for head in range(4):axes[layer,head].imshow(np.asarray(att[layer])[0,head],vmin=0,vmax=1,cmap='magma',origin='upper');axes[layer,head].set_title(f'layer {layer+1}, head {head+1}',fontsize=9)
 fig.suptitle('Optional: fixed first held-out window; where heads read, not an explanation');fig.savefig(e/'part5-attention-heads.png',dpi=140);plt.close(fig)
 print(json.dumps({'mlp':result['mlp'],'transformer':{k:v for k,v in result['transformer'].items() if not k.startswith('position')},'samplers':stats,'entropy':entropy,'gates':gates},indent=2),flush=True)
if __name__=='__main__':main()
