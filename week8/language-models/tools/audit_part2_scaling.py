from pathlib import Path
import sys,json,hashlib
import numpy as np
r=Path(__file__).resolve().parents[1];s=r/'artifacts'/'independent-audit'
a=r/'artifacts/part2/formal-02';x=json.loads((a/'results.json').read_text());refs=[4.839399030993707,3.038191771052508,2.0943404017522624,1.862754323955017,2.082535004940124,2.5503980670551534]
for n,ref in enumerate(refs,1):assert abs(x['models'][str(n)]['nats_per_token']-ref)<1e-12
for f,g in [('bigram-P.npy','bigram-probabilities.npy'),('sample-seed531501.npy','independent-bigram-samples.npy'),('sample-twolag-seed531501.npy','independent-bigram-badlag-samples.npy')]:assert np.array_equal(np.load(a/f),np.load(s/'oracle-data'/g)),f
assert x['sampler']['pooled_chi2']['df']==1207 and x['sampler']['pooled_chi2']['p_value']==.1509067499432535
assert x['sampler']['pooled_chi2']['meta']['minimum_expected']>=5
assert all(x['gates'].values())
for fn,h in x['settings']['source_hashes'].items():assert hashlib.sha256((r/'src'/fn).read_bytes()).hexdigest()==h
out={'part2':'PASS','all6_losses_match_independent_oracle_at_1e-12':True,'probability_matrix_and_both_raw_samples_bitwise_match':True,'chi2_df':1207,'p_value':.1509067499432535,'validation_targets':1078995,'all_mandatory_gates':True}
# Independent scaling formula (direct powers; does not import fitter).
b=r/'artifacts/part6/formal-01';z=json.loads((b/'results.json').read_text());raw=np.load(b/'filtered-data-and-fit.npz');N,D,L,th=[raw[k] for k in ['N','D','L','theta']];E,A,B=np.exp(th[:3]);al,be=th[3:];pred=E+A*N**-al+B*D**-be;res=np.log(pred/L);delta=.001;obj=np.mean(np.where(abs(res)<=delta,.5*res**2,delta*abs(res)-.5*delta**2))/delta**2
assert abs(obj-z['best']['fun'])<1e-11
assert abs(res.mean()-z['mean_log_residual_own'])<1e-14
C=5.76e23;optN=((al*A)/(be*B)*(C/6)**be)**(1/(al+be));optD=C/(6*optN);assert np.isclose(optD/optN,z['optimal_at_C5.76e23']['D_over_N'],rtol=1e-12)
# Optimal stationarity and neighboring choices on fixed compute.
assert np.isclose(al*A*optN**-al,be*B*optD**-be,rtol=1e-12)
for scale in [.8,1.2]:assert E+A*(optN*scale)**-al+B*(optD/scale)**-be> E+A*optN**-al+B*optD**-be
fits=[json.loads(t) for t in (b/'multistart.jsonl').read_text().splitlines()];boots=[json.loads(t) for t in (b/'bootstrap.jsonl').read_text().splitlines()];assert len(fits)==243 and len(boots)==400 and [t['seed'] for t in boots]==list(range(400))
rat=[]
for t in boots:
 seed=t['seed'];assert t['resample_indices']==np.random.default_rng(seed).integers(0,240,240).tolist();assert t['success'];q=t['params'];nn=(q['alpha']*q['A']/(q['beta']*q['B'])*(C/6)**q['beta'])**(1/(q['alpha']+q['beta']));rat.append(C/(6*nn**2))
assert np.allclose(np.percentile(rat,[2.5,97.5]),z['optimal_ratio_bootstrap_95pct'],rtol=1e-12)
assert all(z['mandatory_gates'].values())
out['part6']={'independent_objective':float(obj),'mean_log_residual':float(res.mean()),'optimum_ratio':float(optD/optN),'bootstrap_95pct':np.percentile(rat,[2.5,97.5]).tolist(),'all643_fit_statuses_success':all(t['success'] for t in fits+boots),'termination':'relative objective reduction; best projected gradient infinity 2.25e-6, NOT gtol 1e-12 reached','gate_status':'PASS'}
(s/'independent-part2-part6-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
