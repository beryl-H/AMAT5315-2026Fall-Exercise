"""Independent NumPy checks against saved Rust evidence, then publication-style figures."""
from pathlib import Path
import argparse,csv,json,math,subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from study import run_one,curvature_peak
ROOT=Path(__file__).resolve().parent;ART=ROOT/'artifacts';ART.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130})
def load(p):return json.loads(Path(p).read_text())
def spin_enumeration(l,k,periodic):
    """Direct configuration sum; each positive-direction edge counted separately."""
    n=l*l;states=np.arange(1<<n,dtype=np.uint64);spins=1-2*((states[:,None]>>np.arange(n,dtype=np.uint64))&1).astype(np.int16)
    h=np.zeros(len(states),dtype=np.int32)
    for y in range(l):
        for x in range(l):
            i=y*l+x
            if periodic or x+1<l:h-=spins[:,i]*spins[:,y*l+(x+1)%l]
            if periodic or y+1<l:h-=spins[:,i]*spins[:,((y+1)%l)*l+x]
    a=-k*h;shift=a.max();w=np.exp(a-shift);z=w.sum();mean=np.dot(w,h)/z;var=np.dot(w,(h-mean)**2)/z
    return float((shift+np.log(z))/n),float(k*k*var/n)
def svd_checks():
    p=ROOT/'runs/svd';r=load(p/'run.json');a=np.fromfile(p/'matrix.f64',dtype='<f8').reshape(512,512);s=np.array(load(p/'singular-values.json'))
    checks=[]
    for rank in [16,64,128,256,512]:
        rec=np.fromfile(p/f'rank-{rank}.f64',dtype='<f8').reshape(a.shape);measured=float(np.linalg.norm(a-rec)/np.linalg.norm(a));pred=float(np.linalg.norm(s[rank:])/np.linalg.norm(s))
        assert abs(measured-pred)<1e-8
        if rank==512:assert measured<1e-8
        checks.append({'rank':rank,'measured_error':measured,'tail_error':pred,'difference':abs(measured-pred),'factor_storage_ratio':rank/256})
    tails=np.sqrt(np.r_[np.cumsum(s[::-1]**2)[::-1],0.]/np.sum(s*s));cross=int(np.flatnonzero(tails<=.1)[0]);assert tails[256]>.1
    fig,ax=plt.subplots(1,2,figsize=(10,3.8),layout='constrained');ax[0].plot(np.arange(1,513),s,color='#255c99');ax[0].set(xlabel='Singular-value index',ylabel='Singular value',title='(a) Gaussian singular spectrum')
    ax[1].plot(np.arange(513),tails,label='Saved spectrum tail',color='#255c99');ax[1].scatter([c['rank'] for c in checks],[c['measured_error'] for c in checks],s=35,facecolors='none',edgecolors='#d45d33',label='Saved reconstructions',zorder=4)
    ax[1].axhline(.1,color='#d45d33',ls=':',label='10% error');ax[1].axvline(256,color='gray',ls='--',label='Storage break-even');ax[1].axvspan(256,512,color='gray',alpha=.1);ax[1].annotate(f'First 10% rank: {cross}\nstorage: {cross/256:.3f}×',(cross,.1),xytext=(285,.5),arrowprops={'arrowstyle':'->','color':'gray'})
    ax[1].set(xlabel='Retained rank χ',ylabel='Relative Frobenius error',title='(b) Accuracy and factor storage',xlim=(0,512),ylim=(-.02,1.03));ax[1].legend(fontsize=8);fig.savefig(ART/'svd.png');plt.close(fig)
    result={'checks':checks,'first_rank_10_percent':cross,'storage_ratio_at_crossing':cross/256,'storage_saving_at_10_percent':False,'rng':r['generator']};(ART/'svd-checks.json').write_text(json.dumps(result,indent=2));return result
def small_checks(binary):
    checks=[]
    for method,l,chi,ks,tol in [('boundary',4,8,[0,352,560],1e-8),('trg',2,16,[352],1e-8),('trg',4,8,[352],2e-3)]:
        for ki in ks:
            r=run_one(binary,method,l,chi,ki,namespace=f'checks/{method}-L{l}-chi{chi}-K{ki}');g,cv=spin_enumeration(l,ki/800,method=='trg');err=abs(r['g']-g);assert err<tol
            check={'method':method,'l':l,'chi':chi,'k':ki/800,'g_rust':r['g'],'g_enumeration':g,'g_error':err,'g_tolerance':tol,'path':r['path']}
            if ki==352 and (method=='boundary' or l==2):
                minus=run_one(binary,method,l,chi,350,namespace=f'checks/{method}-L{l}-chi{chi}-K350');plus=run_one(binary,method,l,chi,354,namespace=f'checks/{method}-L{l}-chi{chi}-K354');curv=(352/800)**2*(plus['g']-2*r['g']+minus['g'])/.0025**2
                assert abs(curv-cv)<1e-3;check.update(cv_curvature=curv,cv_variance=cv,cv_error=abs(curv-cv))
            checks.append(check)
    invalid=[['--method','boundary','--l','1','--k','.44','--chi','8'],['--method','trg','--l','3','--k','.44','--chi','8'],['--method','boundary','--l','4','--k','NaN','--chi','8'],['--method','boundary','--l','4','--k','.44','--chi','0'],['--method','trg','--l','4','--k','.44']]
    negative=[]
    for i,args in enumerate(invalid):
        out=ROOT/'runs'/f'invalid-{i}';p=subprocess.run([binary,*args,'--out',str(out)],capture_output=True,text=True);assert p.returncode!=0 and not out.exists();negative.append({'args':args,'returncode':p.returncode})
    existing=ROOT/checks[0]['path'];before=(existing/'run.json').read_bytes();p=subprocess.run([binary,'--method','boundary','--l','4','--k','0','--chi','8','--out',str(existing)],capture_output=True);assert p.returncode!=0 and before==(existing/'run.json').read_bytes();negative.append({'case':'existing output refuses overwrite','returncode':p.returncode})
    return {'physics':checks,'negative_cli':negative}
def boundary_replay(path):
    r=load(ROOT/path/'run.json');e=load(ROOT/path/'boundary-evidence.json');b=[np.array(c['data']).reshape(c['shape']) for c in e['cores']];v=np.ones(1);edge=np.array(e['edge_vector'])
    for i,c in enumerate(b):
        if i<len(b)-1:
            q=c.reshape(-1,c.shape[-1]);assert np.max(np.abs(q.T@q-np.eye(q.shape[1])))<1e-10
        v=np.einsum('a,apb,p->b',v,c,edge)
    assert abs(np.linalg.norm(b[-1])-1)<1e-12
    g=(sum(math.log(x) for x in e['scales'])+math.log(v[0]))/r['sites'];assert abs(g-r['g'])<1e-12
    for t in e['truncations']:
        s=np.array(t['singular_values']);lost=np.sum(s[t['kept']:]**2);assert abs(lost-t['discarded_squared_weight'])<1e-9*max(1,lost)
    return {'path':str(path),'g_replay_error':abs(g-r['g']),'left_isometry':'pass','truncation_records':len(e['truncations'])}
def tensor(t):return np.array(t['data']).reshape(t['shape'])
def trg_replay(path,full=False):
    r=load(ROOT/path/'run.json');e=load(ROOT/path/'trg-evidence.json');a=tensor(e['final_tensor']);z=np.einsum('ijij->',a);ell=sum(st['n_sites_before']*math.log(st['scale_c']) for st in e['steps']);g=(ell+math.log(z))/r['sites'];assert abs(g-r['g'])<1e-12;assert len(e['steps'])==2*int(math.log2(r['l']));assert e['steps'][-1]['n_sites_before']==2
    if full:
        k=r['k'];w=np.array([[np.sqrt(np.cosh(k)),np.sqrt(np.sinh(k))],[np.sqrt(np.cosh(k)),-np.sqrt(np.sinh(k))]]);previous=np.einsum('sr,su,sl,sd->ruld',w,w,w,w)
        for st in e['steps']:
            saved=load(ROOT/path/f"step-{st['step']:04d}.json");s1,s2,s3,s4=[tensor(saved[f's{i}']) for i in range(1,5)];c=float(np.max(np.abs(previous)));assert abs(c-saved['scale_c'])<1e-9*max(1,c)
            # Check both saved split pairs against their own discarded spectral weights.
            norm=previous/c
            for rec,key in [(np.einsum('luq,qrd->ruld',s1,s3),'ma'),(np.einsum('dlq,qur->ruld',s2,s4),'mb')]:
                err=float(np.linalg.norm(rec-norm)**2/np.linalg.norm(norm)**2);assert abs(err-saved[f'discarded_rel_sq_{key}'])<1e-9
            previous=np.einsum('war,abu,lbg,dgw->ruld',s1,s2,s3,s4,optimize='optimal')
        assert np.max(np.abs(previous-a))<1e-9
    return {'path':str(path),'g_replay_error':abs(g-r['g']),'full_round_replay':full,'rounds':len(e['steps'])}
def study_plot(method):
    d=load(ART/f'{method}-study.json');tc=d['onsager_tc'];colors=['#255c99','#e09c36','#38967b'];fig,ax=plt.subplots(1,2,figsize=(11,4.1),layout='constrained');table=[]
    for rec,color in zip(d['records'],colors):
        l=rec['l'];coarse=rec['coarse_chi16'];ref=rec['refined_chi32'];ks=np.array(coarse['curvature_k']);order=np.argsort(1/ks);ax[0].plot((1/ks)[order],np.array(coarse['curvature_cv'])[order],color=color,label=f'L={l}, χ=16');ks=np.array(ref['curvature_k']);order=np.argsort(1/ks);ax[0].plot((1/ks)[order],np.array(ref['curvature_cv'])[order],'o-',ms=3,color=color,lw=1,label=f'L={l}, χ=32 refined')
        ax[1].scatter(1/l,ref['t_peak'],color=color,label=f'L={l}',zorder=4)
        table.append({'L':l,'T_coarse_chi16':coarse['t_peak'],'T_refined_chi16':rec['refined_chi16']['t_peak'],'T_refined_chi32':ref['t_peak'],'T_wide_chi32':rec['wide_chi32']['t_peak'],'cv_peak_chi32':ref['cv_peak'],'rank_shift_percent':100*rec['relative_rank_shift'],'step_shift_percent':100*rec['relative_step_shift']})
    x=np.linspace(0,1/d['records'][0]['l'],100)
    for f,style,label in zip(d['fits'],['-','--'],['all sizes','largest two']):ax[1].plot(x,f['tc']+f['slope']*x,style,color='#555555',label=f"{label}: Tc={f['tc']:.6f}")
    ax[0].axvline(tc,color='#ae4545',ls=':',label='Onsager Tc');ax[1].axhline(tc,color='#ae4545',ls=':',label='Onsager Tc');ax[0].set(xlabel='Temperature T = 1/K',ylabel='Specific heat per spin',title=f"(a) {d['geometry'].capitalize()} heat peaks");ax[1].set(xlabel='Inverse side length 1/L',ylabel='Peak temperature',title='(b) Peak extrapolation (χ=32)')
    ax[0].set_xlim((2.07,2.38) if method=='boundary' else (2.22,2.34));ax[0].legend(fontsize=7);ax[1].legend(fontsize=8);fig.savefig(ART/f'{method}-critical.png');plt.close(fig)
    with (ART/f'{method}-peaks.csv').open('w',newline='') as f:wr=csv.DictWriter(f,fieldnames=list(table[0]));wr.writeheader();wr.writerows(table)
    # Raw tabular g(K), timing, and exact output namespaces for every sampled point.
    rows={r['path']:r for r in d['coarse_runs']+d['refined_runs']}
    with (ART/f'{method}-g.csv').open('w',newline='') as f:
        keys=['l','chi','k','g','log_z','solver_seconds','path'];wr=csv.DictWriter(f,fieldnames=keys);wr.writeheader();wr.writerows({k:r[k] for k in keys} for r in sorted(rows.values(),key=lambda r:(r['l'],r['chi'],r['k'])))
    return d,table
def independent_peak_checks(d):
    """Closed-form parabolic vertices and linear regression, independent of study.py fits."""
    errors=[]
    for record in d['records']:
        l=record['l']
        for key,chi,h,refine,stride in [('coarse_chi16',16,.0025,False,1),('refined_chi16',16,.00125,True,1),('refined_chi32',32,.00125,True,1),('wide_chi32',32,.0025,True,2)]:
            rows=sorted([r for r in d['refined_runs' if refine else 'coarse_runs'] if r['l']==l and r['chi']==chi],key=lambda r:r['k'])[::stride]
            assert len(rows)==(5 if stride==2 else 9 if refine else 41)
            for r in rows:
                saved=load(ROOT/r['path']/'run.json');assert all(saved[k]==r[k] for k in ['l','chi','k','g','solver_seconds'])
            ks=np.array([r['k'] for r in rows]);gs=np.array([r['g'] for r in rows]);cv=(gs[2:]-2*gs[1:-1]+gs[:-2])*ks[1:-1]**2/h**2;j=int(np.argmax(cv));ym,y0,yp=cv[j-1:j+2];kp=ks[j+1]+.5*h*(ym-yp)/(ym-2*y0+yp);tp=1/kp
            error=abs(tp-record[key]['t_peak']);assert error<1e-9;errors.append(error)
    x=np.array([1/r['l'] for r in d['records']]);y=np.array([r['refined_chi32']['t_peak'] for r in d['records']]);fit_errors=[]
    for first,fit in enumerate(d['fits']):
        xx=x[first:];yy=y[first:];slope=np.sum((xx-xx.mean())*(yy-yy.mean()))/np.sum((xx-xx.mean())**2);intercept=yy.mean()-slope*xx.mean();error=abs(intercept-fit['tc']);assert error<1e-12;fit_errors.append(error)
    return {'max_peak_temperature_recompute_error':max(errors),'max_tc_fit_recompute_error':max(fit_errors)}
def main():
    p=argparse.ArgumentParser();p.add_argument('--bin',default='thermo');p.add_argument('--phase',choices=['svd','small','all'],default='all');a=p.parse_args();result={}
    if a.phase in ['svd','all']:result['svd']=svd_checks()
    if a.phase in ['small','all']:result['small_checks']=small_checks(a.bin)
    if a.phase=='all':
        replays=[]
        for method in ['boundary','trg']:
            d,table=study_plot(method);result[method]={'fits':d['fits'],'peaks':table,'unique_compute_seconds':d['unique_compute_seconds'],'independent_peak_checks':independent_peak_checks(d)};maximum=max(r['l'] for r in d['records']);rows=[r for r in d['refined_runs'] if r['l']==maximum]
            for i,r in enumerate(rows):
                replay=boundary_replay(r['path']) if method=='boundary' else trg_replay(r['path'],full=(i==0 or i==9));replays.append(replay)
        result['saved_evidence_replay']=replays
    (ART/f'verification-{a.phase}.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
