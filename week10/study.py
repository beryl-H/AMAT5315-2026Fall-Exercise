"""Execute the sheet's prescribed grids; resume completed runs without overwriting."""
from pathlib import Path
import argparse,concurrent.futures,json,os,subprocess,time
import numpy as np
ROOT=Path(__file__).resolve().parent
def run_one(binary,method,l=None,chi=None,ki=None,namespace=None):
    out=ROOT/'runs'/(namespace or (f'{method}/L{l}/chi{chi}/K{ki:04d}' if ki is not None else method))
    done=out/'run.json'
    if done.exists():
        r=json.loads(done.read_text())
        if method!='svd': assert (r['method'],r['l'],r['chi'],r['k'])==(method,l,chi,ki/800)
        return r|{'path':str(out.relative_to(ROOT))}
    if out.exists(): raise RuntimeError(f'Incomplete prior output preserved; inspect and choose a fresh namespace: {out}')
    cmd=[binary,'--method',method,'--out',str(out)]
    if method!='svd':cmd+=['--l',str(l),'--k',format(ki/800,'.12g'),'--chi',str(chi)]
    result=subprocess.run(cmd,capture_output=True,text=True,env=os.environ|{'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
    if result.returncode:raise RuntimeError(f'{cmd}: {result.stderr[-3000:]}')
    return json.loads(done.read_text())|{'path':str(out.relative_to(ROOT))}
def batch(binary,jobs,workers):
    results=[]; start=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(run_one,binary,*job) for job in jobs]
        for i,f in enumerate(concurrent.futures.as_completed(futures)):
            r=f.result();results.append(r)
            if (i+1)%6==0 or i+1==len(jobs): print(f"{i+1}/{len(jobs)} completed in {time.monotonic()-start:.1f}s; {r.get('path')}",flush=True)
    return results
def curvature_peak(rows,step):
    rows=sorted(rows,key=lambda r:r['k']);k=np.array([r['k'] for r in rows]);g=np.array([r['g'] for r in rows])
    assert np.allclose(np.diff(k),step,rtol=0,atol=1e-12)
    cv=k[1:-1]**2*(g[2:]-2*g[1:-1]+g[:-2])/step**2
    j=int(np.argmax(cv));assert 0<j<len(cv)-1,'Peak lacks a complete three-point curvature stencil'
    x=k[1:-1][j-1:j+2];y=cv[j-1:j+2];p=np.polyfit(x-x[1],y,2);assert p[0]<0
    kp=x[1]-p[1]/(2*p[0]);assert x[0]<=kp<=x[-1]
    return {'k_peak':float(kp),'t_peak':float(1/kp),'cv_peak':float(np.polyval(p,kp-x[1])),'grid_peak_k':float(k[1:-1][j]),'peak_stencil_paths':[r['path'] for r in rows[j-1:j+4]],'curvature_k':k[1:-1].tolist(),'curvature_cv':cv.tolist()}
def study(binary,method,workers):
    sizes=[16,32,64] if method=='boundary' else [64,256,1024]
    coarse=batch(binary,[(method,l,16,ki) for l in sizes for ki in range(320,401,2)],workers)
    centers={l:round(curvature_peak([r for r in coarse if r['l']==l],.0025)['grid_peak_k']*800) for l in sizes}
    refined=batch(binary,[(method,l,chi,ki) for l in sizes for chi in [16,32] for ki in range(centers[l]-4,centers[l]+5)],workers)
    records=[]
    for l in sizes:
        c=curvature_peak([r for r in coarse if r['l']==l],.0025)
        r16=sorted([r for r in refined if r['l']==l and r['chi']==16],key=lambda r:r['k'])
        r32=sorted([r for r in refined if r['l']==l and r['chi']==32],key=lambda r:r['k'])
        a=curvature_peak(r16,.00125);b=curvature_peak(r32,.00125);w=curvature_peak(r32[::2],.0025)
        rank=abs(a['t_peak']/b['t_peak']-1);step=abs(w['t_peak']/b['t_peak']-1)
        assert rank<.01 and step<.01,(method,l,rank,step)
        records.append({'l':l,'coarse_chi16':c,'refined_chi16':a,'refined_chi32':b,'wide_chi32':w,'relative_rank_shift':rank,'relative_step_shift':step})
    xs=np.array([1/r['l'] for r in records]);ys=np.array([r['refined_chi32']['t_peak'] for r in records]);tc=2/np.log(1+np.sqrt(2))
    fits=[]
    for start in [0,1]:
        slope,intercept=np.polyfit(xs[start:],ys[start:],1);err=abs(intercept/tc-1)
        assert err<.01,(method,'Tc outside 1%',intercept)
        fits.append({'sizes':sizes[start:],'tc':float(intercept),'slope':float(slope),'relative_onsager_error':float(err)})
    result={'method':method,'geometry':'open' if method=='boundary' else 'periodic','onsager_tc':float(tc),'records':records,'fits':fits,'coarse_runs':coarse,'refined_runs':refined,'unique_compute_seconds':sum(r['solver_seconds'] for r in {r['path']:r for r in coarse+refined}.values())}
    art=ROOT/'artifacts';art.mkdir(exist_ok=True);(art/f'{method}-study.json').write_text(json.dumps(result,indent=2));
    print(json.dumps({'method':method,'fits':fits,'peak_temperatures':ys.tolist(),'seconds':result['unique_compute_seconds']}),flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--bin',default='thermo');p.add_argument('--phase',choices=['svd','boundary','trg','all'],default='all');p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    if a.phase in ['svd','all']:run_one(a.bin,'svd')
    if a.phase in ['boundary','all']:
        run_one(a.bin,'boundary',64,16,360,'boundary64');study(a.bin,'boundary',a.workers)
    if a.phase in ['trg','all']:
        run_one(a.bin,'trg',1024,16,352,'trg1024');study(a.bin,'trg',a.workers)
if __name__=='__main__':main()
