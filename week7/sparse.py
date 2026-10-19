"""Part 1: observation-only basis pursuit in an orthonormal DCT-II basis."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.optimize import linprog

RATES = np.array([.1,.2,.4,.6,.8], dtype=np.float64)


def dct_basis(n):
    if n < 1:
        raise ValueError('size must be positive')
    psi = np.sqrt(2/n)*np.cos(np.pi*(np.arange(n)[:,None]+.5)*np.arange(n)/n)
    psi[:,0] = 1/np.sqrt(n)
    return psi


def basis_pursuit(matrix, measured):
    matrix=np.asarray(matrix,dtype=np.float64)
    measured=np.asarray(measured,dtype=np.float64)
    if matrix.ndim!=2 or measured.shape!=(matrix.shape[0],):
        raise ValueError('incompatible observation matrix and vector')
    n=matrix.shape[1]
    if not len(measured):
        return np.zeros(n,dtype=np.float64)
    result=linprog(np.ones(2*n),A_eq=np.concatenate([matrix,-matrix],axis=1),
                   b_eq=measured,bounds=(0,None),method='highs',
                   options={'primal_feasibility_tolerance':1e-9,
                            'dual_feasibility_tolerance':1e-9})
    if not result.success:
        raise RuntimeError(f'basis pursuit failed: {result.message}')
    return result.x[:n]-result.x[n:]


def run_study(size, sparsity, repeats, seed):
    if not 1<=sparsity<=size or repeats<1:
        raise ValueError('invalid sparsity or repeats')
    psi=dct_basis(size);rng=np.random.default_rng(seed)
    shape=(len(RATES),repeats,size)
    truth=np.empty(shape);masks=np.empty(shape,dtype=bool)
    obs=np.zeros(shape);recon=np.empty(shape)
    coefficients=np.zeros(shape);recovered=np.empty(shape)
    for i,p in enumerate(RATES):
        for j in range(repeats):
            support=rng.choice(size,sparsity,replace=False)
            coefficients[i,j,support]=rng.normal(size=sparsity)
            truth[i,j]=psi@coefficients[i,j]
            masks[i,j]=rng.random(size)<p
            obs[i,j]=np.where(masks[i,j],truth[i,j],0.)
            recovered[i,j]=basis_pursuit(psi[masks[i,j]],obs[i,j,masks[i,j]])
            recon[i,j]=psi@recovered[i,j]
    errors=np.linalg.norm(recon-truth,axis=-1)/np.linalg.norm(truth,axis=-1)
    return dict(truth=truth,masks=masks,observations=obs,reconstruction=recon,
                relative_error=errors,rates=RATES.copy(),
                true_coefficients=coefficients,recovered_coefficients=recovered)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--size',type=int,required=True)
    p.add_argument('--sparsity',type=int,required=True)
    p.add_argument('--repeats',type=int,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if (a.out/'synthetic.npz').exists():
        raise FileExistsError('Use a new output directory; synthetic.npz already exists')
    start=time.perf_counter();data=run_study(a.size,a.sparsity,a.repeats,a.seed)
    np.savez_compressed(a.out/'synthetic.npz',**data)
    counts=(data['relative_error']<.001).sum(axis=1)
    metadata=dict(size=a.size,sparsity=a.sparsity,repeats=a.repeats,seed=a.seed,
                  success_threshold=.001,counts=counts.tolist(),solver='scipy.optimize.linprog/highs',
                  elapsed_seconds=time.perf_counter()-start)
    (a.out/'sparse-run.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('p\tsuccesses\ttrials')
    for rate,count in zip(RATES,counts):print(f'{rate:.1f}\t{count}\t{a.repeats}')


if __name__=='__main__':main()
