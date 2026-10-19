"""Evaluate fixed reconstruction priors; complete truth stays outside solvers."""
import argparse,json,hashlib
import time,platform
from pathlib import Path
import numpy as np
from inpainting import load_images,make_masks,dct_inpaint,hidden_mse


def fixed_study(images):
    truth=images['test'].copy();masks=make_masks()
    observations=np.where(masks,truth[None],np.float32(0))
    fixed=np.stack([[dct_inpaint(observations[m,i],masks[m,i],20,100) for i in range(10)] for m in range(2)])
    return dict(truth=truth,masks=masks,observations=observations,fixed=fixed,
                fixed_mse=hidden_mse(fixed,truth[None],masks))


def diffusion_study(data,checkpoint,out):
    from repaint import load_prior,repaint_sample
    import jax
    prior=load_prior(checkpoint)
    y=data['observations'].reshape(20,32,32,1)
    masks=data['masks'].reshape(20,32,32,1)
    outputs=[];timings=[]
    for seed in [7201,7202,7203,7204]:
        started=time.perf_counter()
        samples,trace=repaint_sample(prior,y,masks,100,3,seed)
        timings.append(time.perf_counter()-started)
        outputs.append(samples[:,:,:,0].reshape(2,10,32,32))
        seed_file=out/f'diffusion-seed-{seed}.npz'
        if seed_file.exists():raise FileExistsError('Seed result already exists; use a new output directory')
        np.savez_compressed(seed_file,samples=samples)
        if seed==7201:np.savez_compressed(out/'trace.npz',**trace)
        print(f'seed {seed} completed, seconds {timings[-1]:.3f}',flush=True)
    samples=np.stack(outputs,axis=2).astype(np.float32)
    sample64=samples.astype(np.float64);mean=sample64.mean(axis=2)
    data.update(diffusion=samples,
                diffusion_mse=hidden_mse(sample64,data['truth'][None,:,None],data['masks'][:,:,None]),
                mean_completion_mse=hidden_mse(mean,data['truth'][None],data['masks']),
                diversity=(np.where(~data['masks'],sample64.var(axis=2,ddof=0),0).sum((-1,-2))/400))
    print('mask\tDCT\tmean_sample_MSE\tmean_completion_MSE\tvariance',flush=True)
    for m,name in enumerate(['random','central-hole']):
        print(name,*[f'{v:.10f}' for v in [data['fixed_mse'][m].mean(),data['diffusion_mse'][m].mean(),data['mean_completion_mse'][m].mean(),data['diversity'][m].mean()]],sep='\t',flush=True)
    residual=data['diffusion_mse'].mean(2)-data['mean_completion_mse']-data['diversity']
    print('max_identity_residual',float(np.abs(residual).max()),flush=True)
    return dict(steps=100,resamples=3,seeds=[7201,7202,7203,7204],checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                trace=dict(mask_index=0,image_index=0,sample_index=0,seed=7201,batch_shape=[20,32,32,1],batch_index=0),
                runtime=dict(python=platform.python_version(),jax=jax.__version__,numpy=np.__version__,backend=jax.default_backend(),seconds_per_seed=timings))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path)
    a=p.parse_args();images=load_images(a.input)
    if (a.out/'evidence.npz').exists():
        if a.checkpoint is None:raise FileExistsError('Use a new output directory')
        with np.load(a.out/'evidence.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
        if 'diffusion' in data:raise FileExistsError('Diffusion study already exists; use a new output directory')
        metadata=json.loads((a.out/'run.json').read_text())
        if hashlib.sha256(a.input.read_bytes()).hexdigest()!=metadata['input_sha256']:raise ValueError('Image archive changed since DCT study')
        before=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
        for name in ['evidence.npz','run.json']:
            backup=a.out/('dct-'+name)
            if backup.exists():raise FileExistsError('DCT backup already exists; preserve previous attempt and use a new output path')
            backup.write_bytes((a.out/name).read_bytes())
        metadata.update(diffusion_study(data,a.checkpoint,a.out))
        assert before==hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
        np.savez_compressed(a.out/'evidence.npz',**data)
        (a.out/'run.json').write_text(json.dumps(metadata,indent=2)+'\n')
        return
    data=fixed_study(images)
    case=a.input.parent/'case.npz'
    if case.exists():raise FileExistsError('case.npz already exists')
    np.savez_compressed(case,observations=data['observations'][0,0],mask=data['masks'][0,0])
    a.out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.out/'evidence.npz',**data)
    metadata=dict(schema='mnist-inpainting-v1',n=32,mask_names=['random','central-hole'],mask_seed=7001,
                  image_ids=images['test_ids'].tolist(),validation_ids=images['validation_ids'].tolist(),
                  labels=images['labels'].tolist(),fixed_k=20,fixed_iterations=100,
                  input_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest())
    (a.out/'run.json').write_text(json.dumps(metadata,indent=2)+'\n')
    if a.checkpoint is not None:
        metadata.update(diffusion_study(data,a.checkpoint,a.out))
        np.savez_compressed(a.out/'evidence.npz',**data)
        (a.out/'run.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('mask\tmean_missing_mse')
    for name,mse in zip(metadata['mask_names'],data['fixed_mse'].mean(axis=1)):print(f'{name}\t{mse:.10f}')
    print('maximum_observed_error',float(np.max(np.abs(data['fixed'][data['masks']]-data['observations'][data['masks']]))))


if __name__=='__main__':main()
