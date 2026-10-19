"""Evaluate fixed reconstruction priors; complete truth stays outside solvers."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from inpainting import load_images,make_masks,dct_inpaint,hidden_mse


def fixed_study(images):
    truth=images['test'].copy();masks=make_masks()
    observations=np.where(masks,truth[None],np.float32(0))
    fixed=np.stack([[dct_inpaint(observations[m,i],masks[m,i],20,100) for i in range(10)] for m in range(2)])
    return dict(truth=truth,masks=masks,observations=observations,fixed=fixed,
                fixed_mse=hidden_mse(fixed,truth[None],masks))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();images=load_images(a.input)
    if (a.out/'evidence.npz').exists():raise FileExistsError('Use a new output directory')
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
    print('mask\tmean_missing_mse')
    for name,mse in zip(metadata['mask_names'],data['fixed_mse'].mean(axis=1)):print(f'{name}\t{mse:.10f}')
    print('maximum_observed_error',float(np.max(np.abs(data['fixed'][data['masks']]-data['observations'][data['masks']]))))


if __name__=='__main__':main()
