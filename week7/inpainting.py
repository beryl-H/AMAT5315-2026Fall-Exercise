"""Observation-only DCT reconstruction and evaluation utilities."""
import numpy as np
from scipy.fft import dctn, idctn

TEST_IDS=np.array([3,2,1,18,4,8,11,0,61,7])
VALIDATION_IDS=np.array([10,5,35,30,6,15,21,17,84,9])


def validate_observations(observations,mask):
    y=np.asarray(observations);m=np.asarray(mask)
    if y.shape!=m.shape or y.ndim<2 or m.dtype!=np.bool_:
        raise ValueError('observations and boolean mask must have identical image shapes')
    if not np.isfinite(y).all():raise ValueError('observations must be finite')
    if np.any(y[~m]!=0):raise ValueError('observations must be zero at missing pixels')
    return y,m


def hard_threshold(coefficients,k):
    c=np.asarray(coefficients,dtype=np.float64)
    if not 1<=k<=c.size:raise ValueError('k must be between 1 and image size')
    cutoff=np.partition(np.abs(c).ravel(),c.size-k)[c.size-k]
    return np.where(np.abs(c)>=cutoff*(1-1e-12),c,0.)


def dct_inpaint(observations,mask,k,iterations):
    y,m=validate_observations(observations,mask)
    if y.ndim!=2:raise ValueError('DCT solver expects one 2D image')
    if iterations<0:raise ValueError('iterations must be nonnegative')
    if not 1<=k<=y.size:raise ValueError('invalid coefficient budget')
    y=np.asarray(y,dtype=np.float64);x=y.copy()
    for _ in range(iterations):
        c=hard_threshold(dctn(x,type=2,norm='ortho'),k)
        x=np.where(m,y,idctn(c,type=2,norm='ortho'))
    return x


def make_masks():
    masks=np.ones((2,10,32,32),dtype=bool)
    interior=np.arange(1024).reshape(32,32)[2:30,2:30].ravel()
    rng=np.random.default_rng(7001)
    for i in range(10):masks[0,i].reshape(-1)[rng.choice(interior,400,replace=False)]=False
    masks[1,:,6:26,6:26]=False
    return masks


def hidden_mse(reconstruction,truth,mask):
    delta=np.asarray(reconstruction,dtype=np.float64)-np.asarray(truth,dtype=np.float64)
    hidden=~np.asarray(mask,dtype=bool)
    count=hidden.sum(axis=(-1,-2))
    if np.any(count==0):raise ValueError('hidden MSE requires at least one missing pixel')
    return np.sum(np.where(hidden,delta**2,0.),axis=(-1,-2))/count


def load_images(path):
    with np.load(path,allow_pickle=False) as z:data={k:z[k] for k in z.files}
    for key in ['test','validation']:
        a=data[key]
        if a.shape!=(10,32,32) or a.dtype!=np.float32 or not np.isfinite(a).all() or a.min()<-1 or a.max()>1:
            raise ValueError('image archive does not match the selected local MNIST lecture')
    for key,expected in [('test_ids',TEST_IDS),('validation_ids',VALIDATION_IDS),('labels',np.arange(10))]:
        if not np.array_equal(data[key],expected):raise ValueError(f'incorrect {key}')
    return data
