"""Breaks caught: tie loss, clipping, missing restore, mask geometry, diluted MSE."""
import importlib
import numpy as np
import pytest


def api():
    try:
        return importlib.import_module('inpainting')
    except ModuleNotFoundError:
        raise AssertionError('Part 2 inpainting solver is not implemented')


def test_hard_threshold_keeps_all_ties():
    mod=api()
    np.testing.assert_array_equal(mod.hard_threshold(np.array([4.,-4.,1.,-1.]),1),[4,-4,0,0])
    np.testing.assert_array_equal(mod.hard_threshold(np.array([4.,-4.,1.,-1.]),3),[4,-4,1,-1])
    # Floating allowance must preserve an almost equal cutoff.
    np.testing.assert_allclose(mod.hard_threshold(np.array([1.,1.-5e-13,.2]),1),[1,1.-5e-13,0],atol=0)


def test_dct_one_step_matches_independent_cosine_matrices():
    n=4;psi=np.array([[.5 if l==0 else np.sqrt(.5)*np.cos(np.pi*(j+.5)*l/4)
                       for l in range(n)] for j in range(n)])
    obs=np.zeros((n,n));obs[0,0]=3.;obs[3,1]=-2.
    mask=obs!=0
    coeff=psi.T@obs@psi
    cutoff=np.sort(np.abs(coeff).ravel())[-3]
    coeff[np.abs(coeff)<cutoff*(1-1e-12)]=0
    want=psi@coeff@psi.T;want[mask]=obs[mask]
    got=api().dct_inpaint(obs,mask,k=3,iterations=1)
    assert got.dtype==np.float64
    np.testing.assert_allclose(got,want,atol=3e-15)
    assert got[0,0]==3. # observed units must not be clipped


def test_observations_only_and_iteration_zero():
    obs=np.zeros((4,4));obs[1,1]=.3;mask=obs!=0
    np.testing.assert_array_equal(api().dct_inpaint(obs,mask,2,0),obs)
    poisoned=obs.copy();poisoned[0,0]=.9
    with pytest.raises(ValueError,match='zero'):
        api().dct_inpaint(poisoned,mask,2,2)


def test_exact_mask_geometry_and_rng():
    masks=api().make_masks()
    assert masks.shape==(2,10,32,32) and masks.dtype==bool
    np.testing.assert_array_equal((~masks).sum(axis=(-1,-2)),400)
    assert masks[:,:,:2].all() and masks[:,:,30:].all()
    assert masks[:,:,:,:2].all() and masks[:,:,:,30:].all()
    expected=np.ones((32,32),bool);expected[6:26,6:26]=False
    np.testing.assert_array_equal(masks[1],np.broadcast_to(expected,(10,32,32)))
    rng=np.random.default_rng(7001);interior=np.array([32*r+c for r in range(2,30) for c in range(2,30)])
    for i in range(10):
        hidden=rng.choice(interior,400,replace=False)
        np.testing.assert_array_equal(np.sort(hidden),np.flatnonzero(~masks[0,i]))


def test_hidden_mse_excludes_observed_entries():
    truth=np.zeros((2,2));rec=np.full((2,2),99.);rec[0,0]=2.
    mask=np.ones((2,2),bool);mask[0,0]=False
    assert api().hidden_mse(rec,truth,mask)==4.


def test_dct_invalid_budget_is_rejected():
    obs=np.zeros((4,4));mask=np.ones_like(obs,dtype=bool)
    with pytest.raises(ValueError):api().dct_inpaint(obs,mask,0,2)
    with pytest.raises(ValueError):api().dct_inpaint(obs,mask,17,2)
