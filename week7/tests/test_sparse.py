"""Breaks caught: transposed DCT, non-L1 recovery, wrong RNG order, leaked truth."""
import importlib
import numpy as np
from scipy.fft import idct


def api():
    try:
        return importlib.import_module('sparse')
    except ModuleNotFoundError:
        raise AssertionError('Part 1 sparse solver is not implemented')


def test_dct_synthesis_orientation_and_normalization():
    psi = api().dct_basis(16)
    np.testing.assert_allclose(psi.T @ psi, np.eye(16), atol=4e-15)
    np.testing.assert_allclose(psi, idct(np.eye(16), type=2, norm='ortho', axis=0), atol=2e-15)
    np.testing.assert_allclose(psi[:, 0], .25, atol=1e-16)


def test_basis_pursuit_signed_l1_optimum():
    # Unique L1 optimum is (0,0,-1), rather than a minimum-L2 fit.
    A = np.array([[1.,0.,1.],[0.,1.,1.]])
    a = api().basis_pursuit(A, np.array([-1.,-1.]))
    np.testing.assert_allclose(a, [0,0,-1], atol=1e-8)
    np.testing.assert_allclose(A @ a, [-1,-1], atol=1e-8)


def test_no_measurements_minimizes_at_zero():
    np.testing.assert_array_equal(api().basis_pursuit(np.empty((0,8)), np.empty(0)), np.zeros(8))


def test_study_rng_order_shapes_and_missing_values():
    data = api().run_study(16, 3, 2, 7101)
    assert data['truth'].shape == (5,2,16)
    assert data['masks'].dtype == np.bool_
    assert data['truth'].dtype == np.float64
    rng = np.random.default_rng(7101)
    # Independent trigonometric synthesis, not the production helper.
    psi = np.array([[np.sqrt(1/16) if l==0 else np.sqrt(2/16)*np.cos(np.pi*(j+.5)*l/16)
                     for l in range(16)] for j in range(16)])
    for pidx,p in enumerate([.1,.2,.4,.6,.8]):
        for trial in range(2):
            support=rng.choice(16,3,replace=False)
            coef=np.zeros(16);coef[support]=rng.normal(size=3)
            mask=rng.random(16)<p
            np.testing.assert_allclose(data['truth'][pidx,trial],psi@coef,atol=1e-14)
            np.testing.assert_array_equal(data['masks'][pidx,trial],mask)
    assert np.count_nonzero(data['observations'][~data['masks']]) == 0
    np.testing.assert_allclose(data['reconstruction'][data['masks']], data['observations'][data['masks']],atol=1e-7)
