use thermo::{boundary, linalg, model, trg};
#[test]
fn svd_known_spectrum_and_reconstruction() {
    let a = vec![3., 1., 0., 1., 3., 0., 0., 0., 1.];
    let d = linalg::svd(&a, 3, 3);
    for (x, y) in d.s.iter().zip([4., 2., 1.]) {
        assert!((x - y).abs() < 1e-12);
    }
    let mut us = d.u.clone();
    for i in 0..3 {
        for j in 0..3 {
            us[i * 3 + j] *= d.s[j];
        }
    }
    let r = linalg::mm(&us, &d.vt, 3, 3, 3);
    for (x, y) in r.iter().zip(a) {
        assert!((x - y).abs() < 1e-12);
    }
}
#[test]
fn open_small_against_spins() {
    for k in [0., 0.44, 0.7] {
        let g = boundary::run(4, k, 8, None).unwrap();
        let (exact, _) = model::enumerate(4, k, false);
        assert!((g - exact).abs() < 1e-8, "k={k} g={g} exact={exact}");
    }
}
#[test]
fn periodic_small_against_spins() {
    for (l, chi, tol) in [(2, 16, 1e-8), (4, 8, 2e-3)] {
        let g = trg::run(l, 0.44, chi, None).unwrap();
        let (exact, _) = model::enumerate(l, 0.44, true);
        assert!((g - exact).abs() < tol, "l={l} g={g} exact={exact}");
    }
}
#[test]
fn heat_curvature_vs_variance() {
    for (l, chi, p) in [(4, 8, false), (2, 16, true)] {
        let k = 0.44;
        let h = 0.0025;
        let solve = |x| {
            if p {
                trg::run(l, x, chi, None).unwrap()
            } else {
                boundary::run(l, x, chi, None).unwrap()
            }
        };
        let cv = k * k * (solve(k + h) - 2. * solve(k) + solve(k - h)) / (h * h);
        let (_, exact) = model::enumerate(l, k, p);
        assert!((cv - exact).abs() < 1e-3, "cv={cv} exact={exact}");
    }
}
#[test]
fn independent_periodic_parallel_bond_count() {
    let (g, _) = model::enumerate(2, 0., true);
    assert!((g - 2_f64.ln()).abs() < 1e-14);
    let (g, _) = model::enumerate(2, 0.44, true);
    let z = 2. * (8_f64 * 0.44).exp() + 12. + 2. * (-8_f64 * 0.44).exp();
    assert!((g - z.ln() / 4.).abs() < 1e-14);
}
#[test]
fn zero_coupling_large() {
    for l in [2, 8, 16] {
        assert!((boundary::run(l, 0., 4, None).unwrap() - 2_f64.ln()).abs() < 1e-12);
        assert!((trg::run(l, 0., 4, None).unwrap() - 2_f64.ln()).abs() < 1e-12);
    }
}
