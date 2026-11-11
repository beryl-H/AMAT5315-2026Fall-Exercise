//! Zero-field Ising model. Independent spin enumeration includes both L=2 torus bonds.
pub fn site(k: f64) -> Vec<f64> {
    let c = k.cosh().sqrt();
    let s = k.sinh().sqrt();
    let w = [[c, s], [c, -s]];
    let mut a = vec![0.; 16];
    for r in 0..2 {
        for u in 0..2 {
            for l in 0..2 {
                for d in 0..2 {
                    a[((r * 2 + u) * 2 + l) * 2 + d] =
                        (0..2).map(|i| w[i][r] * w[i][u] * w[i][l] * w[i][d]).sum();
                }
            }
        }
    }
    a
}
pub fn enumerate(l: usize, k: f64, periodic: bool) -> (f64, f64) {
    assert!((2..=4).contains(&l));
    let n = l * l;
    let mut hist = std::collections::BTreeMap::<i32, u64>::new();
    for state in 0..(1_u64 << n) {
        let spin = |i: usize| if state >> i & 1 == 0 { 1 } else { -1 };
        let mut h = 0;
        for y in 0..l {
            for x in 0..l {
                let s = spin(y * l + x);
                if periodic || x + 1 < l {
                    h -= s * spin(y * l + (x + 1) % l);
                }
                if periodic || y + 1 < l {
                    h -= s * spin(((y + 1) % l) * l + x);
                }
            }
        }
        *hist.entry(h).or_default() += 1;
    }
    let shift = -k * (*hist.keys().next().unwrap() as f64);
    let (mut z, mut e, mut e2) = (0., 0., 0.);
    for (h, c) in hist {
        let h = h as f64;
        let w = c as f64 * (-k * h - shift).exp();
        z += w;
        e += h * w;
        e2 += h * h * w;
    }
    (
        (shift + z.ln()) / n as f64,
        k * k * (e2 / z - (e / z).powi(2)) / n as f64,
    )
}
