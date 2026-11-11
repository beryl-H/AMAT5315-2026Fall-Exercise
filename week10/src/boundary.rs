//! Algorithms 3 and 4: exact right-canonicalization then left truncation, once per row.
use crate::linalg::{mm, svd};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{fs::File, io::BufWriter, path::Path};
#[derive(Clone, Serialize, Deserialize)]
pub struct Core {
    pub shape: [usize; 3],
    pub data: Vec<f64>,
}
fn compress(b: &mut [Core], chi: usize, row: usize, save: bool, diag: &mut Vec<serde_json::Value>) {
    let n = b.len();
    for i in (1..n).rev() {
        let [a, p, c] = b[i].shape;
        let s = svd(&b[i].data, a, p * c);
        let q = s.s.len();
        let mut us = s.u;
        for j in 0..a {
            for t in 0..q {
                us[j * q + t] *= s.s[t];
            }
        }
        let [la, lp, lb] = b[i - 1].shape;
        assert_eq!(a, lb);
        b[i - 1] = Core {
            shape: [la, lp, q],
            data: mm(&b[i - 1].data, &us, la * lp, a, q),
        };
        b[i] = Core {
            shape: [q, p, c],
            data: s.vt,
        };
    }
    for i in 0..n - 1 {
        let [a, p, c] = b[i].shape;
        let s = svd(&b[i].data, a * p, c);
        let q = chi.min(s.s.len());
        let full = s.s.len();
        if save {
            let total: f64 = s.s.iter().map(|x| x * x).sum();
            let lost: f64 = s.s[q..].iter().map(|x| x * x).sum();
            diag.push(json!({"row":row,"bond":i,"matrix_shape":[a*p,c],"kept":q,"singular_values":s.s,"discarded_squared_weight":lost,"relative_discarded_squared_weight":if total>0.{lost/total}else{0.}}));
        }
        let mut u = vec![0.; a * p * q];
        for j in 0..a * p {
            for t in 0..q {
                u[j * q + t] = s.u[j * full + t];
            }
        }
        let mut sv = vec![0.; q * c];
        for j in 0..q {
            for t in 0..c {
                sv[j * c + t] = s.s[j] * s.vt[j * c + t];
            }
        }
        let [na, np, nc] = b[i + 1].shape;
        assert_eq!(c, na);
        b[i + 1] = Core {
            shape: [q, np, nc],
            data: mm(&sv, &b[i + 1].data, q, c, np * nc),
        };
        b[i] = Core {
            shape: [a, p, q],
            data: u,
        };
    }
}
pub fn run(l: usize, k: f64, chi: usize, out: Option<&Path>) -> Result<f64> {
    ensure!(
        l >= 2 && k.is_finite() && k >= 0. && chi >= 1,
        "invalid boundary arguments"
    );
    let a = crate::model::site(k);
    let edge = 1. / k.cosh().sqrt();
    let mut b = vec![
        Core {
            shape: [1, 2, 1],
            data: vec![edge, 0.]
        };
        l
    ];
    let mut scales = Vec::new();
    let mut diag = Vec::new();
    let mut ell = 0.;
    for y in 0..l {
        let mut grown = Vec::with_capacity(l);
        for (x, core) in b.iter().enumerate() {
            let [la, _, rb] = core.shape;
            let dl = if x == 0 { 1 } else { 2 };
            let dr = if x + 1 == l { 1 } else { 2 };
            let mut v = vec![0.; la * dl * 2 * rb * dr];
            let fac = if x == 0 || x + 1 == l { edge } else { 1. };
            for al in 0..la {
                for le in 0..dl {
                    for d in 0..2 {
                        for be in 0..rb {
                            for re in 0..dr {
                                let val = (0..2)
                                    .map(|u| {
                                        core.data[(al * 2 + u) * rb + be]
                                            * a[((re * 2 + u) * 2 + le) * 2 + d]
                                    })
                                    .sum::<f64>()
                                    * fac;
                                v[(((al * dl + le) * 2 + d) * (rb * dr)) + be * dr + re] = val;
                            }
                        }
                    }
                }
            }
            grown.push(Core {
                shape: [la * dl, 2, rb * dr],
                data: v,
            });
        }
        b = grown;
        compress(&mut b, chi, y, out.is_some(), &mut diag);
        let last = b.last_mut().unwrap();
        let c = last.data.iter().map(|x| x * x).sum::<f64>().sqrt();
        ensure!(c > 0. && c.is_finite(), "invalid boundary scale");
        for v in &mut last.data {
            *v /= c;
        }
        ell += c.ln();
        scales.push(c);
    }
    let mut v = vec![1.];
    for core in &b {
        let [la, _, rb] = core.shape;
        assert_eq!(v.len(), la);
        let mut next = vec![0.; rb];
        for al in 0..la {
            for be in 0..rb {
                next[be] += v[al] * core.data[al * 2 * rb + be] * edge;
            }
        }
        v = next;
    }
    ensure!(
        v.len() == 1 && v[0] > 0. && v[0].is_finite(),
        "invalid boundary closure"
    );
    let log_z = ell + v[0].ln();
    let g = log_z / (l * l) as f64;
    if let Some(dir) = out {
        serde_json::to_writer(
            BufWriter::new(File::create(dir.join("boundary-evidence.json"))?),
            &json!({"cores":b,"edge_vector":[edge,0.],"scales":scales,"log_scale":ell,"bottom_closure":v[0],"log_z":log_z,"g":g,"truncations":diag,"normalization":"last core only; right exact SVD then left truncated SVD"}),
        )?;
    }
    Ok(g)
}
