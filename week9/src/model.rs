use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Factor {
    pub scope: Vec<usize>,
    pub values: Vec<f64>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Model {
    pub kind: String,
    pub dims: Vec<usize>,
    pub factors: Vec<Factor>,
}
pub fn decode(mut i: usize, scope: &[usize], dims: &[usize]) -> Vec<usize> {
    let mut x = vec![0; scope.len()];
    for k in (0..scope.len()).rev() {
        x[k] = i % dims[scope[k]];
        i /= dims[scope[k]];
    }
    x
}
pub fn entries(scope: &[usize], dims: &[usize]) -> usize {
    scope.iter().map(|&v| dims[v]).product()
}
impl Model {
    pub fn parse(text: &str) -> Result<Self> {
        let mut t = text.split_whitespace();
        let kind = t.next().context("missing UAI kind")?.to_string();
        ensure!(kind == "MARKOV" || kind == "BAYES", "unsupported UAI kind");
        fn num<'a>(t: &mut impl Iterator<Item = &'a str>) -> Result<usize> {
            Ok(t.next().context("truncated UAI")?.parse()?)
        }
        let n = num(&mut t)?;
        let dims = (0..n).map(|_| num(&mut t)).collect::<Result<Vec<_>>>()?;
        ensure!(dims.iter().all(|&d| d > 0), "zero dimension");
        let nf = num(&mut t)?;
        ensure!(nf > 0, "no factors");
        let mut scopes = Vec::new();
        for _ in 0..nf {
            let k = num(&mut t)?;
            let s = (0..k).map(|_| num(&mut t)).collect::<Result<Vec<_>>>()?;
            ensure!(s.iter().all(|&v| v < n), "variable out of range");
            let mut u = s.clone();
            u.sort();
            u.dedup();
            ensure!(u.len() == s.len(), "duplicate scope axis");
            scopes.push(s);
        }
        let mut factors = Vec::new();
        for scope in scopes {
            let k = num(&mut t)?;
            ensure!(
                k == entries(&scope, &dims),
                "table count disagrees with scope"
            );
            let values = (0..k)
                .map(|_| Ok(t.next().context("truncated table")?.parse::<f64>()?))
                .collect::<Result<Vec<_>>>()?;
            ensure!(
                values.iter().all(|x| x.is_finite() && *x >= 0.0),
                "invalid potential"
            );
            factors.push(Factor { scope, values });
        }
        ensure!(t.next().is_none(), "trailing UAI data");
        ensure!(
            (0..n).all(|v| factors.iter().any(|f| f.scope.contains(&v))),
            "isolated variables unsupported: each variable must occur in a factor"
        );
        Ok(Self {
            kind,
            dims,
            factors,
        })
    }
    /// Preserve all original tensor axes and select observations with unary
    /// projectors, matching the ASIA diagrams and order-cost representation.
    pub fn with_evidence_caps(&self, e: &BTreeMap<usize, usize>) -> Result<Vec<Factor>> {
        let mut factors = self.factors.clone();
        for (&v, &state) in e {
            ensure!(
                v < self.dims.len() && state < self.dims[v],
                "invalid evidence"
            );
            let mut values = vec![0.0; self.dims[v]];
            values[state] = 1.0;
            factors.push(Factor {
                scope: vec![v],
                values,
            });
        }
        Ok(factors)
    }
    pub fn condition(&self, e: &BTreeMap<usize, usize>) -> Result<Vec<Factor>> {
        for (&v, &s) in e {
            ensure!(v < self.dims.len() && s < self.dims[v], "invalid evidence");
        }
        Ok(self
            .factors
            .iter()
            .map(|f| {
                let scope = f
                    .scope
                    .iter()
                    .copied()
                    .filter(|v| !e.contains_key(v))
                    .collect::<Vec<_>>();
                let mut values = vec![0.0; entries(&scope, &self.dims)];
                for (i, &p) in f.values.iter().enumerate() {
                    let x = decode(i, &f.scope, &self.dims);
                    if f.scope
                        .iter()
                        .zip(&x)
                        .all(|(v, s)| e.get(v).is_none_or(|a| a == s))
                    {
                        let mut j = 0;
                        for (&v, &s) in f.scope.iter().zip(&x) {
                            if !e.contains_key(&v) {
                                j = j * self.dims[v] + s;
                            }
                        }
                        values[j] = p;
                    }
                }
                Factor { scope, values }
            })
            .collect())
    }
    pub fn enumerate(
        &self,
        e: &BTreeMap<usize, usize>,
        query: &[usize],
    ) -> Result<(f64, Vec<f64>)> {
        let scope = (0..self.dims.len()).collect::<Vec<_>>();
        let total = self
            .dims
            .iter()
            .try_fold(1usize, |a, &b| a.checked_mul(b))
            .context("enumeration overflow")?;
        ensure!(total <= 1 << 24, "enumeration restricted to small models");
        let mut p = vec![0.0; entries(query, &self.dims)];
        for i in 0..total {
            let x = decode(i, &scope, &self.dims);
            if e.iter().any(|(&v, &s)| x[v] != s) {
                continue;
            }
            let w = self
                .factors
                .iter()
                .map(|f| {
                    let j = f.scope.iter().fold(0, |a, &v| a * self.dims[v] + x[v]);
                    f.values[j]
                })
                .product::<f64>();
            let q = query.iter().fold(0, |a, &v| a * self.dims[v] + x[v]);
            p[q] += w;
        }
        let z = p.iter().sum::<f64>();
        ensure!(z > 0.0, "zero evidence mass");
        for x in &mut p {
            *x /= z;
        }
        Ok((z, p))
    }
}
pub fn parse_evidence(text: &str) -> Result<BTreeMap<usize, usize>> {
    let a = text
        .split_whitespace()
        .map(str::parse::<usize>)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    ensure!(!a.is_empty(), "empty evidence");
    let (offset, n) = if a.len() == 1 + 2 * a[0] {
        (1, a[0])
    } else if a.len() > 1 && a[0] == 1 && a.len() == 2 + 2 * a[1] {
        (2, a[1])
    } else {
        bail!("invalid UAI evidence format")
    };
    let mut e = BTreeMap::new();
    for i in 0..n {
        ensure!(
            e.insert(a[offset + 2 * i], a[offset + 2 * i + 1]).is_none(),
            "duplicate evidence"
        );
    }
    Ok(e)
}
