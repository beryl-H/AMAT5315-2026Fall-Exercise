use anyhow::{Context, Result, ensure};
use clap::Parser;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde_json::json;
use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};
#[derive(Parser)]
#[command(about = "Week10 zero-field Ising tensor contractions; explicit geometry and rank")]
struct Args {
    #[arg(long,value_parser=["svd","boundary","trg"])]
    method: String,
    #[arg(long)]
    l: Option<usize>,
    #[arg(long)]
    k: Option<f64>,
    #[arg(long)]
    chi: Option<usize>,
    #[arg(long)]
    out: PathBuf,
}
fn binary(path: &Path, data: &[f64]) -> Result<()> {
    let mut f = BufWriter::new(File::create(path)?);
    for x in data {
        f.write_all(&x.to_le_bytes())?;
    }
    f.flush()?;
    Ok(())
}
fn svd_build(out: &Path) -> Result<serde_json::Value> {
    let n = 512;
    let mut rng = ChaCha8Rng::seed_from_u64(2026);
    let mut a = Vec::with_capacity(n * n);
    while a.len() < n * n {
        let u = 1. - rng.random::<f64>();
        let v = rng.random::<f64>();
        let radius = (-2. * u.ln()).sqrt();
        let phase = std::f64::consts::TAU * v;
        a.push(radius * phase.cos());
        a.push(radius * phase.sin());
    }
    binary(&out.join("matrix.f64"), &a)?;
    let t = Instant::now();
    let s = thermo::linalg::svd(&a, n, n);
    let svd_seconds = t.elapsed().as_secs_f64();
    binary(&out.join("u.f64"), &s.u)?;
    binary(&out.join("vt.f64"), &s.vt)?;
    fs::write(out.join("singular-values.json"), serde_json::to_vec(&s.s)?)?;
    let total: f64 = s.s.iter().map(|x| x * x).sum();
    let mut checks = Vec::new();
    for rank in [16, 64, 128, 256, 512] {
        let mut us = vec![0.; n * rank];
        for i in 0..n {
            for j in 0..rank {
                us[i * rank + j] = s.u[i * n + j] * s.s[j];
            }
        }
        let reconstructed = thermo::linalg::mm(&us, &s.vt[..rank * n], n, rank, n);
        binary(&out.join(format!("rank-{rank}.f64")), &reconstructed)?;
        let error = (a
            .iter()
            .zip(&reconstructed)
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            / a.iter().map(|x| x * x).sum::<f64>())
        .sqrt();
        let predicted = (s.s[rank..].iter().map(|x| x * x).sum::<f64>() / total).sqrt();
        checks.push(json!({"rank":rank,"relative_error":error,"spectrum_prediction":predicted,"factor_storage_entries":2*n*rank,"storage_ratio":2.*rank as f64/n as f64}));
    }
    Ok(
        json!({"shape":[n,n],"dtype":"little-endian-f64","order":"C","seed":2026,"generator":"rand_chacha 0.9 ChaCha8Rng + Box-Muller; u=1-random; paired cos/sin","svd_seconds":svd_seconds,"checks":checks,"original_entries":n*n}),
    )
}
fn main() -> Result<()> {
    let args = Args::parse();
    let start = Instant::now();
    if args.method != "svd" {
        let l = args.l.context("--l required")?;
        let k = args.k.context("--k required")?;
        let chi = args.chi.context("--chi required")?;
        ensure!(
            l >= 2 && k.is_finite() && k >= 0. && chi >= 1,
            "require l>=2, finite k>=0, chi>=1"
        );
        ensure!(
            args.method != "trg" || l.is_power_of_two(),
            "TRG requires a power-of-two L"
        );
    } else {
        ensure!(
            args.l.is_none() && args.k.is_none() && args.chi.is_none(),
            "SVD uses prescribed 512x512 seed2026; omit l,k,chi"
        );
    }
    if let Some(parent) = args.out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir(&args.out)
        .context("output must be a new directory; existing evidence is preserved")?;
    let result = if args.method == "svd" {
        svd_build(&args.out)?
    } else {
        let (l, k, chi) = (args.l.unwrap(), args.k.unwrap(), args.chi.unwrap());
        let solve = Instant::now();
        let g = if args.method == "boundary" {
            thermo::boundary::run(l, k, chi, Some(&args.out))?
        } else {
            thermo::trg::run(l, k, chi, Some(&args.out))?
        };
        json!({"model":"H=-sum_edges s_i*s_j; s=+-1; J=kB=1; field=0","method":args.method,"boundary":if args.method=="boundary"{"open"}else{"periodic"},"l":l,"sites":l*l,"bonds":if args.method=="boundary"{2*l*(l-1)}else{2*l*l},"k":k,"chi":chi,"g":g,"log_z":g*(l*l)as f64,"solver_seconds":solve.elapsed().as_secs_f64(),"timing_scope":"solver including full-precision evidence serialization; excludes process startup and compilation","precision":"f64","backend":if cfg!(scipy_blas){"SciPy LP64 OpenBLAS dgesdd/dgemm"}else{"system LAPACK/BLAS dgesdd/dgemm"}})
    };
    fs::write(
        args.out.join("run.json"),
        serde_json::to_vec_pretty(&result)?,
    )?;
    println!(
        "{}",
        json!({"out":args.out,"elapsed_seconds":start.elapsed().as_secs_f64(),"result":result})
    );
    Ok(())
}
