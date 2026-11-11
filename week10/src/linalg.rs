//! Thin LAPACK SVD and BLAS GEMM. Public arrays are always C-order f64.
pub struct Svd {
    pub u: Vec<f64>,
    pub s: Vec<f64>,
    pub vt: Vec<f64>,
    pub m: usize,
    pub n: usize,
}
unsafe extern "C" {
    #[cfg_attr(scipy_blas, link_name = "scipy_dgesdd_")]
    fn dgesdd_(
        job: *const u8,
        m: *const i32,
        n: *const i32,
        a: *mut f64,
        lda: *const i32,
        s: *mut f64,
        u: *mut f64,
        ldu: *const i32,
        vt: *mut f64,
        ldvt: *const i32,
        work: *mut f64,
        lwork: *const i32,
        iwork: *mut i32,
        info: *mut i32,
    );
    #[cfg_attr(scipy_blas, link_name = "scipy_dgemm_")]
    fn dgemm_(
        ta: *const u8,
        tb: *const u8,
        m: *const i32,
        n: *const i32,
        k: *const i32,
        alpha: *const f64,
        a: *const f64,
        lda: *const i32,
        b: *const f64,
        ldb: *const i32,
        beta: *const f64,
        c: *mut f64,
        ldc: *const i32,
    );
}
pub fn svd(a: &[f64], m: usize, n: usize) -> Svd {
    assert_eq!(a.len(), m * n);
    assert!(m > 0 && n > 0 && a.iter().all(|x| x.is_finite()));
    let (mi, ni) = (i32::try_from(m).unwrap(), i32::try_from(n).unwrap());
    let q = m.min(n);
    let qi = q as i32;
    let mut col = vec![0.; m * n];
    for i in 0..m {
        for j in 0..n {
            col[j * m + i] = a[i * n + j];
        }
    }
    let (mut s, mut u, mut vt) = (vec![0.; q], vec![0.; m * q], vec![0.; q * n]);
    let mut iw = vec![0; 8 * q];
    let mut info = 0;
    let mut query = [0.];
    unsafe {
        dgesdd_(
            b"S".as_ptr(),
            &mi,
            &ni,
            col.as_mut_ptr(),
            &mi,
            s.as_mut_ptr(),
            u.as_mut_ptr(),
            &mi,
            vt.as_mut_ptr(),
            &qi,
            query.as_mut_ptr(),
            &-1,
            iw.as_mut_ptr(),
            &mut info,
        );
    }
    assert_eq!(info, 0, "LAPACK workspace query");
    let lw = query[0].ceil() as i32;
    let mut work = vec![0.; lw as usize];
    unsafe {
        dgesdd_(
            b"S".as_ptr(),
            &mi,
            &ni,
            col.as_mut_ptr(),
            &mi,
            s.as_mut_ptr(),
            u.as_mut_ptr(),
            &mi,
            vt.as_mut_ptr(),
            &qi,
            work.as_mut_ptr(),
            &lw,
            iw.as_mut_ptr(),
            &mut info,
        );
    }
    assert_eq!(info, 0, "LAPACK SVD failed");
    let mut ur = vec![0.; m * q];
    let mut vr = vec![0.; q * n];
    for i in 0..m {
        for j in 0..q {
            ur[i * q + j] = u[j * m + i];
        }
    }
    for i in 0..q {
        for j in 0..n {
            vr[i * n + j] = vt[j * q + i];
        }
    }
    Svd {
        u: ur,
        s,
        vt: vr,
        m,
        n,
    }
}
pub fn mm(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    let (mi, ki, ni) = (m as i32, k as i32, n as i32);
    let mut c = vec![0.; m * n];
    // Column-major C^T = B^T A^T implements row-major C = A B.
    unsafe {
        dgemm_(
            b"N".as_ptr(),
            b"N".as_ptr(),
            &ni,
            &mi,
            &ki,
            &1.,
            b.as_ptr(),
            &ni,
            a.as_ptr(),
            &ki,
            &0.,
            c.as_mut_ptr(),
            &ni,
        );
    }
    c
}
