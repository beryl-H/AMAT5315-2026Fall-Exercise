fn main() {
    println!("cargo:rustc-check-cfg=cfg(scipy_blas)");
    println!("cargo:rerun-if-env-changed=THERMO_OPENBLAS");
    if let Ok(path) = std::env::var("THERMO_OPENBLAS") {
        let dir = std::path::Path::new(&path).parent().unwrap();
        println!("cargo:rustc-cfg=scipy_blas");
        println!("cargo:rustc-link-arg={path}");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
    } else {
        println!("cargo:rustc-link-arg=-l:liblapack.so.3");
        println!("cargo:rustc-link-arg=-l:libblas.so.3");
    }
}
