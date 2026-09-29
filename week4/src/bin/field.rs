//! Field initial-condition binary (field.design.toml): subcommands
//! `taylor-green` and `random` write one JSON object
//! {case, n, seed, k_band, u, v} to stdout; fluid reads it on stdin.
//!
//! Usage: field taylor-green --n 64 --nu 0.1 --t 1
//!        field random --n 128 --seed 2026 --k-min 2 --k-max 6

#[cfg(test)]
mod tests {
    use super::{random_output, taylor_green_output};

    #[test]
    fn taylor_green_output_matches_the_contract() {
        let s = taylor_green_output(8, 0.0, None).unwrap();
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["case"], "taylor-green");
        assert_eq!(v["n"], 8);
        assert_eq!(v["seed"], 0);
        assert_eq!(v["k_band"], serde_json::json!([0, 2]));
        // u = cos(x) sin(y) at grid point (i=1, j=1): x = y = pi/4 -> 0.5
        let u = v["u"].as_array().unwrap();
        assert!((u[1 * 8 + 1].as_f64().unwrap() - 0.5).abs() < 1e-12);
        let vv = v["v"].as_array().unwrap();
        assert!((vv[1 * 8 + 1].as_f64().unwrap() + 0.5).abs() < 1e-12);
    }

    #[test]
    fn nu_is_required_when_t_is_positive() {
        assert!(taylor_green_output(8, 0.0, None).is_ok());
        assert!(taylor_green_output(8, 1.0, None).is_err());
        assert!(taylor_green_output(8, 1.0, Some(0.1)).is_ok());
    }

    #[test]
    fn random_output_delivers_half_unit_energy() {
        let s = random_output(32, 7, 2.0, 6.0).unwrap();
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["case"], "random");
        assert_eq!(v["seed"], 7);
        assert_eq!(v["k_band"], serde_json::json!([2.0, 6.0]));
        let u: Vec<f64> = v["u"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        let vv: Vec<f64> = v["v"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        let omega = advdiff::fluid::vorticity_from_velocity(32, &u, &vv);
        assert!((advdiff::fluid::energy(32, &omega) - 0.5).abs() < 1e-12);
    }
}
