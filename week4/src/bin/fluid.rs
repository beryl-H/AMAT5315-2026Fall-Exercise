//! Fluid binary (fluid.design.toml): reads the field JSON from stdin,
//! integrates the vorticity equation with the chosen method, records
//! snapshots.
//!
//! Usage: field taylor-green --n 64 | fluid --method rk4 --nu 0.1 --dt 0.01
//!            --t-end 1 --every 0.1 --out artifacts/taylor-green

#[cfg(test)]
mod tests {
    use super::{parse_field_input, snapshot_schedule};

    #[test]
    fn parse_field_input_reads_the_contract_json() {
        let s = r#"{"case":"taylor-green","n":4,"seed":0,"k_band":[0,1],"u":[0,0,0,0,0,0.5,0,0,0,0,0,0,0,0,0,0],"v":[0,0,0,0,0,-0.5,0,0,0,0,0,0,0,0,0,0]}"#;
        let f = parse_field_input(s).unwrap();
        assert_eq!(f.case, "taylor-green");
        assert_eq!(f.n, 4);
        assert_eq!(f.seed, 0);
        assert_eq!(f.k_band, serde_json::json!([0, 1]));
        assert_eq!(f.u[5], 0.5);
        assert_eq!(f.v[5], -0.5);
    }

    #[test]
    fn snapshot_schedule_starts_at_zero_and_steps_by_s() {
        assert_eq!(snapshot_schedule(7, 3), vec![0, 3, 6]);
        assert_eq!(snapshot_schedule(9, 4), vec![0, 4, 8]);
        assert_eq!(snapshot_schedule(3, 5), vec![0]);
    }
}
