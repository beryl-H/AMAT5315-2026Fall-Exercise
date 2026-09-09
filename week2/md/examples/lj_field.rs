//! Print the Lennard-Jones radial field used by `week2/plot_lj_field.py`.
//!
//! Outputs three whitespace-separated columns: radius, pair energy, and
//! scalar force in reduced units, for radii from 0.05 to 5.0.

fn main() {
    let samples = 1000;
    for i in 0..=samples {
        let r = 0.05 + (5.0 - 0.05) * i as f64 / samples as f64;
        println!("{r:.10e} {} {}", md::lj_energy(r), md::lj_force(r));
    }
}
