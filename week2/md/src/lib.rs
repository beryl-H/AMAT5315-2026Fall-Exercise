pub fn greeting() -> &'static str {
    "Hello, world!"
}

/// Lennard-Jones pair energy in reduced units (epsilon = sigma = 1):
/// `4 * (r^-12 - r^-6)`.
pub fn lj_energy(r: f64) -> f64 {
    let inv_r6 = r.powi(-6);
    4.0 * inv_r6 * (inv_r6 - 1.0)
}

/// Lennard-Jones pair force (placeholder until the next step).
pub fn lj_force(_r: f64) -> f64 {
    todo!("lj_force is implemented in the next step")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greeting_says_hello_world() {
        assert_eq!(greeting(), "Hello, world!");
    }
}
