//! Minimal dynamical state for point atoms moving in 2D.

/// A 2D vector stored as `[x, y]`.
pub type Vec2 = [f64; 2];

/// Complete dynamical state of the simulated atoms.
///
/// Mass is implicitly `1` throughout this crate, matching the Week 2 spec.
#[derive(Clone, Debug, PartialEq)]
pub struct State {
    pub positions: Vec<Vec2>,
    pub velocities: Vec<Vec2>,
}

/// The fixed initial state from the Week 2 spec:
/// atoms at `(0.0, 0.0)` and `(1.2, 0.0)`, both at rest.
pub fn two_atom_initial_state() -> State {
    State {
        positions: vec![[0.0, 0.0], [1.2, 0.0]],
        velocities: vec![[0.0, 0.0], [0.0, 0.0]],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_state_matches_spec() {
        let state = two_atom_initial_state();
        assert_eq!(state.positions, vec![[0.0, 0.0], [1.2, 0.0]]);
        assert_eq!(state.velocities, vec![[0.0, 0.0], [0.0, 0.0]]);
    }
}
