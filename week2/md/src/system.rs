//! Periodic geometry: triangular lattice, box, minimum image, wrapping.

use crate::state::State;
use crate::Vec2;

/// Triangular-lattice constants (a, h) at density `rho`.
/// a = sqrt(2/(sqrt(3)*rho)), h = sqrt(3)/2 * a  [Course Requirement].
pub fn lattice_constants(rho: f64) -> (f64, f64) {
    let a = (2.0 / (3f64.sqrt() * rho)).sqrt();
    let h = 3f64.sqrt() / 2.0 * a;
    (a, h)
}

/// Side length of the sqrt(N) x sqrt(N) lattice, or None if N is not a
/// perfect square. [Suggestion] non-square N is rejected.
pub fn side(n: usize) -> Option<usize> {
    let s = (n as f64).sqrt() as usize;
    if s * s == n { Some(s) } else { None }
}

/// Rectangular periodic box.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Box2 {
    pub lx: f64,
    pub ly: f64,
}

impl Box2 {
    /// Box for n atoms at density rho: Lx = sqrt(N)*a, Ly = sqrt(N)*h.
    pub fn new(n: usize, rho: f64) -> Box2 {
        let s = side(n).expect("n must be a perfect square");
        let (a, h) = lattice_constants(rho);
        Box2 { lx: s as f64 * a, ly: s as f64 * h }
    }
}

/// Initial triangular-lattice state: x = (i + 0.5*(j%2))*a, y = j*h,
/// i,j = 0..sqrt(N)-1; zero velocities. Atom index = j*n_side + i.
pub fn lattice_state(n: usize, rho: f64) -> State {
    let s = side(n).expect("n must be a perfect square");
    let (a, h) = lattice_constants(rho);
    let mut positions = Vec::with_capacity(n);
    for j in 0..s {
        for i in 0..s {
            positions.push(Vec2::from([
                (i as f64 + 0.5 * ((j % 2) as f64)) * a,
                j as f64 * h,
            ]));
        }
    }
    State { positions, velocities: vec![[0.0, 0.0]; n] }
}

/// Minimum-image displacement on one axis: d - L*round(d/L).
pub fn minimum_image(d: f64, l: f64) -> f64 {
    d - l * (d / l).round()
}

/// Wrap a coordinate into [0, L). Velocities are never wrapped.
pub fn wrap(p: f64, l: f64) -> f64 {
    let w = p - l * p.div_euclid(l);
    if w < 0.0 { w + l } else { w }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-12;

    #[test]
    fn lattice_constants_match_the_sheet() {
        let (a, h) = lattice_constants(0.8);
        let a_expected = (2.0 / (3f64.sqrt() * 0.8)).sqrt();
        assert!((a - a_expected).abs() < EPS);
        assert!((h - 3f64.sqrt() / 2.0 * a).abs() < EPS);
        // rho = 1/(a*h).
        assert!((1.0 / (a * h) - 0.8).abs() < 1e-12);
    }

    #[test]
    fn default_box_is_ten_by_ten_lattice() {
        let bx = Box2::new(100, 0.8);
        let (a, h) = lattice_constants(0.8);
        assert!((bx.lx - 10.0 * a).abs() < EPS);
        assert!((bx.ly - 10.0 * h).abs() < EPS);
    }

    #[test]
    fn lattice_coordinates_follow_the_sheet_formula() {
        let state = lattice_state(100, 0.8);
        let (a, h) = lattice_constants(0.8);
        // Atom (i=0,j=0): (0,0). Index = j*10 + i.
        assert!((state.positions[0][0] - 0.0).abs() < EPS);
        // Atom (i=0,j=1): x = 0.5*a, y = h.
        assert!((state.positions[10][0] - 0.5 * a).abs() < EPS);
        assert!((state.positions[10][1] - h).abs() < EPS);
        // Atom (i=3,j=2): x = 3*a (j even), y = 2*h.
        assert!((state.positions[23][0] - 3.0 * a).abs() < EPS);
        assert!((state.positions[23][1] - 2.0 * h).abs() < EPS);
        assert_eq!(state.positions.len(), 100);
        assert!(state.velocities.iter().all(|v| *v == [0.0, 0.0]));
    }

    #[test]
    fn side_detects_perfect_squares() {
        assert_eq!(side(100), Some(10));
        assert_eq!(side(16), Some(4));
        assert_eq!(side(50), None);
        assert_eq!(side(2), None);
    }

    #[test]
    fn minimum_image_and_wrap_properties() {
        // d - L*round(d/L), per the course definition.
        assert!((minimum_image(2.7, 5.0) + 2.3).abs() < EPS); // 2.7 - 5*1
        assert!((minimum_image(-2.7, 5.0) - 2.3).abs() < EPS); // -2.7 + 5*1
        assert!((minimum_image(2.6, 5.0) + 2.4).abs() < EPS); // 2.6 - 5*1
        // 0 <= wrap(p) < L for a sweep of values.
        for p in [-0.3, 0.0, 2.5, 5.0, 7.3, 12.9] {
            let w = wrap(p, 5.0);
            assert!((0.0..5.0).contains(&w), "wrap({p}) = {w}");
        }
    }
}