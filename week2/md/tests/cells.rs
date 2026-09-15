//! Cell-list vs naive equality and coverage tests.
//!
//! Course Requirement: the cells implementation must reproduce the naive
//! reference to rounding on every required configuration.

use md::fluid::{fluid_accelerations, fluid_potential_energy, ForceMethod};
use md::system::{Box2, lattice_state, minimum_image, wrap};
use md::State;

/// Scale-aware rounding comparison [Suggestion]; fixed before implementation.
const ATOL: f64 = 1e-10;
const RTOL: f64 = 1e-9;

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= ATOL + RTOL * a.abs().max(b.abs())
}

fn assert_forces_close(naive: &[[f64; 2]], cells: &[[f64; 2]]) {
    assert_eq!(naive.len(), cells.len());
    for k in 0..naive.len() {
        assert!(close(naive[k][0], cells[k][0]),
            "atom {k} Fx: naive {} vs cells {}", naive[k][0], cells[k][0]);
        assert!(close(naive[k][1], cells[k][1]),
            "atom {k} Fy: naive {} vs cells {}", naive[k][1], cells[k][1]);
    }
}

fn assert_total_force_zero(acc: &[[f64; 2]]) {
    let mut s = [0.0, 0.0];
    for a in acc {
        s[0] += a[0];
        s[1] += a[1];
    }
    assert!(s[0].abs() < 1e-9, "sum Fx = {}", s[0]);
    assert!(s[1].abs() < 1e-9, "sum Fy = {}", s[1]);
}

fn assert_energy_close(naive: f64, cells: f64) {
    assert!(close(naive, cells), "energy: naive {naive} vs cells {cells}");
}

#[test]
fn forces_and_energy_agree_on_lattice() {
    let state = lattice_state(100, 0.8);
    let bx = Box2::new(100, 0.8);
    let naive = fluid_accelerations(&state, &bx);
    let cells = ForceMethod::Cells.accelerations(&state, &bx);
    assert_forces_close(&naive, &cells);
    assert_total_force_zero(&cells);
    assert_energy_close(
        fluid_potential_energy(&state, &bx),
        ForceMethod::Cells.potential_energy(&state, &bx),
    );
}

#[test]
fn agree_on_perturbed_configuration() {
    let mut state = lattice_state(100, 0.8);
    let bx = Box2::new(100, 0.8);
    for (i, p) in state.positions.iter_mut().enumerate() {
        if i % 2 == 0 {
            p[0] = wrap(p[0] + 0.05 * (i as f64 % 3.0), bx.lx);
            p[1] = wrap(p[1] + 0.11, bx.ly);
        }
    }
    let naive = fluid_accelerations(&state, &bx);
    let cells = ForceMethod::Cells.accelerations(&state, &bx);
    assert_forces_close(&naive, &cells);
    assert_total_force_zero(&cells);
    assert_energy_close(
        fluid_potential_energy(&state, &bx),
        ForceMethod::Cells.potential_energy(&state, &bx),
    );
}

#[test]
fn agree_across_periodic_boundary() {
    // Pairs straddling x = 0/Lx and y = 0/Ly with minimum-image r < rc.
    let bx = Box2 { lx: 10.0, ly: 10.0 };
    let state = State {
        positions: vec![
            [0.3, 5.0], [9.7, 5.0], [5.0, 0.3], [5.0, 9.7], [2.0, 2.0], [8.0, 8.0],
        ],
        velocities: vec![[0.0, 0.0]; 6],
    };
    // Sanity: the x-straddling pair really is within rc via minimum image.
    let dx = minimum_image(0.3 - 9.7, bx.lx);
    assert!(dx.abs() < 2.5, "boundary pair must interact, dx = {dx}");
    let naive = fluid_accelerations(&state, &bx);
    let cells = ForceMethod::Cells.accelerations(&state, &bx);
    assert_forces_close(&naive, &cells);
    assert_energy_close(
        fluid_potential_energy(&state, &bx),
        ForceMethod::Cells.potential_energy(&state, &bx),
    );
}

#[test]
fn agree_at_exact_cutoff() {
    // A pair at exactly r = rc (zero contribution) plus a sub-rc pair, so the
    // test is not trivially all-zero.
    let bx = Box2 { lx: 10.0, ly: 10.0 };
    let state = State {
        positions: vec![[1.0, 5.0], [3.5, 5.0], [5.0, 1.0], [5.0, 8.0], [1.0, 3.0]],
        velocities: vec![[0.0, 0.0]; 5],
    };
    let naive = fluid_accelerations(&state, &bx);
    let cells = ForceMethod::Cells.accelerations(&state, &bx);
    assert_forces_close(&naive, &cells);
    assert_energy_close(
        fluid_potential_energy(&state, &bx),
        ForceMethod::Cells.potential_energy(&state, &bx),
    );
}

#[test]
fn agree_at_cutoff_minus_epsilon() {
    // [Suggestion] A pair just inside the cutoff must actually be found by
    // cells (nonzero contribution) and agree with naive.
    let eps = 1e-6;
    let bx = Box2 { lx: 10.0, ly: 10.0 };
    let state = State {
        positions: vec![[1.0, 5.0], [1.0 + 2.5 - eps, 5.0], [5.0, 1.0], [5.0, 8.0]],
        velocities: vec![[0.0, 0.0]; 4],
    };
    let naive = fluid_accelerations(&state, &bx);
    let cells = ForceMethod::Cells.accelerations(&state, &bx);
    assert_forces_close(&naive, &cells);
    assert_energy_close(
        fluid_potential_energy(&state, &bx),
        ForceMethod::Cells.potential_energy(&state, &bx),
    );
}

#[test]
fn agree_in_two_cell_wide_box_with_dedup() {
    // nx = ny = 2: wrapped 3x3 neighbor indices duplicate and must be
    // deduplicated or pairs would be double-counted.
    let bx = Box2 { lx: 5.0, ly: 5.0 };
    let mut pos = Vec::new();
    for i in 0..4 {
        for j in 0..4 {
            pos.push([i as f64 + 0.6, j as f64 + 0.6]);
        }
    }
    let state = State {
        positions: pos,
        velocities: vec![[0.0, 0.0]; 16],
    };
    let naive = fluid_accelerations(&state, &bx);
    let cells = ForceMethod::Cells.accelerations(&state, &bx);
    assert_forces_close(&naive, &cells);
    assert_total_force_zero(&cells);
    assert_energy_close(
        fluid_potential_energy(&state, &bx),
        ForceMethod::Cells.potential_energy(&state, &bx),
    );
}
