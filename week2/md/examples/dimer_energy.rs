//! Print the two-atom dimer energy-error series used by
//! `week2/plot_dimer.py`.
//!
//! Runs the shared `run_experiment` driver with both integrators at the
//! course parameters (`dt = 0.01`) and prints one whitespace-separated
//! block per run. Columns: run index, step, time, relative energy error.
//! Each block starts with a `# run=` comment line naming the run.

use md::{Euler, ExperimentResult, VelocityVerlet, run_experiment};

const DT: f64 = 0.01;

fn print_run(run: usize, name: &str, result: &ExperimentResult) {
    println!("# run={run} name={name}");
    for (i, &error) in result.relative_energy_errors.iter().enumerate() {
        println!(
            "{run} {} {:.10e} {:.10e}",
            result.steps[i], result.times[i], error
        );
    }
}

fn main() {
    let euler_500 = run_experiment(&mut Euler::default(), 500, DT);
    let verlet_500 = run_experiment(&mut VelocityVerlet::default(), 500, DT);
    let verlet_5000 = run_experiment(&mut VelocityVerlet::default(), 5000, DT);

    print_run(0, "euler-500", &euler_500);
    print_run(1, "verlet-500", &verlet_500);
    print_run(2, "verlet-5000", &verlet_5000);
}
