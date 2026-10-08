use crate::{Experiment, advance, jvp, vjp};
use serde_json::{Value, json};

pub struct Output {
    pub array: Vec<f64>,
    pub frames: Vec<f32>,
    pub recording_steps: Vec<usize>,
    pub statistics: Option<Value>,
    pub shot_l2: Vec<f64>,
}

pub fn run(exp: &Experiment, mode: &str, weights: &[f64], every: usize) -> Output {
    let n = exp.cells();
    let nr = exp.receivers.len();
    let velocity = exp.velocity();
    let direction = exp.perturbation();
    let sigma = exp.sigma();
    let mut result = Output { array: vec![0.0; if mode == "adjoint" { n } else { exp.shots.len() * exp.steps * nr }],
        frames: Vec::new(), recording_steps: Vec::new(), statistics: None, shot_l2: Vec::new() };
    let mut per_shot = Vec::new();
    for (shot, &position) in exp.shots.iter().enumerate() {
        let footprint = exp.footprint(position);
        let mut source = vec![0.0; n];
        let mut state = vec![0.0; 2 * n];
        let mut next = state.clone();
        let mut tangent = state.clone();
        let mut next_tangent = state.clone();
        let mut history = Vec::new();
        if mode == "adjoint" { history.push(state.clone()); }
        for step in 0..exp.steps {
            exp.source(step, &footprint, &mut source);
            if mode == "born" {
                // One Enzyme call computes this timestep's primal and JVP.
                jvp(exp, &state, &tangent, &velocity, &direction, &sigma, &source, &mut next, &mut next_tangent);
                std::mem::swap(&mut tangent, &mut next_tangent);
            } else {
                advance(exp, &state, &velocity, &sigma, &source, &mut next);
            }
            std::mem::swap(&mut state, &mut next);
            if mode == "adjoint" { history.push(state.clone()); continue; }
            let field = if mode == "born" { &tangent[n..] } else { &state[n..] };
            for (r, &[x, z]) in exp.receivers.iter().enumerate() {
                result.array[(shot * exp.steps + step) * nr + r] = field[z * exp.nx + x];
            }
            if shot == 0 && every > 0 && (step + 1) % every == 0 {
                result.recording_steps.push(step + 1);
                result.frames.extend(field.iter().map(|&v| v as f32));
            }
        }
        if mode == "adjoint" {
            let start = shot * exp.steps * nr;
            let mut reverse = Reverse::new(exp, &velocity, &sigma, &footprint,
                &weights[start..start + exp.steps * nr], if shot == 0 { every } else { 0 });
            for step in (0..exp.steps).rev() { reverse.grad(step, &history[step]); }
            result.shot_l2.push(norm(&reverse.image));
            for (total, value) in result.array.iter_mut().zip(&reverse.image) { *total += value; }
            if shot == 0 { result.frames = reverse.frames; result.recording_steps = reverse.recording_steps; }
            per_shot.push(json!({"scheduler_forward_calls": exp.steps, "reverse_calls": exp.steps,
                "peak_saved_states": history.len()}));
        } else {
            let slice = &result.array[shot * exp.steps * nr..(shot + 1) * exp.steps * nr];
            result.shot_l2.push(norm(slice));
        }
    }
    if mode == "adjoint" {
        let bytes = 2 * n * 8;
        result.statistics = Some(json!({"storage": "full", "checkpoints": null, "state_wavefields": 2,
            "saved_state_bytes": bytes, "peak_saved_states": exp.steps + 1,
            "peak_saved_bytes": (exp.steps + 1) * bytes,
            "scheduler_forward_calls": exp.steps * exp.shots.len(),
            "reverse_calls": exp.steps * exp.shots.len(), "per_shot": per_shot}));
    }
    result
}

pub fn norm(x: &[f64]) -> f64 { x.iter().map(|v| v * v).sum::<f64>().sqrt() }

// VJP scratch, receiver weights and adjoints are separate from saved states.
pub struct Reverse<'a> {
    exp: &'a Experiment, velocity: &'a [f64], sigma: &'a [f64], footprint: &'a [f64], weights: &'a [f64],
    source: Vec<f64>, lambda: Vec<f64>, input_bar: Vec<f64>, scratch: Vec<f64>,
    pub image: Vec<f64>, pub frames: Vec<f32>, pub recording_steps: Vec<usize>, every: usize,
}

impl<'a> Reverse<'a> {
    pub fn new(exp: &'a Experiment, velocity: &'a [f64], sigma: &'a [f64], footprint: &'a [f64],
               weights: &'a [f64], every: usize) -> Self {
        let n = exp.cells();
        Self { exp, velocity, sigma, footprint, weights, source: vec![0.0; n],
            lambda: vec![0.0; 2 * n], input_bar: vec![0.0; 2 * n], scratch: vec![0.0; 2 * n],
            image: vec![0.0; n], frames: Vec::new(), recording_steps: Vec::new(), every }
    }
    pub fn grad(&mut self, step: usize, state: &[f64]) {
        let n = self.exp.cells();
        let nr = self.exp.receivers.len();
        for (r, &[x, z]) in self.exp.receivers.iter().enumerate() {
            // += is essential when more than one receiver occupies the same cell.
            self.lambda[n + z * self.exp.nx + x] += self.weights[step * nr + r];
        }
        self.exp.source(step, self.footprint, &mut self.source);
        self.input_bar.fill(0.0);
        // The velocity shadow accumulates time contributions; the output seed is consumed.
        vjp(self.exp, state, &mut self.input_bar, self.velocity, &mut self.image,
            self.sigma, &self.source, &mut self.scratch, &mut self.lambda);
        std::mem::swap(&mut self.lambda, &mut self.input_bar);
        if self.every > 0 && step % self.every == 0 {
            self.recording_steps.push(step);
            self.frames.extend(self.lambda[n..].iter().map(|&v| v as f32));
        }
    }
}
