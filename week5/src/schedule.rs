use crate::{Experiment, advance, simulation::Reverse};
use serde::Serialize;
use std::collections::BTreeMap;

#[derive(Serialize)]
pub struct Action {
    pub action: &'static str,
    pub step: usize,
    pub saved_states: usize,
}

pub struct Counters {
    pub scheduler_forward_calls: usize,
    pub reverse_calls: usize,
    pub peak_saved_states: usize,
}

fn capacity(d: usize, t: usize, limit: usize) -> usize {
    let total = d as u128 + t as u128;
    let mut value = 1_u128;
    for i in 1..=d.min(t) {
        value = value.saturating_mul(total - i as u128 + 1) / i as u128;
        if value >= limit as u128 { return limit; }
    }
    value as usize
}

fn split(s: usize, e: usize, d: usize, t: usize) -> usize {
    if d + t == 0 { return e; }
    let numerator = d as u128 * s as u128 + t as u128 * e as u128;
    let mut k = numerator.div_ceil(d as u128 + t as u128) as usize;
    if k >= e && d > 0 { k = (s + 1).max(e - 1); }
    k
}

// Saved slots, replay workspace, and reverse scratch are three separate owners.
pub struct Treeverse<'a> {
    exp: &'a Experiment, velocity: &'a [f64], sigma: &'a [f64], footprint: &'a [f64],
    saved: BTreeMap<usize, Vec<f64>>, workspace: Vec<f64>, next: Vec<f64>, source: Vec<f64>,
    workspace_step: usize, budget: usize,
    pub reverse: Reverse<'a>, pub actions: Vec<Action>, pub counters: Counters,
}

impl<'a> Treeverse<'a> {
    pub fn new(exp: &'a Experiment, velocity: &'a [f64], sigma: &'a [f64], footprint: &'a [f64],
               reverse: Reverse<'a>, budget: usize) -> Self {
        let n = exp.cells();
        let mut saved = BTreeMap::new();
        saved.insert(0, vec![0.0; 2 * n]);
        Self { exp, velocity, sigma, footprint, saved, workspace: vec![0.0; 2 * n],
            next: vec![0.0; 2 * n], source: vec![0.0; n], workspace_step: 0, budget,
            reverse, actions: Vec::new(),
            counters: Counters { scheduler_forward_calls: 0, reverse_calls: 0, peak_saved_states: 1 } }
    }

    pub fn run(&mut self) {
        assert!(self.budget > 0);
        self.actions = construct(self.exp.steps, self.budget);
        for index in 0..self.actions.len() {
            let action = self.actions[index].action;
            let step = self.actions[index].step;
            match action {
                "restore" => {
                    self.workspace.copy_from_slice(self.saved.get(&step).expect("saved replay base"));
                    self.workspace_step = step;
                }
                "call" => {
                    assert_eq!(self.workspace_step, step);
                    self.exp.source(step, self.footprint, &mut self.source);
                    // Replay runs only the primal; the image and adjoints are untouched.
                    advance(self.exp, &self.workspace, self.velocity, self.sigma, &self.source, &mut self.next);
                    std::mem::swap(&mut self.workspace, &mut self.next);
                    self.workspace_step = step + 1;
                    self.counters.scheduler_forward_calls += 1;
                }
                "store" => {
                    assert_eq!(self.workspace_step, step);
                    assert!(!self.saved.contains_key(&step));
                    self.saved.insert(step, self.workspace.clone());
                    assert!(self.saved.len() <= self.budget + 1);
                    self.counters.peak_saved_states = self.counters.peak_saved_states.max(self.saved.len());
                }
                "grad" => {
                    self.reverse.grad(step, self.saved.get(&step).expect("saved reverse input"));
                    self.counters.reverse_calls += 1;
                }
                "fetch" => {
                    assert!(step != 0);
                    self.saved.remove(&step).expect("live checkpoint to release");
                }
                _ => unreachable!(),
            }
            assert_eq!(self.actions[index].saved_states, self.saved.len());
        }
        assert_eq!(self.saved.len(), 1);
        assert!(self.saved.contains_key(&0));
        assert_eq!(self.counters.reverse_calls, self.exp.steps);
    }
}

// These owned frames evaluate the lecture's recursive visit construction.
// The pinned nightly eliminated a changing scalar replay-base argument in the
// direct recursive method. Explicit frames preserve every call's local values;
// tests enforce the original splits, work counts, memory budget and image.
enum Frame {
    Visit { b: usize, s: usize, e: usize, d: usize, t: usize },
    Loop { b: usize, s: usize, e: usize, d: usize, t: usize },
}

fn construct(steps: usize, budget: usize) -> Vec<Action> {
    let mut t = 1;
    while capacity(budget, t, steps) < steps { t += 1; }
    let mut frames = vec![Frame::Visit { b: 0, s: 0, e: steps, d: budget, t }];
    let mut actions = Vec::new();
    let mut live = 1;
    while let Some(frame) = frames.pop() {
        match frame {
            Frame::Visit { b, s, e, mut d, t } => {
                if s > b {
                    assert!(d > 0);
                    d -= 1;
                    actions.push(Action { action: "restore", step: b, saved_states: live });
                    for step in b..s { actions.push(Action { action: "call", step, saved_states: live }); }
                    live += 1;
                    assert!(live <= budget + 1);
                    actions.push(Action { action: "store", step: s, saved_states: live });
                }
                frames.push(Frame::Loop { b, s, e, d, t });
            }
            Frame::Loop { b, s, e, d, t } => {
                let k = split(s, e, d, t);
                if t > 0 && k < e {
                    frames.push(Frame::Loop { b, s, e: k, d, t: t - 1 });
                    frames.push(Frame::Visit { b: s, s: k, e, d, t });
                } else {
                    actions.push(Action { action: "grad", step: s, saved_states: live });
                    if s > b {
                        live -= 1;
                        actions.push(Action { action: "fetch", step: s, saved_states: live });
                    }
                }
            }
        }
    }
    assert_eq!(live, 1);
    actions
}
