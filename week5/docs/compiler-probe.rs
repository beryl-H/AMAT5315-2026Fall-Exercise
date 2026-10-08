// Diagnostic source retained for the pinned compiler's recursive scalar issue.
// No Enzyme, FFI, unsafe operations, or production solver code is used here.
fn split(s: usize, e: usize, d: usize, t: usize) -> usize {
    if d + t == 0 { return e; }
    let numerator = d as u128 * s as u128 + t as u128 * e as u128;
    let mut k = numerator.div_ceil(d as u128 + t as u128) as usize;
    if k >= e && d > 0 { k = (s + 1).max(e - 1); }
    k
}

#[inline(never)]
fn visit(out: &mut Vec<(u8, usize)>, b: usize, s: usize,
         mut e: usize, mut d: usize, mut t: usize) {
    let b = std::hint::black_box(b);
    if s > b {
        assert!(d > 0);
        d -= 1;
        out.push((0, b));
        for step in b..s { out.push((1, step)); }
        out.push((2, s));
    }
    let mut k = split(s, e, d, t);
    while t > 0 && k < e {
        visit(out, s, k, e, d, t);
        t -= 1;
        e = k;
        k = split(s, e, d, t);
    }
    out.push((3, s));
    if s > b { out.push((4, s)); }
}

#[unsafe(no_mangle)]
pub fn schedule(e: usize, d: usize, t: usize) -> Vec<(u8, usize)> {
    let mut out = Vec::new();
    visit(&mut out, 0, 0, e, d, t);
    out
}

fn main() {
    let steps: usize = std::env::args().nth(1).unwrap().parse().unwrap();
    let actions = schedule(steps, 2, 2);
    let calls = actions.iter().filter(|(action, _)| *action == 1).count();
    let restores: Vec<_> = actions.iter().filter(|(action, _)| *action == 0)
        .map(|(_, step)| *step).collect();
    println!("calls={calls}; restore bases={restores:?}");
}
