use anyhow::{Context, Result, bail};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};

type MemoKey = (usize, u64, u64, u64);

struct CarrySolver {
    n: usize,
    full: u64,
    nonzero: HashMap<MemoKey, u64>,
    zero: HashSet<MemoKey>,
}

impl CarrySolver {
    fn new(n: usize) -> Self {
        Self {
            n,
            full: (1_u64 << n) - 1,
            nonzero: HashMap::new(),
            zero: HashSet::new(),
        }
    }

    fn transition(
        &self,
        columns: u64,
        down_left: u64,
        down_right: u64,
        queen: usize,
    ) -> (u64, u64, u64) {
        let bit = 1_u64 << queen;

        // T(in, x, out) has support
        // (0,0,0), (0,1,1), (1,0,1): a line carry may start at an occupied
        // square or pass through an empty square. It cannot pass two queens.
        // The horizontal 0->1 boundary therefore leaves exactly one queen per row.
        let next_columns = columns | bit;

        // A queen starts both diagonal carries. Existing diagonal carries move
        // one column toward the next row; shifts beyond the board are dropped.
        let next_down_left = ((down_left | bit) >> 1) & self.full;
        let next_down_right = ((down_right | bit) << 1) & self.full;

        (next_columns, next_down_left, next_down_right)
    }

    fn suffix_count(
        &mut self,
        row: usize,
        columns: u64,
        down_left: u64,
        down_right: u64,
    ) -> Result<u64> {
        if row == self.n {
            // At the lower boundary a diagonal carry may have exited or may
            // still be present, so either final diagonal carry is allowed.
            // Every vertical column carry, however, must be one.
            return Ok(if columns == self.full { 1 } else { 0 });
        }

        let key = (row, columns, down_left, down_right);
        if let Some(&count) = self.nonzero.get(&key) {
            return Ok(count);
        }
        if self.zero.contains(&key) {
            return Ok(0);
        }

        let mut total = 0_u64;
        for queen in 0..self.n {
            let bit = 1_u64 << queen;
            if ((columns | down_left | down_right) & bit) != 0 {
                continue;
            }

            let (next_columns, next_down_left, next_down_right) =
                self.transition(columns, down_left, down_right, queen);
            let child =
                self.suffix_count(row + 1, next_columns, next_down_left, next_down_right)?;
            total = total
                .checked_add(child)
                .context("N-queens suffix-count overflow")?;
        }

        if total == 0 {
            self.zero.insert(key);
        } else {
            self.nonzero.insert(key, total);
        }
        Ok(total)
    }

    fn completion_weight(&self, key: MemoKey) -> Result<u64> {
        let (row, columns, _, _) = key;
        if row == self.n {
            return Ok(u64::from(columns == self.full));
        }
        if let Some(&weight) = self.nonzero.get(&key) {
            return Ok(weight);
        }
        if self.zero.contains(&key) {
            return Ok(0);
        }
        bail!("sampling reached a carry state missing from the contraction")
    }

    fn sample_board(&self, rng: &mut ChaCha8Rng) -> Result<Vec<usize>> {
        let mut board = Vec::with_capacity(self.n);
        let mut row = 0;
        let (mut columns, mut down_left, mut down_right) = (0, 0, 0);

        while row < self.n {
            let mut choices = Vec::new();

            for queen in 0..self.n {
                let bit = 1_u64 << queen;
                if ((columns | down_left | down_right) & bit) != 0 {
                    continue;
                }

                let (next_columns, next_down_left, next_down_right) =
                    self.transition(columns, down_left, down_right, queen);
                let weight = self.completion_weight((
                    row + 1,
                    next_columns,
                    next_down_left,
                    next_down_right,
                ))?;
                if weight != 0 {
                    choices.push((queen, next_columns, next_down_left, next_down_right, weight));
                }
            }

            let total = choices.iter().try_fold(0_u64, |sum, choice| {
                sum.checked_add(choice.4)
                    .context("N-queens sampling-weight overflow")
            })?;
            if total == 0 {
                bail!("no positive-weight N-queens continuation");
            }

            // Select directly with integer completion weights. No complete
            // boards are enumerated and no invalid-board rejection is used.
            let mut ticket = rng.random_range(0..total);
            let mut selected = None;
            for choice in &choices {
                if ticket < choice.4 {
                    selected = Some(*choice);
                    break;
                }
                ticket -= choice.4;
            }
            let (queen, next_columns, next_down_left, next_down_right, _) =
                selected.context("failed to select an N-queens continuation")?;

            board.push(queen);
            columns = next_columns;
            down_left = next_down_left;
            down_right = next_down_right;
            row += 1;
        }

        Ok(board)
    }
}

pub fn analyze(n: usize, samples: usize, seed: u64) -> Result<(Value, Vec<(&'static str, Value)>)> {
    if n == 0 || n > 12 {
        bail!("n must be between 1 and 12");
    }

    let mut solver = CarrySolver::new(n);
    let board_count = solver.suffix_count(0, 0, 0, 0)?;
    let stored_entries = solver.nonzero.len();

    // One deterministic ChaCha8 stream is seeded for this sampling run.
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    if samples > 0 && board_count == 0 {
        bail!("N={n} has no valid boards to sample");
    }
    let sampled_boards = (0..samples)
        .map(|_| solver.sample_board(&mut rng))
        .collect::<Result<Vec<_>>>()?;

    let value = json!({
        "n": n,
        "board_count": board_count,
        "samples": sampled_boards,
        "stored_entries": stored_entries,
        "strategy": "sparse_row_carry"
    });
    let metrics = vec![
        ("board_count", json!(board_count)),
        ("stored_entries", json!(stored_entries)),
    ];

    Ok((value, metrics))
}
