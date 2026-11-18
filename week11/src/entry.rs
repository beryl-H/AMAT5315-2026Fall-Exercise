use anyhow::{Context, Result};

#[derive(Clone, Copy, Debug)]
pub struct Entry {
    pub energy: i64,
    pub count: u128,
    pub log_relative: f64,
}

impl Entry {
    pub fn leaf(energy: i64) -> Self {
        Self {
            energy,
            count: 1,
            log_relative: 0.0,
        }
    }

    pub fn product(self, other: Self) -> Result<Self> {
        Ok(Self {
            energy: self
                .energy
                .checked_add(other.energy)
                .context("entry energy overflow during product")?,
            count: self
                .count
                .checked_mul(other.count)
                .context("entry count overflow during product")?,
            log_relative: self.log_relative + other.log_relative,
        })
    }

    pub fn add_alternative(self, other: Self, beta: f64) -> Result<Self> {
        let min_energy = self.energy.min(other.energy);
        let count = match self.energy.cmp(&other.energy) {
            std::cmp::Ordering::Less => self.count,
            std::cmp::Ordering::Greater => other.count,
            std::cmp::Ordering::Equal => self
                .count
                .checked_add(other.count)
                .context("entry count overflow while joining alternatives")?,
        };

        let log_relative = log_add_exp(
            shifted_log(self, min_energy, beta),
            shifted_log(other, min_energy, beta),
        );

        Ok(Self {
            energy: min_energy,
            count,
            log_relative,
        })
    }

    pub fn log_z(self, beta: f64) -> f64 {
        self.log_relative - beta * (self.energy as f64)
    }

    pub fn ground_probability(self) -> f64 {
        (self.count as f64) * (-self.log_relative).exp()
    }
}

fn shifted_log(entry: Entry, min_energy: i64, beta: f64) -> f64 {
    let gap = i128::from(entry.energy) - i128::from(min_energy);
    if gap == 0 {
        entry.log_relative
    } else {
        entry.log_relative - beta * (gap as f64)
    }
}

fn log_add_exp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    if a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    if a == f64::INFINITY || b == f64::INFINITY {
        return f64::INFINITY;
    }

    let hi = a.max(b);
    let lo = a.min(b);
    hi + (lo - hi).exp().ln_1p()
}
