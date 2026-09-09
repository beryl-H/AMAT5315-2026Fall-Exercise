//! Trajectory file IO: the v1 text format written by `md run`.

use std::fmt::Write as _;

/// Fixed run metadata stored in the header.
pub struct Meta {
    pub n: usize,
    pub box_l: f64,
    pub temp: f64,
    pub dt: f64,
}

/// All frames of a run; each frame holds one `[x, y, vx, vy]` per atom.
pub struct Trajectory {
    pub meta: Meta,
    pub frames: Vec<Vec<[f64; 4]>>,
}

/// Write the v1 text format (header + N lines per frame).
pub fn write(path: &str, t: &Trajectory) -> std::io::Result<()> {
    let mut s = String::new();
    s.push_str("# md trajectory v1\n");
    let _ = writeln!(s, "# N {}", t.meta.n);
    let _ = writeln!(s, "# box {:.10}", t.meta.box_l);
    let _ = writeln!(s, "# temp {:.10}", t.meta.temp);
    let _ = writeln!(s, "# dt {:.10}", t.meta.dt);
    let _ = writeln!(s, "# frames {}", t.frames.len());
    for frame in &t.frames {
        for a in frame {
            let _ = writeln!(s, "{:.10} {:.10} {:.10} {:.10}", a[0], a[1], a[2], a[3]);
        }
    }
    std::fs::write(path, s)
}

/// Parse a v1 trajectory; errors carry the offending line number.
pub fn read(path: &str) -> Result<Trajectory, String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let mut n = None;
    let mut box_l = None;
    let mut temp = None;
    let mut dt = None;
    let mut frames_expected = None;
    for line in text.lines().filter(|l| l.starts_with('#')) {
        let f: Vec<&str> = line[1..].split_whitespace().collect();
        let val = |i: usize| f.get(i).and_then(|v| v.parse::<f64>().ok());
        match f.first().copied().unwrap_or("") {
            "md" => {}
            "N" => n = f.get(1).and_then(|v| v.parse::<usize>().ok()),
            "box" => box_l = val(1),
            "temp" => temp = val(1),
            "dt" => dt = val(1),
            "frames" => frames_expected = f.get(1).and_then(|v| v.parse::<usize>().ok()),
            _ => return Err(format!("unknown header line: {line}")),
        }
    }
    let meta = Meta {
        n: n.ok_or("missing # N")?,
        box_l: box_l.ok_or("missing # box")?,
        temp: temp.ok_or("missing # temp")?,
        dt: dt.ok_or("missing # dt")?,
    };
    let frames_expected = frames_expected.ok_or("missing # frames")?;
    let mut frames = Vec::new();
    let mut current: Option<Vec<[f64; 4]>> = None;
    for (no, line) in text.lines().enumerate() {
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        let v: Vec<f64> = line
            .split_whitespace()
            .map(|w| w.parse::<f64>())
            .collect::<Result<_, _>>()
            .map_err(|e| format!("line {}: {e}", no + 1))?;
        if v.len() != 4 {
            return Err(format!("line {}: expected 4 numbers", no + 1));
        }
        let cur = current.get_or_insert_with(Vec::new);
        cur.push([v[0], v[1], v[2], v[3]]);
        if cur.len() == meta.n {
            frames.push(current.take().unwrap());
        }
    }
    if frames.len() != frames_expected {
        return Err(format!(
            "header promises {frames_expected} frames, file has {}",
            frames.len()
        ));
    }
    Ok(Trajectory { meta, frames })
}

#[cfg(test)]
mod tests {
    use super::{read, write, Meta, Trajectory};

    fn sample() -> Trajectory {
        Trajectory {
            meta: Meta { n: 2, box_l: 10.0, temp: 0.7, dt: 0.005 },
            frames: vec![
                vec![[1.0, 2.0, 0.1, -0.2], [3.0, 4.0, 0.0, 0.5]],
                vec![[1.1, 2.1, 0.1, -0.2], [3.1, 4.1, 0.0, 0.5]],
            ],
        }
    }

    #[test]
    fn round_trip() {
        let path = std::env::temp_dir().join("md_traj_round_trip.txt");
        let path = path.to_str().unwrap();
        write(path, &sample()).unwrap();
        let t = read(path).unwrap();
        assert_eq!(t.meta.n, 2);
        assert_eq!(t.frames.len(), 2);
        assert_eq!(t.frames[0][1][3], 0.5);
        for (a, b) in t.frames.iter().flatten().zip(sample().frames.iter().flatten()) {
            for k in 0..4 {
                assert!((a[k] - b[k]).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn rejects_malformed() {
        let path = std::env::temp_dir().join("md_traj_bad.txt");
        std::fs::write(&path, "not a trajectory\n").unwrap();
        assert!(read(path.to_str().unwrap()).is_err());
        // Frame count mismatch: header promises 3, file has 1.
        let body = "# md trajectory v1\n# N 1\n# box 10\n# temp 1\n# dt 0.01\n# frames 3\n0 0 0 0\n";
        std::fs::write(&path, body).unwrap();
        assert!(read(path.to_str().unwrap()).is_err());
    }
}
