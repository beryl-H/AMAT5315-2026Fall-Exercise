use crate::Result;
use std::{fs::File, io::{Read, Write}, path::Path};

fn header(file: &mut File, dtype: &str, shape: &[usize]) -> Result<()> {
    let mut dimensions = shape.iter().map(usize::to_string).collect::<Vec<_>>().join(", ");
    if shape.len() == 1 { dimensions.push(','); }
    let mut text = format!("{{'descr': '{dtype}', 'fortran_order': False, 'shape': ({dimensions}), }}");
    let padding = (64 - ((10 + text.len() + 1) % 64)) % 64;
    text.push_str(&" ".repeat(padding)); text.push('\n');
    file.write_all(b"\x93NUMPY\x01\x00")?;
    file.write_all(&(text.len() as u16).to_le_bytes())?;
    file.write_all(text.as_bytes())?;
    Ok(())
}

pub fn write_f64(path: &Path, shape: &[usize], data: &[f64]) -> Result<()> {
    if shape.iter().product::<usize>() != data.len() { return Err("NPY shape does not match data".into()); }
    let mut file = File::create(path)?;
    header(&mut file, "<f8", shape)?;
    let mut bytes = Vec::with_capacity(data.len() * 8);
    for x in data { bytes.extend_from_slice(&x.to_le_bytes()); }
    file.write_all(&bytes)?;
    Ok(())
}

pub fn write_f32(path: &Path, shape: &[usize], data: &[f32]) -> Result<()> {
    if shape.iter().product::<usize>() != data.len() { return Err("NPY shape does not match data".into()); }
    let mut file = File::create(path)?;
    header(&mut file, "<f4", shape)?;
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for x in data { bytes.extend_from_slice(&x.to_le_bytes()); }
    file.write_all(&bytes)?;
    Ok(())
}

pub fn read_f64(path: &Path, expected: &[usize]) -> Result<Vec<f64>> {
    let mut file = File::open(path)?;
    let mut magic = [0; 8]; file.read_exact(&mut magic)?;
    if &magic != b"\x93NUMPY\x01\x00" { return Err("expected NumPy v1.0 format".into()); }
    let mut size = [0; 2]; file.read_exact(&mut size)?;
    let mut bytes = vec![0; u16::from_le_bytes(size) as usize]; file.read_exact(&mut bytes)?;
    let text = std::str::from_utf8(&bytes)?;
    if !text.contains("'descr': '<f8'") || !text.contains("'fortran_order': False") {
        return Err("receiver data must be C-order little-endian float64".into());
    }
    let shape_text = text.split("'shape': (").nth(1).ok_or("missing NPY shape")?
        .split(')').next().ok_or("invalid NPY shape")?;
    let shape = shape_text.split(',').filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse::<usize>()).collect::<std::result::Result<Vec<_>, _>>()?;
    if shape != expected { return Err(format!("data shape {shape:?} does not match {expected:?}").into()); }
    bytes.clear(); file.read_to_end(&mut bytes)?;
    if bytes.len() != expected.iter().product::<usize>() * 8 { return Err("NPY data byte count mismatch".into()); }
    let data = bytes.chunks_exact(8).map(|b| f64::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
    if data.iter().any(|v| !v.is_finite()) { return Err("receiver data must be finite".into()); }
    Ok(data)
}
