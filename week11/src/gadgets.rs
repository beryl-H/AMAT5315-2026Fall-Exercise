use serde_json::{Value, json};

fn spin(bit: u8) -> i64 {
    if bit == 0 { -1 } else { 1 }
}

fn bits(mask: u8, width: usize) -> String {
    format!("{:0width$b}", mask, width = width)
}

fn row(bits: String, energy: i64, valid: bool) -> Value {
    json!({"bits": bits, "energy": energy, "valid": valid})
}

fn and_energy(a: i64, b: i64, z: i64) -> i64 {
    a * b - 2 * a * z - 2 * b * z - a - b + 2 * z
}

pub fn evidence() -> Value {
    let mut and_rows = Vec::new();
    for mask in 0u8..8 {
        let (ab, bb, zb) = ((mask >> 2) & 1, (mask >> 1) & 1, mask & 1);
        let (a, b, z) = (spin(ab), spin(bb), spin(zb));
        let energy = and_energy(a, b, z);
        and_rows.push(row(bits(mask, 3), energy, zb == (ab & bb)));
    }

    let mut or_rows = Vec::new();
    for mask in 0u8..8 {
        let (ab, bb, zb) = ((mask >> 2) & 1, (mask >> 1) & 1, mask & 1);
        let (a, b, z) = (spin(ab), spin(bb), spin(zb));
        let energy = a * b - 2 * a * z - 2 * b * z + a + b - 2 * z;
        or_rows.push(row(bits(mask, 3), energy, zb == (ab | bb)));
    }

    let mut not_rows = Vec::new();
    let mut copy_rows = Vec::new();
    for mask in 0u8..4 {
        let (ab, zb) = ((mask >> 1) & 1, mask & 1);
        let (a, z) = (spin(ab), spin(zb));
        not_rows.push(row(bits(mask, 2), a * z, zb == (1 - ab)));
        copy_rows.push(row(bits(mask, 2), -a * z, zb == ab));
    }

    let mut set0_rows = Vec::new();
    let mut set1_rows = Vec::new();
    for mask in 0u8..2 {
        let z = spin(mask);
        set0_rows.push(row(bits(mask, 1), z, mask == 0));
        set1_rows.push(row(bits(mask, 1), -z, mask == 1));
    }

    let mut xor_rows = Vec::new();
    for mask in 0u8..8 {
        let (ab, bb, zb) = ((mask >> 2) & 1, (mask >> 1) & 1, mask & 1);
        let (a, b, z) = (spin(ab), spin(bb), spin(zb));
        let energy = [-1_i64, 1]
            .iter()
            .map(|&r| a * b - a * z - 2 * a * r - b * z - 2 * b * r + 2 * z * r - a - b + z + 2 * r)
            .min()
            .unwrap();
        xor_rows.push(row(bits(mask, 3), energy, zb == (ab ^ bb)));
    }

    let mut circuit_rows = Vec::new();
    for mask in 0u8..16 {
        let (ab, bb, tb, ob) = ((mask >> 3) & 1, (mask >> 2) & 1, (mask >> 1) & 1, mask & 1);
        let (a, b, t, out) = (spin(ab), spin(bb), spin(tb), spin(ob));
        let not_energy = t * out;
        let set1_energy = -out;
        let valid = tb == (ab & bb) && ob == (1 - tb) && ob == 1;
        circuit_rows.push(row(
            bits(mask, 4),
            and_energy(a, b, t) + not_energy + set1_energy,
            valid,
        ));
    }

    json!({
        "gates": {
            "AND": and_rows,
            "OR": or_rows,
            "NOT": not_rows,
            "COPY": copy_rows,
            "SET0": set0_rows,
            "SET1": set1_rows,
            "XOR": xor_rows
        },
        "circuit": circuit_rows
    })
}
