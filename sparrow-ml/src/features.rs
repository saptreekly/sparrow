use std::collections::HashSet;

use sparrow_core::{Complex64, SparseStateVector};

use crate::gate::GateSpec;

pub const FEATURE_COUNT: usize = 8;
pub const FEATURE_NAMES: [&str; FEATURE_COUNT] = [
    "source_norm_sqr",
    "source_re",
    "source_im",
    "partner_norm_sqr",
    "partner_re",
    "partner_im",
    "global_density",
    "local_density",
];

#[derive(Clone, Debug)]
pub struct FeatureContext {
    pub target_basis: u128,
    pub source_basis: u128,
    pub features: [f64; FEATURE_COUNT],
}

pub fn visit_candidates<F>(state: &SparseStateVector, gate: &GateSpec, mut visitor: F)
where
    F: FnMut(FeatureContext),
{
    match gate {
        GateSpec::Cnot { control, target } => {
            visit_cnot_candidates(state, *control, *target, &mut visitor)
        }
    }
}

fn visit_cnot_candidates<F>(state: &SparseStateVector, control: u8, target: u8, visitor: &mut F)
where
    F: FnMut(FeatureContext),
{
    let entries = state.amplitudes();
    if entries.is_empty() {
        return;
    }

    let qubits = state.qubit_count();
    let len = entries.len();
    let control_mask = 1u128 << control;
    let target_mask = 1u128 << target;
    let basis_set: HashSet<u128> = entries.iter().map(|(basis, _)| *basis).collect();

    let global_density = if qubits == 0 {
        1.0
    } else {
        let total_states = 2f64.powi(qubits as i32);
        (len as f64) / total_states
    };

    for (basis, amplitude) in entries.iter() {
        if basis & control_mask == 0 {
            continue;
        }

        let partner = basis ^ target_mask;
        if basis_set.contains(&partner) {
            continue;
        }

        let partner_amp = state.amplitude(partner);
        let local_density = local_density(&basis_set, *basis, qubits);

        let features = build_feature_vector(*amplitude, partner_amp, global_density, local_density);

        visitor(FeatureContext {
            target_basis: partner,
            source_basis: *basis,
            features,
        });
    }
}

fn build_feature_vector(
    source_amp: Complex64,
    partner_amp: Complex64,
    global_density: f64,
    local_density: f64,
) -> [f64; FEATURE_COUNT] {
    [
        source_amp.norm_sqr(),
        source_amp.re,
        source_amp.im,
        partner_amp.norm_sqr(),
        partner_amp.re,
        partner_amp.im,
        global_density,
        local_density,
    ]
}

fn local_density(basis_set: &HashSet<u128>, basis: u128, qubits: u8) -> f64 {
    if qubits == 0 {
        return 0.0;
    }

    let mut total = 1usize;
    let mut count = if basis_set.contains(&basis) {
        1usize
    } else {
        0
    };

    for bit in 0..qubits {
        total += 1;
        let neighbor = basis ^ (1u128 << bit);
        if basis_set.contains(&neighbor) {
            count += 1;
        }
    }

    for i in 0..qubits {
        for j in (i + 1)..qubits {
            total += 1;
            let neighbor = basis ^ (1u128 << i) ^ (1u128 << j);
            if basis_set.contains(&neighbor) {
                count += 1;
            }
        }
    }

    count as f64 / total as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn visit_candidates_skips_existing_partner() {
        let mut state = SparseStateVector::new(2);
        state.set_amplitude(2, Complex64::new(1.0, 0.0)).unwrap();
        state.set_amplitude(3, Complex64::new(0.5, 0.0)).unwrap();

        let mut count = 0;
        visit_candidates(
            &state,
            &GateSpec::Cnot {
                control: 1,
                target: 0,
            },
            |_| count += 1,
        );

        assert_eq!(
            count, 0,
            "partner already present should produce no candidates"
        );
    }
}
