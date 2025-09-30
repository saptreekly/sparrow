use std::collections::HashSet;

use rand::distributions::{Distribution, Uniform};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_distr::Uniform as UniformF64;

use sparrow_core::{Complex64, SparseStateVector};

use crate::features::{FeatureContext, visit_candidates};
use crate::gate::GateSpec;

#[derive(Clone, Debug)]
pub struct Sample {
    pub features: Vec<f64>,
    pub label: bool,
    pub target_basis: u128,
    pub source_basis: u128,
}

pub struct DatasetConfig {
    pub qubits: u8,
    pub num_states: usize,
    pub min_non_zero: usize,
    pub max_non_zero: usize,
    pub label_threshold: f64,
    pub gate: GateSpec,
    pub seed: u64,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            qubits: 12,
            num_states: 500,
            min_non_zero: 8,
            max_non_zero: 32,
            label_threshold: 1e-8,
            gate: GateSpec::Cnot {
                control: 3,
                target: 1,
            },
            seed: 0xC0FFEE,
        }
    }
}

pub fn generate_dataset(config: &DatasetConfig) -> Vec<Sample> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);

    let mut samples = Vec::new();

    let capacity = if config.qubits as u32 >= usize::BITS {
        usize::MAX
    } else {
        1usize << config.qubits
    };

    for _ in 0..config.num_states {
        let desired = rng
            .gen_range(config.min_non_zero..=config.max_non_zero)
            .max(1);
        let count = desired.min(capacity);
        let mut state = random_state(config.qubits, count, &mut rng);
        if state.is_empty() {
            continue;
        }
        state.normalize();

        let mut feature_buffer: Vec<FeatureContext> = Vec::new();
        visit_candidates(&state, &config.gate, |context| feature_buffer.push(context));
        if feature_buffer.is_empty() {
            continue;
        }

        let mut after = state.clone();
        config
            .gate
            .apply(&mut after)
            .expect("gate application should succeed");

        for ctx in feature_buffer {
            let amplitude = after.amplitude(ctx.target_basis);
            let label = amplitude.norm_sqr() > config.label_threshold;
            samples.push(Sample {
                features: ctx.features.to_vec(),
                label,
                target_basis: ctx.target_basis,
                source_basis: ctx.source_basis,
            });
        }
    }

    samples
}

pub fn shuffle_split(
    samples: Vec<Sample>,
    test_ratio: f64,
    seed: u64,
) -> (Vec<Sample>, Vec<Sample>) {
    assert!((0.0..1.0).contains(&test_ratio));
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut data = samples;
    data.shuffle(&mut rng);
    let test_len = ((data.len() as f64) * test_ratio).round() as usize;
    let split_index = data.len().saturating_sub(test_len);
    let test = data.split_off(split_index);
    (data, test)
}

fn random_state(qubits: u8, non_zero: usize, rng: &mut impl Rng) -> SparseStateVector {
    let mut state = SparseStateVector::new(qubits);
    if non_zero == 0 {
        return state;
    }

    let max_states = if qubits == 128 {
        u128::MAX
    } else {
        (1u128 << qubits).saturating_sub(1)
    };

    let available = if qubits as u32 >= usize::BITS {
        usize::MAX
    } else {
        1usize << qubits
    };
    let target = non_zero.min(available);

    let mut chosen = HashSet::with_capacity(target);
    let basis_dist = Uniform::new_inclusive(0u128, max_states);

    while chosen.len() < target {
        let basis = basis_dist.sample(rng);
        chosen.insert(basis);
    }

    let log_dist = UniformF64::new_inclusive(-9.0, 0.0);
    let phase_dist = UniformF64::new_inclusive(0.0, std::f64::consts::TAU);

    for basis in chosen {
        let log_mag = log_dist.sample(rng);
        let magnitude = 10f64.powf(log_mag);
        let phase = phase_dist.sample(rng);
        let amplitude = Complex64::new(phase.cos() * magnitude, phase.sin() * magnitude);
        state
            .set_amplitude(basis, amplitude)
            .expect("basis within range");
    }

    state
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dataset_generation_produces_samples() {
        let config = DatasetConfig {
            num_states: 50,
            ..DatasetConfig::default()
        };
        let samples = generate_dataset(&config);
        assert!(!samples.is_empty());
    }
}
