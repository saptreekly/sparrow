use core::mem;

use log::{debug, info, trace};
use sparrow_core::{Circuit, Complex64, Gate, SparseStateError, SparseStateVector};
use sparrow_ml::gate::GateSpec as MlGate;
use sparrow_ml::predictor::SplashPredictor;
use thiserror::Error;

/// Execution modes supported by the simulator.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SimulationMode {
    Predictive,
    Sparse,
    Dense,
}

/// Configuration options that control the simulator runtime.
#[derive(Clone, Debug)]
pub struct SimulationConfig {
    pub mode: SimulationMode,
    pub prune_threshold: f64,
    pub prediction_threshold: f64,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            mode: SimulationMode::Predictive,
            prune_threshold: 1e-9,
            prediction_threshold: 0.05,
        }
    }
}

/// Errors that can occur when orchestrating a simulation.
#[derive(Error, Debug)]
pub enum OrchestratorError {
    #[error("sparse state error: {0:?}")]
    Sparse(SparseStateError),
    #[error("dense simulation exceeds host limits for {0} qubits")]
    DenseTooLarge(u8),
}

pub type OrchestratorResult<T> = Result<T, OrchestratorError>;

impl From<SparseStateError> for OrchestratorError {
    fn from(value: SparseStateError) -> Self {
        OrchestratorError::Sparse(value)
    }
}

/// High-level simulation engine that coordinates predictive inference and sparse evolution.
pub struct Orchestrator {
    config: SimulationConfig,
    predictor: Option<SplashPredictor>,
    scratch: Vec<u128>,
}

impl Orchestrator {
    pub fn new(config: SimulationConfig) -> Self {
        let predictor = match config.mode {
            SimulationMode::Predictive => Some(
                SplashPredictor::pretrained().with_threshold(config.prediction_threshold),
            ),
            _ => None,
        };
        Self {
            config,
            predictor,
            scratch: Vec::new(),
        }
    }

    pub fn run(&mut self, circuit: &Circuit, state: &mut SparseStateVector) -> OrchestratorResult<()> {
        match self.config.mode {
            SimulationMode::Predictive => self.run_predictive(circuit, state),
            SimulationMode::Sparse => self.run_sparse(circuit, state),
            SimulationMode::Dense => self.run_dense(circuit, state),
        }
    }

    fn run_sparse(&mut self, circuit: &Circuit, state: &mut SparseStateVector) -> OrchestratorResult<()> {
        info!(
            target: "sparrow_sim::orchestrator",
            "Executing circuit in sparse mode with {} gates",
            circuit.gates().len()
        );
        for (index, gate) in circuit.gates().iter().enumerate() {
            trace!(
                target: "sparrow_sim::orchestrator",
                "Applying sparse gate {}: {:?}",
                index,
                gate
            );
            state.apply_gate(gate)?;
            self.prune_state(state);
        }
        Ok(())
    }

    fn run_predictive(
        &mut self,
        circuit: &Circuit,
        state: &mut SparseStateVector,
    ) -> OrchestratorResult<()> {
        info!(
            target: "sparrow_sim::orchestrator",
            "Executing circuit in predictive mode with {} gates",
            circuit.gates().len()
        );
        let predictor = self
            .predictor
            .as_ref()
            .expect("predictive mode requires predictor");

        for (index, gate) in circuit.gates().iter().enumerate() {
            if let Some(spec) = convert_gate_for_prediction(gate) {
                self.scratch.clear();
                predictor.predict_splash_zone_into(&spec, state, &mut self.scratch);
                if !self.scratch.is_empty() {
                    debug!(
                        target: "sparrow_sim::orchestrator",
                        "Gate {} predicted {} amplitudes",
                        index,
                        self.scratch.len()
                    );
                    state.reserve_additional(self.scratch.len());
                }
            }

            trace!(
                target: "sparrow_sim::orchestrator",
                "Applying predictive gate {}: {:?}",
                index,
                gate
            );
            state.apply_gate(gate)?;
            self.prune_state(state);
        }

        Ok(())
    }

    fn run_dense(&mut self, circuit: &Circuit, state: &mut SparseStateVector) -> OrchestratorResult<()> {
        let qubits = state.qubit_count();
        let limit_bits = usize::BITS - 1;
        if qubits as u32 >= limit_bits {
            return Err(OrchestratorError::DenseTooLarge(qubits));
        }
        let total_states = 1usize << qubits;
        info!(
            target: "sparrow_sim::orchestrator",
            "Executing circuit in dense mode with {} gates over {} amplitudes",
            circuit.gates().len(),
            total_states
        );

        let mut dense = vec![Complex64::new(0.0, 0.0); total_states];
        for &(basis, amplitude) in state.amplitudes().iter() {
            dense[basis as usize] = amplitude;
        }

        for (index, gate) in circuit.gates().iter().enumerate() {
            trace!(
                target: "sparrow_sim::orchestrator",
                "Applying dense gate {}: {:?}",
                index,
                gate
            );
            match gate {
                Gate::SingleQubit { qubit, matrix } => {
                    apply_dense_single_qubit(&mut dense, *qubit, matrix)
                }
                Gate::Cnot { control, target } => {
                    apply_dense_cnot(&mut dense, *control, *target)
                }
                Gate::Cz { control, target } => {
                    apply_dense_cz(&mut dense, *control, *target)
                }
                Gate::ControlledPhase {
                    control,
                    target,
                    phase,
                } => apply_dense_cphase(&mut dense, *control, *target, *phase),
            }
        }

        let mut rebuilt = sparrow_core::SparseStateVector::from_dense(
            qubits,
            &dense,
            self.config.prune_threshold,
        );
        mem::swap(state, &mut rebuilt);
        self.prune_state(state);
        Ok(())
    }

    fn prune_state(&self, state: &mut SparseStateVector) {
        if self.config.prune_threshold > 0.0 {
            state.prune(self.config.prune_threshold);
        }
    }
}

fn convert_gate_for_prediction(gate: &Gate) -> Option<MlGate> {
    match gate {
        Gate::Cnot { control, target } => Some(MlGate::Cnot {
            control: *control,
            target: *target,
        }),
        _ => None,
    }
}

fn apply_dense_single_qubit(state: &mut [Complex64], qubit: u8, gate: &[[Complex64; 2]; 2]) {
    let step = 1usize << qubit;
    let period = step << 1;
    let len = state.len();
    for base in (0..len).step_by(period) {
        for offset in 0..step {
            let i0 = base + offset;
            let i1 = i0 + step;
            if i1 >= len {
                continue;
            }
            let amp0 = state[i0];
            let amp1 = state[i1];
            state[i0] = gate[0][0] * amp0 + gate[0][1] * amp1;
            state[i1] = gate[1][0] * amp0 + gate[1][1] * amp1;
        }
    }
}

fn apply_dense_cnot(state: &mut [Complex64], control: u8, target: u8) {
    if control == target {
        return;
    }
    let control_mask = 1usize << control;
    let target_mask = 1usize << target;
    for basis in 0..state.len() {
        if (basis & control_mask) != 0 {
            let partner = basis ^ target_mask;
            if basis < partner {
                state.swap(basis, partner);
            }
        }
    }
}

fn apply_dense_cz(state: &mut [Complex64], control: u8, target: u8) {
    if control == target {
        return;
    }
    let control_mask = 1usize << control;
    let target_mask = 1usize << target;
    for (index, amplitude) in state.iter_mut().enumerate() {
        if (index & control_mask != 0) && (index & target_mask != 0) {
            *amplitude = -*amplitude;
        }
    }
}

fn apply_dense_cphase(state: &mut [Complex64], control: u8, target: u8, phase: f64) {
    if control == target {
        return;
    }
    let control_mask = 1usize << control;
    let target_mask = 1usize << target;
    let factor = Complex64::new(phase.cos(), phase.sin());
    for (index, amplitude) in state.iter_mut().enumerate() {
        if (index & control_mask != 0) && (index & target_mask != 0) {
            *amplitude *= factor;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::FRAC_1_SQRT_2;

    fn sample_circuit() -> Circuit {
        let coeff = Complex64::new(FRAC_1_SQRT_2, 0.0);
        Circuit::new(vec![
            Gate::SingleQubit {
                qubit: 0,
                matrix: [[coeff, coeff], [coeff, -coeff]],
            },
            Gate::Cnot { control: 0, target: 1 },
        ])
    }

    fn simulate(mode: SimulationMode) -> SparseStateVector {
        let config = SimulationConfig {
            mode,
            prune_threshold: 0.0,
            prediction_threshold: 0.0,
        };
        let mut orchestrator = Orchestrator::new(config);
        let mut state = SparseStateVector::new(2);
        state
            .set_amplitude(0, Complex64::new(1.0, 0.0))
            .expect("initialize");
        let circuit = sample_circuit();
        orchestrator
            .run(&circuit, &mut state)
            .expect("simulation succeeds");
        state
    }

    #[test]
    fn predictive_matches_sparse() {
        let sparse = simulate(SimulationMode::Sparse);
        let predictive = simulate(SimulationMode::Predictive);
        assert!(sparse.is_close_to(&predictive, 1e-9));
    }

    #[test]
    fn dense_matches_sparse() {
        let sparse = simulate(SimulationMode::Sparse);
        let dense = simulate(SimulationMode::Dense);
        assert!(sparse.is_close_to(&dense, 1e-9));
    }
}
