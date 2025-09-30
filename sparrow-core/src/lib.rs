#![no_std]

extern crate alloc;

use alloc::vec::Vec;
use core::mem;
use core::mem::size_of;
use core::result::Result as CoreResult;

use libm::{cos, sin, sqrt};
use log::trace;
use num_complex::Complex;

/// Alias for complex amplitudes used throughout the sparse state vector.
pub type Complex64 = Complex<f64>;

const ZERO: Complex64 = Complex64::new(0.0, 0.0);

/// Errors that can occur while manipulating a [`SparseStateVector`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseStateError {
    /// The provided qubit index is outside the range of the state.
    QubitOutOfRange,
    /// Control and target qubits overlap for a multi-qubit operation.
    ControlTargetOverlap,
    /// The requested computational basis state index exceeds the state size.
    BasisOutOfRange,
}

/// Convenience result type used by the sparse simulator.
pub type SparseResult<T> = CoreResult<T, SparseStateError>;

/// Sparse representation of a quantum state storing only significant amplitudes.
#[derive(Clone, Debug)]
pub struct SparseStateVector {
    qubit_count: u8,
    amplitudes: Vec<(u128, Complex64)>,
}

/// Enumeration of supported quantum gates for circuit execution.
#[derive(Clone, Debug)]
pub enum Gate {
    /// Arbitrary single-qubit unitary gate.
    SingleQubit {
        qubit: u8,
        matrix: [[Complex64; 2]; 2],
    },
    /// Controlled-NOT gate.
    Cnot { control: u8, target: u8 },
    /// Controlled-Z gate.
    Cz { control: u8, target: u8 },
    /// Controlled phase rotation gate.
    ControlledPhase { control: u8, target: u8, phase: f64 },
}

/// Container for an ordered sequence of gates.
#[derive(Clone, Debug)]
pub struct Circuit {
    gates: Vec<Gate>,
}

impl Circuit {
    /// Creates a new circuit from an explicit list of gates.
    pub fn new(gates: Vec<Gate>) -> Self {
        Self { gates }
    }

    /// Returns the gates stored in the circuit.
    pub fn gates(&self) -> &[Gate] {
        &self.gates
    }

    /// Appends a gate to the end of the circuit.
    pub fn push(&mut self, gate: Gate) {
        self.gates.push(gate);
    }

    /// Executes the circuit against the provided state.
    pub fn run(&self, state: &mut SparseStateVector) -> SparseResult<()> {
        log::info!(
            target: "sparrow_core::circuit",
            "Running circuit with {} gates.",
            self.gates.len()
        );
        for (index, gate) in self.gates.iter().enumerate() {
            log::trace!(
                target: "sparrow_core::circuit",
                "Applying gate {}: {:?}",
                index,
                gate
            );
            state.apply_gate(gate)?;
            #[cfg(debug_assertions)]
            debug_assert!(
                state.check_norm(1e-9),
                "state became non-normalized after gate {}",
                index
            );
        }
        Ok(())
    }
}

impl SparseStateVector {
    /// Creates a new empty sparse state with the given number of qubits.
    pub fn new(qubit_count: u8) -> Self {
        Self {
            qubit_count,
            amplitudes: Vec::new(),
        }
    }

    /// Constructs a sparse state from already-sorted amplitudes.
    pub fn from_sorted_amplitudes(
        qubit_count: u8,
        mut amplitudes: Vec<(u128, Complex64)>,
    ) -> SparseResult<Self> {
        finalize(&mut amplitudes);
        if let Some(&(basis, _)) = amplitudes.last() {
            if !basis_in_range(qubit_count, basis) {
                return Err(SparseStateError::BasisOutOfRange);
            }
        }

        Ok(Self {
            qubit_count,
            amplitudes,
        })
    }

    /// Returns the number of qubits in the state.
    pub fn qubit_count(&self) -> u8 {
        self.qubit_count
    }

    /// Returns the number of stored non-zero amplitudes.
    pub fn len(&self) -> usize {
        self.amplitudes.len()
    }

    /// Returns true if the sparse representation holds no amplitudes.
    pub fn is_empty(&self) -> bool {
        self.amplitudes.is_empty()
    }

    /// Returns an immutable view of the internal sparse entries.
    pub fn amplitudes(&self) -> &[(u128, Complex64)] {
        &self.amplitudes
    }

    /// Ensures the sparse vector has spare capacity for at least `additional` entries.
    pub fn reserve_additional(&mut self, additional: usize) {
        self.amplitudes.reserve(additional);
    }

    /// Returns the amplitude for the requested basis state or zero when absent.
    pub fn amplitude(&self, basis: u128) -> Complex64 {
        if let Ok(index) = self.position(basis) {
            self.amplitudes[index].1
        } else {
            ZERO
        }
    }

    /// Returns true when the provided computational basis state is stored.
    pub fn contains_basis(&self, basis: u128) -> bool {
        if !basis_in_range(self.qubit_count, basis) {
            return false;
        }
        self.amplitudes
            .binary_search_by(|entry| entry.0.cmp(&basis))
            .is_ok()
    }

    /// Returns the total squared magnitude of the state.
    pub fn norm_sqr(&self) -> f64 {
        self.amplitudes.iter().map(|(_, amp)| amp.norm_sqr()).sum()
    }

    /// Returns true when the state's norm is within `tolerance` of unity.
    pub fn check_norm(&self, tolerance: f64) -> bool {
        let norm = self.norm_sqr();
        (norm - 1.0).abs() <= tolerance
    }

    /// Normalizes the state to unit length if it has non-zero norm.
    pub fn normalize(&mut self) {
        let norm = sqrt(self.norm_sqr());
        if norm == 0.0 {
            return;
        }
        for (_, amplitude) in self.amplitudes.iter_mut() {
            *amplitude /= norm;
        }
    }

    /// Returns true when the amplitudes in `other` match this state within the supplied tolerance.
    pub fn is_close_to(&self, other: &Self, tolerance: f64) -> bool {
        if self.qubit_count != other.qubit_count {
            return false;
        }

        let mut lhs = 0usize;
        let mut rhs = 0usize;
        let mut error = 0.0f64;

        while lhs < self.amplitudes.len() || rhs < other.amplitudes.len() {
            match (self.amplitudes.get(lhs), other.amplitudes.get(rhs)) {
                (Some(&(basis_a, amp_a)), Some(&(basis_b, amp_b))) => {
                    if basis_a == basis_b {
                        let difference = amp_a - amp_b;
                        error += difference.norm_sqr();
                        lhs += 1;
                        rhs += 1;
                    } else if basis_a < basis_b {
                        error += amp_a.norm_sqr();
                        lhs += 1;
                    } else {
                        error += amp_b.norm_sqr();
                        rhs += 1;
                    }
                }
                (Some(&(_, amp)), None) => {
                    error += amp.norm_sqr();
                    lhs += 1;
                }
                (None, Some(&(_, amp))) => {
                    error += amp.norm_sqr();
                    rhs += 1;
                }
                (None, None) => break,
            }
            if error > tolerance * tolerance {
                return false;
            }
        }

        error <= tolerance * tolerance
    }

    /// Estimates the heap memory used by the sparse representation in bytes.
    pub fn memory_bytes(&self) -> usize {
        size_of::<Self>() + self.amplitudes.capacity() * size_of::<(u128, Complex64)>()
    }

    /// Inserts or updates the amplitude for a specific basis state.
    ///
    /// A zero amplitude removes the entry entirely.
    pub fn set_amplitude(&mut self, basis: u128, amplitude: Complex64) -> SparseResult<()> {
        self.ensure_basis(basis)?;
        match self.position(basis) {
            Ok(index) => {
                if is_zero(&amplitude) {
                    self.amplitudes.remove(index);
                } else {
                    self.amplitudes[index].1 = amplitude;
                }
            }
            Err(index) => {
                if !is_zero(&amplitude) {
                    self.amplitudes.insert(index, (basis, amplitude));
                }
            }
        }
        Ok(())
    }

    /// Constructs a sparse state from a dense representation.
    ///
    /// Entries with squared magnitude below `threshold` are discarded.
    pub fn from_dense(qubit_count: u8, dense: &[Complex64], threshold: f64) -> Self {
        let expected_len = dense_length(qubit_count);
        assert_eq!(
            dense.len(),
            expected_len,
            "dense input length does not match qubit count"
        );

        let mut amplitudes = Vec::new();
        for (index, value) in dense.iter().enumerate() {
            if value.norm_sqr() > threshold && !is_zero(value) {
                amplitudes.push((index as u128, *value));
            }
        }

        let mut state = Self {
            qubit_count,
            amplitudes,
        };
        finalize(&mut state.amplitudes);
        state
    }

    /// Applies a structured gate to the sparse state.
    pub fn apply_gate(&mut self, gate: &Gate) -> SparseResult<()> {
        match gate {
            Gate::SingleQubit { qubit, matrix } => self.apply_single_qubit(*qubit, *matrix),
            Gate::Cnot { control, target } => self.apply_cnot(*control, *target),
            Gate::Cz { control, target } => self.apply_cz(*control, *target),
            Gate::ControlledPhase {
                control,
                target,
                phase,
            } => self.apply_controlled_phase(*control, *target, *phase),
        }
    }

    /// Applies an arbitrary 2x2 single-qubit gate to the specified qubit.
    pub fn apply_single_qubit(&mut self, qubit: u8, gate: [[Complex64; 2]; 2]) -> SparseResult<()> {
        self.ensure_qubit(qubit)?;
        trace!(
            target: "sparrow_core::gates",
            "Applying single-qubit gate on qubit {} with {} amplitudes",
            qubit,
            self.amplitudes.len()
        );
        let mask = 1u128 << qubit;

        let source = mem::take(&mut self.amplitudes);
        let mut scratch = Vec::with_capacity(source.len().saturating_mul(2));

        let mut index = 0;
        while index < source.len() {
            let (state, amplitude) = source[index];
            let base = state & !mask;

            let mut amp0 = ZERO;
            let mut amp1 = ZERO;

            if state & mask == 0 {
                amp0 = amplitude;
            } else {
                amp1 = amplitude;
            }

            if index + 1 < source.len() {
                let (other_state, other_amp) = source[index + 1];
                if (other_state & !mask) == base {
                    if other_state & mask == 0 {
                        amp0 = other_amp;
                    } else {
                        amp1 = other_amp;
                    }
                    index += 1;
                }
            }

            let new0 = gate[0][0] * amp0 + gate[0][1] * amp1;
            let new1 = gate[1][0] * amp0 + gate[1][1] * amp1;

            if !is_zero(&new0) {
                scratch.push((base, new0));
            }
            if !is_zero(&new1) {
                scratch.push((base | mask, new1));
            }

            index += 1;
        }

        finalize(&mut scratch);
        self.amplitudes = scratch;
        Ok(())
    }

    /// Applies a CNOT gate with the provided control and target qubits.
    pub fn apply_cnot(&mut self, control: u8, target: u8) -> SparseResult<()> {
        if control == target {
            return Err(SparseStateError::ControlTargetOverlap);
        }
        self.ensure_qubit(control)?;
        self.ensure_qubit(target)?;

        let control_mask = 1u128 << control;
        let target_mask = 1u128 << target;

        trace!(
            target: "sparrow_core::apply_cnot",
            "Applying CNOT(c={}, t={}) to state with {} amplitudes",
            control,
            target,
            self.amplitudes.len()
        );

        let source = mem::take(&mut self.amplitudes);
        let mut scratch = Vec::with_capacity(source.len());

        let mut index = 0;
        while index < source.len() {
            let (state, amplitude) = source[index];

            if state & control_mask == 0 {
                if !is_zero(&amplitude) {
                    scratch.push((state, amplitude));
                }
                index += 1;
                continue;
            }

            let base = state & !target_mask;
            let mut amp0 = ZERO;
            let mut amp1 = ZERO;

            if state & target_mask == 0 {
                amp0 = amplitude;
            } else {
                amp1 = amplitude;
            }

            if index + 1 < source.len() {
                let (other_state, other_amp) = source[index + 1];
                if (other_state & !target_mask) == base && (other_state & control_mask) != 0 {
                    if other_state & target_mask == 0 {
                        amp0 = other_amp;
                    } else {
                        amp1 = other_amp;
                    }
                    index += 1;
                }
            }

            let swapped0 = amp1;
            let swapped1 = amp0;

            if !is_zero(&swapped0) {
                scratch.push((base, swapped0));
            }
            if !is_zero(&swapped1) {
                scratch.push((base | target_mask, swapped1));
            }

            index += 1;
        }

        finalize(&mut scratch);
        self.amplitudes = scratch;
        Ok(())
    }

    /// Applies a controlled-Z gate, flipping the phase when both qubits are 1.
    pub fn apply_cz(&mut self, control: u8, target: u8) -> SparseResult<()> {
        if control == target {
            return Err(SparseStateError::ControlTargetOverlap);
        }
        self.ensure_qubit(control)?;
        self.ensure_qubit(target)?;

        let control_mask = 1u128 << control;
        let target_mask = 1u128 << target;

        trace!(
            target: "sparrow_core::gates",
            "Applying CZ(c={}, t={}) to state with {} amplitudes",
            control,
            target,
            self.amplitudes.len()
        );

        for (state, amplitude) in self.amplitudes.iter_mut() {
            if (*state & control_mask != 0) && (*state & target_mask != 0) {
                *amplitude = -*amplitude;
            }
        }

        self.amplitudes.retain(|(_, amp)| !is_zero(amp));
        Ok(())
    }

    /// Applies a controlled phase rotation where both qubits being 1 multiplies by `exp(i * phase)`.
    pub fn apply_controlled_phase(
        &mut self,
        control: u8,
        target: u8,
        phase: f64,
    ) -> SparseResult<()> {
        if control == target {
            return Err(SparseStateError::ControlTargetOverlap);
        }
        self.ensure_qubit(control)?;
        self.ensure_qubit(target)?;

        let control_mask = 1u128 << control;
        let target_mask = 1u128 << target;
        let factor = Complex64::new(cos(phase), sin(phase));

        for (state, amplitude) in self.amplitudes.iter_mut() {
            if (*state & control_mask != 0) && (*state & target_mask != 0) {
                *amplitude *= factor;
            }
        }

        self.amplitudes.retain(|(_, amp)| !is_zero(amp));
        Ok(())
    }

    /// Removes amplitudes whose squared magnitude falls below the provided threshold.
    pub fn prune(&mut self, threshold: f64) {
        self.amplitudes.retain(|(_, amp)| {
            if is_zero(amp) {
                return false;
            }
            if threshold <= 0.0 {
                true
            } else {
                amp.norm_sqr() > threshold
            }
        });
    }

    fn ensure_qubit(&self, qubit: u8) -> SparseResult<()> {
        if qubit < self.qubit_count {
            Ok(())
        } else {
            Err(SparseStateError::QubitOutOfRange)
        }
    }

    fn ensure_basis(&self, basis: u128) -> SparseResult<()> {
        if basis_in_range(self.qubit_count, basis) {
            Ok(())
        } else {
            Err(SparseStateError::BasisOutOfRange)
        }
    }

    fn position(&self, basis: u128) -> CoreResult<usize, usize> {
        self.amplitudes
            .binary_search_by(|entry| entry.0.cmp(&basis))
    }
}

fn basis_in_range(qubits: u8, basis: u128) -> bool {
    basis <= basis_limit(qubits)
}

fn basis_limit(qubits: u8) -> u128 {
    match qubits {
        0 => 0,
        128 => u128::MAX,
        _ => (1u128 << qubits) - 1,
    }
}

fn dense_length(qubits: u8) -> usize {
    if qubits as u32 >= usize::BITS {
        panic!(
            "dense representations are not supported above {} qubits",
            usize::BITS - 1
        );
    }
    1usize << qubits
}

fn is_zero(value: &Complex64) -> bool {
    value.re == 0.0 && value.im == 0.0
}

fn finalize(entries: &mut Vec<(u128, Complex64)>) {
    if entries.is_empty() {
        return;
    }

    entries.sort_unstable_by(|a, b| a.0.cmp(&b.0));

    let mut write = 0usize;
    for read in 0..entries.len() {
        let (basis, amplitude) = entries[read];
        if is_zero(&amplitude) {
            continue;
        }

        if write > 0 && entries[write - 1].0 == basis {
            entries[write - 1].1 += amplitude;
            if is_zero(&entries[write - 1].1) {
                write -= 1;
            }
        } else {
            if write != read {
                entries[write] = (basis, amplitude);
            }
            write += 1;
        }
    }

    entries.truncate(write);
}

#[cfg(test)]
extern crate std;

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use core::f64::consts::FRAC_1_SQRT_2;

    fn assert_close(lhs: Complex64, rhs: Complex64) {
        let delta = lhs - rhs;
        assert!(
            delta.norm_sqr() < 1e-24,
            "left={lhs:?}, right={rhs:?}, delta={:?}",
            delta
        );
    }

    fn is_sorted(entries: &[(u128, Complex64)]) -> bool {
        entries.windows(2).all(|window| window[0].0 < window[1].0)
    }

    #[test]
    fn hadamard_single_qubit() {
        let mut state = SparseStateVector::new(1);
        state
            .set_amplitude(0, Complex64::new(1.0, 0.0))
            .expect("valid amplitude");

        let coeff = Complex64::new(FRAC_1_SQRT_2, 0.0);
        let gate = [[coeff, coeff], [coeff, -coeff]];

        state
            .apply_single_qubit(0, gate)
            .expect("hadamard application");

        assert_eq!(state.len(), 2);
        assert_close(state.amplitude(0), coeff);
        assert_close(state.amplitude(1), coeff);
        assert!(is_sorted(state.amplitudes()));
    }

    #[test]
    fn cnot_moves_amplitude_when_control_active() {
        let mut state = SparseStateVector::new(2);
        state
            .set_amplitude(1, Complex64::new(1.0, 0.0))
            .expect("valid amplitude");

        state.apply_cnot(0, 1).expect("cnot application");

        assert_eq!(state.len(), 1);
        assert_eq!(state.amplitudes()[0].0, 3);
        assert_close(state.amplitudes()[0].1, Complex64::new(1.0, 0.0));
        assert!(is_sorted(state.amplitudes()));
    }

    #[test]
    fn cnot_handles_missing_partner_state() {
        let mut state = SparseStateVector::new(2);
        state
            .set_amplitude(3, Complex64::new(0.5, 0.5))
            .expect("valid amplitude");

        state.apply_cnot(0, 1).expect("cnot application");

        assert_eq!(state.len(), 1);
        assert_eq!(state.amplitudes()[0].0, 1);
        assert_close(state.amplitudes()[0].1, Complex64::new(0.5, 0.5));
        assert!(is_sorted(state.amplitudes()));
    }

    #[test]
    fn cz_applies_negative_phase() {
        let mut state = SparseStateVector::new(2);
        state
            .set_amplitude(3, Complex64::new(1.0, 0.0))
            .expect("valid amplitude");

        state.apply_cz(0, 1).expect("cz application");

        assert_close(state.amplitudes()[0].1, Complex64::new(-1.0, 0.0));
    }

    #[test]
    fn controlled_phase_uses_supplied_angle() {
        let mut state = SparseStateVector::new(2);
        state
            .set_amplitude(3, Complex64::new(1.0, 0.0))
            .expect("valid amplitude");

        state
            .apply_controlled_phase(0, 1, core::f64::consts::PI / 2.0)
            .expect("phase application");

        assert_close(state.amplitudes()[0].1, Complex64::new(0.0, 1.0));
    }

    #[test]
    fn prune_removes_small_entries() {
        let mut state = SparseStateVector::new(1);
        state
            .set_amplitude(0, Complex64::new(1.0, 0.0))
            .expect("valid amplitude");
        state
            .set_amplitude(1, Complex64::new(1e-10, 0.0))
            .expect("valid amplitude");

        state.prune(1e-12);

        assert_eq!(state.len(), 1);
        assert_eq!(state.amplitudes()[0].0, 0);
    }

    #[test]
    fn from_dense_respects_threshold() {
        let dense = vec![Complex64::new(1.0, 0.0), Complex64::new(1e-9, 0.0)];
        let state = SparseStateVector::from_dense(1, &dense, 1e-12);
        assert_eq!(state.len(), 1);
        assert_eq!(state.amplitudes()[0].0, 0);
    }

    #[test]
    fn from_sorted_amplitudes_normalizes_input() {
        let entries = vec![
            (2, Complex64::new(1.0, 0.0)),
            (2, Complex64::new(1.0, 0.0)),
            (1, Complex64::new(1.0, 0.0)),
        ];
        let state = SparseStateVector::from_sorted_amplitudes(2, entries).expect("builder");
        assert_eq!(state.len(), 2);
        assert_eq!(state.amplitudes()[0].0, 1);
        assert_eq!(state.amplitudes()[1].0, 2);
    }

    #[test]
    fn normalize_scales_amplitudes() {
        let mut state = SparseStateVector::new(1);
        state
            .set_amplitude(0, Complex64::new(3.0, 0.0))
            .expect("valid amplitude");
        state
            .set_amplitude(1, Complex64::new(4.0, 0.0))
            .expect("valid amplitude");

        state.normalize();
        let norm = state.norm_sqr();
        assert!((norm - 1.0).abs() < 1e-12);
        assert!(state.check_norm(1e-12));
    }

    #[test]
    fn contains_basis_checks_membership() {
        let mut state = SparseStateVector::new(2);
        state
            .set_amplitude(2, Complex64::new(1.0, 0.0))
            .expect("valid amplitude");
        assert!(state.contains_basis(2));
        assert!(!state.contains_basis(1));
    }

    #[test]
    fn reserve_additional_does_not_change_length() {
        let mut state = SparseStateVector::new(3);
        state.reserve_additional(16);
        assert_eq!(state.len(), 0);
    }

    #[test]
    fn circuit_executes_gate_sequence() {
        let coeff = Complex64::new(FRAC_1_SQRT_2, 0.0);
        let hadamard = Gate::SingleQubit {
            qubit: 0,
            matrix: [[coeff, coeff], [coeff, -coeff]],
        };
        let cnot = Gate::Cnot { control: 0, target: 1 };

        let circuit = Circuit::new(vec![hadamard, cnot]);

        let mut state = SparseStateVector::new(2);
        state
            .set_amplitude(0, Complex64::new(1.0, 0.0))
            .expect("valid amplitude");

        circuit.run(&mut state).expect("circuit execution");

        assert_eq!(state.len(), 2);
        assert_close(state.amplitude(0), coeff);
        assert_close(state.amplitude(3), coeff);
        assert!(state.check_norm(1e-12));
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn complex_strategy() -> impl Strategy<Value = Complex64> {
        let range = -1.0f64..1.0f64;
        (range.clone(), range).prop_map(|(re, im)| Complex64::new(re, im))
    }

    fn unitary_strategy() -> impl Strategy<Value = [[Complex64; 2]; 2]> {
        prop::collection::vec(complex_strategy(), 4).prop_map(|values| {
            let mut col0 = [values[0], values[1]];
            if is_zero(&col0[0]) && is_zero(&col0[1]) {
                col0 = [Complex64::new(1.0, 0.0), ZERO];
            }
            let norm0 = sqrt(col0[0].norm_sqr() + col0[1].norm_sqr());
            if norm0 != 0.0 {
                col0[0] /= norm0;
                col0[1] /= norm0;
            } else {
                col0 = [Complex64::new(1.0, 0.0), ZERO];
            }

            let mut col1 = [values[2], values[3]];
            let projection = col0[0].conj() * col1[0] + col0[1].conj() * col1[1];
            col1[0] -= projection * col0[0];
            col1[1] -= projection * col0[1];
            let norm1 = sqrt(col1[0].norm_sqr() + col1[1].norm_sqr());
            if norm1 != 0.0 {
                col1[0] /= norm1;
                col1[1] /= norm1;
            } else {
                col1 = [-col0[1].conj(), col0[0].conj()];
            }

            [[col0[0], col1[0]], [col0[1], col1[1]]]
        })
    }

    fn sparse_state_strategy() -> impl Strategy<Value = SparseStateVector> {
        (1u8..=6).prop_flat_map(|qubits| {
            let mask = if qubits == 128 {
                u128::MAX
            } else {
                (1u128 << qubits) - 1
            };
            let basis = proptest::num::u128::ANY.prop_map(move |value| value & mask);
            let complex = complex_strategy();
            prop::collection::btree_map(basis, complex, 1..=16).prop_map(move |map| {
                let mut entries: Vec<(u128, Complex64)> = map.into_iter().collect();
                if entries
                    .iter()
                    .all(|(_, amplitude)| is_zero(amplitude))
                {
                    entries.clear();
                }

                let mut state = if entries.is_empty() {
                    SparseStateVector::new(qubits)
                } else {
                    SparseStateVector::from_sorted_amplitudes(qubits, entries).expect("valid state")
                };

                if state.is_empty() {
                    state
                        .set_amplitude(0, Complex64::new(1.0, 0.0))
                        .expect("baseline amplitude");
                }

                state.normalize();
                state
            })
        })
    }

    fn conjugate_transpose(matrix: [[Complex64; 2]; 2]) -> [[Complex64; 2]; 2] {
        [
            [matrix[0][0].conj(), matrix[1][0].conj()],
            [matrix[0][1].conj(), matrix[1][1].conj()],
        ]
    }

    proptest! {
        #[test]
        fn single_qubit_gate_inverse_is_identity(matrix in unitary_strategy(), state in sparse_state_strategy()) {
            prop_assert!(state.check_norm(1e-9));

            let inverse = conjugate_transpose(matrix);

            let mut forward = state.clone();
            forward.apply_single_qubit(0, matrix).expect("forward gate");
            forward.apply_single_qubit(0, inverse).expect("inverse gate");

            prop_assert!(forward.check_norm(1e-9));
            prop_assert!(forward.is_close_to(&state, 1e-8));
        }
    }
}
