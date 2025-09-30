use sparrow_core::{Complex64, SparseStateVector};

use plotters::prelude::*;
use std::error::Error;
use std::fs::{File, create_dir_all};
use std::io::Write;

const OUTPUT_DIR: &str = "benchmarks";
const CSV_NAME: &str = "fidelity_frontier.csv";
const PLOT_NAME: &str = "fidelity_frontier.png";

const QUBITS: u8 = 8;

fn main() -> Result<(), Box<dyn Error>> {
    create_dir_all(OUTPUT_DIR)?;

    let initial = prepare_initial_state(QUBITS);
    let dense = qft_dense_reference(QUBITS);

    let thresholds = linear_space(0.0, 1e-8, 100);
    let mut records = Vec::with_capacity(thresholds.len());

    for threshold in thresholds {
        let mut state = initial.clone();
        apply_qft_sparse(&mut state, threshold);
        let fidelity = fidelity_against(&state, &dense);
        records.push((threshold, fidelity));
    }

    write_csv(&records)?;
    plot_fidelity(&records)?;

    Ok(())
}

fn prepare_initial_state(qubits: u8) -> SparseStateVector {
    let mut state = SparseStateVector::new(qubits);
    state
        .set_amplitude(1, Complex64::new(1.0, 0.0))
        .expect("initial amplitude assignment");
    state
}

fn apply_qft_sparse(state: &mut SparseStateVector, threshold: f64) {
    let n = state.qubit_count();
    for target in 0..n {
        for control in 0..target {
            let angle = std::f64::consts::TAU / (1u64 << ((target - control) as u32 + 1)) as f64;
            state
                .apply_controlled_phase(control, target, angle)
                .expect("controlled phase within range");
            if threshold > 0.0 {
                state.prune(threshold);
            }
        }
        state
            .apply_single_qubit(target, hadamard())
            .expect("hadamard application");
        if threshold > 0.0 {
            state.prune(threshold);
        }
    }

    let half = n / 2;
    for i in 0..half {
        let a = i;
        let b = n - 1 - i;
        state.apply_cnot(a, b).expect("swap step 1");
        state.apply_cnot(b, a).expect("swap step 2");
        state.apply_cnot(a, b).expect("swap step 3");
        if threshold > 0.0 {
            state.prune(threshold);
        }
    }

    state.normalize();
}

fn hadamard() -> [[Complex64; 2]; 2] {
    let coeff = 1.0 / (2.0_f64).sqrt();
    [
        [Complex64::new(coeff, 0.0), Complex64::new(coeff, 0.0)],
        [Complex64::new(coeff, 0.0), Complex64::new(-coeff, 0.0)],
    ]
}

fn qft_dense_reference(qubits: u8) -> Vec<Complex64> {
    let n = 1usize << qubits;
    let norm = (n as f64).sqrt();
    (0..n)
        .map(|k| {
            let phase = std::f64::consts::TAU * k as f64 / n as f64;
            Complex64::new(phase.cos(), phase.sin()) / norm
        })
        .collect()
}

fn fidelity_against(state: &SparseStateVector, dense: &[Complex64]) -> f64 {
    let overlap = state
        .amplitudes()
        .iter()
        .fold(Complex64::new(0.0, 0.0), |acc, (index, amplitude)| {
            acc + dense[*index as usize].conj() * *amplitude
        });
    overlap.norm_sqr()
}

fn write_csv(records: &[(f64, f64)]) -> Result<(), Box<dyn Error>> {
    let path = format!("{OUTPUT_DIR}/{CSV_NAME}");
    let mut file = File::create(path)?;
    writeln!(file, "threshold,fidelity")?;
    for (threshold, fidelity) in records {
        writeln!(file, "{threshold:.10},{fidelity:.10}")?;
    }
    Ok(())
}

fn plot_fidelity(records: &[(f64, f64)]) -> Result<(), Box<dyn Error>> {
    let path = format!("{OUTPUT_DIR}/{PLOT_NAME}");
    let root = BitMapBackend::new(&path, (1280, 720)).into_drawing_area();
    root.fill(&WHITE)?;

    let y_min = records
        .iter()
        .map(|(_, fidelity)| *fidelity)
        .fold(1.0_f64, f64::min)
        .min(0.0);

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Fidelity vs Pruning Threshold (QFT on |1〉)",
            ("sans-serif", 30).into_font(),
        )
        .margin(20)
        .x_label_area_size(60)
        .y_label_area_size(70)
        .build_cartesian_2d(0.0_f64..1e-8_f64, y_min..1.02)?;

    chart
        .configure_mesh()
        .x_desc("Pruning threshold")
        .y_desc("Fidelity")
        .x_label_formatter(&|value| format!("{value:0.2e}"))
        .y_label_formatter(&|value| format!("{value:.3}"))
        .draw()?;

    chart.draw_series(LineSeries::new(records.iter().cloned(), &BLUE))?;
    chart.draw_series(
        records
            .iter()
            .map(|(x, y)| Circle::new((*x, *y), 3, BLUE.filled())),
    )?;

    root.present()?;
    Ok(())
}

fn linear_space(start: f64, end: f64, samples: usize) -> Vec<f64> {
    if samples <= 1 {
        return vec![start];
    }
    (0..samples)
        .map(|i| start + (end - start) * i as f64 / (samples as f64 - 1.0))
        .collect()
}
