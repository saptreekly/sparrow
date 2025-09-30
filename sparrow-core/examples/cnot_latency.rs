use sparrow_core::{Complex64, SparseStateVector};

use plotters::prelude::*;
use std::error::Error;
use std::fs::{File, create_dir_all};
use std::io::Write;
use std::time::Instant;

const OUTPUT_DIR: &str = "benchmarks";
const CSV_NAME: &str = "cnot_latency.csv";
const PLOT_NAME: &str = "cnot_latency.png";

const QUBITS: u8 = 28;
const CONTROL: u8 = 3;
const TARGET: u8 = 0;

fn main() -> Result<(), Box<dyn Error>> {
    create_dir_all(OUTPUT_DIR)?;

    let dense_per_op = dense_time_per_amplitude(20, CONTROL, TARGET);
    let dense_states = 1u64 << QUBITS;
    let dense_estimate_us = dense_per_op * dense_states as f64 * 1e6;

    let k_values: Vec<usize> = (10..=20).map(|exp| 1usize << exp).collect();

    let mut records = Vec::with_capacity(k_values.len());
    for k in k_values {
        let state = make_balanced_state(QUBITS, k, CONTROL, TARGET);
        let avg_us = measure_sparse_cnot(&state, CONTROL, TARGET);
        records.push((k as f64, avg_us));
    }

    write_csv(&records, dense_estimate_us)?;
    plot_latency_curve(&records, dense_estimate_us)?;

    Ok(())
}

fn dense_time_per_amplitude(qubits: u8, control: u8, target: u8) -> f64 {
    let len = 1usize << qubits;
    let mut data = vec![Complex64::new(1.0, 0.0); len];
    let iterations = 8;

    let start = Instant::now();
    for _ in 0..iterations {
        apply_dense_cnot(&mut data, control, target);
    }
    let elapsed = start.elapsed().as_secs_f64();
    elapsed / (iterations as f64 * len as f64)
}

fn apply_dense_cnot(state: &mut [Complex64], control: u8, target: u8) {
    let control_mask = 1usize << control;
    let target_mask = 1usize << target;
    for index in 0..state.len() {
        if index & control_mask != 0 && index & target_mask == 0 {
            let partner = index ^ target_mask;
            state.swap(index, partner);
        }
    }
}

fn make_balanced_state(qubits: u8, k: usize, control: u8, target: u8) -> SparseStateVector {
    let mut entries = Vec::with_capacity(k);
    let control_mask = 1u128 << control;
    let target_mask = 1u128 << target;

    let pairs = k / 2;
    for i in 0..pairs {
        let base = (i as u128) << (target.max(control) + 1);
        entries.push((base | control_mask, Complex64::new(0.7, 0.0)));
        entries.push((base | control_mask | target_mask, Complex64::new(0.7, 0.0)));
    }

    if k % 2 == 1 {
        let base = (pairs as u128) << (target.max(control) + 1);
        entries.push((base | control_mask, Complex64::new(1.0, 0.0)));
    }

    entries.shrink_to_fit();
    SparseStateVector::from_sorted_amplitudes(qubits, entries)
        .expect("balanced state construction failed")
}

fn measure_sparse_cnot(state: &SparseStateVector, control: u8, target: u8) -> f64 {
    let mut working = state.clone();
    let iterations = 100u32;
    let start = Instant::now();
    for _ in 0..iterations {
        working.apply_cnot(control, target).unwrap();
    }
    let elapsed = start.elapsed().as_secs_f64();
    elapsed * 1e6 / iterations as f64
}

fn write_csv(records: &[(f64, f64)], dense_estimate_us: f64) -> Result<(), Box<dyn Error>> {
    let path = format!("{OUTPUT_DIR}/{CSV_NAME}");
    let mut file = File::create(path)?;
    writeln!(
        file,
        "non_zero_amplitudes,sparse_latency_us,dense_estimate_us"
    )?;
    for (k, sparse) in records {
        writeln!(file, "{k:.0},{sparse:.6},{dense_estimate_us:.6}")?;
    }
    Ok(())
}

fn plot_latency_curve(
    records: &[(f64, f64)],
    dense_estimate_us: f64,
) -> Result<(), Box<dyn Error>> {
    let path = format!("{OUTPUT_DIR}/{PLOT_NAME}");
    let root = BitMapBackend::new(&path, (1280, 720)).into_drawing_area();
    root.fill(&WHITE)?;

    let x_range = records
        .iter()
        .map(|(k, _)| k.log10())
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), x| {
            (min.min(x), max.max(x))
        });
    let y_max = records
        .iter()
        .map(|(_, sparse)| *sparse)
        .fold(dense_estimate_us, f64::max)
        * 1.2;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "CNOT Latency Scaling (28 qubits)",
            ("sans-serif", 32).into_font(),
        )
        .margin(20)
        .x_label_area_size(60)
        .y_label_area_size(70)
        .build_cartesian_2d(x_range.0..x_range.1, 0.0_f64..y_max)?;

    chart
        .configure_mesh()
        .x_desc("log10(non-zero amplitudes)")
        .y_desc("Latency (microseconds)")
        .x_label_formatter(&|value| format!("{value:.2}"))
        .draw()?;

    let sparse_series: Vec<(f64, f64)> = records
        .iter()
        .map(|(k, sparse)| (k.log10(), *sparse))
        .collect();

    chart
        .draw_series(LineSeries::new(sparse_series.clone(), &BLUE))?
        .label("Sparse CNOT")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], BLUE.filled()));

    chart.draw_series(
        sparse_series
            .iter()
            .map(|(x, y)| Circle::new((*x, *y), 4, BLUE.filled())),
    )?;

    chart
        .draw_series(LineSeries::new(
            vec![
                (x_range.0, dense_estimate_us),
                (x_range.1, dense_estimate_us),
            ],
            &RED,
        ))?
        .label("Dense estimate")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], RED.filled()));

    chart
        .configure_series_labels()
        .border_style(&BLACK)
        .draw()?;
    root.present()?;
    Ok(())
}
