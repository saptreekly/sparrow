use sparrow_core::{Complex64, SparseStateVector};

use plotters::prelude::*;
use std::error::Error;
use std::fs::{File, create_dir_all};
use std::io::Write;
use std::mem::size_of;

const OUTPUT_DIR: &str = "benchmarks";
const CSV_NAME: &str = "memory_curve.csv";
const PLOT_NAME: &str = "memory_curve.png";

fn main() -> Result<(), Box<dyn Error>> {
    create_dir_all(OUTPUT_DIR)?;

    let qubits = 20u8;
    let total_states = 1usize << qubits;
    let dense_bytes = total_states * size_of::<Complex64>();
    let dense_mb = bytes_to_mb(dense_bytes as u64);

    let fractions = logspace(0.0001, 1.0, 32);
    let mut records = Vec::with_capacity(fractions.len());

    for fraction in fractions {
        let non_zero = ((fraction * total_states as f64).round() as usize).max(1);
        let mut entries = Vec::with_capacity(non_zero);
        for basis in 0..non_zero {
            entries.push((basis as u128, Complex64::new(1.0, 0.0)));
        }
        entries.shrink_to_fit();
        let state = SparseStateVector::from_sorted_amplitudes(qubits, entries)
            .expect("failed to build sparse state");
        let sparse_mb = bytes_to_mb(state.memory_bytes() as u64);
        let sparsity_percent = fraction * 100.0;

        records.push((sparsity_percent, sparse_mb));
    }

    write_csv(&records, dense_mb)?;
    plot_memory_curve(&records, dense_mb)?;

    Ok(())
}

fn write_csv(records: &[(f64, f64)], dense_mb: f64) -> Result<(), Box<dyn Error>> {
    let path = format!("{OUTPUT_DIR}/{CSV_NAME}");
    let mut file = File::create(path)?;
    writeln!(file, "sparsity_percent,sparse_memory_mb,dense_memory_mb")?;
    for (sparsity, sparse_mb) in records {
        writeln!(file, "{sparsity:.6},{sparse_mb:.6},{dense_mb:.6}")?;
    }
    Ok(())
}

fn plot_memory_curve(records: &[(f64, f64)], dense_mb: f64) -> Result<(), Box<dyn Error>> {
    let path = format!("{OUTPUT_DIR}/{PLOT_NAME}");
    let root = BitMapBackend::new(&path, (1280, 720)).into_drawing_area();
    root.fill(&WHITE)?;

    let x_max = records
        .iter()
        .map(|(sparsity, _)| *sparsity)
        .fold(0.0_f64, f64::max)
        .max(100.0);
    let y_max = records
        .iter()
        .map(|(_, sparse_mb)| *sparse_mb)
        .fold(dense_mb, f64::max)
        * 1.1;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Memory Footprint vs Sparsity (20 qubits)",
            ("sans-serif", 32).into_font(),
        )
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d(0.01_f64..x_max, 0.0_f64..y_max)?;

    chart
        .configure_mesh()
        .x_desc("Non-zero amplitudes (%)")
        .y_desc("Memory (MB)")
        .x_label_formatter(&|value| format!("{value:.2}"))
        .draw()?;

    let series: Vec<(f64, f64)> = records.to_vec();
    chart
        .draw_series(LineSeries::new(series.clone(), &BLUE))?
        .label("Sparse state")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE.filled()));

    chart.draw_series(
        records
            .iter()
            .map(|(x, y)| Circle::new((*x, *y), 3, BLUE.filled())),
    )?;

    chart
        .draw_series(LineSeries::new(
            vec![(0.01_f64, dense_mb), (x_max, dense_mb)],
            &RED,
        ))?
        .label("Dense state")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.filled()));

    chart
        .configure_series_labels()
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}

fn logspace(start: f64, end: f64, samples: usize) -> Vec<f64> {
    let log_start = start.log10();
    let log_end = end.log10();
    (0..samples)
        .map(|i| {
            let t = i as f64 / (samples.saturating_sub(1) as f64);
            10f64.powf(log_start + (log_end - log_start) * t)
        })
        .collect()
}

fn bytes_to_mb(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}
