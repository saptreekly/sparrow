use std::error::Error;
use std::fs::{File, create_dir_all};
use std::io::Write;

use plotters::prelude::*;

use log::info;
use sparrow_ml::dataset::{DatasetConfig, generate_dataset};
use sparrow_ml::predictor::SplashPredictor;

fn main() -> Result<(), Box<dyn Error>> {
    let _ = env_logger::builder().format_timestamp_millis().try_init();

    info!("generating precision-recall dataset");
    let config = DatasetConfig {
        qubits: 22,
        num_states: 600,
        min_non_zero: 20,
        max_non_zero: 140,
        label_threshold: 1e-10,
        gate: sparrow_ml::gate::GateSpec::Cnot {
            control: 6,
            target: 1,
        },
        seed: 0xDEADBEEF,
    };

    let dataset = generate_dataset(&config);
    info!("precision-recall evaluation on {} samples", dataset.len());

    let thresholds = logspace(1e-4, 0.2, 60);

    let mut records = Vec::new();
    for threshold in thresholds {
        let (precision, recall) = evaluate(&dataset, threshold);
        records.push((threshold, precision, recall));
    }

    create_dir_all("benchmarks")?;
    write_csv(&records)?;
    plot_precision_recall(&records)?;

    Ok(())
}

fn evaluate(dataset: &[sparrow_ml::dataset::Sample], threshold: f64) -> (f64, f64) {
    let predictor = SplashPredictor::pretrained().with_threshold(threshold);
    let mut tp = 0.0;
    let mut fp = 0.0;
    let mut fn_ = 0.0;

    for sample in dataset {
        let probability = predictor.model().predict_probability(&sample.features);
        let predicted = probability >= threshold;
        match (predicted, sample.label) {
            (true, true) => tp += 1.0,
            (true, false) => fp += 1.0,
            (false, true) => fn_ += 1.0,
            _ => {}
        }
    }

    let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 1.0 };
    let recall = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 1.0 };
    (precision, recall)
}

fn write_csv(records: &[(f64, f64, f64)]) -> Result<(), Box<dyn Error>> {
    let mut file = File::create("benchmarks/precision_recall.csv")?;
    writeln!(file, "threshold,precision,recall")?;
    for (threshold, precision, recall) in records {
        writeln!(file, "{threshold:.8},{precision:.6},{recall:.6}")?;
    }
    Ok(())
}

fn plot_precision_recall(records: &[(f64, f64, f64)]) -> Result<(), Box<dyn Error>> {
    let root =
        BitMapBackend::new("benchmarks/precision_recall.png", (1280, 720)).into_drawing_area();
    root.fill(&WHITE)?;

    let recall_min = records
        .iter()
        .map(|(_, _, recall)| *recall)
        .fold(1.0, f64::min)
        .min(0.9);

    let mut chart = ChartBuilder::on(&root)
        .caption("Precision vs Recall", ("sans-serif", 32).into_font())
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(70)
        .build_cartesian_2d(recall_min..1.01, 0.0..1.05)?;

    chart
        .configure_mesh()
        .x_desc("Recall")
        .y_desc("Precision")
        .y_labels(11)
        .x_label_formatter(&|value| format!("{value:.4}"))
        .y_label_formatter(&|value| format!("{value:.2}"))
        .draw()?;

    chart.draw_series(LineSeries::new(
        records
            .iter()
            .map(|(_, precision, recall)| (*recall, *precision)),
        &BLUE,
    ))?;

    chart.draw_series(
        records
            .iter()
            .map(|(_, precision, recall)| Circle::new((*recall, *precision), 3, BLUE.filled())),
    )?;

    root.present()?;
    Ok(())
}

fn logspace(start: f64, end: f64, samples: usize) -> Vec<f64> {
    assert!(start > 0.0 && end > start);
    let log_start = start.log10();
    let log_end = end.log10();
    (0..samples)
        .map(|i| {
            let t = i as f64 / ((samples - 1) as f64);
            10f64.powf(log_start + (log_end - log_start) * t)
        })
        .collect()
}
