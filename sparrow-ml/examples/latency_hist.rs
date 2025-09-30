use std::error::Error;
use std::fs::{File, create_dir_all};
use std::io::Write;
use std::time::Instant;

use plotters::prelude::*;
use rand::distributions::{Distribution, Uniform};
use rand::{Rng, SeedableRng};

use log::{debug, info};
use sparrow_core::{Complex64, SparseStateVector};
use sparrow_ml::gate::GateSpec;
use sparrow_ml::predictor::SplashPredictor;

const ITERATIONS: usize = 250_000;
const STATE_POOL: usize = 128;

fn main() -> Result<(), Box<dyn Error>> {
    let _ = env_logger::builder().format_timestamp_millis().try_init();

    info!("running latency histogram benchmark");
    let predictor = SplashPredictor::pretrained();
    let gate = GateSpec::Cnot {
        control: 6,
        target: 1,
    };

    let mut rng = rand::rngs::StdRng::seed_from_u64(0xA17CE);
    let states = build_state_pool(24, STATE_POOL, &mut rng);
    let mut buffer = Vec::new();
    let mut durations = Vec::with_capacity(ITERATIONS);

    for i in 0..ITERATIONS {
        let state = &states[i % states.len()];
        let start = Instant::now();
        predictor.predict_splash_zone_into(&gate, state, &mut buffer);
        let elapsed = start.elapsed().as_nanos() as u64;
        durations.push(elapsed);
    }

    let cnot_baseline = measure_cnot_baseline(&states[0], &gate);
    let p99 = percentile(&mut durations.clone(), 0.99);
    debug!(
        target: "sparrow_ml::latency",
        "baseline seconds per CNOT: {:.6}",
        cnot_baseline
    );
    info!(
        "Splash prediction p99 latency: {} ns (baseline CNOT: {:.0} ns)",
        p99,
        cnot_baseline * 1e9
    );

    create_dir_all("benchmarks")?;
    write_histogram_csv(&durations, "benchmarks/latency_histogram.csv")?;
    plot_histogram(&durations, p99 as f64, cnot_baseline * 1e9)?;

    Ok(())
}

fn build_state_pool(
    qubits: u8,
    pool_size: usize,
    rng: &mut rand::rngs::StdRng,
) -> Vec<SparseStateVector> {
    let mut states = Vec::with_capacity(pool_size);
    let basis_dist = Uniform::new(0u128, 1u128 << qubits);
    let log_dist = Uniform::new(-9.0, 0.0);
    let phase_dist = Uniform::new(0.0, std::f64::consts::TAU);

    for _ in 0..pool_size {
        let mut state = SparseStateVector::new(qubits);
        let entries = rng.gen_range(200..600);
        for _ in 0..entries {
            let basis = basis_dist.sample(rng);
            let log_mag = log_dist.sample(rng);
            let magnitude = 10f64.powf(log_mag);
            let phase = phase_dist.sample(rng);
            let amplitude = Complex64::new(phase.cos() * magnitude, phase.sin() * magnitude);
            let _ = state.set_amplitude(basis, amplitude);
        }
        state.normalize();
        state.prune(1e-14);
        states.push(state);
    }

    states
}

fn measure_cnot_baseline(state: &SparseStateVector, gate: &GateSpec) -> f64 {
    let iterations = 1000;
    let mut total = 0.0;
    for _ in 0..iterations {
        let mut working = state.clone();
        let start = Instant::now();
        gate.apply(&mut working).unwrap();
        total += start.elapsed().as_secs_f64();
    }
    total / iterations as f64
}

fn percentile(values: &mut [u64], quantile: f64) -> u64 {
    values.sort_unstable();
    let position = ((values.len() as f64) * quantile).ceil() as usize;
    let index = position.clamp(1, values.len()) - 1;
    values[index]
}

fn write_histogram_csv(data: &[u64], path: &str) -> Result<(), Box<dyn Error>> {
    let bins = histogram(data, 40);
    let mut file = File::create(path)?;
    writeln!(file, "bin_start_ns,bin_end_ns,count")?;
    for bin in bins {
        writeln!(file, "{},{},{}", bin.start, bin.end, bin.count)?;
    }
    Ok(())
}

fn plot_histogram(data: &[u64], p99: f64, baseline_ns: f64) -> Result<(), Box<dyn Error>> {
    let bins = histogram(data, 40);
    let max_count = bins.iter().map(|bin| bin.count).max().unwrap_or(1) as f64;

    let root =
        BitMapBackend::new("benchmarks/latency_histogram.png", (1280, 720)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_ns = bins.last().map(|bin| bin.end).unwrap_or(1);

    let mut chart = ChartBuilder::on(&root)
        .caption("Splash Prediction Latency", ("sans-serif", 32).into_font())
        .margin(20)
        .x_label_area_size(60)
        .y_label_area_size(70)
        .build_cartesian_2d(0f64..max_ns as f64, 0f64..(max_count * 1.1))?;

    chart
        .configure_mesh()
        .x_desc("Latency (ns)")
        .y_desc("Count")
        .y_labels(10)
        .x_label_formatter(&|v| format!("{v:.0}"))
        .draw()?;

    chart.draw_series(bins.iter().map(|bin| {
        let x0 = bin.start as f64;
        let x1 = bin.end as f64;
        let y = bin.count as f64;
        Rectangle::new([(x0, 0.0), (x1, y)], BLUE.mix(0.6).filled())
    }))?;

    chart
        .draw_series(
            [p99].iter().map(|value| {
                PathElement::new(vec![(*value, 0.0), (*value, max_count * 1.1)], &RED)
            }),
        )?
        .label("99th percentile")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 40, y)], RED));

    chart
        .draw_series(
            [baseline_ns].iter().map(|value| {
                PathElement::new(vec![(*value, 0.0), (*value, max_count * 1.1)], &BLACK)
            }),
        )?
        .label("CNOT baseline")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 40, y)], &BLACK));

    chart
        .configure_series_labels()
        .border_style(&BLACK)
        .draw()?;
    root.present()?;
    Ok(())
}

#[derive(Clone)]
struct Bin {
    start: u64,
    end: u64,
    count: u64,
}

fn histogram(data: &[u64], bins: usize) -> Vec<Bin> {
    if data.is_empty() {
        return Vec::new();
    }
    let min = data.iter().copied().min().unwrap();
    let max = data.iter().copied().max().unwrap().max(1);
    let span = (max - min).max(1);
    let width = span as f64 / bins as f64;

    let mut histogram = vec![
        Bin {
            start: 0,
            end: 0,
            count: 0,
        };
        bins
    ];

    for (i, bin) in histogram.iter_mut().enumerate() {
        let start = min as f64 + width * i as f64;
        bin.start = start.round() as u64;
        bin.end = (start + width).round() as u64;
    }

    for value in data {
        let index = (((*value as f64) - min as f64) / width)
            .floor()
            .clamp(0.0, (bins - 1) as f64) as usize;
        histogram[index].count += 1;
    }

    histogram
}
