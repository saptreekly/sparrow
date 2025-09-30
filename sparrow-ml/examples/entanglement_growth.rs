use std::collections::HashSet;
use std::error::Error;
use std::fs::{File, create_dir_all};
use std::io::Write;

use plotters::prelude::*;
use rand::{Rng, SeedableRng, seq::SliceRandom};

use log::info;
use sparrow_core::{Complex64, SparseStateVector};
use sparrow_ml::gate::GateSpec;
use sparrow_ml::predictor::SplashPredictor;

const STEPS: usize = 200;

fn main() -> Result<(), Box<dyn Error>> {
    let _ = env_logger::builder().format_timestamp_millis().try_init();

    info!("running entanglement growth benchmark");
    let mut state = SparseStateVector::new(16);
    state
        .set_amplitude(0, Complex64::new(1.0, 0.0))
        .expect("initial basis");

    let predictor = SplashPredictor::pretrained().with_threshold(0.02);
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xF1E1D1);

    let mut records = Vec::with_capacity(STEPS);

    for step in 0..STEPS {
        inject_superposition(&mut state, &mut rng);

        let (control, target) = random_pair(state.qubit_count(), &mut rng);
        let gate = GateSpec::Cnot { control, target };

        let predicted = predictor.predict_splash_zone(&gate, &state).len();

        let before: HashSet<u128> = state.amplitudes().iter().map(|(basis, _)| *basis).collect();
        let mut after = state.clone();
        gate.apply(&mut after).unwrap();
        after.prune(1e-12);
        after.normalize();

        let actual_new = after
            .amplitudes()
            .iter()
            .filter(|(basis, _)| !before.contains(basis))
            .count();

        records.push((
            step as u32,
            predicted as u32,
            actual_new as u32,
            after.len() as u32,
        ));
        state = after;
    }

    create_dir_all("benchmarks")?;
    write_csv(&records)?;
    plot_growth(&records)?;

    Ok(())
}

fn inject_superposition(state: &mut SparseStateVector, rng: &mut rand::rngs::StdRng) {
    let qubits = state.qubit_count();
    if qubits == 0 {
        return;
    }
    let qubit = rng.gen_range(0..qubits);
    state
        .apply_single_qubit(qubit, hadamard())
        .expect("hadamard application");
    state.prune(1e-12);
    state.normalize();
}

fn random_pair(qubits: u8, rng: &mut rand::rngs::StdRng) -> (u8, u8) {
    if qubits < 2 {
        return (0, 0);
    }
    let mut indices: Vec<u8> = (0..qubits).collect();
    indices.shuffle(rng);
    (indices[0], indices[1])
}

fn hadamard() -> [[Complex64; 2]; 2] {
    let coeff = 1.0 / (2.0_f64).sqrt();
    [
        [Complex64::new(coeff, 0.0), Complex64::new(coeff, 0.0)],
        [Complex64::new(coeff, 0.0), Complex64::new(-coeff, 0.0)],
    ]
}

fn write_csv(records: &[(u32, u32, u32, u32)]) -> Result<(), Box<dyn Error>> {
    let mut file = File::create("benchmarks/entanglement_growth.csv")?;
    writeln!(file, "step,predicted,actual_new,total_non_zero")?;
    for (step, predicted, actual_new, total) in records {
        writeln!(file, "{step},{predicted},{actual_new},{total}")?;
    }
    Ok(())
}

fn plot_growth(records: &[(u32, u32, u32, u32)]) -> Result<(), Box<dyn Error>> {
    let root =
        BitMapBackend::new("benchmarks/entanglement_growth.png", (1280, 720)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_step = records.last().map(|(step, _, _, _)| *step).unwrap_or(0);
    let max_predicted = records
        .iter()
        .map(|(_, predicted, _, total)| (*predicted).max(*total))
        .max()
        .unwrap_or(1);

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Predicted Splash Zone vs Entanglement",
            ("sans-serif", 32).into_font(),
        )
        .margin(20)
        .x_label_area_size(60)
        .y_label_area_size(70)
        .build_cartesian_2d(0u32..(max_step + 1), 0u32..(max_predicted + 10))?;

    chart
        .configure_mesh()
        .x_desc("Entangling gates applied")
        .y_desc("Count")
        .y_label_formatter(&|value| format!("{value}"))
        .draw()?;

    chart
        .draw_series(LineSeries::new(
            records
                .iter()
                .map(|(step, predicted, _, _)| (*step, *predicted)),
            &BLUE,
        ))?
        .label("Predicted splash")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], BLUE));

    chart
        .draw_series(LineSeries::new(
            records.iter().map(|(step, _, actual, _)| (*step, *actual)),
            &RED,
        ))?
        .label("Actual new states")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], RED));

    chart
        .draw_series(LineSeries::new(
            records.iter().map(|(step, _, _, total)| (*step, *total)),
            &GREEN,
        ))?
        .label("Total non-zero amplitudes")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], GREEN));

    chart
        .configure_series_labels()
        .border_style(&BLACK)
        .draw()?;
    root.present()?;
    Ok(())
}
