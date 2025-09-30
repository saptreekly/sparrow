use std::fs;
use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use clap::{ArgAction, Parser, ValueEnum};
use log::LevelFilter;

use sparrow_core::{Complex64, SparseStateVector};

use crate::orchestrator::{Orchestrator, SimulationConfig, SimulationMode};
use crate::parser::parse_program;

mod orchestrator;
mod parser;

#[derive(Parser, Debug)]
#[command(author, version, about = "sparrow-sim: AI-assisted sparse quantum simulator")]
struct Cli {
    /// Path to the OpenQASM 2.0 input file
    #[arg(value_name = "QASM")] 
    input: PathBuf,

    /// Simulation strategy to use
    #[arg(long, value_enum, default_value_t = ModeArg::Predictive)]
    mode: ModeArg,

    /// Pruning threshold for sparse amplitudes
    #[arg(long, default_value = "1e-9")]
    threshold: f64,

    /// ML prediction probability threshold (predictive mode)
    #[arg(long, default_value = "0.05")]
    prediction_threshold: f64,

    /// Increase output verbosity (-v info, -vv debug, -vvv trace)
    #[arg(short, long, action = ArgAction::Count)]
    verbose: u8,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum ModeArg {
    Predictive,
    Sparse,
    Dense,
}

impl From<ModeArg> for SimulationMode {
    fn from(value: ModeArg) -> Self {
        match value {
            ModeArg::Predictive => SimulationMode::Predictive,
            ModeArg::Sparse => SimulationMode::Sparse,
            ModeArg::Dense => SimulationMode::Dense,
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    init_logging(cli.verbose)?;

    let source = fs::read_to_string(&cli.input)
        .with_context(|| format!("failed to read {}", cli.input.display()))?;
    let parsed = parse_program(&source)?;
    anyhow::ensure!(
        parsed.total_qubits > 0,
        "the circuit does not declare any qubits"
    );

    let mut state = SparseStateVector::new(parsed.total_qubits);
    state
        .set_amplitude(0, Complex64::new(1.0, 0.0))
        .map_err(|err| anyhow!("failed to initialize |0> state: {:?}", err))?;

    let config = SimulationConfig {
        mode: cli.mode.into(),
        prune_threshold: cli.threshold,
        prediction_threshold: cli.prediction_threshold,
    };

    let mut orchestrator = Orchestrator::new(config);
    orchestrator
        .run(&parsed.circuit, &mut state)
        .context("simulation failed")?;

    println!(
        "Simulation complete: {} amplitudes above threshold {:.3e}",
        state.len(),
        cli.threshold
    );
    println!("State norm squared: {:.6}", state.norm_sqr());

    Ok(())
}

fn init_logging(verbosity: u8) -> Result<()> {
    let level = match verbosity {
        0 => LevelFilter::Warn,
        1 => LevelFilter::Info,
        2 => LevelFilter::Debug,
        _ => LevelFilter::Trace,
    };

    let mut builder = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn"));
    builder.filter_level(level);
    builder.try_init().map_err(|err| err.into())
}
