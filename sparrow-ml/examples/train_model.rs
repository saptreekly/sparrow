use std::env;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{ensure, Context, Result};
use log::info;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::Deserialize;
use serde_json::to_writer_pretty;
use time::format_description::well_known::Rfc3339;
use time::OffsetDateTime;

type StdResult<T> = Result<T>;

use sparrow_ml::dataset::{generate_dataset, shuffle_split, DatasetConfig, Sample};
use sparrow_ml::features::FEATURE_NAMES;
use sparrow_ml::gate::GateSpec;
use sparrow_ml::logistic::{train, ModelArtifact, ModelMetadata, TrainingSettings};
use sparrow_ml::predictor::{PredictionSettings, SplashPredictor};

fn main() -> StdResult<()> {
    let _ = env_logger::builder().format_timestamp_millis().try_init();

    let config_path = env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("config/train_model.toml"));
    let config = load_config(&config_path)?;
    config.training.validate()?;

    info!("starting training run using {}", config_path.display());

    let RunConfig {
        dataset,
        training,
        output,
    } = config;

    let dataset_config = dataset.into_dataset_config();
    info!(
        "generating dataset (qubits={}, states={})",
        dataset_config.qubits, dataset_config.num_states
    );
    let dataset = generate_dataset(&dataset_config);
    info!("collected {} samples", dataset.len());

    let training_settings = training.to_training_settings();
    let cv_scores = k_fold_cross_validation(
        &dataset,
        training_settings,
        training.k_folds,
        training.cross_validation_seed,
        training.probability_threshold,
    );
    if !cv_scores.is_empty() {
        let mean = mean(&cv_scores);
        let std_dev = standard_deviation(&cv_scores, mean);
        info!(
            "{}-fold cross-validation accuracy {:.4} ± {:.4}",
            cv_scores.len(),
            mean,
            std_dev
        );
    } else {
        info!("cross-validation skipped (insufficient samples)");
    }

    let (train_set, test_set) = shuffle_split(dataset, training.test_ratio, training.cross_validation_seed);
    info!(
        "training with {} samples, testing on {} samples",
        train_set.len(),
        test_set.len()
    );

    let model = train(&train_set, training_settings);

    let metadata = ModelMetadata {
        model_version: output.model_version.clone(),
        training_date: OffsetDateTime::now_utc()
            .format(&Rfc3339)
            .context("format training timestamp")?,
        features: FEATURE_NAMES.iter().map(|name| name.to_string()).collect(),
    };

    let predictor = SplashPredictor::with_metadata(
        model.clone(),
        metadata.clone(),
        PredictionSettings {
            probability_threshold: training.probability_threshold,
        },
    );
    let (precision, recall) = evaluate(&predictor, &test_set);
    info!(
        "test precision {:.4}, recall {:.5} @ threshold {:.3}",
        precision,
        recall,
        predictor.settings().probability_threshold
    );

    let artifact = ModelArtifact::new(metadata, model);
    write_artifact(&output.model_path, &artifact)?;
    info!("saved model to {}", output.model_path.display());

    Ok(())
}

fn load_config(path: &Path) -> StdResult<RunConfig> {
    let content = fs::read_to_string(path).with_context(|| format!("read config at {}", path.display()))?;
    toml::from_str(&content).context("parse TOML configuration")
}

fn k_fold_cross_validation(
    samples: &[Sample],
    settings: TrainingSettings,
    k: usize,
    seed: u64,
    threshold: f64,
) -> Vec<f64> {
    if k < 2 || samples.len() < k {
        return Vec::new();
    }

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut shuffled = samples.to_vec();
    shuffled.shuffle(&mut rng);

    let fold_size = (shuffled.len() + k - 1) / k;
    let mut accuracies = Vec::with_capacity(k);

    for fold in 0..k {
        let start = fold * fold_size;
        if start >= shuffled.len() {
            break;
        }
        let end = ((fold + 1) * fold_size).min(shuffled.len());
        let test_slice = &shuffled[start..end];
        if test_slice.is_empty() {
            continue;
        }
        let mut train_data = Vec::with_capacity(shuffled.len() - test_slice.len());
        train_data.extend_from_slice(&shuffled[..start]);
        train_data.extend_from_slice(&shuffled[end..]);
        if train_data.is_empty() {
            continue;
        }

        let model = train(&train_data, settings);
        let accuracy = classification_accuracy(&model, test_slice, threshold);
        accuracies.push(accuracy);
    }

    accuracies
}

fn classification_accuracy(model: &sparrow_ml::logistic::LogisticModel, data: &[Sample], threshold: f64) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    let mut correct = 0usize;
    for sample in data {
        let prob = model.predict_probability(&sample.features);
        let predicted = prob >= threshold;
        if predicted == sample.label {
            correct += 1;
        }
    }

    correct as f64 / data.len() as f64
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().copied().sum::<f64>() / values.len() as f64
}

fn standard_deviation(values: &[f64], mean: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let variance = values
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / values.len() as f64;
    variance.sqrt()
}

fn evaluate(predictor: &SplashPredictor, data: &[Sample]) -> (f64, f64) {
    let threshold = predictor.settings().probability_threshold;
    let mut tp = 0.0;
    let mut fp = 0.0;
    let mut fn_ = 0.0;

    for sample in data {
        let prob = predictor.model().predict_probability(&sample.features);
        let predicted = prob >= threshold;
        match (predicted, sample.label) {
            (true, true) => tp += 1.0,
            (true, false) => fp += 1.0,
            (false, true) => fn_ += 1.0,
            (false, false) => {}
        }
    }

    let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 1.0 };
    let recall = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 1.0 };
    (precision, recall)
}

fn write_artifact(path: &Path, artifact: &ModelArtifact) -> StdResult<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).with_context(|| format!("create directory {}", parent.display()))?;
        }
    }
    let mut file = File::create(path).with_context(|| format!("create model file {}", path.display()))?;
    to_writer_pretty(&mut file, artifact).context("serialize model artifact")?;
    file.write_all(b"\n").context("flush model artifact")?;
    Ok(())
}

#[derive(Debug, Deserialize)]
struct RunConfig {
    dataset: DatasetSection,
    training: TrainingSection,
    #[serde(default)]
    output: OutputSection,
}

#[derive(Debug, Deserialize)]
struct DatasetSection {
    qubits: u8,
    num_states: usize,
    min_non_zero: usize,
    max_non_zero: usize,
    label_threshold: f64,
    gate: GateSpec,
    seed: u64,
}

impl DatasetSection {
    fn into_dataset_config(self) -> DatasetConfig {
        DatasetConfig {
            qubits: self.qubits,
            num_states: self.num_states,
            min_non_zero: self.min_non_zero,
            max_non_zero: self.max_non_zero,
            label_threshold: self.label_threshold,
            gate: self.gate,
            seed: self.seed,
        }
    }
}

#[derive(Debug, Deserialize)]
struct TrainingSection {
    epochs: usize,
    learning_rate: f64,
    l2_regularization: f64,
    #[serde(default = "default_test_ratio")]
    test_ratio: f64,
    #[serde(default = "default_probability_threshold")]
    probability_threshold: f64,
    #[serde(default = "default_k_folds")]
    k_folds: usize,
    #[serde(default = "default_cv_seed")]
    cross_validation_seed: u64,
}

impl TrainingSection {
    fn to_training_settings(&self) -> TrainingSettings {
        TrainingSettings {
            epochs: self.epochs,
            learning_rate: self.learning_rate,
            l2: self.l2_regularization,
        }
    }

    fn validate(&self) -> StdResult<()> {
        ensure!(
            (0.0..1.0).contains(&self.test_ratio),
            "test_ratio must be between 0 and 1 (exclusive)"
        );
        ensure!(self.k_folds >= 2, "k_folds must be at least 2");
        ensure!(
            (0.0..=1.0).contains(&self.probability_threshold),
            "probability_threshold must be within [0, 1]"
        );
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
struct OutputSection {
    #[serde(default = "default_model_path")]
    model_path: PathBuf,
    #[serde(default = "default_model_version")]
    model_version: String,
}

impl Default for OutputSection {
    fn default() -> Self {
        Self {
            model_path: default_model_path(),
            model_version: default_model_version(),
        }
    }
}

fn default_model_path() -> PathBuf {
    PathBuf::from("assets/splash_model.json")
}

fn default_model_version() -> String {
    "0.1.0".to_string()
}

fn default_test_ratio() -> f64 {
    0.2
}

fn default_probability_threshold() -> f64 {
    0.02
}

fn default_k_folds() -> usize {
    5
}

fn default_cv_seed() -> u64 {
    0xABCD
}
