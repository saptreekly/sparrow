use serde::{Deserialize, Serialize};

use crate::dataset::Sample;

#[derive(Clone, Copy, Debug)]
pub struct TrainingSettings {
    pub epochs: usize,
    pub learning_rate: f64,
    pub l2: f64,
}

impl Default for TrainingSettings {
    fn default() -> Self {
        Self {
            epochs: 200,
            learning_rate: 0.3,
            l2: 1e-4,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LogisticModel {
    pub weights: Vec<f64>,
    pub bias: f64,
}

impl LogisticModel {
    pub fn new(weights: Vec<f64>, bias: f64) -> Self {
        Self { weights, bias }
    }

    pub fn feature_count(&self) -> usize {
        self.weights.len()
    }

    pub fn predict_probability(&self, features: &[f64]) -> f64 {
        debug_assert_eq!(features.len(), self.weights.len());
        let score = dot(&self.weights, features) + self.bias;
        sigmoid(score)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub model_version: String,
    pub training_date: String,
    pub features: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelArtifact {
    pub metadata: ModelMetadata,
    pub model: LogisticModel,
}

impl ModelArtifact {
    pub fn new(metadata: ModelMetadata, model: LogisticModel) -> Self {
        Self { metadata, model }
    }
}

pub fn train(samples: &[Sample], settings: TrainingSettings) -> LogisticModel {
    assert!(!samples.is_empty(), "training requires at least one sample");
    let feature_len = samples[0].features.len();
    let mut weights = vec![0.0; feature_len];
    let mut bias = 0.0;

    let lr = settings.learning_rate;
    let l2 = settings.l2;
    let m = samples.len() as f64;

    for _ in 0..settings.epochs {
        let mut grad_w = vec![0.0; feature_len];
        let mut grad_b = 0.0;

        for sample in samples {
            debug_assert_eq!(sample.features.len(), feature_len);
            let prediction = sigmoid(dot(&weights, &sample.features) + bias);
            let label = if sample.label { 1.0 } else { 0.0 };
            let error = prediction - label;

            for (i, value) in sample.features.iter().enumerate() {
                grad_w[i] += error * value;
            }
            grad_b += error;
        }

        for (i, weight) in weights.iter_mut().enumerate() {
            let grad = grad_w[i] / m + l2 * *weight;
            *weight -= lr * grad;
        }
        bias -= lr * grad_b / m;
    }

    LogisticModel { weights, bias }
}

pub fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        let z = (-value).exp();
        1.0 / (1.0 + z)
    } else {
        let z = value.exp();
        z / (1.0 + z)
    }
}

fn dot(weights: &[f64], features: &[f64]) -> f64 {
    weights
        .iter()
        .zip(features.iter())
        .map(|(w, f)| w * f)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::FEATURE_COUNT;

    #[test]
    fn logistic_model_can_fit_simple_data() {
        let dataset: Vec<Sample> = (0..100)
            .map(|i| {
                let x = i as f64 / 100.0;
                let label = x > 0.5;
                Sample {
                    features: vec![x; FEATURE_COUNT],
                    label,
                    target_basis: 0,
                    source_basis: 0,
                }
            })
            .collect();

        let model = train(&dataset, TrainingSettings::default());
        assert!(model.predict_probability(&dataset[10].features) < 0.5);
        assert!(model.predict_probability(&dataset[90].features) > 0.5);
    }
}
