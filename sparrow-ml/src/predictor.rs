use log::{debug, trace};
use once_cell::sync::Lazy;
use serde_json::from_str;
use sparrow_core::SparseStateVector;

use crate::features::{FEATURE_COUNT, FEATURE_NAMES, visit_candidates};
use crate::gate::GateSpec;
use crate::logistic::{LogisticModel, ModelArtifact, ModelMetadata};

static PRETRAINED_ARTIFACT: Lazy<ModelArtifact> = Lazy::new(|| {
    const DATA: &str = include_str!("../assets/splash_model.json");
    from_str(DATA).expect("invalid pretrained model JSON")
});

#[derive(Clone, Copy, Debug)]
pub struct PredictionSettings {
    pub probability_threshold: f64,
}

impl Default for PredictionSettings {
    fn default() -> Self {
        Self {
            probability_threshold: 0.05,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SplashPredictor {
    model: LogisticModel,
    metadata: Option<ModelMetadata>,
    settings: PredictionSettings,
}

impl SplashPredictor {
    pub fn new(model: LogisticModel) -> Self {
        debug_assert_eq!(
            model.feature_count(),
            FEATURE_COUNT,
            "predictor instantiated with mismatched feature count"
        );
        Self {
            model,
            metadata: None,
            settings: PredictionSettings::default(),
        }
    }

    pub fn with_settings(model: LogisticModel, settings: PredictionSettings) -> Self {
        debug_assert_eq!(
            model.feature_count(),
            FEATURE_COUNT,
            "predictor instantiated with mismatched feature count"
        );
        Self {
            model,
            metadata: None,
            settings,
        }
    }

    pub fn with_metadata(
        model: LogisticModel,
        metadata: ModelMetadata,
        settings: PredictionSettings,
    ) -> Self {
        ensure_compatible(&metadata, &model);
        Self {
            model,
            metadata: Some(metadata),
            settings,
        }
    }

    pub fn from_artifact(artifact: ModelArtifact) -> Self {
        Self::from_artifact_with_settings(artifact, PredictionSettings::default())
    }

    pub fn from_artifact_with_settings(
        artifact: ModelArtifact,
        settings: PredictionSettings,
    ) -> Self {
        ensure_compatible(&artifact.metadata, &artifact.model);
        Self {
            model: artifact.model,
            metadata: Some(artifact.metadata),
            settings,
        }
    }

    pub fn pretrained() -> Self {
        Self::from_artifact(PRETRAINED_ARTIFACT.clone())
    }

    pub fn settings(&self) -> PredictionSettings {
        self.settings
    }

    pub fn with_threshold(mut self, probability_threshold: f64) -> Self {
        self.settings.probability_threshold = probability_threshold;
        self
    }

    pub fn probability(&self, features: &[f64; FEATURE_COUNT]) -> f64 {
        self.model.predict_probability(features.as_slice())
    }

    pub fn predict_splash_zone(&self, gate: &GateSpec, state: &SparseStateVector) -> Vec<u128> {
        let mut buffer = Vec::new();
        self.predict_splash_zone_into(gate, state, &mut buffer);
        buffer
    }

    pub fn predict_splash_zone_into(
        &self,
        gate: &GateSpec,
        state: &SparseStateVector,
        output: &mut Vec<u128>,
    ) {
        output.clear();
        let threshold = self.settings.probability_threshold;
        let (control, target) = gate.control_target();
        debug!(
            target: "sparrow_ml::predictor",
            "Predicting splash zone for {}({}, {}) over {} amplitudes (threshold={:.3})",
            gate.label(),
            control,
            target,
            state.len(),
            threshold
        );

        visit_candidates(state, gate, |ctx| {
            let probability = self.model.predict_probability(ctx.features.as_slice());
            trace!(
                target: "sparrow_ml::predictor",
                "Candidate target={} source={} probability={:.4} features={:?}",
                ctx.target_basis,
                ctx.source_basis,
                probability,
                ctx.features
            );
            if probability >= threshold {
                output.push(ctx.target_basis);
            }
        });
    }

    pub fn model(&self) -> &LogisticModel {
        &self.model
    }

    pub fn metadata(&self) -> Option<&ModelMetadata> {
        self.metadata.as_ref()
    }
}

fn ensure_compatible(metadata: &ModelMetadata, model: &LogisticModel) {
    assert_eq!(
        model.feature_count(),
        FEATURE_COUNT,
        "model trained with {} features but {} are required",
        model.feature_count(),
        FEATURE_COUNT
    );
    assert_eq!(
        metadata.features.len(),
        FEATURE_COUNT,
        "metadata features length {} does not match expected {}",
        metadata.features.len(),
        FEATURE_COUNT
    );

    for (expected, actual) in FEATURE_NAMES.iter().zip(metadata.features.iter()) {
        assert_eq!(
            actual.as_str(),
            *expected,
            "metadata feature mismatch: expected `{}`, found `{}`",
            expected,
            actual
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sparrow_core::{Complex64, SparseStateVector};

    #[test]
    fn predictor_returns_candidates() {
        let mut state = SparseStateVector::new(3);
        state
            .set_amplitude(0b110, Complex64::new(0.2, 0.0))
            .unwrap();
        state.normalize();

        let predictor = SplashPredictor::pretrained().with_threshold(0.0);
        assert!(predictor.metadata().is_some());
        let predictions = predictor.predict_splash_zone(
            &GateSpec::Cnot {
                control: 2,
                target: 0,
            },
            &state,
        );
        assert!(!predictions.is_empty());
    }
}
