pub mod dataset;
pub mod features;
pub mod gate;
pub mod logistic;
pub mod predictor;

pub use dataset::{DatasetConfig, Sample};
pub use features::{FEATURE_COUNT, FeatureContext};
pub use gate::GateSpec;
pub use logistic::{LogisticModel, TrainingSettings};
pub use predictor::{PredictionSettings, SplashPredictor};
