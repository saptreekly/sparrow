use serde::{Deserialize, Serialize};

use sparrow_core::{SparseResult, SparseStateVector};

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum GateSpec {
    Cnot { control: u8, target: u8 },
}

impl GateSpec {
    pub fn apply(&self, state: &mut SparseStateVector) -> SparseResult<()> {
        match self {
            GateSpec::Cnot { control, target } => state.apply_cnot(*control, *target),
        }
    }

    pub fn control_target(&self) -> (u8, u8) {
        match self {
            GateSpec::Cnot { control, target } => (*control, *target),
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            GateSpec::Cnot { .. } => "cnot",
        }
    }
}
