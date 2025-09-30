use std::collections::HashMap;
use std::convert::TryFrom;

use anyhow::{anyhow, bail, ensure, Context, Result};
use pest::iterators::Pair;
use pest::Parser;
use sparrow_core::{Circuit, Complex64, Gate};

#[derive(pest_derive::Parser)]
#[grammar = "qasm.pest"]
struct QasmParser;

#[derive(Debug, Clone)]
struct Register {
    offset: u8,
    size: u8,
}

#[derive(Debug, Default)]
struct ParseContext {
    registers: HashMap<String, Register>,
    total_qubits: u8,
}

impl ParseContext {
    fn declare_register(&mut self, name: &str, size: u32) -> Result<()> {
        if self.registers.contains_key(name) {
            bail!("register `{}` declared more than once", name);
        }
        ensure!(size > 0, "register `{}` must contain at least one qubit", name);
        let size_u8 = u8::try_from(size).context("register size exceeds supported range")?;
        ensure!(
            size_u8 <= 128,
            "register `{}` of size {} exceeds 128-qubit limit",
            name,
            size
        );
        let offset = self.total_qubits;
        self.total_qubits = self
            .total_qubits
            .checked_add(size_u8)
            .context("total qubit count exceeds 255")?;
        ensure!(
            self.total_qubits <= 128,
            "total qubit count {} exceeds 128-qubit limit",
            self.total_qubits
        );
        self.registers.insert(
            name.to_string(),
            Register {
                offset,
                size: size_u8,
            },
        );
        Ok(())
    }

    fn resolve(&self, name: &str, index: u32) -> Result<u8> {
        let register = self
            .registers
            .get(name)
            .with_context(|| format!("unknown register `{}`", name))?;
        ensure!(
            index < register.size as u32,
            "qubit index {} out of range for register `{}` (size {})",
            index,
            name,
            register.size
        );
        Ok(register.offset + index as u8)
    }
}

#[derive(Debug)]
pub struct ParsedProgram {
    pub circuit: Circuit,
    pub total_qubits: u8,
}

pub fn parse_program(input: &str) -> Result<ParsedProgram> {
    let mut pairs =
        QasmParser::parse(Rule::program, input).context("failed to parse OpenQASM source")?;
    let program = pairs
        .next()
        .ok_or_else(|| anyhow!("program rule produced no pairs"))?;

    let mut context = ParseContext::default();
    let mut gates = Vec::new();

    for pair in program.into_inner() {
        match pair.as_rule() {
            Rule::header => handle_header(pair)?,
            Rule::statement => handle_statement(pair, &mut context, &mut gates)?,
            _ => {}
        }
    }

    Ok(ParsedProgram {
        circuit: Circuit::new(gates),
        total_qubits: context.total_qubits,
    })
}

fn handle_header(pair: Pair<Rule>) -> Result<()> {
    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::openqasm_decl => {
                let mut decl_inner = inner.into_inner();
                let version = decl_inner
                    .next()
                    .ok_or_else(|| anyhow!("missing version in OPENQASM declaration"))?
                    .as_str();
                ensure!(
                    version.starts_with('2'),
                    "unsupported OpenQASM version `{}`",
                    version
                );
            }
            Rule::include_decl => {
                // Nothing to validate for the include path at the moment.
            }
            _ => {}
        }
    }
    Ok(())
}

fn handle_statement(pair: Pair<Rule>, ctx: &mut ParseContext, gates: &mut Vec<Gate>) -> Result<()> {
    let mut inner = pair.into_inner();
    if let Some(node) = inner.next() {
        match node.as_rule() {
            Rule::qreg_decl => handle_qreg(node, ctx)?,
            Rule::gate_u => gates.push(parse_u_gate(node, ctx)?),
            Rule::gate_cx => gates.push(parse_cx_gate(node, ctx)?),
            Rule::barrier => validate_barrier(node, ctx)?,
            other => bail!("unsupported statement: {:?}", other),
        }
    }
    Ok(())
}

fn handle_qreg(pair: Pair<Rule>, ctx: &mut ParseContext) -> Result<()> {
    let mut inner = pair.into_inner();
    let name = inner
        .next()
        .ok_or_else(|| anyhow!("missing identifier in qreg declaration"))?
        .as_str();
    let size_text = inner
        .next()
        .ok_or_else(|| anyhow!("missing size in qreg declaration"))?
        .as_str();
    let size: u32 = size_text
        .parse()
        .with_context(|| format!("invalid register size `{}`", size_text))?;
    ctx.declare_register(name, size)
}

fn parse_u_gate(pair: Pair<Rule>, ctx: &ParseContext) -> Result<Gate> {
    let mut inner = pair.into_inner();
    let theta = parse_angle(inner.next().ok_or_else(|| anyhow!("missing theta"))?)?;
    let phi = parse_angle(inner.next().ok_or_else(|| anyhow!("missing phi"))?)?;
    let lambda = parse_angle(inner.next().ok_or_else(|| anyhow!("missing lambda"))?)?;
    let qubit = parse_qubit(inner.next().ok_or_else(|| anyhow!("missing qubit"))?, ctx)?;

    let half_theta = theta / 2.0;
    let cos = half_theta.cos();
    let sin = half_theta.sin();
    let exp_i_phi = Complex64::new(phi.cos(), phi.sin());
    let exp_i_lambda = Complex64::new(lambda.cos(), lambda.sin());
    let exp_i_phi_lambda = Complex64::new((phi + lambda).cos(), (phi + lambda).sin());

    let matrix = [
        [Complex64::new(cos, 0.0), -exp_i_lambda * sin],
        [exp_i_phi * sin, exp_i_phi_lambda * cos],
    ];

    Ok(Gate::SingleQubit { qubit, matrix })
}

fn parse_cx_gate(pair: Pair<Rule>, ctx: &ParseContext) -> Result<Gate> {
    let mut inner = pair.into_inner();
    let control = parse_qubit(inner.next().ok_or_else(|| anyhow!("missing control"))?, ctx)?;
    let target = parse_qubit(inner.next().ok_or_else(|| anyhow!("missing target"))?, ctx)?;
    Ok(Gate::Cnot { control, target })
}

fn validate_barrier(pair: Pair<Rule>, ctx: &ParseContext) -> Result<()> {
    for qubit in pair.into_inner() {
        if qubit.as_rule() == Rule::qubit_ref {
            let _ = parse_qubit(qubit, ctx)?;
        }
    }
    Ok(())
}

fn parse_qubit(pair: Pair<Rule>, ctx: &ParseContext) -> Result<u8> {
    let mut inner = pair.into_inner();
    let name = inner
        .next()
        .ok_or_else(|| anyhow!("missing register name in qubit reference"))?
        .as_str();
    let index_text = inner
        .next()
        .ok_or_else(|| anyhow!("missing index in qubit reference"))?
        .as_str();
    let index: u32 = index_text
        .parse()
        .with_context(|| format!("invalid qubit index `{}`", index_text))?;
    ctx.resolve(name, index)
}

fn parse_angle(pair: Pair<Rule>) -> Result<f64> {
    let expr = pair.as_str().replace('^', "**");
    meval::eval_str(expr.as_str()).map_err(|err| anyhow!("invalid angle expression: {}", err))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_basic_program() {
        let source = r#"
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
u(pi/2,0,pi) q[0];
cx q[0], q[1];
barrier q[0], q[1];
"#;

        let parsed = parse_program(source).expect("parse program");
        assert_eq!(parsed.total_qubits, 2);
        assert_eq!(parsed.circuit.gates().len(), 2);

        match &parsed.circuit.gates()[0] {
            Gate::SingleQubit { qubit, .. } => assert_eq!(*qubit, 0),
            other => panic!("unexpected gate: {other:?}"),
        }
        match &parsed.circuit.gates()[1] {
            Gate::Cnot { control, target } => {
                assert_eq!((*control, *target), (0, 1));
            }
            other => panic!("unexpected gate: {other:?}"),
        }
    }
}
