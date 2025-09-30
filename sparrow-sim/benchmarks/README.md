# sparrow-sim Benchmark Guide

This document collects the commands and helper scripts that underpin the
validation and performance benchmarking plan for **sparrow-sim**. It is meant
to be a living reference so future contributors can reproduce the published
results quickly.

## 1. Circuit Generators

Two helper scripts live under `sparrow-sim/tools`:

- `generate_qft_qasm.py` – emits an *n*-qubit Quantum Fourier Transform using
  only `u` and `cx` gates.
- `generate_vqe_qasm.py` – emits a hardware-efficient VQE-style ansatz with
  configurable depth/seed.

Examples:

```bash
# 16-qubit QFT
python3 sparrow-sim/tools/generate_qft_qasm.py 16 --output qft_16.qasm

# 20-qubit VQE circuit with 6 entangling layers
python3 sparrow-sim/tools/generate_vqe_qasm.py 20 --depth 6 --seed 1337 --output vqe_20.qasm
```

## 2. Functional Validation (QFT)

Run the simulator in all three modes against the QFT circuit. Set the pruning
threshold to zero so all amplitudes are preserved.

```bash
cargo run --manifest-path sparrow-sim/Cargo.toml -- --mode dense qft_16.qasm
cargo run --manifest-path sparrow-sim/Cargo.toml -- --mode sparse --threshold 0 qft_16.qasm
cargo run --manifest-path sparrow-sim/Cargo.toml -- --mode predictive --threshold 0 \
    --prediction-threshold 0 qft_16.qasm
```

For fidelity checks, modify `sparrow-sim/src/main.rs` to dump `state.amplitudes()`
to disk or augment the CLI with an `--output-state` flag. Compare each state to
the dense baseline using a small Python helper:

```python
import numpy as np
d = np.loadtxt("dense_state.csv", dtype=complex)
s = np.loadtxt("sparse_state.csv", dtype=complex)
fidelity = abs(np.vdot(d, s))**2
print(f"Fidelity: {fidelity:.12f}")
```

## 3. Performance “Money Shot” (VQE Sweeps)

Generate a family of VQE circuits (e.g., 12, 16, 20, 24 qubits) and benchmark
each mode. On macOS, `/usr/bin/time -l` provides both wall-clock time and peak
RSS (bytes):

```bash
for mode in dense sparse predictive; do
  /usr/bin/time -l cargo run --manifest-path sparrow-sim/Cargo.toml -- \
      --mode "$mode" --threshold 1e-9 vqe_20.qasm 2>&1 | tee logs/vqe_20_${mode}.log
done
```

For repeated measurements, consider `hyperfine`:

```bash
hyperfine --warmup 1 --runs 5 \
  'cargo run --manifest-path sparrow-sim/Cargo.toml -- --mode sparse --threshold 1e-9 vqe_20.qasm'
```

Collect the data into a CSV (`qubits,mode,time_seconds,memory_mb`) using a
small script, then plot Time-vs-Qubits and Memory-vs-Qubits using your preferred
tool (Matplotlib, gnuplot, Plotters, etc.).

## 4. AI Stress Test (Random Circuits)

1. Generate a chaotic circuit (e.g., adapt the VQE generator to alternate
   Hadamards and random CNOTs) for ~20 qubits.
2. Run predictive mode with verbose logging enabled:

   ```bash
   RUST_LOG=sparrow_sim::orchestrator=trace cargo run --manifest-path sparrow-sim/Cargo.toml -- \
       --mode predictive --threshold 1e-9 random_20.qasm 2>&1 | tee logs/random_predictive.log
   ```

3. Inspect logs for per-gate prediction counts and ensure runtime remains
   comparable to sparse mode.

## 5. Suggested Directory Layout

- `sparrow-sim/benchmarks/` – this guide, CSV outputs, plots
- `sparrow-sim/logs/` – raw `/usr/bin/time` output per run
- `sparrow-sim/circuits/` – generated `.qasm` inputs

Keeping artefacts organised makes it easy to regenerate results or compare
future optimisations.

---

Feel free to extend this guide with new benchmark scenarios or automation
scripts (e.g., Rust/Python harnesses that sweep gate counts and emit plots
directly).
