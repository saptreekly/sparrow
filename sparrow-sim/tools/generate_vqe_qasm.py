#!/usr/bin/env python3
"""Generate a hardware-efficient VQE-style OpenQASM 2.0 circuit.

The emitted ansatz consists of layers of single-qubit rotations followed by a
nearest-neighbour entangling pattern. Random angles are sampled to mimic a
converged VQE snapshot; this produces realistic sparsity patterns without
requiring an upstream chemistry package.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Iterable, List


def u_gate(theta: float, phi: float, lamb: float, qubit: int) -> str:
    return f"u({theta:.12f},{phi:.12f},{lamb:.12f}) q[{qubit}];"


def rz(angle: float, qubit: int) -> str:
    return u_gate(0.0, 0.0, angle, qubit)


def rx(angle: float, qubit: int) -> str:
    # Rx(angle) = U(angle, -pi/2, pi/2)
    return u_gate(angle, -math.pi / 2.0, math.pi / 2.0, qubit)


def ry(angle: float, qubit: int) -> str:
    # Ry(angle) = U(angle, 0, 0)
    return u_gate(angle, 0.0, 0.0, qubit)


def cnot(control: int, target: int) -> str:
    return f"cx q[{control}], q[{target}];"


def layer_single_qubit(qubits: int, rng: random.Random) -> Iterable[str]:
    for q in range(qubits):
        yield rz(rng.uniform(0.0, 2.0 * math.pi), q)
        yield ry(rng.uniform(0.0, 2.0 * math.pi), q)
        yield rz(rng.uniform(0.0, 2.0 * math.pi), q)


def layer_entangle(qubits: int) -> Iterable[str]:
    for q in range(qubits - 1):
        yield cnot(q, q + 1)
    if qubits > 2:
        # add reverse chain to mimic hardware-efficient topology
        for q in range(qubits - 1, 0, -1):
            yield cnot(q, q - 1)


def build_vqe(qubits: int, depth: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    lines: List[str] = []

    # Initial layer to break symmetry
    lines.extend(layer_single_qubit(qubits, rng))

    for layer in range(depth):
        lines.extend(layer_entangle(qubits))
        lines.extend(layer_single_qubit(qubits, rng))

    return lines


def write_qasm(qubits: int, depth: int, seed: int, path: Path) -> None:
    header = [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        f"qreg q[{qubits}];",
    ]
    body = build_vqe(qubits, depth, seed)
    path.write_text("\n".join(header + body) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a randomised hardware-efficient VQE circuit in OpenQASM 2.0 format."
    )
    parser.add_argument("qubits", type=int, help="Number of qubits in the ansatz")
    parser.add_argument(
        "--depth",
        type=int,
        default=4,
        help="Number of entangling layers (default: 4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0x5151AA,
        help="Random seed used to sample rotation angles",
    )
    parser.add_argument("--output", "-o", type=Path, required=True, help="Destination .qasm file")

    args = parser.parse_args()

    if args.qubits <= 0:
        raise SystemExit("Number of qubits must be positive")
    if args.depth < 0:
        raise SystemExit("Depth must be non-negative")

    write_qasm(args.qubits, args.depth, args.seed, args.output)


if __name__ == "__main__":
    main()
