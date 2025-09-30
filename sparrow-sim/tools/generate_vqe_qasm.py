#!/usr/bin/env python3
"""Generate OpenQASM 2.0 circuits for VQE-style or chaotic benchmarks.

By default the script emits a hardware-efficient VQE ansatz composed of random
single-qubit rotations and nearest-neighbour entangling layers. When the
``--chaotic`` flag is supplied the generator produces a highly entangling
random circuit built from Hadamard/CNOT layers, suitable for stress-testing the
predictive engine with dense states.
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


def hadamard(qubit: int) -> str:
    # Hadamard = U(pi/2, 0, pi)
    return "u(pi/2,0,pi) q[{0}];".format(qubit)


def build_vqe(qubits: int, depth: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    lines: List[str] = []

    # Initial layer to break symmetry
    lines.extend(layer_single_qubit(qubits, rng))

    for layer in range(depth):
        lines.extend(layer_entangle(qubits))
        lines.extend(layer_single_qubit(qubits, rng))

    return lines


def build_chaotic(qubits: int, depth: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    lines: List[str] = []

    for _ in range(max(depth, 1)):
        # Apply Hadamards to all qubits to spread amplitude mass
        for q in range(qubits):
            lines.append(hadamard(q))

        # Randomised CNOT pairing
        ordering = list(range(qubits))
        rng.shuffle(ordering)
        for i in range(0, qubits - 1, 2):
            control = ordering[i]
            target = ordering[i + 1]
            if rng.random() < 0.5:
                control, target = target, control
            lines.append(cnot(control, target))

        # Inject random single-qubit phase kicks to avoid periodicity
        for q in range(qubits):
            lines.append(rz(rng.uniform(0.0, 2.0 * math.pi), q))

    return lines


def write_qasm(qubits: int, depth: int, seed: int, path: Path, chaotic: bool) -> None:
    header = [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        f"qreg q[{qubits}];",
    ]
    if chaotic:
        body = build_chaotic(qubits, depth, seed)
    else:
        body = build_vqe(qubits, depth, seed)
    path.write_text("\n".join(header + body) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate hardware-efficient VQE or chaotic random circuits in OpenQASM 2.0 format."
    )
    parser.add_argument("qubits", type=int, help="Number of qubits in the ansatz")
    parser.add_argument(
        "--depth",
        type=int,
        default=4,
        help="Number of layers (default: 4). Interpreted as entangling depth for VQE,"
        " or Hadamard/CNOT rounds for chaotic circuits.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0x5151AA,
        help="Random seed used to sample rotation angles",
    )
    parser.add_argument(
        "--chaotic",
        action="store_true",
        help="Generate a chaotic Hadamard/CNOT stress-test circuit instead of a VQE ansatz",
    )
    parser.add_argument("--output", "-o", type=Path, required=True, help="Destination .qasm file")

    args = parser.parse_args()

    if args.qubits <= 0:
        raise SystemExit("Number of qubits must be positive")
    if args.depth < 0:
        raise SystemExit("Depth must be non-negative")

    write_qasm(args.qubits, args.depth, args.seed, args.output, args.chaotic)


if __name__ == "__main__":
    main()
