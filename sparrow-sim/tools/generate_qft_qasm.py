#!/usr/bin/env python3
"""Generate an OpenQASM 2.0 file implementing an n-qubit QFT.

The emitted circuit only uses `u` and `cx` gates so it is fully compatible
with the sparrow-sim parser and orchestrator.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List


def format_angle(value: float) -> str:
    """Render floating-point angles with enough precision for QFT."""

    return f"{value:.12f}"


def emit_u(theta: float, phi: float, lamb: float, target: int) -> str:
    return (
        f"u({format_angle(theta)},{format_angle(phi)},{format_angle(lamb)}) "
        f"q[{target}];"
    )


def emit_h(target: int) -> str:
    # Hadamard = U(pi/2, 0, pi)
    return "u(pi/2,0,pi) q[{0}];".format(target)


def emit_cnot(control: int, target: int) -> str:
    return f"cx q[{control}], q[{target}];"


def emit_controlled_phase(control: int, target: int, angle: float) -> List[str]:
    half = angle / 2.0
    return [
        emit_u(0.0, 0.0, half, target),
        emit_cnot(control, target),
        emit_u(0.0, 0.0, -half, target),
        emit_cnot(control, target),
        emit_u(0.0, 0.0, half, control),
    ]


def emit_swap(a: int, b: int) -> List[str]:
    return [
        emit_cnot(a, b),
        emit_cnot(b, a),
        emit_cnot(a, b),
    ]


def build_qft(qubits: int) -> List[str]:
    lines: List[str] = []
    for target in range(qubits):
        for control in range(target):
            angle = math.pi / (2 ** (target - control))
            lines.extend(emit_controlled_phase(control, target, angle))
        lines.append(emit_h(target))

    # Reverse the register order with swaps
    for i in range(qubits // 2):
        j = qubits - i - 1
        lines.extend(emit_swap(i, j))

    return lines


def write_qasm(qubits: int, output: Path) -> None:
    header = [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        f"qreg q[{qubits}];",
    ]
    body = build_qft(qubits)
    contents = "\n".join(header + body) + "\n"
    output.write_text(contents)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a QFT circuit in OpenQASM 2.0 format.")
    parser.add_argument("qubits", type=int, help="Number of qubits in the QFT circuit")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Destination .qasm file",
    )
    args = parser.parse_args()

    if args.qubits <= 0:
        raise SystemExit("Number of qubits must be positive")

    write_qasm(args.qubits, args.output)


if __name__ == "__main__":
    main()
