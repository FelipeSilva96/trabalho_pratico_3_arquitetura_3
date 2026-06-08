#!/usr/bin/env python3
"""Executa os experimentos CPU do TP3.

Este script é uma versão organizada da célula de experimentos do notebook.
Ele deve ser usado depois que as funções TODO em src/cpu/gemm_cpu.c forem implementadas.
"""

from __future__ import annotations

import os
import subprocess
from io import StringIO
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXECUTABLE = ROOT / "build" / "gemm_cpu"
OUT_CSV = ROOT / "results" / "csv" / "cpu_results.csv"


def main() -> None:
    if not EXECUTABLE.exists():
        raise FileNotFoundError(f"Executável não encontrado: {EXECUTABLE}. Compile com: make cpu")

    experiments: list[tuple[str, int, int | None, int | None]] = []
    ns = [128, 256, 512]
    repeats = 5

    for n in ns:
        experiments.append(("naive", n, None, None))
        experiments.append(("transposed", n, None, None))

        for bs in [16, 32, 64]:
            experiments.append(("blocked", n, bs, None))

        for threads in [1, 2, 4, 8]:
            experiments.append(("openmp", n, None, threads))
            experiments.append(("blocked_openmp", n, 32, threads))

    rows = []
    for version, n, bs, threads in experiments:
        cmd = [str(EXECUTABLE), version, str(n), str(repeats)]
        if bs is not None:
            cmd.append(str(bs))

        env = os.environ.copy()
        if threads is not None:
            env["OMP_NUM_THREADS"] = str(threads)

        print(f"\n▶ Executando: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
        print(result.stdout)

        if result.returncode != 0:
            print(result.stderr)
            continue

        lines = result.stdout.strip().splitlines()
        if len(lines) >= 2:
            df = pd.read_csv(StringIO("\n".join(lines[-2:])))
            rows.append(df)

    if not rows:
        raise RuntimeError("Nenhum resultado válido foi gerado.")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    cpu_results = pd.concat(rows, ignore_index=True)
    cpu_results.to_csv(OUT_CSV, index=False)
    print(f"\n✅ Resultados CPU salvos em: {OUT_CSV}")
    print(cpu_results)


if __name__ == "__main__":
    main()
