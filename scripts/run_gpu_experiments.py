#!/usr/bin/env python3
"""Executa os experimentos CUDA/GPU do TP3.

Este script é uma versão organizada da célula de experimentos GPU do notebook.
Ele deve ser usado depois que os kernels TODO em src/cuda/gemm_cuda.cu forem implementados.
"""

from __future__ import annotations

import subprocess
from io import StringIO
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXECUTABLE = ROOT / "build" / "gemm_cuda"
OUT_CSV = ROOT / "results" / "csv" / "gpu_results.csv"


def main() -> None:
    if not EXECUTABLE.exists():
        raise FileNotFoundError(f"Executável não encontrado: {EXECUTABLE}. Compile com: make cuda")

    ns = [128, 256, 512, 1024]
    repeats = 5
    rows = []

    for n in ns:
        for version in ["naive", "tiled"]:
            cmd = [str(EXECUTABLE), version, str(n), str(repeats)]
            print(f"\n▶ Executando: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
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
    gpu_results = pd.concat(rows, ignore_index=True)
    gpu_results.to_csv(OUT_CSV, index=False)
    print(f"\n✅ Resultados GPU salvos em: {OUT_CSV}")
    print(gpu_results)


if __name__ == "__main__":
    main()
