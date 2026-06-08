#!/usr/bin/env python3
"""Gera gráficos dos resultados GPU."""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
IN_CSV = ROOT / "results" / "csv" / "gpu_results.csv"
OUT_DIR = ROOT / "results" / "figures"


def main() -> None:
    if not IN_CSV.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {IN_CSV}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gpu_results = pd.read_csv(IN_CSV)

    for n in sorted(gpu_results["N"].unique()):
        df = gpu_results[gpu_results["N"] == n]

        plt.figure(figsize=(8, 4))
        plt.bar(df["version"], df["kernel_ms"])
        plt.ylabel("Tempo do kernel (ms)")
        plt.title(f"Tempo do kernel CUDA — N={n}")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"gpu_kernel_time_N{n}.png", dpi=150)
        plt.close()

        plt.figure(figsize=(8, 4))
        plt.bar(df["version"], df["kernel_GFLOPS"])
        plt.ylabel("Kernel GFLOP/s")
        plt.title(f"Desempenho do kernel CUDA — N={n}")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"gpu_kernel_gflops_N{n}.png", dpi=150)
        plt.close()

    print(f"✅ Gráficos GPU salvos em: {OUT_DIR}")


if __name__ == "__main__":
    main()
