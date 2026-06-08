#!/usr/bin/env python3
"""Gera gráficos dos resultados CPU."""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
IN_CSV = ROOT / "results" / "csv" / "cpu_results.csv"
OUT_DIR = ROOT / "results" / "figures"


def main() -> None:
    if not IN_CSV.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {IN_CSV}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cpu_results = pd.read_csv(IN_CSV)

    for n in sorted(cpu_results["N"].unique()):
        df = cpu_results[cpu_results["N"] == n].copy()
        labels = df["version"] + "_T" + df["threads"].astype(str) + "_BS" + df["BS"].astype(str)

        plt.figure(figsize=(12, 5))
        plt.bar(labels, df["avg_time_s"])
        plt.xticks(rotation=90)
        plt.ylabel("Tempo médio (s)")
        plt.title(f"Tempo médio por versão — N={n}")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"cpu_tempo_N{n}.png", dpi=150)
        plt.close()

        plt.figure(figsize=(12, 5))
        plt.bar(labels, df["GFLOPS"])
        plt.xticks(rotation=90)
        plt.ylabel("GFLOP/s")
        plt.title(f"Desempenho por versão — N={n}")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"cpu_gflops_N{n}.png", dpi=150)
        plt.close()

    print(f"✅ Gráficos CPU salvos em: {OUT_DIR}")


if __name__ == "__main__":
    main()
