# 🧮 Código CPU — GEMM em C

Arquivo principal:

```text
gemm_cpu.c
```

Este arquivo contém o código-base em C para as versões CPU do trabalho.

## Funções pendentes

- `gemm_naive`
- `gemm_transposed`
- `gemm_blocked`
- `gemm_openmp`
- `gemm_blocked_openmp`

## Compilação prevista

```bash
gcc gemm_cpu.c -O3 -march=native -fopenmp -fopt-info-vec-optimized -o gemm_cpu 2> vec_report.txt
```
