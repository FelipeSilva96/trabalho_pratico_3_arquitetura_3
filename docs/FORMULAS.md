# 🧮 Fórmulas do Trabalho

## Operações de ponto flutuante em GEMM

Para matrizes quadradas `N × N`:

```text
operações ≈ 2 × N³
```

## GFLOP/s

```text
GFLOP/s = (2 × N³) / (tempo_em_segundos × 10⁹)
```

## Speedup

```text
speedup = tempo_referência / tempo_otimizado
```

## Eficiência paralela

```text
eficiência = speedup / número_de_threads
```

## Tempo total em GPU

```text
tempo_total = tempo_host_para_device + tempo_kernel + tempo_device_para_host
```

## Observação

Para CUDA, é importante distinguir:

- desempenho considerando apenas o kernel;
- desempenho considerando kernel + cópias entre CPU e GPU.
