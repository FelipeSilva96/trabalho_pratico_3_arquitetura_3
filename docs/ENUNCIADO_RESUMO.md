# 📘 Resumo Técnico do Enunciado

O trabalho propõe a otimização progressiva da multiplicação de matrizes densas `C = A × B`.

## Partes do enunciado

1. Configuração do ambiente.
2. Código C base para CPU.
3. Compilação com `gcc`, `-O3`, `-march=native`, OpenMP e relatório de vetorização.
4. Execução automatizada dos experimentos CPU.
5. Geração de gráficos CPU.
6. Código CUDA base.
7. Compilação CUDA com `nvcc`.
8. Execução automatizada dos experimentos GPU.
9. Geração de gráficos GPU.
10. Tabela final de comparação.
11. Questões para discussão.
12. Critérios de avaliação.

## Versões CPU previstas

- `naive`
- `transposed`
- `blocked`
- `openmp`
- `blocked_openmp`

## Versões GPU previstas

- `cuda_naive`
- `cuda_tiled`

## Regras principais

- Não usar BLAS, cuBLAS, Eigen ou equivalentes na parte principal.
- Executar cada experimento pelo menos 5 vezes.
- Usar média dos tempos.
- Validar todas as versões comparando com referência sequencial.
- Usar Colab para GPU, se necessário.
- Preferir máquina local para CPU quando possível.

## Código útil identificado

O notebook contém dois blocos de código diretamente úteis como template:

- programa C completo com funções `TODO` para CPU;
- programa CUDA completo com kernels `TODO` para GPU.

Esses arquivos foram extraídos para:

```text
src/cpu/gemm_cpu.c
src/cuda/gemm_cuda.cu
```
