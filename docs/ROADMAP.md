# 🧭 Roadmap de Implementação — TP3

Este arquivo organiza a execução futura do trabalho em etapas progressivas.

## 1. Ambiente

- [ ] Testar `gcc --version`.
- [ ] Testar suporte a OpenMP.
- [ ] Testar `python3 --version`.
- [ ] Instalar `pandas` e `matplotlib`.
- [ ] Em ambiente GPU, testar `nvidia-smi`.
- [ ] Em ambiente GPU, testar `nvcc --version`.

## 2. CPU Sequencial

- [ ] Implementar `gemm_naive`.
- [ ] Compilar com `make cpu`.
- [ ] Executar com `N=128`.
- [ ] Verificar `max_abs_error`.

## 3. Localidade

- [ ] Implementar `gemm_transposed`.
- [ ] Comparar contra `naive`.
- [ ] Explicar ganho ou ausência de ganho no relatório.

## 4. Blocking / Tiling

- [ ] Implementar `gemm_blocked`.
- [ ] Testar `BS=16`.
- [ ] Testar `BS=32`.
- [ ] Testar `BS=64`.
- [ ] Discutir relação entre `BS` e cache.

## 5. OpenMP

- [ ] Implementar `gemm_openmp`.
- [ ] Testar 1 thread.
- [ ] Testar 2 threads.
- [ ] Testar 4 threads.
- [ ] Testar 8 threads.
- [ ] Calcular speedup.
- [ ] Calcular eficiência.

## 6. Blocking + OpenMP

- [ ] Implementar `gemm_blocked_openmp`.
- [ ] Comparar com `blocked`.
- [ ] Comparar com `openmp`.

## 7. CUDA

- [ ] Implementar `gemm_cuda_naive`.
- [ ] Implementar `gemm_cuda_tiled`.
- [ ] Medir `h2d_ms`.
- [ ] Medir `kernel_ms`.
- [ ] Medir `d2h_ms`.
- [ ] Comparar kernel isolado vs tempo total.

## 8. Relatório

- [ ] Descrever metodologia.
- [ ] Descrever hardware.
- [ ] Incluir tabelas.
- [ ] Incluir gráficos.
- [ ] Discutir resultados.
- [ ] Responder questões do enunciado.
- [ ] Concluir criticamente.
