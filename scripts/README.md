# ⚙️ Scripts Auxiliares

Esta pasta contém scripts de apoio para automatizar experimentos e geração de gráficos.

## Arquivos

- `run_cpu_experiments.py`: executa as versões CPU e salva `results/csv/cpu_results.csv`.
- `run_gpu_experiments.py`: executa as versões CUDA e salva `results/csv/gpu_results.csv`.
- `plot_cpu_results.py`: gera gráficos de tempo e GFLOP/s da CPU.
- `plot_gpu_results.py`: gera gráficos de tempo de kernel e GFLOP/s da GPU.

## Observação

Os scripts assumem que os executáveis já foram compilados em `build/`.
