# 🚀 Trabalho Prático 3 — Otimização Paralela de GEMM em CPU e GPU

<div align="center">

![Status](https://img.shields.io/badge/status-estrutura%20inicial-yellow)
![Disciplina](https://img.shields.io/badge/disciplina-Arquitetura%20de%20Computadores%20III-blue)
![Tema](https://img.shields.io/badge/tema-CPU%20%7C%20OpenMP%20%7C%20SIMD%20%7C%20CUDA-purple)
![Linguagens](https://img.shields.io/badge/linguagens-C%20%7C%20CUDA%20%7C%20Python-green)

</div>

---

## 📌 Visão Geral

Este repositório contém a **estrutura inicial** do **Trabalho Prático 3 de Arquitetura de Computadores III**, cujo tema é a **otimização paralela da multiplicação de matrizes densas**, também conhecida como **GEMM** (*General Matrix-Matrix Multiplication*).

O objetivo do trabalho é partir de uma implementação inicial simples de multiplicação de matrizes e evoluir progressivamente para versões mais eficientes, explorando conceitos centrais de **hierarquia de memória**, **localidade de dados**, **vetorização SIMD**, **paralelismo em CPU com OpenMP** e **aceleração em GPU com CUDA**.

A operação matemática estudada é:

```text
C = A × B
```

Para matrizes quadradas de dimensão `N × N`, o custo computacional aproximado da multiplicação é:

```text
2 × N³ operações de ponto flutuante
```

Assim, o desempenho pode ser medido em **GFLOP/s** pela fórmula:

```text
GFLOP/s = (2 × N³) / (tempo_em_segundos × 10⁹)
```

---

## ⚠️ Status Atual do Repositório

> **Este repositório está apenas com a estrutura inicial do trabalho.**

Neste momento, o objetivo **não é entregar a implementação completa**, mas deixar o projeto organizado para desenvolvimento posterior.

✅ Já foi preparado:

- `README.md` completo e documentado;
- organização inicial de pastas;
- cópia do notebook do enunciado;
- extração dos códigos-base úteis do notebook;
- templates de CPU e CUDA com os `TODOs` originais;
- scripts auxiliares para futura execução dos experimentos;
- estrutura para guardar resultados, gráficos, tabelas e relatório;
- checklist do que deverá ser implementado;
- documentação inicial do plano de execução.

❌ Ainda **não** foi implementado:

- função `gemm_naive`;
- função `gemm_transposed`;
- função `gemm_blocked`;
- função `gemm_openmp`;
- função `gemm_blocked_openmp`;
- kernel `gemm_cuda_naive`;
- kernel `gemm_cuda_tiled`;
- experimentos reais;
- tabelas finais;
- gráficos finais;
- relatório final.

---

## 🎯 Objetivo do Trabalho

O trabalho busca estudar, na prática, como diferentes estratégias arquiteturais e de programação afetam o desempenho de uma aplicação computacional intensiva.

A multiplicação de matrizes é um excelente estudo de caso porque:

- possui custo computacional elevado;
- realiza muitos acessos à memória;
- expõe claramente problemas de localidade;
- permite otimizações por reordenação de laços;
- pode ser vetorizada por compiladores modernos;
- pode ser paralelizada com múltiplas threads;
- pode ser acelerada por GPUs usando milhares de threads;
- permite comparar CPU e GPU de forma quantitativa.

O trabalho também conecta diretamente vários tópicos estudados em Arquitetura de Computadores III, como:

- hierarquia de memória;
- cache;
- localidade temporal;
- localidade espacial;
- miss de cache;
- largura de banda de memória;
- paralelismo em nível de dados;
- paralelismo em nível de threads;
- SIMD;
- OpenMP;
- CUDA;
- GPU como acelerador;
- speedup;
- eficiência paralela;
- custo de comunicação CPU-GPU.

---

## 🧠 Conceitos Envolvidos

### 🧮 GEMM — Multiplicação Geral de Matrizes

GEMM é uma das operações mais importantes da computação científica e de alto desempenho. Ela aparece em:

- aprendizado de máquina;
- redes neurais profundas;
- simulações físicas;
- computação gráfica;
- álgebra linear numérica;
- processamento de imagens;
- modelagem científica;
- aplicações de engenharia.

Neste trabalho, será considerada a forma simplificada:

```text
C[i][j] = soma, para k = 0 até N-1, de A[i][k] × B[k][j]
```

A versão ingênua normalmente utiliza três laços aninhados. Embora seja simples, ela pode apresentar desempenho ruim por não aproveitar adequadamente a cache, especialmente no acesso à matriz `B`.

---

### 🧊 Localidade de Memória

A hierarquia de memória só funciona bem porque programas reais apresentam **localidade**.

- **Localidade temporal:** se um dado foi usado agora, é provável que seja usado novamente em breve.
- **Localidade espacial:** se um dado foi acessado, dados próximos a ele também podem ser acessados em breve.

Na multiplicação de matrizes, a forma como os laços são organizados influencia diretamente o padrão de acesso à memória.

Uma implementação ingênua pode acessar linhas de `A` de forma eficiente, mas acessar colunas de `B` com saltos grandes na memória, prejudicando a localidade espacial.

---

### 🔁 Transposição da Matriz B

A versão `transposed` pretende melhorar o acesso à matriz `B`.

Em vez de acessar `B[k][j]`, que percorre a matriz por colunas, é criada uma matriz transposta `BT`, permitindo acessar os elementos de forma mais contígua.

A ideia central é transformar um acesso ruim para cache em um acesso mais favorável à hierarquia de memória.

---

### 🧱 Blocagem / Tiling

A versão `blocked` utiliza a técnica de **blocagem**, também chamada de **tiling**.

A ideia é dividir as matrizes em blocos menores para aumentar a reutilização dos dados na cache.

Em vez de tentar multiplicar matrizes inteiras diretamente, o algoritmo trabalha com submatrizes. Isso tende a reduzir o número de cache misses e melhorar o aproveitamento da memória rápida.

O tamanho do bloco `BS` é um parâmetro importante do experimento.

---

### ⚙️ Vetorização SIMD

SIMD significa **Single Instruction, Multiple Data**.

Nesse modelo, uma única instrução opera sobre vários dados ao mesmo tempo. Em processadores modernos, isso pode ocorrer por meio de extensões como SSE, AVX e AVX2/AVX-512, dependendo da CPU.

Neste trabalho, a vetorização será analisada principalmente pelo relatório do compilador, usando a flag:

```bash
-fopt-info-vec-optimized
```

O objetivo é verificar se o compilador conseguiu transformar alguns laços em instruções vetoriais.

---

### 🧵 OpenMP

OpenMP é uma API para programação paralela em memória compartilhada.

Neste trabalho, ela será usada para paralelizar versões da multiplicação de matrizes em CPU.

As versões previstas são:

- `openmp`;
- `blocked_openmp`.

A análise deve comparar diferentes números de threads, como:

```text
1, 2, 4 e 8 threads
```

O desempenho esperado depende de fatores como:

- número de núcleos físicos;
- número de threads lógicas;
- balanceamento de carga;
- overhead de criação/sincronização de threads;
- contenção de memória;
- tamanho do problema;
- comportamento da cache.

---

### 🎮 CUDA e GPU

CUDA será usada para implementar duas versões em GPU:

- `cuda_naive`: uma thread calcula um elemento de `C`;
- `cuda_tiled`: usa memória compartilhada para melhorar a reutilização dos dados.

A GPU será avaliada separando:

1. tempo de cópia **host → device**;
2. tempo do **kernel**;
3. tempo de cópia **device → host**;
4. tempo total.

Essa separação é essencial porque uma GPU pode ser muito rápida no kernel, mas o custo de transferência de dados entre CPU e GPU pode reduzir ou até eliminar o ganho total.

---

## 📦 Estrutura do Projeto

```text
trabalho_pratico_3_arquitetura_3/
├── README.md
├── Enunciado.ipynb
├── Makefile
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── README.md
│   ├── cpu/
│   │   ├── README.md
│   │   └── gemm_cpu.c
│   │
│   └── cuda/
│       ├── README.md
│       └── gemm_cuda.cu
│
├── scripts/
│   ├── README.md
│   ├── run_cpu_experiments.py
│   ├── run_gpu_experiments.py
│   ├── plot_cpu_results.py
│   └── plot_gpu_results.py
│
├── docs/
│   ├── ENUNCIADO_RESUMO.md
│   ├── ROADMAP.md
│   ├── CHECKLIST.md
│   └── FORMULAS.md
│
├── report/
│   ├── README.md
│   └── relatorio_template.md
│
├── results/
│   ├── README.md
│   ├── csv/
│   ├── figures/
│   └── tables/
│
├── notebooks/
│   ├── README.md
│   └── Enunciado_original.ipynb
│
├── assets/
│   └── .gitkeep
│
└── build/
    └── .gitkeep
```

---

## 📁 Descrição dos Arquivos e Pastas

### `README.md`

Documento principal do projeto. Explica o objetivo, a estrutura, os conceitos envolvidos, o plano de implementação, os comandos de execução e os entregáveis esperados.

### `Enunciado.ipynb`

Notebook original do enunciado do trabalho. Ele contém:

- descrição geral do trabalho;
- código-base em C;
- código-base em CUDA;
- células de compilação;
- células de execução dos experimentos;
- células de geração de gráficos;
- tabela final de comparação;
- questões para discussão;
- critérios de avaliação.

### `src/cpu/gemm_cpu.c`

Código-base em C extraído do notebook. Contém as funções que deverão ser implementadas futuramente:

```c
void gemm_naive(int N, const float *A, const float *B, float *C);
void gemm_transposed(int N, const float *A, const float *BT, float *C);
void gemm_blocked(int N, int BS, const float *A, const float *B, float *C);
void gemm_openmp(int N, const float *A, const float *B, float *C);
void gemm_blocked_openmp(int N, int BS, const float *A, const float *B, float *C);
```

### `src/cuda/gemm_cuda.cu`

Código-base em CUDA extraído do notebook. Contém os kernels que deverão ser implementados futuramente:

```cuda
__global__ void gemm_cuda_naive(int N, const float *A, const float *B, float *C);
__global__ void gemm_cuda_tiled(int N, const float *A, const float *B, float *C);
```

### `scripts/`

Contém scripts auxiliares para automatizar experimentos e gerar gráficos.

Eles foram preparados como ponto de partida e podem ser adaptados ao longo do desenvolvimento.

### `docs/`

Contém documentação auxiliar:

- resumo técnico do enunciado;
- roadmap de implementação;
- checklist de tarefas;
- fórmulas usadas no relatório.

### `report/`

Contém uma estrutura inicial para o relatório final.

### `results/`

Pasta reservada para armazenar resultados experimentais:

- arquivos `.csv`;
- tabelas;
- gráficos;
- figuras usadas no relatório.

### `build/`

Pasta reservada para arquivos compilados.

---

## 🛠️ Ferramentas Necessárias

### CPU

Para compilar e executar a parte CPU:

- GCC;
- OpenMP;
- Python 3;
- pandas;
- matplotlib.

Em sistemas Ubuntu/Debian:

```bash
sudo apt update
sudo apt install build-essential python3 python3-pip
pip install -r requirements.txt
```

Verificação:

```bash
gcc --version
python3 --version
```

---

### GPU / CUDA

Para a parte CUDA, será necessário:

- GPU NVIDIA compatível;
- driver NVIDIA;
- CUDA Toolkit;
- compilador `nvcc`.

Verificação:

```bash
nvidia-smi
nvcc --version
```

No Google Colab, é necessário ativar GPU em:

```text
Ambiente de execução → Alterar tipo de ambiente de execução → GPU
```

---

## ▶️ Como Compilar

### Compilar versão CPU

```bash
make cpu
```

Comando equivalente:

```bash
gcc src/cpu/gemm_cpu.c -O3 -march=native -fopenmp -fopt-info-vec-optimized -o build/gemm_cpu 2> results/tables/vec_report.txt
```

---

### Compilar versão CUDA

```bash
make cuda
```

Comando equivalente:

```bash
nvcc src/cuda/gemm_cuda.cu -O3 -o build/gemm_cuda
```

---

### Compilar tudo

```bash
make all
```

---

## 🧪 Como Executar Futuramente

> Os comandos abaixo são previstos para depois que as funções `TODO` forem implementadas.

### Executar versão CPU manualmente

```bash
./build/gemm_cpu naive 128 5
./build/gemm_cpu transposed 128 5
./build/gemm_cpu blocked 128 5 32
OMP_NUM_THREADS=4 ./build/gemm_cpu openmp 128 5
OMP_NUM_THREADS=4 ./build/gemm_cpu blocked_openmp 128 5 32
```

### Executar experimentos CPU automatizados

```bash
python3 scripts/run_cpu_experiments.py
```

Saída esperada:

```text
results/csv/cpu_results.csv
```

---

### Executar versão CUDA manualmente

```bash
./build/gemm_cuda naive 128 5
./build/gemm_cuda tiled 128 5
```

### Executar experimentos GPU automatizados

```bash
python3 scripts/run_gpu_experiments.py
```

Saída esperada:

```text
results/csv/gpu_results.csv
```

---

### Gerar gráficos

```bash
python3 scripts/plot_cpu_results.py
python3 scripts/plot_gpu_results.py
```

Saídas esperadas:

```text
results/figures/
```

---

## 📊 Métricas que Serão Avaliadas

### Tempo Médio

Cada experimento deve ser executado pelo menos **5 vezes**.

A métrica principal de tempo será a média:

```text
tempo_médio = soma_dos_tempos / número_de_repetições
```

---

### GFLOP/s

Medida de desempenho bruto em bilhões de operações de ponto flutuante por segundo:

```text
GFLOP/s = (2 × N³) / (tempo × 10⁹)
```

---

### Speedup

Compara uma versão otimizada com uma versão de referência:

```text
speedup = tempo_referência / tempo_otimizado
```

Normalmente, a referência inicial será a versão `CPU naive`.

---

### Eficiência Paralela

Para versões paralelas em CPU:

```text
eficiência = speedup / número_de_threads
```

A eficiência mostra quanto do potencial teórico das threads foi realmente aproveitado.

---

## 🧾 Versões que Serão Implementadas

| Versão | Plataforma | Ideia Principal | Status |
|---|---|---|---|
| `naive` | CPU | Três laços simples | ⏳ Pendente |
| `transposed` | CPU | Melhorar localidade de `B` | ⏳ Pendente |
| `blocked` | CPU | Usar blocagem/tiling para cache | ⏳ Pendente |
| `openmp` | CPU | Paralelizar com threads | ⏳ Pendente |
| `blocked_openmp` | CPU | Combinar cache blocking + OpenMP | ⏳ Pendente |
| `cuda_naive` | GPU | Uma thread por elemento de `C` | ⏳ Pendente |
| `cuda_tiled` | GPU | Usar memória compartilhada | ⏳ Pendente |

---

## 🧭 Plano de Desenvolvimento

### Etapa 1 — Preparação

- [x] Organizar estrutura do repositório.
- [x] Criar README completo.
- [x] Copiar notebook do enunciado.
- [x] Extrair códigos-base úteis.
- [x] Criar scripts auxiliares.
- [ ] Conferir ambiente de CPU.
- [ ] Conferir ambiente CUDA/Colab.

### Etapa 2 — Implementação CPU

- [ ] Implementar `gemm_naive`.
- [ ] Validar versão naive.
- [ ] Implementar `gemm_transposed`.
- [ ] Implementar `gemm_blocked`.
- [ ] Testar diferentes valores de `BS`.
- [ ] Implementar `gemm_openmp`.
- [ ] Implementar `gemm_blocked_openmp`.
- [ ] Testar com diferentes números de threads.

### Etapa 3 — Análise SIMD

- [ ] Compilar com `-O3`.
- [ ] Compilar com `-march=native`.
- [ ] Gerar relatório de vetorização.
- [ ] Identificar quais laços foram vetorizados.
- [ ] Relacionar vetorização com desempenho observado.

### Etapa 4 — Implementação CUDA

- [ ] Implementar `gemm_cuda_naive`.
- [ ] Validar resultado com CPU.
- [ ] Implementar `gemm_cuda_tiled`.
- [ ] Usar memória compartilhada.
- [ ] Medir tempo de cópia e tempo de kernel separadamente.

### Etapa 5 — Experimentos

- [ ] Executar cada experimento pelo menos 5 vezes.
- [ ] Salvar resultados em CSV.
- [ ] Gerar gráficos de tempo.
- [ ] Gerar gráficos de GFLOP/s.
- [ ] Comparar CPU vs GPU.
- [ ] Comparar kernel GPU vs tempo total GPU.

### Etapa 6 — Relatório

- [ ] Descrever metodologia experimental.
- [ ] Informar hardware usado.
- [ ] Apresentar tabelas.
- [ ] Apresentar gráficos.
- [ ] Calcular GFLOP/s.
- [ ] Calcular speedup.
- [ ] Calcular eficiência.
- [ ] Responder às questões do enunciado.
- [ ] Escrever conclusão crítica.

---

## ❓ Questões que Deverão Ser Respondidas no Relatório

O enunciado pede discussão técnica sobre os resultados. Entre as perguntas principais estão:

1. Por que a versão ingênua apresenta baixo desempenho?
2. O impacto da transposição de `B` foi significativo? Por quê?
3. Qual tamanho de bloco apresentou melhor desempenho? Por quê?
4. O compilador conseguiu vetorizar alguma parte do código?
5. O speedup com OpenMP foi próximo do ideal?
6. A eficiência diminuiu com mais threads?
7. A GPU foi mais rápida considerando apenas o kernel?
8. A GPU foi mais rápida considerando o tempo total com cópias?
9. Qual foi a principal limitação observada: computação, memória, comunicação ou overhead?
10. Quais cuidados são necessários ao usar o Google Colab para avaliar desempenho?

---

## 📈 Tabela Final Esperada

Ao final do trabalho, uma tabela como esta deverá ser preenchida:

| Versão | N | Tempo médio | GFLOP/s | Speedup | Eficiência | Observações |
|---|---:|---:|---:|---:|---:|---|
| CPU naive |  |  |  | 1,0 | 1,0 | Referência |
| CPU transposed |  |  |  |  |  | Localidade |
| CPU blocked |  |  |  |  |  | Cache |
| CPU OpenMP 2 threads |  |  |  |  |  | Threads |
| CPU OpenMP 4 threads |  |  |  |  |  | Threads |
| CUDA naive |  |  |  |  |  | Kernel simples |
| CUDA tiled |  |  |  |  |  | Memória compartilhada |

---

## 🧪 Critérios de Avaliação

### Implementação — 5 pontos

| Critério | Pontos |
|---|---:|
| Código sequencial correto e validação dos resultados | 1,0 |
| Otimização de localidade / transposição / blocagem | 1,0 |
| Análise de vetorização SIMD | 1,0 |
| Paralelização com OpenMP | 1,0 |
| Implementação CUDA com medição separada de cópia e kernel | 1,0 |

### Relatório — 5 pontos

| Critério | Pontos |
|---|---:|
| Metodologia experimental clara | 1,0 |
| Tabelas e gráficos bem apresentados | 1,0 |
| Cálculo correto de GFLOP/s, speedup e eficiência | 1,0 |
| Discussão técnica dos resultados | 1,5 |
| Conclusão crítica sobre limitações e trade-offs | 0,5 |

---

## 🔍 Observações Importantes

### Sobre o uso de bibliotecas prontas

O enunciado informa que **não devem ser usadas BLAS, cuBLAS, Eigen ou bibliotecas equivalentes na parte principal**.

Essas bibliotecas podem ser utilizadas apenas como comparação bônus, caso desejado e permitido.

---

### Sobre repetição dos experimentos

Cada experimento deve ser executado pelo menos **5 vezes**.

A média deve ser usada para reduzir ruído experimental.

---

### Sobre validação

Todas as versões devem ser comparadas com uma versão sequencial de referência.

A validação usa erro máximo absoluto:

```text
max_abs_error
```

Isso é importante porque otimizações de desempenho não podem alterar o resultado matemático esperado.

---

### Sobre Google Colab

O Google Colab é útil para executar CUDA, mas pode não ser ideal para testes de CPU porque normalmente oferece poucos núcleos de CPU e ambiente compartilhado.

Por isso, a recomendação do enunciado é:

- usar Colab para GPU;
- usar máquina local para testes CPU, quando possível.

---

## 🧠 Relação com Arquitetura de Computadores III

Este trabalho não é apenas um exercício de programação. Ele é uma aplicação prática de conceitos de arquitetura.

A comparação entre as versões permite observar como decisões de hardware e software afetam o desempenho:

- a versão ingênua evidencia custo de memória e baixa localidade;
- a transposição mostra a importância da organização dos dados;
- a blocagem mostra a relação entre algoritmo e cache;
- a vetorização mostra paralelismo de dados em CPU;
- OpenMP mostra paralelismo de threads em memória compartilhada;
- CUDA mostra paralelismo massivo em GPU;
- a medição separada de cópia e kernel mostra o custo de comunicação em arquiteturas heterogêneas.

---

## 🚧 Próximos Passos

1. Conferir se o código-base compila.
2. Implementar a versão `naive` em CPU.
3. Validar corretamente a referência.
4. Implementar uma otimização por vez.
5. Medir desempenho após cada alteração.
6. Salvar resultados em CSV.
7. Gerar gráficos.
8. Construir análise técnica.
9. Finalizar relatório.
10. Exportar notebook preenchido em PDF.

---

## 📚 Licença e Uso

Projeto acadêmico desenvolvido para a disciplina **Arquitetura de Computadores III**.

Uso destinado a estudo, implementação, experimentação e documentação dos conceitos de processamento paralelo em CPU e GPU.
