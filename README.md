# 🚀 Trabalho Prático 3 — Arquitetura de Computadores III

## GEMM em CPU e GPU

Este repositório contém o material do **Trabalho Prático 3 de Arquitetura de Computadores III**, cujo tema é a análise e otimização da multiplicação de matrizes densas, também chamada de **GEMM** (*General Matrix-Matrix Multiplication*).

A ideia principal do trabalho é comparar diferentes formas de calcular:

```text
C = A × B
```

usando versões sequenciais, otimizações de localidade de memória, paralelismo em CPU com OpenMP e execução em GPU com CUDA.

---

## 📌 Observação importante sobre o repositório

A parte principal que foi trabalhada neste projeto está concentrada principalmente nas pastas:

```text
docs/
notebooks/
```

As demais pastas, como `src/`, `scripts/`, `results/`, `report/`, `assets/` e `build/`, fazem parte da estrutura inicial do projeto ou servem como arquivos auxiliares/base do enunciado. Elas foram mantidas no repositório para preservar a organização original, mas não são o foco principal da versão final documentada aqui.

Em resumo:

* `docs/` contém a documentação auxiliar do trabalho;
* `notebooks/` contém a documentação principal e o material usado para explicar as implementações;
* as demais pastas funcionam principalmente como apoio, template ou estrutura inicial.

---

## 📁 Pastas principais alteradas

### `docs/`

A pasta `docs/` reúne arquivos de apoio usados para organizar e explicar o trabalho.

Nela estão materiais como:

* resumo técnico do enunciado;
* checklist geral do trabalho;
* roadmap de implementação;
* fórmulas usadas nas análises.

Essa pasta ajuda a entender o que o trabalho pedia, quais versões deveriam ser comparadas e quais métricas seriam usadas, como tempo de execução, GFLOP/s, speedup e eficiência.

---

### `notebooks/`

A pasta `notebooks/` é a parte mais importante para compreender o desenvolvimento do trabalho.

Ela contém a documentação principal das implementações e das estratégias usadas, incluindo explicações sobre:

* GEMM sequencial;
* GEMM com matriz transposta;
* GEMM com blocagem/tiling;
* GEMM com OpenMP;
* GEMM com blocagem + OpenMP;
* GEMM em CUDA;
* GEMM CUDA com tiling e memória compartilhada.

Essa pasta deve ser usada como referência principal para entender a lógica do trabalho e as decisões tomadas.

---

## 🧠 O que foi estudado no trabalho

O trabalho analisa como diferentes estratégias afetam o desempenho da multiplicação de matrizes.

Foram considerados principalmente os seguintes pontos:

* custo computacional da multiplicação de matrizes;
* localidade espacial e temporal;
* impacto da cache;
* uso de matriz transposta para melhorar acesso à memória;
* blocagem/tiling para reaproveitamento de dados;
* paralelismo em CPU com OpenMP;
* execução em GPU com CUDA;
* diferença entre tempo de kernel e tempo total em GPU;
* comparação de desempenho por tempo, GFLOP/s e speedup.

---

## 🧮 Fórmulas principais

Para matrizes quadradas de tamanho `N × N`, a quantidade aproximada de operações é:

```text
operações ≈ 2 × N³
```

O desempenho em GFLOP/s é calculado por:

```text
GFLOP/s = (2 × N³) / (tempo_em_segundos × 10⁹)
```

O speedup é calculado por:

```text
speedup = tempo_referência / tempo_otimizado
```

E, para GPU, o tempo total considera:

```text
tempo_total = tempo_host_para_device + tempo_kernel + tempo_device_para_host
```

---

## 📦 Estrutura resumida do repositório

```text
trabalho_pratico_3_arquitetura_3/
│
├── docs/              # Documentação auxiliar do trabalho
├── notebooks/         # Documentação principal e explicação das implementações
├── src/               # Código-base/template do enunciado
├── scripts/           # Scripts auxiliares
├── results/           # Estrutura para resultados
├── report/            # Estrutura inicial para relatório
├── assets/            # Arquivos auxiliares
├── build/             # Pasta de compilação/saída
├── Enunciado.ipynb    # Notebook original do enunciado
└── README.md          # Este arquivo
```

---

## 👥 Integrantes

* Felipe Pereira da Silva
* Rikerson Antônio de Freitas
* Mateus Resende Ottoni
* Kauan Gabriel Pereira
* Diego Feitosa

---

## ✅ Resumo final

Este README foi atualizado para deixar o repositório mais claro e objetivo.

O ponto principal é que o trabalho deve ser entendido principalmente pelas pastas `docs/` e `notebooks/`, pois elas concentram a documentação, a explicação das estratégias e a organização final do que foi desenvolvido.

As outras pastas foram mantidas porque fazem parte da estrutura do projeto, mas não representam necessariamente a parte principal alterada pelo grupo.
